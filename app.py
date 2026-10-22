"""
Main Flask application entry point.

Wires together every module across all phases as Flask routes:
- Vehicle Detection: detection/vehicle_detector.py (COCO YOLO)
- Custom Emergency Vehicle Detection: detection/emergency_detector.py (Fine-Tuned YOLO)
- Traffic Data Science: traffic/density.py, traffic/demand.py
- Green-Time ML: ml/predictor.py
- Siren Audio ML: audio/siren_detector.py
- Traffic Signal & Emergency Controller: traffic/signal_controller.py, traffic/emergency_controller.py
- Comparative Analytics: traffic/comparison.py

Runtime data: in-memory state dict keeps the latest analysis results and emergency event log.
"""

import os
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
)

import uuid

from utils.config import (
    APP_NAME, DEBUG_MODE, SECRET_KEY, UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
    ROADS, EMERGENCY_VEHICLE_TYPES, TRAFFIC_MODEL_PATH, SIREN_MODEL_PATH,
    TRAFFIC_METRICS_PATH, ALLOWED_AUDIO_EXTENSIONS, DATA_FOLDER, FIXED_SIGNAL_TIME,
    WAITING_TIME_PER_CYCLE, MAX_SIMULATED_WAITING_TIME, EMERGENCY_MODEL_PATH,
    EMERGENCY_CONFIDENCE_THRESHOLD, EMERGENCY_CLASSES, MIN_GREEN, MAX_GREEN,
    SAFE_FALLBACK_GREEN_TIME, YELLOW_LIGHT_SECONDS, get_max_green_time,
    SIREN_CONFIDENCE_THRESHOLD, EMERGENCY_GREEN_SECONDS,
    SIREN_MODEL_V2_PATH, SIREN_V2_CONFIDENCE_THRESHOLD, SIREN_ACTIVE_VERSION,
    get_siren_model_v2_path,
)
from utils.validators import is_allowed_image, save_uploaded_image
from detection.vehicle_detector import VehicleDetector
from detection.emergency_detector import EmergencyDetector
from detection.image_processor import draw_detections
from traffic.density import calculate_density
from traffic.demand import calculate_demand
from traffic.timing import clamp_green_time, get_safe_fallback_green_time
from traffic.signal_controller import build_cycle_plan
from traffic.emergency_controller import (
    is_emergency_active, build_emergency_plan, evaluate_emergency_candidate,
    resolve_emergency_priority,
)
import io
from traffic.comparison import compare_fixed_vs_adaptive, evaluate_traffic_impact

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# In-memory runtime state
# ---------------------------------------------------------------------------
STATE = {
    "latest_analysis": None,   # dict: per-road counts/density/demand/green_time/emergency
    "cycle_number": 0,
    "emergency_log": [],       # list of dicts: {time, vehicle, road, confidence, siren, action}
}

# ---------------------------------------------------------------------------
# Pre-initialize detectors once at application startup (singleton pattern)
# ---------------------------------------------------------------------------
try:
    vehicle_detector = VehicleDetector()
except Exception as e:
    vehicle_detector = None
    app.logger.warning(f"Default VehicleDetector could not be initialized at startup: {e}")

try:
    emergency_detector = EmergencyDetector()
except Exception as e:
    emergency_detector = None
    app.logger.warning(f"Custom EmergencyDetector could not be initialized at startup: {e}")


@app.errorhandler(413)
def file_too_large(_err):
    flash("Upload too large. Please use smaller images (total limit is set in utils/config.py).")
    return redirect(url_for("index"))


def _ml_available() -> bool:
    return os.path.exists(TRAFFIC_MODEL_PATH)


def _siren_model_available() -> bool:
    return os.path.exists(SIREN_MODEL_PATH)


def _siren_model_v2_available() -> bool:
    path = get_siren_model_v2_path()
    return bool(path and os.path.exists(path))


def _emergency_model_available() -> bool:
    return EmergencyDetector.is_available()


# ---------------------------------------------------------------------------
# Page 1 — Upload
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        app_name=APP_NAME,
        roads=ROADS,
        emergency_types=EMERGENCY_VEHICLE_TYPES,
        ml_available=_ml_available(),
        siren_model_available=_siren_model_available(),
        siren_model_v2_available=_siren_model_v2_available(),
        siren_active_version=SIREN_ACTIVE_VERSION,
        emergency_model_available=_emergency_model_available(),
        emergency_model_info=EmergencyDetector.get_model_info(),
    )


def execute_analysis(files, form):
    """
    Executes the complete traffic analysis and emergency prioritization pipeline.

    1. Validates that all four road images (North, East, South, West) are provided and valid.
    2. Validates optional emergency siren audio file (.wav) and associates it with the chosen direction.
    3. Runs YOLOv8 vehicle detection and Custom Emergency YOLO detection on all directions.
    4. Evaluates emergency priority with the strict AND-condition.
    5. Calculates density, demand, and ML adaptive green times constrained by [min_green, max_green].
    6. Generates visual detection previews and builds simulation plans.
    """
    if not _ml_available():
        return None, ("Traffic ML model not found. Run: python -m ml.train_traffic_model", 500)

    from ml.predictor import GreenTimePredictor
    predictor = GreenTimePredictor()

    siren_version_requested = form.get("siren_version", form.get("siren_model_version", SIREN_ACTIVE_VERSION)).lower()
    siren_detector = None
    siren_version_used = "v1"

    if siren_version_requested == "v2" and _siren_model_v2_available():
        from audio.siren_detector_v2 import SirenDetectorV2
        try:
            siren_detector = SirenDetectorV2()
            siren_version_used = "v2"
        except Exception as e:
            app.logger.warning(f"Could not initialize SirenDetectorV2, falling back to V1: {e}")
            if _siren_model_available():
                from audio.siren_detector import SirenDetector
                try:
                    siren_detector = SirenDetector()
                    siren_version_used = "v1"
                except Exception as ex:
                    app.logger.warning(f"Could not initialize fallback SirenDetector: {ex}")
    elif _siren_model_available():
        from audio.siren_detector import SirenDetector
        try:
            siren_detector = SirenDetector()
            siren_version_used = "v1"
        except Exception as e:
            app.logger.warning(f"Could not initialize SirenDetector: {e}")
            if _siren_model_v2_available():
                from audio.siren_detector_v2 import SirenDetectorV2
                try:
                    siren_detector = SirenDetectorV2()
                    siren_version_used = "v2"
                except Exception as ex:
                    app.logger.warning(f"Could not initialize fallback SirenDetectorV2: {ex}")

    global vehicle_detector, emergency_detector
    if vehicle_detector is None:
        try:
            vehicle_detector = VehicleDetector()
        except Exception as e:
            app.logger.exception("General YOLO model could not be loaded")
            return None, ("General YOLO model unavailable. Connect to the internet once so weights can download, then retry.", 500)

    if emergency_detector is None and _emergency_model_available():
        try:
            emergency_detector = EmergencyDetector()
        except Exception as e:
            app.logger.warning(f"Emergency detector could not be loaded: {e}")

    # Validate all 4 images
    saved_filenames = {}
    missing_roads = []
    for road in ROADS:
        file = files.get(road.lower()) or files.get(road)
        if file is None or file.filename == "":
            missing_roads.append(road)
        elif not is_allowed_image(file.filename):
            return None, (f"Invalid image format for {road} direction. Allowed formats: JPG, JPEG, PNG.", 400)
        else:
            saved_filenames[road] = save_uploaded_image(file, road)

    if missing_roads:
        missing_str = ", ".join(missing_roads)
        return None, (f"Missing image for direction: {missing_str}. All four direction images (North, East, South, West) are required before analysis.", 400)

    # Process optional siren audio
    # The siren audio represents an omnidirectional acoustic sensor for the intersection.
    audio_file = files.get("siren_audio") or files.get("audio") or files.get("audio_file")
    if not audio_file or audio_file.filename == "":
        for r in ROADS:
            candidate_audio = files.get(f"audio_{r.lower()}")
            if candidate_audio and candidate_audio.filename != "":
                audio_file = candidate_audio
                break

    siren_analysis = {
        "provided": False,
        "filename": None,
        "direction": "Omnidirectional",
        "is_siren": False,
        "confidence": None,
        "predicted_class": None,
        "probabilities": None,
        "siren_type": None,
        "model_version": siren_version_used,
        "error": None,
    }

    if audio_file and audio_file.filename != "":
        if not audio_file.filename.lower().endswith(tuple(ALLOWED_AUDIO_EXTENSIONS)):
            return None, ("Invalid audio format. Emergency siren audio must be a WAV file (.wav).", 400)

        ext = audio_file.filename.rsplit(".", 1)[1].lower()
        unique_audio_name = f"siren_audio_{uuid.uuid4().hex[:6]}.{ext}"
        audio_path = os.path.join(UPLOAD_FOLDER, unique_audio_name)
        audio_file.save(audio_path)
        siren_analysis["provided"] = True
        siren_analysis["filename"] = unique_audio_name

        if siren_detector is not None:
            try:
                siren_pred = siren_detector.predict(audio_path)
                siren_analysis["is_siren"] = bool(siren_pred.get("is_siren", False))
                siren_analysis["confidence"] = siren_pred.get("confidence", 0.0)
                siren_analysis["predicted_class"] = siren_pred.get("predicted_class")
                siren_analysis["probabilities"] = siren_pred.get("probabilities")
                siren_analysis["siren_type"] = siren_pred.get("siren_type")
                siren_analysis["model_version"] = siren_pred.get("model_version", siren_version_used)
                if siren_pred.get("error"):
                    siren_analysis["error"] = siren_pred["error"]
            except Exception as e:
                app.logger.exception(f"Siren detection inference failed: {e}")
                siren_analysis["error"] = f"Acoustic analysis failed: {e}"
        else:
            siren_analysis["error"] = "Siren classification model unavailable."

    STATE["cycle_number"] += 1
    cycle_number = STATE["cycle_number"]

    analysis_results = {}
    green_times = {}
    counts_by_road = {}
    road_candidates = {}

    for road in ROADS:
        filename = saved_filenames[road]
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        processed_filename = f"processed_{filename}"
        max_green = get_max_green_time(road)
        min_green = MIN_GREEN
        safe_fallback = get_safe_fallback_green_time(max_green, min_green)

        # 1. Computer Vision: General Traffic (COCO YOLO)
        try:
            detections = vehicle_detector.detect(image_path)
            counts = VehicleDetector.count_by_type(detections)
        except Exception as e:
            app.logger.exception(f"Vehicle detector error on {road}: {e}")
            detections = []
            counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0}

        # 2. Computer Vision: Custom Emergency Vehicle (Ambulance / Fire Truck)
        em_summary = {
            "detected": False,
            "vehicle_type": "None",
            "confidence": None,
            "box": None,
            "detections": [],
        }
        if emergency_detector is not None:
            try:
                em_summary = emergency_detector.detect_summary(image_path, road=road)
            except Exception as e:
                app.logger.exception(f"Emergency detection error on {road}: {e}")

        # Check manual simulation override if supplied
        form_override = form.get(f"emergency_{road.lower()}")
        if form_override in ("Ambulance", "Fire Truck"):
            emergency_vehicle_type = form_override
            detected_conf = em_summary["confidence"] if em_summary["detected"] else 0.85
            em_detected = True
        else:
            em_detected = em_summary["detected"]
            emergency_vehicle_type = em_summary["vehicle_type"] if em_detected else "None"
            detected_conf = em_summary["confidence"] if em_detected else None

        # 3. Visual Annotation (OpenCV)
        try:
            draw_detections(
                image_path,
                detections,
                processed_filename,
                emergency_detections=em_summary.get("detections"),
            )
        except Exception as e:
            app.logger.warning(f"Could not draw detections for {road}: {e}")
            processed_filename = filename

        # 4. Density + Demand
        waiting_time = min(MAX_SIMULATED_WAITING_TIME, (cycle_number - 1) * WAITING_TIME_PER_CYCLE)
        density = calculate_density(counts["total"])
        demand = calculate_demand(counts, waiting_time=waiting_time)

        # 5. ML Green-Time Prediction
        features = {
            "car_count": counts["car"],
            "motorcycle_count": counts["motorcycle"],
            "bus_count": counts["bus"],
            "truck_count": counts["truck"],
            "total_vehicle_count": counts["total"],
            "density_percentage": density["percentage"],
            "demand_score": demand,
            "waiting_time": waiting_time,
        }

        try:
            prediction = predictor.predict(features)
            raw_pred = prediction.get("raw_prediction", prediction.get("final_green_time"))
            recommended_green = clamp_green_time(raw_pred, max_green_time=max_green, min_green_time=min_green)
            fallback_used = False
        except Exception as e:
            app.logger.exception(f"Prediction failed for {road}: {e}")
            raw_pred = None
            recommended_green = safe_fallback
            fallback_used = True

        # Enforce critical timing rule backend validation:
        # final_green_time = min(max_green_time, max(min_green_time, predicted_green_time))
        final_green_time = min(max_green, max(min_green, recommended_green))
        green_times[road] = final_green_time
        counts_by_road[road] = counts

        road_candidates[road] = {
            "road": road,
            "detected": em_detected,
            "vehicle_type": emergency_vehicle_type,
            "confidence": detected_conf,
            "box": em_summary.get("box"),
            "detections": em_summary.get("detections", []),
            "ambulance_detected": bool(em_summary.get("ambulance_detected", False) or (em_detected and emergency_vehicle_type == "Ambulance")),
            "fire_truck_detected": bool(em_summary.get("fire_truck_detected", False) or (em_detected and emergency_vehicle_type == "Fire Truck")),
            "counts": counts,
            "filename": filename,
            "processed_filename": processed_filename,
            "density": density,
            "demand": demand,
            "waiting_time": waiting_time,
            "max_green": max_green,
            "min_green": min_green,
            "final_green_time": final_green_time,
            "raw_pred": raw_pred,
            "fallback_used": fallback_used,
        }

    # 6. Direction-Independent Siren & Emergency Priority Resolution
    # Evaluates omnidirectional acoustic siren against YOLO detections across ALL 4 approaches
    active_threshold = SIREN_V2_CONFIDENCE_THRESHOLD if siren_analysis.get("model_version") == "v2" else SIREN_CONFIDENCE_THRESHOLD
    siren_confirmed = bool(
        siren_analysis["is_siren"] and (
            siren_analysis["confidence"] is not None and siren_analysis["confidence"] >= active_threshold
        )
    )

    priority_res = resolve_emergency_priority(
        siren_confirmed=siren_confirmed,
        road_candidates=road_candidates,
        siren_confidence=siren_analysis["confidence"],
        siren_type=siren_analysis.get("siren_type"),
        siren_provided=siren_analysis["provided"],
        min_vehicle_conf=EMERGENCY_CONFIDENCE_THRESHOLD,
    )

    emergency_road = priority_res["emergency_road"]
    vehicle_type = priority_res["vehicle_type"]
    vehicle_confidence = priority_res["vehicle_confidence"]

    if emergency_road:
        siren_analysis["direction"] = emergency_road

    # Populate directional analysis results using the unified resolution
    for road in ROADS:
        cand = road_candidates[road]
        road_status = priority_res["road_statuses"][road]
        em_detected = cand["detected"]

        emergency_eval = evaluate_emergency_candidate(
            vehicle_type=cand["vehicle_type"],
            vehicle_confidence=cand["confidence"],
            siren_detected=siren_analysis["is_siren"],
            siren_confidence=siren_analysis["confidence"],
            siren_type=siren_analysis.get("siren_type"),
            road=road,
        )

        analysis_results[road] = {
            "direction": road,
            "original": cand["filename"],
            "original_url": url_for("uploaded_file", filename=cand["filename"]),
            "processed": cand["processed_filename"],
            "processed_url": url_for("uploaded_file", filename=cand["processed_filename"]),
            "counts": cand["counts"],
            "total_vehicles": cand["counts"]["total"],
            "ordinary_vehicles": cand["counts"]["total"],
            "emergency_vehicle_count": 1 if em_detected else 0,
            "no_vehicles": cand["counts"]["total"] == 0,
            "density": cand["density"],
            "demand_score": cand["demand"],
            "waiting_time": cand["waiting_time"],
            "max_green_time": cand["max_green"],
            "recommended_green_time": cand["final_green_time"],
            "green_time": cand["final_green_time"],
            "raw_predicted_green_time": cand["raw_pred"],
            "prediction": {
                "raw_prediction": cand["raw_pred"],
                "final_green_time": cand["final_green_time"],
                "fallback_used": cand["fallback_used"],
            },
            "emergency_detected": em_detected,
            "emergency_vehicle_detected": em_detected,
            "emergency_candidate_pending": bool(em_detected and not road_status["emergency_priority_confirmed"]),
            "ambulance_detected": cand["ambulance_detected"],
            "fire_truck_detected": cand["fire_truck_detected"],
            "emergency_vehicle_type": cand["vehicle_type"],
            "emergency_confidence": cand["confidence"],
            "emergency_box": cand["box"],
            "emergency_eval": emergency_eval,
            "siren": {
                "provided": siren_analysis["provided"],
                "is_siren": siren_analysis["is_siren"],
                "confidence": siren_analysis["confidence"],
                "predicted_class": siren_analysis.get("predicted_class"),
                "probabilities": siren_analysis.get("probabilities"),
                "siren_type": siren_analysis.get("siren_type"),
                "model_version": siren_analysis.get("model_version"),
                "direction": emergency_road or "Omnidirectional",
            },
            "emergency_active": road_status["emergency_active"],
            "emergency_priority_confirmed": road_status["emergency_priority_confirmed"],
            "status_color": road_status["status_color"],
            "status_label": road_status["status_label"],
            "status_message": road_status["status_message"],
        }

    # Recalculate signal allocation whenever emergency priority is confirmed:
    # Set the prioritized direction to EMERGENCY_GREEN_SECONDS (25s), strictly bounded by [min_green, max_green].
    # Conflicting directions retain their bounded adaptive green times for the cycle sequence.
    if emergency_road:
        max_green = get_max_green_time(emergency_road)
        allocated_emergency_green = min(max_green, max(MIN_GREEN, EMERGENCY_GREEN_SECONDS))
        green_times[emergency_road] = allocated_emergency_green
        analysis_results[emergency_road]["green_time"] = allocated_emergency_green
        analysis_results[emergency_road]["recommended_green_time"] = allocated_emergency_green
        if "prediction" in analysis_results[emergency_road]:
            analysis_results[emergency_road]["prediction"]["final_green_time"] = allocated_emergency_green

        STATE["emergency_log"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "vehicle": vehicle_type,
            "road": emergency_road,
            "vehicle_confidence": f"{vehicle_confidence:.2f}" if vehicle_confidence else "N/A",
            "siren": "ON",
            "siren_confidence": f"{siren_analysis['confidence']:.2f}" if siren_analysis["confidence"] else "N/A",
            "siren_type": siren_analysis.get("siren_type") or "Generic Siren",
            "siren_model_version": siren_analysis.get("model_version", "v1"),
            "action": "SAFETY TRANSITION -> GREEN",
        })

    cycle_plan = build_cycle_plan(green_times, cycle_number, emergency_road=emergency_road)
    emergency_plan = None
    if emergency_road:
        emergency_plan = build_emergency_plan(
            current_road=ROADS[0],
            emergency_road=emergency_road,
            vehicle_type=vehicle_type or "Ambulance",
            vehicle_confidence=vehicle_confidence,
        )

    # Build dedicated emergency analysis summary for dashboard and audit
    if emergency_road:
        prioritized_res = analysis_results.get(emergency_road, {})
        emergency_analysis = {
            "selected_direction": emergency_road,
            "ambulance_detected": prioritized_res.get("ambulance_detected", False),
            "fire_truck_detected": prioritized_res.get("fire_truck_detected", False),
            "emergency_detected": True,
            "emergency_vehicle_type": vehicle_type or "Ambulance",
            "emergency_confidence": vehicle_confidence,
            "siren_provided": siren_analysis["provided"],
            "siren_detected": siren_analysis["is_siren"],
            "siren_confidence": siren_analysis["confidence"],
            "siren_type": siren_analysis.get("siren_type"),
            "siren_model_version": siren_analysis.get("model_version"),
            "confirmed": True,
            "emergency_road": emergency_road,
            "outcome_case": priority_res["outcome_case"],
            "is_ambiguous": False,
            "reason": priority_res["reason"],
            "detailed_reason": priority_res["detailed_reason"],
            "annotated_url": prioritized_res.get("processed_url"),
            "original_url": prioritized_res.get("original_url"),
            "box": prioritized_res.get("emergency_box"),
        }
    else:
        top_cand = priority_res["confirmed_candidates"][0] if priority_res["confirmed_candidates"] else (
            priority_res["all_detected_vehicles"][0] if priority_res["all_detected_vehicles"] else None
        )
        cand_res = analysis_results.get(top_cand["road"], {}) if top_cand else {}
        selected_dir = (
            ", ".join(priority_res["confirmed_directions"])
            if priority_res["confirmed_directions"]
            else "All Approaches (Auto-Scan)"
        )

        emergency_analysis = {
            "selected_direction": selected_dir,
            "ambulance_detected": any(r.get("ambulance_detected", False) for r in analysis_results.values()),
            "fire_truck_detected": any(r.get("fire_truck_detected", False) for r in analysis_results.values()),
            "emergency_detected": bool(top_cand is not None),
            "emergency_vehicle_type": top_cand["vehicle_type"] if top_cand else "None",
            "emergency_confidence": top_cand["confidence"] if top_cand else None,
            "siren_provided": siren_analysis["provided"],
            "siren_detected": siren_analysis["is_siren"],
            "siren_confidence": siren_analysis["confidence"],
            "siren_type": siren_analysis.get("siren_type"),
            "siren_model_version": siren_analysis.get("model_version"),
            "confirmed": False,
            "emergency_road": None,
            "outcome_case": priority_res["outcome_case"],
            "is_ambiguous": priority_res.get("is_ambiguous", False),
            "reason": priority_res["reason"],
            "detailed_reason": priority_res["detailed_reason"],
            "annotated_url": cand_res.get("processed_url"),
            "original_url": cand_res.get("original_url"),
            "box": cand_res.get("emergency_box"),
        }

    latest_analysis = {
        "cycle_number": cycle_number,
        "results": analysis_results,
        "green_times": green_times,
        "counts_by_road": counts_by_road,
        "siren_analysis": siren_analysis,
        "emergency_active": bool(emergency_road is not None),
        "emergency_road": emergency_road,
        "emergency_analysis": emergency_analysis,
        "cycle_plan": cycle_plan,
        "emergency_plan": emergency_plan,
    }

    STATE["latest_analysis"] = latest_analysis
    return latest_analysis, None


@app.route("/upload", methods=["POST"])
def upload():
    """
    Form submission upload route.
    Supports both traditional browser form redirects and JSON responses.
    """
    result_data, err = execute_analysis(request.files, request.form)
    if err:
        error_msg, status_code = err
        if request.is_json or request.headers.get("Accept") == "application/json" or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": error_msg}), status_code
        flash(error_msg)
        return redirect(url_for("index"))

    if request.is_json or request.headers.get("Accept") == "application/json" or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": True, "data": result_data}), 200

    return render_template(
        "analysis.html",
        app_name=APP_NAME,
        roads=ROADS,
        results=result_data["results"],
        cycle_number=result_data["cycle_number"],
        emergency_model_available=_emergency_model_available(),
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Dedicated JSON REST endpoint for traffic & emergency analysis.
    Accepts multipart form with:
    - north, east, south, west image files
    - optional siren_audio / audio WAV file
    - optional siren_version (v1, v2)
    Returns structured JSON with all metrics, annotations, and simulation plans.
    """
    result_data, err = execute_analysis(request.files, request.form)
    if err:
        error_msg, status_code = err
        return jsonify({"success": False, "error": error_msg}), status_code

    return jsonify({"success": True, "data": result_data}), 200


@app.route("/api/siren_model_info", methods=["GET"])
def api_siren_model_info():
    """Returns availability and metadata for both Siren V1 and V2 models."""
    from audio.siren_detector_v2 import SirenDetectorV2
    return jsonify({
        "success": True,
        "active_version": SIREN_ACTIVE_VERSION,
        "v1": {
            "available": _siren_model_available(),
            "path": SIREN_MODEL_PATH,
            "classes": ["no_siren", "siren"],
            "features": 26,
        },
        "v2": SirenDetectorV2.get_model_info(),
    })



@app.route("/api/demo_samples", methods=["GET"])
def api_demo_samples():
    """Returns metadata for sample demo images and audio to support 1-click college project demos."""
    return jsonify({
        "success": True,
        "emergency_demo": {
            "name": "Ambulance Emergency Scenario (South Direction)",
            "north": "/api/sample_media/images/1478019957687018435_jpg.rf.1337211ed85fd3098d01e02cfd8d68d8.jpg",
            "east": "/api/sample_media/images/1478020431196343477_jpg.rf.86bf50c272bf7e4ccb6404974247b130.jpg",
            "south": "/api/sample_media/images/back_134_jpeg_jpg.rf.771c3c3c2cc150b90705d8d6e96473f1.jpg",
            "west": "/api/sample_media/images/1478020491698266386_jpg.rf.cc7f10b1511352c8c8b2ae377747dc5e.jpg",
            "siren_audio": "/api/sample_media/audio/siren/siren_000.wav",
            "siren_direction": "South",
        },
        "normal_demo": {
            "name": "Standard Adaptive Traffic (No Siren)",
            "north": "/api/sample_media/images/1478019957687018435_jpg.rf.1337211ed85fd3098d01e02cfd8d68d8.jpg",
            "east": "/api/sample_media/images/1478020431196343477_jpg.rf.86bf50c272bf7e4ccb6404974247b130.jpg",
            "south": "/api/sample_media/images/1478020512707283286_jpg.rf.9f3d00b99058a0675b8a638096accabf.jpg",
            "west": "/api/sample_media/images/1478020491698266386_jpg.rf.cc7f10b1511352c8c8b2ae377747dc5e.jpg",
            "siren_audio": None,
            "siren_direction": "South",
        }
    })


@app.route("/api/sample_media/<media_type>/<path:filename>")
def api_sample_media(media_type, filename):
    """Serves bundled sample media for easy client loading."""
    if media_type == "images":
        directory = os.path.join(DATA_FOLDER, "emergency_yolo", "test", "images")
    elif media_type == "audio":
        directory = os.path.join(DATA_FOLDER, "audio")
    else:
        return jsonify({"error": "Invalid media type"}), 404

    return send_from_directory(directory, filename)



@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/plots/<path:filename>")
def plot_file(filename):
    plots_dir = os.path.join(DATA_FOLDER, "traffic", "plots")
    return send_from_directory(plots_dir, filename)


# ---------------------------------------------------------------------------
# Page 3 — Simulation
# ---------------------------------------------------------------------------
def load_and_execute_demo(scenario="normal"):
    """Loads bundled benchmark demo assets and runs the AI analysis pipeline."""
    from werkzeug.datastructures import FileStorage
    img_dir = os.path.join(DATA_FOLDER, "emergency_yolo", "test", "images")
    audio_dir = os.path.join(DATA_FOLDER, "audio")

    if scenario == "emergency":
        files_map = {
            "north": os.path.join(img_dir, "1478019957687018435_jpg.rf.1337211ed85fd3098d01e02cfd8d68d8.jpg"),
            "east": os.path.join(img_dir, "1478020431196343477_jpg.rf.86bf50c272bf7e4ccb6404974247b130.jpg"),
            "south": os.path.join(img_dir, "back_134_jpeg_jpg.rf.771c3c3c2cc150b90705d8d6e96473f1.jpg"),
            "west": os.path.join(img_dir, "1478020491698266386_jpg.rf.cc7f10b1511352c8c8b2ae377747dc5e.jpg"),
            "siren_audio": os.path.join(audio_dir, "siren", "siren_000.wav"),
        }
    else:
        files_map = {
            "north": os.path.join(img_dir, "1478019957687018435_jpg.rf.1337211ed85fd3098d01e02cfd8d68d8.jpg"),
            "east": os.path.join(img_dir, "1478020431196343477_jpg.rf.86bf50c272bf7e4ccb6404974247b130.jpg"),
            "south": os.path.join(img_dir, "1478020512707283286_jpg.rf.9f3d00b99058a0675b8a638096accabf.jpg"),
            "west": os.path.join(img_dir, "1478020491698266386_jpg.rf.cc7f10b1511352c8c8b2ae377747dc5e.jpg"),
        }

    file_storages = {}
    for key, fpath in files_map.items():
        if fpath and os.path.exists(fpath):
            with open(fpath, "rb") as fp:
                file_storages[key] = FileStorage(
                    stream=io.BytesIO(fp.read()),
                    filename=os.path.basename(fpath),
                    content_type="audio/wav" if key == "siren_audio" else "image/jpeg",
                )

    form_dict = {
        "emergency_north": "Auto-Detect",
        "emergency_east": "Auto-Detect",
        "emergency_south": "Auto-Detect",
        "emergency_west": "Auto-Detect",
        "siren_version": "v1",
    }
    result_data, err = execute_analysis(file_storages, form_dict)
    if result_data:
        result_data["is_demo"] = True
        result_data["demo_name"] = (
            "Emergency Priority Scenario (Ambulance + Siren)"
            if scenario == "emergency"
            else "Standard Adaptive Traffic Scenario"
        )
    return result_data, err


@app.route("/simulate")
def simulate():
    demo_type = request.args.get("demo")
    if demo_type in ["normal", "emergency"]:
        result_data, err = load_and_execute_demo(demo_type)
        if err:
            flash(f"Error loading demo scenario: {err[0]}")
        else:
            flash(
                f"Demo benchmark dataset loaded: "
                f"{'Emergency Priority (Ambulance + Siren)' if demo_type == 'emergency' else 'Standard Adaptive Traffic'}"
            )
        return redirect(url_for("simulate"))

    analysis = STATE["latest_analysis"]
    if analysis is None:
        return render_template(
            "simulation.html",
            app_name=APP_NAME,
            analysis=None,
            cycle_plan=None,
            emergency_plan=None,
            cycle_number=0,
            impact_assessment=None,
            is_demo=False,
            fixed_signal_time=FIXED_SIGNAL_TIME,
        )

    emergency_road = analysis.get("emergency_road")
    cycle_plan = analysis.get("cycle_plan") or build_cycle_plan(
        analysis["green_times"], analysis["cycle_number"], emergency_road=emergency_road
    )
    emergency_plan = analysis.get("emergency_plan")

    if emergency_road and not emergency_plan:
        prioritized_res = analysis["results"].get(emergency_road, {})
        vehicle_type = prioritized_res.get("emergency_vehicle_type", "Ambulance")
        vehicle_confidence = prioritized_res.get("emergency_confidence")
        emergency_plan = build_emergency_plan(
            current_road=ROADS[0],
            emergency_road=emergency_road,
            vehicle_type=vehicle_type,
            vehicle_confidence=vehicle_confidence,
        )

    impact_assessment = evaluate_traffic_impact(analysis)

    return render_template(
        "simulation.html",
        app_name=APP_NAME,
        analysis=analysis,
        cycle_plan=cycle_plan,
        emergency_plan=emergency_plan,
        cycle_number=analysis["cycle_number"],
        impact_assessment=impact_assessment,
        is_demo=analysis.get("is_demo", False),
        fixed_signal_time=FIXED_SIGNAL_TIME,
    )


# ---------------------------------------------------------------------------
# Page 4 — Data Science Dashboard
# ---------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    import json

    model_metrics = None
    if os.path.exists(TRAFFIC_METRICS_PATH):
        with open(TRAFFIC_METRICS_PATH) as f:
            model_metrics = json.load(f)

    siren_metrics_path = os.path.join(DATA_FOLDER, "audio", "siren_model_metrics.json")
    siren_metrics = None
    if os.path.exists(siren_metrics_path):
        with open(siren_metrics_path) as f:
            siren_metrics = json.load(f)

    analysis = STATE["latest_analysis"]
    comparison = None
    if analysis:
        comparison = compare_fixed_vs_adaptive(analysis["green_times"], analysis["counts_by_road"])

    emergency_model_info = EmergencyDetector.get_model_info()

    return render_template(
        "dashboard.html",
        app_name=APP_NAME,
        model_metrics=model_metrics,
        siren_metrics=siren_metrics,
        analysis=analysis,
        comparison=comparison,
        emergency_log=STATE["emergency_log"],
        fixed_signal_time=FIXED_SIGNAL_TIME,
        emergency_model_info=emergency_model_info,
    )


# ---------------------------------------------------------------------------
# Page 5 — What-If Simulation
# ---------------------------------------------------------------------------
@app.route("/whatif", methods=["GET", "POST"])
def whatif():
    analysis = STATE["latest_analysis"]
    if analysis is None:
        flash("Run an analysis first (upload images) before using What-If simulation.")
        return redirect(url_for("index"))

    if not _ml_available():
        flash("Traffic ML model not found. Run: python -m ml.train_traffic_model")
        return redirect(url_for("index"))

    from ml.predictor import GreenTimePredictor
    predictor = GreenTimePredictor()

    result = None
    selected_road = request.values.get("road", ROADS[0])

    if request.method == "POST":
        base_counts = dict(analysis["counts_by_road"][selected_road])
        pct_change = float(request.form.get("percent_change", 0))
        extra_buses = int(request.form.get("extra_buses", 0) or 0)
        extra_motorcycles = int(request.form.get("extra_motorcycles", 0) or 0)

        new_counts = {
            "car": max(0, round(base_counts["car"] * (1 + pct_change / 100))),
            "motorcycle": max(0, round(base_counts["motorcycle"] * (1 + pct_change / 100)) + extra_motorcycles),
            "bus": max(0, base_counts["bus"] + extra_buses),
            "truck": base_counts["truck"],
        }
        new_counts["total"] = sum(new_counts.values())

        density = calculate_density(new_counts["total"])
        demand = calculate_demand(new_counts, waiting_time=analysis["results"][selected_road]["waiting_time"])

        features = {
            "car_count": new_counts["car"],
            "motorcycle_count": new_counts["motorcycle"],
            "bus_count": new_counts["bus"],
            "truck_count": new_counts["truck"],
            "total_vehicle_count": new_counts["total"],
            "density_percentage": density["percentage"],
            "demand_score": demand,
            "waiting_time": analysis["results"][selected_road]["waiting_time"],
        }
        prediction = predictor.predict(features)

        result = {
            "new_counts": new_counts,
            "density": density,
            "demand_score": demand,
            "prediction": prediction,
            "original_green_time": analysis["results"][selected_road]["prediction"]["final_green_time"],
        }

    return render_template(
        "what_if.html",
        app_name=APP_NAME,
        roads=ROADS,
        selected_road=selected_road,
        current=analysis["results"][selected_road],
        result=result,
    )


# ---------------------------------------------------------------------------
# REST API Endpoints for Integration & Testing
# ---------------------------------------------------------------------------
@app.route("/api/model_info")
def api_model_info():
    """Returns metadata for all ML and CV models in the system."""
    return jsonify({
        "emergency_model": EmergencyDetector.get_model_info(),
        "traffic_model_available": _ml_available(),
        "siren_model_available": _siren_model_available(),
        "roads": ROADS,
        "default_emergency_confidence_threshold": EMERGENCY_CONFIDENCE_THRESHOLD,
    })


@app.route("/api/detect_emergency", methods=["POST"])
def api_detect_emergency():
    """
    Direct image detection endpoint.
    Accepts multipart image upload, optional 'direction'/'road', and optional 'conf'.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' file provided in request"}), 400

    file = request.files["image"]
    if not is_allowed_image(file.filename):
        return jsonify({"success": False, "error": "Invalid image format. Allowed: jpg, jpeg, png"}), 400

    direction = request.form.get("direction", request.form.get("road", "North"))
    conf_override = request.form.get("conf", type=float)

    # Save to temp location in uploads
    temp_filename = f"api_test_{int(datetime.now().timestamp())}_{file.filename}"
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
    file.save(temp_path)

    detector = EmergencyDetector()
    summary = detector.detect_summary(temp_path, conf=conf_override, road=direction)

    return jsonify({
        "success": True,
        "direction": direction,
        "emergency_detected": summary["detected"],
        "vehicle_type": summary["vehicle_type"],
        "confidence": summary["confidence"],
        "bounding_box": summary["box"],
        "detections": summary["detections"],
        "model_available": summary["model_available"],
    })


@app.route("/api/evaluate_emergency", methods=["POST"])
def api_evaluate_emergency():
    """
    Evaluates vehicle detection + audio siren inputs.
    Strict rule: Emergency vehicle AND Siren detected = Emergency candidate.
    """
    data = request.get_json(silent=True) or request.form

    vehicle_type = data.get("vehicle_type", "None")
    vehicle_conf = data.get("vehicle_confidence")
    if vehicle_conf is not None:
        try:
            vehicle_conf = float(vehicle_conf)
        except ValueError:
            vehicle_conf = None

    siren_detected = str(data.get("siren_detected", "false")).lower() in ("true", "1", "yes")
    siren_conf = data.get("siren_confidence")
    if siren_conf is not None:
        try:
            siren_conf = float(siren_conf)
        except ValueError:
            siren_conf = None

    road = data.get("road", data.get("direction", "South"))
    if road not in ROADS:
        road = "South"

    evaluation = evaluate_emergency_candidate(
        vehicle_type=vehicle_type,
        vehicle_confidence=vehicle_conf,
        siren_detected=siren_detected,
        siren_confidence=siren_conf,
        road=road,
    )

    emergency_plan = None
    if evaluation["is_emergency"]:
        emergency_plan = build_emergency_plan(
            current_road=ROADS[0],
            emergency_road=road,
            vehicle_type=evaluation["vehicle_type"],
            vehicle_confidence=vehicle_conf,
        )

    return jsonify({
        "success": True,
        "evaluation": evaluation,
        "emergency_plan": emergency_plan,
    })


if __name__ == "__main__":
    app.run(debug=DEBUG_MODE, port=5000)
