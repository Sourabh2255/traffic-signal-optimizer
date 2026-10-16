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
    is_emergency_active, build_emergency_plan, evaluate_emergency_candidate
)
from traffic.comparison import compare_fixed_vs_adaptive

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

    siren_detector = None
    if _siren_model_available():
        from audio.siren_detector import SirenDetector
        try:
            siren_detector = SirenDetector()
        except Exception as e:
            app.logger.warning(f"Could not initialize SirenDetector: {e}")

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
    # Accept siren_audio, audio, or direction-specific audio_south, etc.
    audio_file = files.get("siren_audio") or files.get("audio") or files.get("audio_file")
    siren_direction = form.get("siren_direction", form.get("audio_direction", "South"))
    if not audio_file or audio_file.filename == "":
        for r in ROADS:
            candidate_audio = files.get(f"audio_{r.lower()}")
            if candidate_audio and candidate_audio.filename != "":
                audio_file = candidate_audio
                siren_direction = r
                break

    siren_direction = str(siren_direction).strip().title()
    if siren_direction not in ROADS:
        siren_direction = "South"

    siren_analysis = {
        "provided": False,
        "filename": None,
        "direction": siren_direction,
        "is_siren": False,
        "confidence": None,
        "error": None,
    }

    if audio_file and audio_file.filename != "":
        if not audio_file.filename.lower().endswith(tuple(ALLOWED_AUDIO_EXTENSIONS)):
            return None, ("Invalid audio format. Emergency siren audio must be a WAV file (.wav).", 400)

        ext = audio_file.filename.rsplit(".", 1)[1].lower()
        unique_audio_name = f"siren_{siren_direction.lower()}_{uuid.uuid4().hex[:6]}.{ext}"
        audio_path = os.path.join(UPLOAD_FOLDER, unique_audio_name)
        audio_file.save(audio_path)
        siren_analysis["provided"] = True
        siren_analysis["filename"] = unique_audio_name

        if siren_detector is not None:
            try:
                siren_pred = siren_detector.predict(audio_path)
                siren_analysis["is_siren"] = bool(siren_pred.get("is_siren", False))
                siren_analysis["confidence"] = siren_pred.get("confidence", 0.0)
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

        # 6. Strict AND-Condition Emergency Priority Check
        is_audio_direction = siren_analysis["provided"] and (siren_direction == road)
        has_em_vehicle = em_detected and (detected_conf is not None and detected_conf >= EMERGENCY_CONFIDENCE_THRESHOLD)
        has_siren_audio = is_audio_direction and siren_analysis["is_siren"] and (
            siren_analysis["confidence"] is not None and siren_analysis["confidence"] >= SIREN_CONFIDENCE_THRESHOLD
        )

        emergency_priority_confirmed = False
        if siren_analysis["provided"]:
            if is_audio_direction:
                if has_em_vehicle and has_siren_audio:
                    emergency_priority_confirmed = True
                    status_color = "green"
                    status_label = "Emergency Priority Confirmed"
                    status_message = "Emergency confirmed — safely prioritizing the selected direction."
                elif has_siren_audio and not has_em_vehicle:
                    emergency_priority_confirmed = False
                    status_color = "amber"
                    status_label = "Siren Detected (No Vehicle)"
                    status_message = "Siren detected, but no emergency vehicle was confirmed in the selected direction."
                elif has_em_vehicle and not has_siren_audio:
                    emergency_priority_confirmed = False
                    status_color = "amber"
                    status_label = "Vehicle Detected (No Siren)"
                    status_message = "Emergency vehicle detected, but siren confirmation is missing."
                else:
                    emergency_priority_confirmed = False
                    status_color = "blue"
                    status_label = "Normal Adaptive Mode"
                    status_message = "Normal adaptive mode: No emergency vehicle or siren detected."
            else:
                # Other directions when siren audio was provided for a different road
                if has_em_vehicle:
                    emergency_priority_confirmed = False
                    status_color = "amber"
                    status_label = "Vehicle Detected (No Siren Audio)"
                    status_message = "Emergency vehicle detected, but siren confirmation is missing."
                else:
                    emergency_priority_confirmed = False
                    status_color = "blue"
                    status_label = "Normal Adaptive Mode"
                    status_message = "Normal adaptive mode."
        else:
            # No siren WAV was provided at all
            if has_em_vehicle:
                emergency_priority_confirmed = False
                status_color = "amber"
                status_label = "Vehicle Detected (No Siren Audio)"
                status_message = "Emergency vehicle detected, but siren confirmation is missing."
            else:
                emergency_priority_confirmed = False
                status_color = "blue"
                status_label = "Normal Adaptive Mode"
                status_message = "No siren audio provided — normal adaptive mode."

        # Evaluate candidate for structured log compatibility
        emergency_eval = evaluate_emergency_candidate(
            vehicle_type=emergency_vehicle_type,
            vehicle_confidence=detected_conf,
            siren_detected=siren_analysis["is_siren"] if is_audio_direction else False,
            siren_confidence=siren_analysis["confidence"] if is_audio_direction else None,
            road=road,
        )

        analysis_results[road] = {
            "direction": road,
            "original": filename,
            "original_url": url_for("uploaded_file", filename=filename),
            "processed": processed_filename,
            "processed_url": url_for("uploaded_file", filename=processed_filename),
            "counts": counts,
            "total_vehicles": counts["total"],
            "ordinary_vehicles": counts["total"],
            "emergency_vehicle_count": 1 if em_detected else 0,
            "no_vehicles": counts["total"] == 0,
            "density": density,
            "demand_score": demand,
            "waiting_time": waiting_time,
            "max_green_time": max_green,
            "recommended_green_time": final_green_time,
            "green_time": final_green_time,
            "raw_predicted_green_time": raw_pred,
            "prediction": {
                "raw_prediction": raw_pred,
                "final_green_time": final_green_time,
                "fallback_used": fallback_used,
            },
            "emergency_detected": em_detected,
            "emergency_vehicle_detected": em_detected,
            "emergency_candidate_pending": bool(em_detected and not emergency_priority_confirmed),
            "ambulance_detected": em_summary.get("ambulance_detected", False),
            "fire_truck_detected": em_summary.get("fire_truck_detected", False),
            "emergency_vehicle_type": emergency_vehicle_type,
            "emergency_confidence": detected_conf,
            "emergency_box": em_summary.get("box"),
            "emergency_eval": emergency_eval,
            "siren": {
                "provided": siren_analysis["provided"],
                "is_siren": siren_analysis["is_siren"] if is_audio_direction else False,
                "confidence": siren_analysis["confidence"] if is_audio_direction else None,
                "direction": siren_direction if is_audio_direction else None,
            },
            "emergency_active": emergency_priority_confirmed,
            "emergency_priority_confirmed": emergency_priority_confirmed,
            "status_color": status_color,
            "status_label": status_label,
            "status_message": status_message,
        }

    # Identify confirmed emergency candidate road
    emergency_road = None
    vehicle_type = None
    vehicle_confidence = None

    for road, r in analysis_results.items():
        if r["emergency_active"]:
            emergency_road = road
            vehicle_type = r["emergency_vehicle_type"]
            vehicle_confidence = r["emergency_confidence"]
            STATE["emergency_log"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "vehicle": vehicle_type,
                "road": road,
                "vehicle_confidence": f"{vehicle_confidence:.2f}" if vehicle_confidence else "N/A",
                "siren": "ON",
                "siren_confidence": f"{siren_analysis['confidence']:.2f}" if siren_analysis["confidence"] else "N/A",
                "action": "SAFETY TRANSITION -> GREEN",
            })
            break

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

    cycle_plan = build_cycle_plan(green_times, cycle_number, emergency_road=emergency_road)
    emergency_plan = None
    if emergency_road:
        emergency_plan = build_emergency_plan(
            current_road=ROADS[0],
            emergency_road=emergency_road,
            vehicle_type=vehicle_type or "Ambulance",
            vehicle_confidence=vehicle_confidence,
        )

    # Build dedicated emergency analysis summary for debugging panel (Task 5)
    selected_result = analysis_results.get(siren_direction, {})
    confirmed = bool(emergency_road is not None)

    if confirmed:
        analysis_reason = "Emergency confirmed — safely prioritizing the selected direction."
    elif siren_analysis["is_siren"] and not selected_result.get("emergency_detected", False):
        analysis_reason = "Siren detected, but no emergency vehicle was confirmed in the selected direction."
    elif selected_result.get("emergency_detected", False) and not siren_analysis["is_siren"]:
        analysis_reason = "Emergency vehicle detected, but siren confirmation is missing."
    else:
        analysis_reason = "No emergency vehicle or siren detected in the selected direction."

    emergency_analysis = {
        "selected_direction": siren_direction,
        "ambulance_detected": selected_result.get("ambulance_detected", False),
        "fire_truck_detected": selected_result.get("fire_truck_detected", False),
        "emergency_detected": selected_result.get("emergency_detected", False),
        "emergency_vehicle_type": selected_result.get("emergency_vehicle_type", "None"),
        "emergency_confidence": selected_result.get("emergency_confidence"),
        "siren_provided": siren_analysis["provided"],
        "siren_detected": siren_analysis["is_siren"],
        "siren_confidence": siren_analysis["confidence"],
        "confirmed": confirmed,
        "emergency_road": emergency_road,
        "reason": analysis_reason,
        "annotated_url": selected_result.get("processed_url"),
        "original_url": selected_result.get("original_url"),
        "box": selected_result.get("emergency_box"),
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
    - optional siren_direction (North, East, South, West)
    Returns structured JSON with all metrics, annotations, and simulation plans.
    """
    result_data, err = execute_analysis(request.files, request.form)
    if err:
        error_msg, status_code = err
        return jsonify({"success": False, "error": error_msg}), status_code

    return jsonify({"success": True, "data": result_data}), 200


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
@app.route("/simulate")
def simulate():
    analysis = STATE["latest_analysis"]
    if analysis is None:
        flash("Run an analysis first (upload images) before starting a simulation.")
        return redirect(url_for("index"))

    cycle_plan = build_cycle_plan(analysis["green_times"], analysis["cycle_number"])

    emergency_road = None
    emergency_plan = None
    vehicle_type = None
    vehicle_confidence = None

    for road, r in analysis["results"].items():
        if r["emergency_active"]:
            emergency_road = road
            vehicle_type = r["emergency_vehicle_type"]
            vehicle_confidence = r.get("emergency_confidence")
            break

    if emergency_road:
        emergency_plan = build_emergency_plan(
            current_road=ROADS[0],
            emergency_road=emergency_road,
            vehicle_type=vehicle_type,
            vehicle_confidence=vehicle_confidence,
        )

    return render_template(
        "simulation.html",
        app_name=APP_NAME,
        cycle_plan=cycle_plan,
        emergency_plan=emergency_plan,
        cycle_number=analysis["cycle_number"],
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
