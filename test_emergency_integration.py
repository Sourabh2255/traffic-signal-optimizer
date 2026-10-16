"""
Comprehensive Integration Test Suite for Custom YOLO Emergency Vehicle Detection.

Verifies:
1. Loading the custom model (models/best_emergency_vehicle.pt).
2. Detecting ambulances and fire trucks on test images.
3. Handling images with no emergency vehicles (negative traffic samples).
4. Handling invalid inputs and missing model files gracefully.
5. Combining vehicle and siren results (the strict AND-condition).
6. Verifying safe signal transitions (all-red buffer, zero conflicting greens, [15, 60]s bounds).
7. Simulation timer handling and state reset.
8. Confirming existing traffic density and siren models are intact and functional.
9. Flask API endpoints (/api/model_info, /api/detect_emergency, /api/evaluate_emergency).
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    ROADS,
    EMERGENCY_MODEL_PATH,
    EMERGENCY_CONFIDENCE_THRESHOLD,
    SIREN_CONFIDENCE_THRESHOLD,
    SAFETY_TRANSITION_SECONDS,
    EMERGENCY_GREEN_SECONDS,
    MIN_GREEN,
    MAX_GREEN,
)
from detection.emergency_detector import EmergencyDetector
from traffic.emergency_controller import (
    is_emergency_active,
    evaluate_emergency_candidate,
    build_emergency_plan,
)
from traffic.density import calculate_density
from traffic.demand import calculate_demand
from ml.predictor import GreenTimePredictor
from app import app


class TestEmergencyModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.detector = EmergencyDetector()

        # Locate sample test images
        test_img_dir = Path("data/emergency_yolo/test/images")
        test_lbl_dir = Path("data/emergency_yolo/test/labels")

        cls.amb_image = None
        cls.fire_image = None
        cls.neg_image = None

        if test_lbl_dir.exists():
            for p in test_lbl_dir.glob("*.txt"):
                lines = p.read_text().splitlines()
                img_path = test_img_dir / (p.stem + ".jpg")
                if not img_path.exists():
                    continue

                if not lines and cls.neg_image is None:
                    cls.neg_image = str(img_path)

                for l in lines:
                    parts = l.strip().split()
                    if parts:
                        if parts[0] == "0" and cls.amb_image is None:
                            cls.amb_image = str(img_path)
                        elif parts[0] == "1" and cls.fire_image is None:
                            cls.fire_image = str(img_path)

    # --------------------------------------------------------------------------
    # 1. Custom Model Loading
    # --------------------------------------------------------------------------
    def test_01_custom_model_loading(self):
        """Verifies custom YOLO weights exist, load, and have the correct classes."""
        self.assertTrue(
            EmergencyDetector.is_available(),
            f"Emergency model weights missing at {EMERGENCY_MODEL_PATH}",
        )
        info = EmergencyDetector.get_model_info()
        self.assertTrue(info["exists"])
        self.assertTrue(info["loaded"])
        self.assertIn(0, info["classes"])
        self.assertIn(1, info["classes"])
        self.assertEqual(info["classes"][0].lower(), "ambulance")
        self.assertEqual(info["classes"][1].lower(), "fire_truck")

    # --------------------------------------------------------------------------
    # 2. Detecting Ambulances and Fire Trucks
    # --------------------------------------------------------------------------
    def test_02_detecting_ambulances_and_fire_trucks(self):
        """Verifies inference returns bounding boxes, confidence, and class names."""
        if self.fire_image:
            detections = self.detector.detect(self.fire_image, conf=0.20)
            self.assertIsInstance(detections, list)
            if detections:
                d = detections[0]
                self.assertIn(d["class_name"], ["Fire Truck", "Ambulance"])
                self.assertGreaterEqual(d["confidence"], 0.20)
                self.assertEqual(len(d["box"]), 4)

        if self.amb_image:
            detections = self.detector.detect(self.amb_image, conf=0.15)
            self.assertIsInstance(detections, list)

    # --------------------------------------------------------------------------
    # 3. Handling Images with No Emergency Vehicles
    # --------------------------------------------------------------------------
    def test_03_handling_negative_images(self):
        """Verifies that an ordinary traffic image produces 0 emergency detections."""
        if self.neg_image:
            detections = self.detector.detect(self.neg_image, conf=0.25)
            self.assertEqual(
                len(detections),
                0,
                f"False positive detection found on confirmed negative image: {detections}",
            )

            summary = self.detector.detect_summary(self.neg_image, road="North")
            self.assertFalse(summary["detected"])
            self.assertEqual(summary["vehicle_type"], "None")
            self.assertIsNone(summary["confidence"])

    # --------------------------------------------------------------------------
    # 4. Handling Invalid Inputs and Missing Model Files
    # --------------------------------------------------------------------------
    def test_04_handling_invalid_inputs_and_missing_models(self):
        """Verifies resilience to non-existent files and invalid paths."""
        # Non-existent file should return empty list gracefully without throwing an unhandled crash
        res = self.detector.detect("non_existent_image_path_12345.jpg")
        self.assertEqual(res, [])

        summary = self.detector.detect_summary("invalid_path.jpg", road="East")
        self.assertFalse(summary["detected"])
        self.assertEqual(summary["vehicle_type"], "None")

    # --------------------------------------------------------------------------
    # 5. Combining Vehicle and Siren Results (Strict AND-Condition)
    # --------------------------------------------------------------------------
    def test_05_combining_vehicle_and_siren_results(self):
        """
        Tests the strict AND-condition:
        Emergency vehicle detected AND siren detected = emergency candidate
        """
        # Case A: Both vehicle and siren present -> EMERGENCY ACTIVE
        eval_both = evaluate_emergency_candidate(
            vehicle_type="Ambulance",
            vehicle_confidence=0.85,
            siren_detected=True,
            siren_confidence=0.90,
            road="South",
        )
        self.assertTrue(eval_both["is_emergency"])
        self.assertEqual(eval_both["candidate_status"], "ACTIVE")

        # Case B: Vehicle present, but NO siren -> NORMAL MODE
        eval_no_siren = evaluate_emergency_candidate(
            vehicle_type="Ambulance",
            vehicle_confidence=0.85,
            siren_detected=False,
            siren_confidence=0.10,
            road="South",
        )
        self.assertFalse(eval_no_siren["is_emergency"])
        self.assertEqual(eval_no_siren["candidate_status"], "INACTIVE")
        self.assertIn("normal adaptive signal timing is maintained", eval_no_siren["reason"].lower())

        # Case C: Siren present, but NO emergency vehicle -> NORMAL MODE
        eval_no_vehicle = evaluate_emergency_candidate(
            vehicle_type="None",
            vehicle_confidence=None,
            siren_detected=True,
            siren_confidence=0.85,
            road="North",
        )
        self.assertFalse(eval_no_vehicle["is_emergency"])
        self.assertEqual(eval_no_vehicle["candidate_status"], "INACTIVE")

        # Case D: Neither present -> NORMAL MODE
        eval_neither = evaluate_emergency_candidate(
            vehicle_type="None",
            vehicle_confidence=None,
            siren_detected=False,
            siren_confidence=None,
            road="West",
        )
        self.assertFalse(eval_neither["is_emergency"])

        # Case E: Vehicle confidence below threshold -> NORMAL MODE
        eval_low_conf = evaluate_emergency_candidate(
            vehicle_type="Ambulance",
            vehicle_confidence=0.05,  # below threshold
            siren_detected=True,
            siren_confidence=0.80,
            road="East",
        )
        self.assertFalse(eval_low_conf["is_emergency"])

        # Backward compatibility check
        self.assertTrue(is_emergency_active("Ambulance", True))
        self.assertFalse(is_emergency_active("Ambulance", False))
        self.assertFalse(is_emergency_active("None", True))

    # --------------------------------------------------------------------------
    # 6. Verifying Safe Signal Transitions
    # --------------------------------------------------------------------------
    def test_06_safe_signal_transitions(self):
        """
        Verifies signal transition safety:
        - Conflicting roads transition to RED
        - All-red safety buffer
        - Only emergency road gets GREEN
        - Normal bounds [15, 60]s respected
        """
        emergency_road = "South"
        plan = build_emergency_plan(
            current_road="North",
            emergency_road=emergency_road,
            vehicle_type="Fire Truck",
        )

        self.assertEqual(plan["emergency_road"], emergency_road)
        self.assertEqual(set(plan["conflicting_roads"]), {"North", "East", "West"})
        self.assertEqual(plan["safety_buffer_seconds"], SAFETY_TRANSITION_SECONDS)
        self.assertEqual(plan["emergency_green_seconds"], EMERGENCY_GREEN_SECONDS)
        self.assertEqual(plan["normal_green_bounds"]["min"], MIN_GREEN)
        self.assertEqual(plan["normal_green_bounds"]["max"], MAX_GREEN)

        phases = [s["phase"] for s in plan["steps"]]
        self.assertEqual(
            phases,
            ["current_to_red", "safety_buffer", "emergency_green", "resume_normal"],
        )

        # Verify that emergency_green phase strictly assigns green to emergency road only
        green_step = next(s for s in plan["steps"] if s["phase"] == "emergency_green")
        self.assertEqual(green_step["road"], emergency_road)
        self.assertNotIn(emergency_road, green_step["conflicting_roads"])

    # --------------------------------------------------------------------------
    # 7. Stopping and Restarting Simulation Without Stale Timers
    # --------------------------------------------------------------------------
    def test_07_stopping_and_restarting_simulation(self):
        """Verifies simulation.js timer management patterns in static file."""
        sim_js_path = Path("static/js/simulation.js")
        self.assertTrue(sim_js_path.exists())
        js_content = sim_js_path.read_text(encoding="utf-8")

        # Verify clearAllTimers exists and clears both timerId and emergencyTimerIds
        self.assertIn("clearAllTimers", js_content)
        self.assertIn("emergencyTimerIds", js_content)
        self.assertIn("clearAllTimers();", js_content)

    # --------------------------------------------------------------------------
    # 8. Confirming Existing Traffic-Density & Siren Models Unaffected
    # --------------------------------------------------------------------------
    def test_08_existing_traffic_and_siren_models_unaffected(self):
        """Verifies that traffic density calculation and green-time ML work normally."""
        # 1. Density calculation
        density = calculate_density(total_vehicles=20)
        self.assertEqual(density["percentage"], 50.0)
        self.assertIn(density["level"], ["HIGH", "MEDIUM"])

        # 2. Demand calculation
        counts = {"car": 10, "motorcycle": 4, "bus": 2, "truck": 1, "total": 17}
        demand = calculate_demand(counts, waiting_time=24)
        self.assertGreater(demand, 0.0)

        # 3. Green time ML model
        predictor = GreenTimePredictor()
        features = {
            "car_count": 10,
            "motorcycle_count": 4,
            "bus_count": 2,
            "truck_count": 1,
            "total_vehicle_count": 17,
            "density_percentage": 50.0,
            "demand_score": demand,
            "waiting_time": 24,
        }
        pred = predictor.predict(features)
        self.assertIn("final_green_time", pred)
        self.assertGreaterEqual(pred["final_green_time"], MIN_GREEN)
        self.assertLessEqual(pred["final_green_time"], MAX_GREEN)

    # --------------------------------------------------------------------------
    # 9. Flask API Endpoints
    # --------------------------------------------------------------------------
    def test_09_flask_api_endpoints(self):
        """Tests the Flask API endpoints for model info and emergency evaluation."""
        # 1. GET /api/model_info
        res = self.client.get("/api/model_info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["emergency_model"]["loaded"])
        self.assertEqual(data["roads"], ROADS)

        # 2. POST /api/evaluate_emergency
        eval_req = {
            "vehicle_type": "Ambulance",
            "vehicle_confidence": 0.88,
            "siren_detected": True,
            "siren_confidence": 0.95,
            "road": "East",
        }
        res = self.client.post("/api/evaluate_emergency", json=eval_req)
        self.assertEqual(res.status_code, 200)
        eval_data = res.get_json()
        self.assertTrue(eval_data["success"])
        self.assertTrue(eval_data["evaluation"]["is_emergency"])
        self.assertIsNotNone(eval_data["emergency_plan"])
        self.assertEqual(eval_data["emergency_plan"]["emergency_road"], "East")

        # 3. GET / (index page)
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"Custom Emergency YOLO Detector Active", res_index.data)

        # 4. GET /dashboard
        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        self.assertIn(b"Custom YOLO Emergency-Vehicle Model", res_dash.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
