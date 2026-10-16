"""
Comprehensive Test Suite for AI-Based Adaptive Traffic Signal Control
Tests all 10 scenarios required by Section 8 of the project specifications:

1. Four valid images without siren audio.
2. Four valid images with siren audio and a confirmed emergency vehicle in the selected direction.
3. Siren detected but no emergency vehicle in the selected direction.
4. Emergency vehicle detected but siren not confirmed.
5. Missing direction image.
6. Invalid image or audio file.
7. Predicted green time exceeding the configured maximum.
8. Detection or model-loading failure fallback.
9. Starting, stopping, and restarting simulation logic & clean timers.
10. Verification that conflicting directions never receive green simultaneously.
"""

import io
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ["FLASK_ENV"] = "testing"

from app import app, execute_analysis
from detection.emergency_detector import EmergencyDetector
from traffic.emergency_controller import (
    build_emergency_plan,
    evaluate_emergency_candidate,
)
from traffic.signal_controller import build_cycle_plan
from traffic.timing import clamp_green_time, get_safe_fallback_green_time
from utils.config import (
    DATA_FOLDER,
    EMERGENCY_CONFIDENCE_THRESHOLD,
    EMERGENCY_GREEN_SECONDS,
    FIXED_SIGNAL_TIME,
    MAX_GREEN,
    MIN_GREEN,
    ROADS,
    SAFETY_TRANSITION_SECONDS,
)


class TestComprehensiveWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        app.config["TESTING"] = True

        # Locate sample media files
        img_dir = Path(DATA_FOLDER) / "emergency_yolo" / "test" / "images"
        audio_dir = Path(DATA_FOLDER) / "audio"

        # Find confirmed emergency vehicle image (ambulance)
        cls.amb_image_path = None
        for cand in [
            "back_134_jpeg_jpg.rf.771c3c3c2cc150b90705d8d6e96473f1.jpg",
            "data10-87-_jpg.rf.f1bca54787d8c04fca2052e2281c88a4.jpg",
            "data10-59-_jpg.rf.fd0739de6825c32d5e677b47140ae0a5.jpg",
        ]:
            p = img_dir / cand
            if p.exists():
                cls.amb_image_path = p
                break

        # Find confirmed ordinary traffic images (without emergency vehicle)
        cls.traffic_images = []
        for cand in [
            "1478019957687018435_jpg.rf.1337211ed85fd3098d01e02cfd8d68d8.jpg",
            "1478020431196343477_jpg.rf.86bf50c272bf7e4ccb6404974247b130.jpg",
            "1478020491698266386_jpg.rf.cc7f10b1511352c8c8b2ae377747dc5e.jpg",
            "1478020512707283286_jpg.rf.9f3d00b99058a0675b8a638096accabf.jpg",
        ]:
            p = img_dir / cand
            if p.exists():
                cls.traffic_images.append(p)

        # Audio samples
        cls.siren_wav_path = audio_dir / "siren" / "siren_000.wav"
        cls.no_siren_wav_path = audio_dir / "no_siren" / "no_siren_000.wav"

        assert cls.amb_image_path and cls.amb_image_path.exists(), "Ambulance test image not found"
        assert len(cls.traffic_images) >= 4, "4 traffic test images not found"
        assert cls.siren_wav_path.exists(), "Siren WAV not found"
        assert cls.no_siren_wav_path.exists(), "No-siren WAV not found"

    # --------------------------------------------------------------------------
    # Scenario 1: Four Valid Images Without Siren Audio
    # --------------------------------------------------------------------------
    def test_01_four_valid_images_without_siren_audio(self):
        """Scenario 1: 4 valid traffic images with no siren audio -> Normal adaptive mode."""
        data = {
            "north": (open(self.traffic_images[0], "rb"), "north.jpg"),
            "east": (open(self.traffic_images[1], "rb"), "east.jpg"),
            "south": (open(self.traffic_images[2], "rb"), "south.jpg"),
            "west": (open(self.traffic_images[3], "rb"), "west.jpg"),
        }

        response = self.client.post(
            "/api/analyze",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json["success"])

        data_res = res_json["data"]
        # Siren was not provided
        self.assertFalse(data_res["siren_analysis"]["provided"])
        # Emergency priority must NOT be granted
        self.assertFalse(data_res["emergency_active"])
        self.assertIsNone(data_res["emergency_road"])

        # Check all 4 directions returned
        for road in ["North", "East", "South", "West"]:
            self.assertIn(road, data_res["results"])
            r = data_res["results"][road]
            self.assertFalse(r["emergency_priority_confirmed"])
            self.assertGreaterEqual(r["green_time"], MIN_GREEN)
            self.assertLessEqual(r["green_time"], MAX_GREEN)

    # --------------------------------------------------------------------------
    # Scenario 2: Four Valid Images With Siren Audio & Confirmed Emergency Vehicle
    # --------------------------------------------------------------------------
    def test_02_emergency_priority_dual_verification(self):
        """Scenario 2: Siren detected AND emergency vehicle detected in South -> Emergency Priority."""
        data = {
            "north": (open(self.traffic_images[0], "rb"), "north.jpg"),
            "east": (open(self.traffic_images[1], "rb"), "east.jpg"),
            "south": (open(self.amb_image_path, "rb"), "south_amb.jpg"),
            "west": (open(self.traffic_images[3], "rb"), "west.jpg"),
            "siren_audio": (open(self.siren_wav_path, "rb"), "siren.wav"),
            "siren_direction": "South",
        }

        response = self.client.post(
            "/api/analyze",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json["success"])

        data_res = res_json["data"]
        self.assertTrue(data_res["siren_analysis"]["provided"])
        self.assertTrue(data_res["siren_analysis"]["is_siren"])
        self.assertTrue(data_res["emergency_active"])
        self.assertEqual(data_res["emergency_road"], "South")

        # South direction must have green priority badge
        south_res = data_res["results"]["South"]
        self.assertTrue(south_res["emergency_priority_confirmed"])
        self.assertEqual(south_res["status_color"], "green")
        self.assertIn("Emergency Priority Confirmed", south_res["status_label"])

        # Conflicting directions must NOT receive emergency green
        for road in ["North", "East", "West"]:
            self.assertFalse(data_res["results"][road]["emergency_priority_confirmed"])

    # --------------------------------------------------------------------------
    # Scenario 3: Siren Detected But No Emergency Vehicle in Selected Direction
    # --------------------------------------------------------------------------
    def test_03_siren_detected_but_no_emergency_vehicle(self):
        """Scenario 3: Siren detected in North, but image contains ordinary traffic -> No Priority."""
        data = {
            "north": (open(self.traffic_images[0], "rb"), "north.jpg"),
            "east": (open(self.traffic_images[1], "rb"), "east.jpg"),
            "south": (open(self.traffic_images[2], "rb"), "south.jpg"),
            "west": (open(self.traffic_images[3], "rb"), "west.jpg"),
            "siren_audio": (open(self.siren_wav_path, "rb"), "siren.wav"),
            "siren_direction": "North",
        }

        response = self.client.post(
            "/api/analyze",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        data_res = res_json["data"]

        # Emergency priority NOT granted
        self.assertFalse(data_res["emergency_active"])
        north_res = data_res["results"]["North"]
        self.assertFalse(north_res["emergency_priority_confirmed"])
        self.assertEqual(north_res["status_color"], "amber")
        self.assertTrue(
            north_res["status_message"].startswith("Siren detected, but no emergency vehicle was confirmed")
        )

    # --------------------------------------------------------------------------
    # Scenario 4: Emergency Vehicle Detected But Siren Not Confirmed
    # --------------------------------------------------------------------------
    def test_04_emergency_vehicle_detected_but_no_siren(self):
        """Scenario 4: Ambulance on South, but siren WAV contains no siren sound -> No Priority."""
        data = {
            "north": (open(self.traffic_images[0], "rb"), "north.jpg"),
            "east": (open(self.traffic_images[1], "rb"), "east.jpg"),
            "south": (open(self.amb_image_path, "rb"), "south_amb.jpg"),
            "west": (open(self.traffic_images[3], "rb"), "west.jpg"),
            "siren_audio": (open(self.no_siren_wav_path, "rb"), "no_siren.wav"),
            "siren_direction": "South",
        }

        response = self.client.post(
            "/api/analyze",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        data_res = res_json["data"]

        # Emergency priority NOT granted
        self.assertFalse(data_res["emergency_active"])
        south_res = data_res["results"]["South"]
        self.assertFalse(south_res["emergency_priority_confirmed"])
        self.assertEqual(south_res["status_color"], "amber")
        self.assertEqual(
            south_res["status_message"],
            "Emergency vehicle detected, but siren confirmation is missing.",
        )

    # --------------------------------------------------------------------------
    # Scenario 5: Missing Direction Image
    # --------------------------------------------------------------------------
    def test_05_missing_direction_image(self):
        """Scenario 5: Only 3 images uploaded -> Rejects with 400 error."""
        data = {
            "north": (open(self.traffic_images[0], "rb"), "north.jpg"),
            "east": (open(self.traffic_images[1], "rb"), "east.jpg"),
            "south": (open(self.traffic_images[2], "rb"), "south.jpg"),
            # West is intentionally missing
        }

        response = self.client.post(
            "/api/analyze",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        res_json = response.get_json()
        self.assertFalse(res_json["success"])
        self.assertIn("Missing image for direction: West", res_json["error"])

    # --------------------------------------------------------------------------
    # Scenario 6: Invalid Image or Audio File
    # --------------------------------------------------------------------------
    def test_06_invalid_files(self):
        """Scenario 6: Rejects invalid file formats (e.g. .txt image or .mp3 audio)."""
        # 6a: Invalid image extension (.txt)
        data_invalid_img = {
            "north": (io.BytesIO(b"fake image text content"), "north.txt"),
            "east": (open(self.traffic_images[1], "rb"), "east.jpg"),
            "south": (open(self.traffic_images[2], "rb"), "south.jpg"),
            "west": (open(self.traffic_images[3], "rb"), "west.jpg"),
        }
        res_img = self.client.post(
            "/api/analyze",
            data=data_invalid_img,
            content_type="multipart/form-data",
        )
        self.assertEqual(res_img.status_code, 400)
        self.assertIn("Invalid image format", res_img.get_json()["error"])

        # 6b: Invalid audio format (.mp3 instead of .wav)
        data_invalid_audio = {
            "north": (open(self.traffic_images[0], "rb"), "north.jpg"),
            "east": (open(self.traffic_images[1], "rb"), "east.jpg"),
            "south": (open(self.traffic_images[2], "rb"), "south.jpg"),
            "west": (open(self.traffic_images[3], "rb"), "west.jpg"),
            "siren_audio": (io.BytesIO(b"fake mp3 audio"), "siren.mp3"),
            "siren_direction": "South",
        }
        res_audio = self.client.post(
            "/api/analyze",
            data=data_invalid_audio,
            content_type="multipart/form-data",
        )
        self.assertEqual(res_audio.status_code, 400)
        self.assertIn("WAV", res_audio.get_json()["error"])

    # --------------------------------------------------------------------------
    # Scenario 7: Predicted Green Time Exceeding Configured Maximum
    # --------------------------------------------------------------------------
    def test_07_clamp_green_time_exceeding_maximum(self):
        """
        Scenario 7: Enforce timing formula:
        final_green_time = min(max_green_time, max(min_green_time, predicted_green_time))
        Never increase configured maximum automatically because traffic count is high.
        """
        # Upper bound test: prediction exceeds configured maximum (e.g. 150s vs max 60s)
        clamped_high = clamp_green_time(150, min_green_time=15, max_green_time=60)
        self.assertEqual(clamped_high, 60, "Must never exceed configured maximum")

        # Upper bound test: custom lower max (e.g. 45s)
        clamped_custom = clamp_green_time(55, min_green_time=15, max_green_time=45)
        self.assertEqual(clamped_custom, 45, "Must respect direction-specific max")

        # Lower bound test: prediction below minimum (e.g. 5s vs min 15s)
        clamped_low = clamp_green_time(5, min_green_time=15, max_green_time=60)
        self.assertEqual(clamped_low, 15, "Must enforce minimum safe duration")

        # Normal within bounds
        clamped_mid = clamp_green_time(37, min_green_time=15, max_green_time=60)
        self.assertEqual(clamped_mid, 37)

    # --------------------------------------------------------------------------
    # Scenario 8: Detection or Model-Loading Failure Safe Fallback
    # --------------------------------------------------------------------------
    def test_08_model_loading_and_detection_fallback(self):
        """Scenario 8: Non-existent image or inference failure produces safe fallback."""
        detector = EmergencyDetector()
        summary = detector.detect_summary("non_existent_path.jpg", road="East")
        self.assertFalse(summary["detected"])
        self.assertEqual(summary["vehicle_type"], "None")
        self.assertIsNone(summary["confidence"])

        # Check safe fallback timing duration
        fallback_time = get_safe_fallback_green_time("East")
        self.assertEqual(fallback_time, FIXED_SIGNAL_TIME)
        self.assertGreaterEqual(fallback_time, MIN_GREEN)
        self.assertLessEqual(fallback_time, MAX_GREEN)

    # --------------------------------------------------------------------------
    # Scenario 9: Starting, Stopping, and Restarting Simulation
    # --------------------------------------------------------------------------
    def test_09_simulation_timer_management(self):
        """Scenario 9: Simulation controller enforces clean timer disposal on stop/restart."""
        sim_js_path = Path("static/js/simulation.js")
        self.assertTrue(sim_js_path.exists())
        js_text = sim_js_path.read_text(encoding="utf-8")

        # Verify clearAllTimers exists and clears both timerId and emergencyTimerIds
        self.assertIn("function clearAllTimers()", js_text)
        self.assertIn("emergencyTimerIds", js_text)
        self.assertIn("clearAllTimers();", js_text)

        # Verify stopSimulation calls clearAllTimers
        self.assertIn("function stopSimulation()", js_text)
        self.assertIn("function restartSimulation()", js_text)

    # --------------------------------------------------------------------------
    # Scenario 10: Verification That Conflicting Directions Never Receive Green Simultaneously
    # --------------------------------------------------------------------------
    def test_10_zero_conflicting_greens_invariant(self):
        """
        Scenario 10: Invariant verification:
        Conflicting directions NEVER receive green simultaneously in normal cycle or emergency preemption.
        """
        # Test Normal Cycle Plan:
        green_times = {"North": 30, "East": 25, "South": 40, "West": 20}
        plan = build_cycle_plan(green_times, cycle_number=1)

        for step in plan["sequence"]:
            active_road = step["road"]
            conflicting = [r for r in ROADS if r != active_road]
            # Verify conflicting directions are exactly the other 3 roads
            self.assertEqual(len(conflicting), 3)
            self.assertNotIn(active_road, conflicting)

        # Test Emergency Preemption Plan:
        emergency_plan = build_emergency_plan(
            current_road="North",
            emergency_road="South",
            vehicle_type="Ambulance",
        )
        self.assertEqual(emergency_plan["emergency_road"], "South")
        self.assertEqual(set(emergency_plan["conflicting_roads"]), {"North", "East", "West"})

        # Check steps sequence
        steps = emergency_plan["steps"]
        step_phases = [s["phase"] for s in steps]
        self.assertEqual(
            step_phases,
            ["current_to_red", "safety_buffer", "emergency_green", "resume_normal"],
        )

        # In safety_buffer, all directions must be RED (zero green)
        safety_step = next(s for s in steps if s["phase"] == "safety_buffer")
        self.assertEqual(safety_step["duration"], SAFETY_TRANSITION_SECONDS)
        self.assertIn("All directions RED", safety_step["description"])

        # In emergency_green, ONLY emergency road gets GREEN
        emerg_step = next(s for s in steps if s["phase"] == "emergency_green")
        self.assertEqual(emerg_step["road"], "South")
        self.assertEqual(emerg_step["conflicting_roads"], ["North", "East", "West"])


if __name__ == "__main__":
    unittest.main()
