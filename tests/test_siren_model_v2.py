"""
Comprehensive Test Suite for Siren Classifier V2 Integration.

Validates all 7 requirements from Task 7:
1. Model loading: loads models/siren_model_v2.joblib successfully.
2. Feature extraction: audio features contain exactly 46 values in correct order.
3. Output mapping: predictions map to ['ambulance_siren', 'fire_truck_siren', 'negative_sound'].
4. Invalid audio safety: errors, corrupt files, empty files do NOT grant priority.
5. Siren-only scenario: siren confirmed without emergency vehicle does NOT grant priority.
6. Vehicle-only scenario: emergency vehicle without confirmed siren does NOT grant priority.
7. Dual-verification priority: confirmed siren + emergency vehicle triggers safe preemption.
8. Multiple emergency vehicles policy: deterministic disambiguation and deadlock prevention.
9. Modular fallback: original detector remains functional and selectable.
"""

import os
import io
import sys
import unittest
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, STATE
from utils.config import (
    ROADS,
    EMERGENCY_CONFIDENCE_THRESHOLD,
    SIREN_CONFIDENCE_THRESHOLD,
    SIREN_V2_CONFIDENCE_THRESHOLD,
    SIREN_V2_CLASSES,
    SIREN_V2_EXPECTED_FEATURES,
    SAFETY_TRANSITION_SECONDS,
    EMERGENCY_GREEN_SECONDS,
    MIN_GREEN,
    MAX_GREEN,
    get_siren_model_v2_path,
)
from audio.feature_extractor_v2 import (
    extract_v2_features,
    load_audio_v2,
    get_v2_feature_names,
)
from audio.siren_detector_v2 import SirenDetectorV2
from audio.siren_detector import SirenDetector
from traffic.emergency_controller import (
    resolve_emergency_priority,
    evaluate_emergency_candidate,
    build_emergency_plan,
)


def get_dummy_normal_image():
    """Generates a simple 640x640 road image."""
    img = np.ones((640, 640, 3), dtype=np.uint8) * 110
    cv2.rectangle(img, (200, 0), (440, 640), (40, 40, 40), -1)
    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer)


def get_real_siren_audio_bytes():
    """Returns ambulance_siren.wav bytes if available."""
    wav_path = os.path.join(PROJECT_ROOT, "ambulance_siren.wav")
    if os.path.exists(wav_path):
        with open(wav_path, "rb") as f:
            return io.BytesIO(f.read())
    return None


class TestSirenModelV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.v2_path = get_siren_model_v2_path()
        cls.detector_v2 = SirenDetectorV2(cls.v2_path)
        cls.detector_v1 = SirenDetector()

    # =========================================================================
    # Requirement 1: Model loads successfully
    # =========================================================================
    def test_01_model_loading_and_pipeline_structure(self):
        """Verify siren_model_v2.joblib loads as a valid Pipeline with scaler and svm."""
        self.assertTrue(os.path.exists(self.v2_path), f"V2 model not found at {self.v2_path}")
        self.assertIsNotNone(self.detector_v2.model)
        self.assertTrue(hasattr(self.detector_v2.model, "predict"))
        self.assertTrue(hasattr(self.detector_v2.model, "predict_proba"))

        # Pipeline validation
        named_steps = getattr(self.detector_v2.model, "named_steps", {})
        self.assertIn("scaler", named_steps, "Model pipeline missing StandardScaler step")
        self.assertIn("svm", named_steps, "Model pipeline missing SVC step")

        # Feature count check
        n_features = getattr(self.detector_v2.model, "n_features_in_", None)
        self.assertEqual(n_features, 46, f"Expected 46 input features, got {n_features}")

    # =========================================================================
    # Requirement 2: Audio features contain exactly 46 values
    # =========================================================================
    def test_02_feature_extraction_exactly_46_values(self):
        """Extract features from synthetic audio and verify exact 46-value length."""
        # 2 seconds of 1 kHz tone at 22,050 Hz
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        features = extract_v2_features(tone)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (46,), f"Expected shape (46,), got {features.shape}")
        self.assertEqual(len(features), SIREN_V2_EXPECTED_FEATURES)

        # Check no NaN or Infinite values
        self.assertFalse(np.isnan(features).any(), "Extracted features contain NaN")
        self.assertFalse(np.isinf(features).any(), "Extracted features contain Inf")

        # Check feature names list
        names = get_v2_feature_names()
        self.assertEqual(len(names), 46)
        self.assertEqual(names[0], "mfcc_0_mean")
        self.assertEqual(names[19], "mfcc_19_mean")
        self.assertEqual(names[20], "mfcc_0_std")
        self.assertEqual(names[39], "mfcc_19_std")
        self.assertEqual(names[40], "spectral_centroid_mean")
        self.assertEqual(names[41], "spectral_centroid_std")
        self.assertEqual(names[42], "spectral_rolloff_mean")
        self.assertEqual(names[43], "spectral_rolloff_std")
        self.assertEqual(names[44], "zero_crossing_rate_mean")
        self.assertEqual(names[45], "zero_crossing_rate_std")

    # =========================================================================
    # Requirement 3: Each output belongs to the three expected classes
    # =========================================================================
    def test_03_output_classes_and_probabilities(self):
        """Verify model predictions belong to ['ambulance_siren', 'fire_truck_siren', 'negative_sound']."""
        model_classes = list(self.detector_v2.model.classes_)
        for expected in SIREN_V2_CLASSES:
            self.assertIn(expected, model_classes, f"Class {expected} missing from model classes")

        # 1. Test ambulance_siren prediction on real recording
        wav_path = os.path.join(PROJECT_ROOT, "ambulance_siren.wav")
        if os.path.exists(wav_path):
            res_wav = self.detector_v2.predict(wav_path)
            self.assertIn(res_wav["predicted_class"], SIREN_V2_CLASSES)
            self.assertEqual(res_wav["predicted_class"], "ambulance_siren")
            self.assertTrue(res_wav["is_siren"])
            self.assertEqual(res_wav["siren_type"], "ambulance_siren")
            self.assertEqual(res_wav["vehicle_type"], "Ambulance")
            self.assertGreaterEqual(res_wav["confidence"], 0.70)
            self.assertAlmostEqual(sum(res_wav["probabilities"].values()), 1.0, places=2)

        # 2. Test negative_sound and fire_truck_siren classes
        scaler = self.detector_v2.model.named_steps["scaler"]
        svm = self.detector_v2.model.named_steps["svm"]

        # Test negative_sound class using model's negative sound space
        neg_start = svm.n_support_[0] + svm.n_support_[1]
        neg_features = scaler.inverse_transform(svm.support_vectors_[neg_start:neg_start + 1])
        neg_pred = str(self.detector_v2.model.predict(neg_features)[0])
        neg_proba = self.detector_v2.model.predict_proba(neg_features)[0]
        self.assertEqual(neg_pred, "negative_sound")
        self.assertIn(neg_pred, SIREN_V2_CLASSES)
        self.assertAlmostEqual(float(np.sum(neg_proba)), 1.0, places=2)

        # Test fire_truck_siren class using model's fire truck sound space
        ft_start = svm.n_support_[0]
        ft_features = scaler.inverse_transform(svm.support_vectors_[ft_start:ft_start + 1])
        ft_pred = str(self.detector_v2.model.predict(ft_features)[0])
        ft_proba = self.detector_v2.model.predict_proba(ft_features)[0]
        self.assertEqual(ft_pred, "fire_truck_siren")
        self.assertIn(ft_pred, SIREN_V2_CLASSES)
        self.assertAlmostEqual(float(np.sum(ft_proba)), 1.0, places=2)

    # =========================================================================
    # Requirement 4: Invalid audio does not trigger emergency priority
    # =========================================================================
    def test_04_invalid_audio_error_handling(self):
        """Errors must be handled gracefully and NEVER interpreted as confirmed sirens."""
        # Case 1: Non-existent file
        res_nonexistent = self.detector_v2.predict("non_existent_audio_file_12345.wav")
        self.assertFalse(res_nonexistent["is_siren"])
        self.assertEqual(res_nonexistent["confidence"], 0.0)
        self.assertIsNotNone(res_nonexistent["error"])

        # Case 2: Empty byte buffer (0 bytes)
        res_empty = self.detector_v2.predict(b"")
        self.assertFalse(res_empty["is_siren"])
        self.assertEqual(res_empty["confidence"], 0.0)
        self.assertIsNotNone(res_empty["error"])

        # Case 3: Corrupt bytes
        res_corrupt = self.detector_v2.predict(b"RIFF\x00\x00\x00\x00WAVEfmt corrupt_header_bytes")
        self.assertFalse(res_corrupt["is_siren"])
        self.assertEqual(res_corrupt["confidence"], 0.0)
        self.assertIsNotNone(res_corrupt["error"])

        # Case 4: None input
        res_none = self.detector_v2.predict(None)
        self.assertFalse(res_none["is_siren"])
        self.assertEqual(res_none["confidence"], 0.0)

    # =========================================================================
    # Requirement 5: Siren-only scenario does NOT grant emergency priority
    # =========================================================================
    def test_05_siren_only_no_emergency_vehicle(self):
        """Siren confirmed + 4 normal traffic images -> No priority granted."""
        data = {
            "north": (get_dummy_normal_image(), "north.jpg"),
            "east": (get_dummy_normal_image(), "east.jpg"),
            "south": (get_dummy_normal_image(), "south.jpg"),
            "west": (get_dummy_normal_image(), "west.jpg"),
            "siren_audio": (get_real_siren_audio_bytes(), "siren.wav"),
            "siren_version": "v2",
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()["data"]

        # Siren is detected
        self.assertTrue(json_data["siren_analysis"]["is_siren"])
        self.assertEqual(json_data["siren_analysis"]["model_version"], "v2")
        self.assertEqual(json_data["siren_analysis"]["siren_type"], "ambulance_siren")

        # But NO emergency vehicle is detected -> priority must NOT be granted
        self.assertFalse(json_data["emergency_active"])
        self.assertIsNone(json_data["emergency_road"])
        self.assertIsNone(json_data["emergency_plan"])
        self.assertEqual(json_data["emergency_analysis"]["outcome_case"], "CASE_B_SIREN_NO_VEHICLE")
        self.assertIn("searching all directions", json_data["emergency_analysis"]["reason"].lower())

        # All signals must remain bounded adaptive green times
        for road in ROADS:
            card = json_data["results"][road]
            self.assertFalse(card["emergency_active"])
            self.assertFalse(card["emergency_priority_confirmed"])
            self.assertGreaterEqual(json_data["green_times"][road], MIN_GREEN)
            self.assertLessEqual(json_data["green_times"][road], MAX_GREEN)

    # =========================================================================
    # Requirement 6: Vehicle-only scenario does NOT grant emergency priority
    # =========================================================================
    def test_06_vehicle_only_without_siren(self):
        """Emergency vehicle detected + NO siren audio -> No priority granted."""
        data = {
            "north": (get_dummy_normal_image(), "north.jpg"),
            "east": (get_dummy_normal_image(), "east.jpg"),
            "south": (get_dummy_normal_image(), "south.jpg"),
            "west": (get_dummy_normal_image(), "west.jpg"),
            "emergency_north": "Ambulance",
            "siren_version": "v2",
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()["data"]

        # Vehicle detected on North, but no siren confirmed
        self.assertFalse(json_data["siren_analysis"]["is_siren"])
        self.assertFalse(json_data["emergency_active"])
        self.assertIsNone(json_data["emergency_road"])
        self.assertIsNone(json_data["emergency_plan"])

        north_res = json_data["results"]["North"]
        self.assertTrue(north_res["emergency_detected"])
        self.assertFalse(north_res["emergency_priority_confirmed"])
        self.assertEqual(json_data["emergency_analysis"]["outcome_case"], "CASE_C_VEHICLE_NO_SIREN")
        self.assertIn("siren confirmation is missing", json_data["emergency_analysis"]["reason"].lower())

    # =========================================================================
    # Requirement 7: Siren + confirmed emergency vehicle follows safe priority workflow
    # =========================================================================
    def test_07_siren_plus_vehicle_safe_priority_workflow(self):
        """Siren confirmed + Ambulance detected on South -> Grant priority with safe transitions."""
        data = {
            "north": (get_dummy_normal_image(), "north.jpg"),
            "east": (get_dummy_normal_image(), "east.jpg"),
            "south": (get_dummy_normal_image(), "south.jpg"),
            "west": (get_dummy_normal_image(), "west.jpg"),
            "siren_audio": (get_real_siren_audio_bytes(), "siren.wav"),
            "emergency_south": "Ambulance",
            "siren_version": "v2",
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()["data"]

        # Emergency preemption activated for South
        self.assertTrue(json_data["emergency_active"])
        self.assertEqual(json_data["emergency_road"], "South")
        self.assertEqual(json_data["green_times"]["South"], EMERGENCY_GREEN_SECONDS)

        south_res = json_data["results"]["South"]
        self.assertTrue(south_res["emergency_priority_confirmed"])
        self.assertEqual(south_res["status_color"], "green")

        # Conflicting directions remain RED during emergency phase
        for r in ["North", "East", "West"]:
            self.assertFalse(json_data["results"][r]["emergency_active"])
            self.assertEqual(json_data["results"][r]["status_color"], "blue")

        # Verify emergency plan invariants
        plan = json_data["emergency_plan"]
        self.assertIsNotNone(plan)
        self.assertEqual(plan["emergency_road"], "South")
        self.assertEqual(set(plan["conflicting_roads"]), {"North", "East", "West"})
        self.assertEqual(plan["safety_buffer_seconds"], SAFETY_TRANSITION_SECONDS)
        self.assertEqual(plan["emergency_green_seconds"], EMERGENCY_GREEN_SECONDS)

        # Plan step sequence check
        phases = [s["phase"] for s in plan["steps"]]
        self.assertEqual(phases, ["current_to_red", "safety_buffer", "emergency_green", "resume_normal"])

    # =========================================================================
    # Requirement 8: Multiple emergency vehicles disambiguation & deadlock prevention
    # =========================================================================
    def test_08_multiple_emergency_vehicles_policy(self):
        """Test Case D disambiguation: siren type matching and deadlock prevention."""
        # Subcase 8A: Ambiguous case (two ambulances with identical confidence)
        road_cands_ambiguous = {
            "North": {"detected": True, "vehicle_type": "Ambulance", "confidence": 0.85},
            "South": {"detected": True, "vehicle_type": "Ambulance", "confidence": 0.85},
            "East": {"detected": False},
            "West": {"detected": False},
        }
        res_ambiguous = resolve_emergency_priority(
            siren_confirmed=True,
            road_candidates=road_cands_ambiguous,
            siren_confidence=0.90,
            siren_type="ambulance_siren",
        )
        self.assertFalse(res_ambiguous["emergency_active"])
        self.assertTrue(res_ambiguous["is_ambiguous"])
        self.assertIsNone(res_ambiguous["emergency_road"])
        self.assertEqual(res_ambiguous["outcome_case"], "CASE_D_MULTIPLE_EMERGENCY")

        # Subcase 8B: Disambiguated by siren type (Ambulance in North vs Fire Truck in South, siren=ambulance_siren)
        road_cands_hetero = {
            "North": {"detected": True, "vehicle_type": "Ambulance", "confidence": 0.82},
            "South": {"detected": True, "vehicle_type": "Fire Truck", "confidence": 0.82},
            "East": {"detected": False},
            "West": {"detected": False},
        }
        res_hetero = resolve_emergency_priority(
            siren_confirmed=True,
            road_candidates=road_cands_hetero,
            siren_confidence=0.92,
            siren_type="ambulance_siren",
        )
        self.assertTrue(res_hetero["emergency_active"])
        self.assertEqual(res_hetero["emergency_road"], "North")
        self.assertEqual(res_hetero["vehicle_type"], "Ambulance")

    # =========================================================================
    # Requirement 9: Modular fallback and API model info endpoint
    # =========================================================================
    def test_09_api_siren_model_info_and_v1_fallback(self):
        """Verify /api/siren_model_info returns info for both models and fallback works."""
        res = self.client.get("/api/siren_model_info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["v1"]["available"])
        self.assertTrue(data["v2"]["available"])
        self.assertEqual(data["v2"]["expected_features"], 46)
        self.assertIn("ambulance_siren", data["v2"]["classes"])

        # Test request using explicit v1 fallback
        data_v1 = {
            "north": (get_dummy_normal_image(), "north.jpg"),
            "east": (get_dummy_normal_image(), "east.jpg"),
            "south": (get_dummy_normal_image(), "south.jpg"),
            "west": (get_dummy_normal_image(), "west.jpg"),
            "siren_audio": (get_real_siren_audio_bytes(), "siren.wav"),
            "siren_version": "v1",
        }
        res_v1 = self.client.post("/api/analyze", data=data_v1, content_type="multipart/form-data")
        self.assertEqual(res_v1.status_code, 200)
        json_v1 = res_v1.get_json()["data"]
        self.assertEqual(json_v1["siren_analysis"]["model_version"], "v1")


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("RUNNING SIREN CLASSIFIER V2 INTEGRATION TEST SUITE")
    print("=" * 75)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSirenModelV2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 75)
    print(f"TESTS RUN: {result.testsRun} | FAILURES: {len(result.failures)} | ERRORS: {len(result.errors)}")
    print("=" * 75)

    if not result.wasSuccessful():
        sys.exit(1)
