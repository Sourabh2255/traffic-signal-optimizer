"""
Comprehensive Test Suite for Emergency Ambulance Detection & Signal Priority.

Validates all 7 scenarios specified in Task 6:
1. Ambulance image without siren.
2. Siren without emergency vehicle.
3. Normal traffic image without siren.
4. Correct ambulance and siren with matching direction.
5. Wrong direction selected.
6. No detections returned by YOLO (e.g. blank/empty image).
7. Signal transitions and simulation restart (safety buffer, zero simultaneous green).
"""

import os
import io
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, STATE
from utils.config import (
    ROADS, EMERGENCY_CONFIDENCE_THRESHOLD, SIREN_CONFIDENCE_THRESHOLD,
    SAFETY_TRANSITION_SECONDS, EMERGENCY_GREEN_SECONDS, MIN_GREEN, MAX_GREEN
)
from detection.emergency_detector import EmergencyDetector
from audio.siren_detector import SirenDetector
from traffic.emergency_controller import evaluate_emergency_candidate, build_emergency_plan

def client_fixture():
    app.config["TESTING"] = True
    return app.test_client()

def get_normal_image_bytes():
    """Generates a simple 640x640 traffic-like image with a few grey rectangles (cars)."""
    img = np.ones((640, 640, 3), dtype=np.uint8) * 120
    # draw road asphalt
    cv2.rectangle(img, (200, 0), (440, 640), (40, 40, 40), -1)
    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer)

def get_blank_image_bytes():
    """Generates a plain black 640x640 image returning 0 detections."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer)

def get_ambulance_image_bytes():
    """Loads the real Ambulance.jpg image."""
    img_path = "Ambulance.jpg"
    if not os.path.exists(img_path):
        img_path = os.path.join("c:", os.sep, "Users", "Saurabh", "Downloads", "Ambulance.jpg")
    with open(img_path, "rb") as f:
        return io.BytesIO(f.read())

def get_siren_audio_bytes():
    """Loads the real ambulance siren WAV file."""
    wav_path = "ambulance_siren.wav"
    if not os.path.exists(wav_path):
        wav_path = os.path.join("c:", os.sep, "Users", "Saurabh", "Downloads", "u_xg7ssi08yr-ambulance-siren-363656.wav")
    with open(wav_path, "rb") as f:
        return io.BytesIO(f.read())


# ===========================================================================
# SCENARIO 1: Ambulance image without siren
# ===========================================================================
def test_scenario_1_ambulance_without_siren(client):
    """Ambulance on South, but NO siren audio uploaded."""
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_direction": "South",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # Ambulance is visually detected on South
    south_res = json_data["results"]["South"]
    assert south_res["ambulance_detected"] is True
    assert south_res["emergency_detected"] is True
    assert south_res["emergency_confidence"] >= EMERGENCY_CONFIDENCE_THRESHOLD

    # But siren was not provided
    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    assert json_data["emergency_plan"] is None
    assert south_res["emergency_active"] is False
    assert "siren confirmation is missing" in south_res["status_message"].lower()

    ea = json_data["emergency_analysis"]
    assert ea["confirmed"] is False
    assert "siren confirmation is missing" in ea["reason"].lower()


# ===========================================================================
# SCENARIO 2: Siren without emergency vehicle
# ===========================================================================
def test_scenario_2_siren_without_emergency_vehicle(client):
    """Siren audio on South, but normal traffic (no ambulance) in all directions."""
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "South",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # Siren is detected
    assert json_data["siren_analysis"]["is_siren"] is True
    assert json_data["siren_analysis"]["confidence"] >= SIREN_CONFIDENCE_THRESHOLD

    # But no emergency vehicle on South
    south_res = json_data["results"]["South"]
    assert south_res["emergency_detected"] is False

    # Priority NOT confirmed
    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    assert json_data["emergency_plan"] is None

    ea = json_data["emergency_analysis"]
    assert ea["confirmed"] is False
    assert "no emergency vehicle was confirmed" in ea["reason"].lower()


# ===========================================================================
# SCENARIO 3: Normal traffic image without siren
# ===========================================================================
def test_scenario_3_normal_traffic_without_siren(client):
    """Standard rush hour: all normal images, no siren audio."""
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    assert json_data["emergency_plan"] is None
    for road in ROADS:
        assert json_data["results"][road]["emergency_active"] is False
        assert json_data["results"][road]["emergency_detected"] is False


# ===========================================================================
# SCENARIO 4: Correct ambulance and siren with matching direction
# ===========================================================================
def test_scenario_4_matching_ambulance_and_siren(client):
    """Ambulance on South + Siren audio assigned to South."""
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "South",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # Both conditions met on South
    south_res = json_data["results"]["South"]
    assert south_res["ambulance_detected"] is True
    assert south_res["emergency_detected"] is True
    assert south_res["emergency_confidence"] >= EMERGENCY_CONFIDENCE_THRESHOLD
    assert json_data["siren_analysis"]["is_siren"] is True
    assert json_data["siren_analysis"]["confidence"] >= SIREN_CONFIDENCE_THRESHOLD

    # Emergency priority CONFIRMED
    assert json_data["emergency_active"] is True
    assert json_data["emergency_road"] == "South"
    assert south_res["emergency_active"] is True
    assert south_res["status_color"] == "green"
    assert "emergency confirmed" in south_res["status_message"].lower()

    # Emergency plan is generated
    plan = json_data["emergency_plan"]
    assert plan is not None
    assert plan["emergency_road"] == "South"
    assert plan["safety_buffer_seconds"] == SAFETY_TRANSITION_SECONDS
    assert plan["emergency_green_seconds"] == EMERGENCY_GREEN_SECONDS

    # Emergency debug panel data
    ea = json_data["emergency_analysis"]
    assert ea["confirmed"] is True
    assert ea["selected_direction"] == "South"
    assert ea["ambulance_detected"] is True
    assert ea["siren_detected"] is True
    assert "emergency confirmed" in ea["reason"].lower()


# ===========================================================================
# SCENARIO 5: Wrong direction selected (mismatch)
# ===========================================================================
def test_scenario_5_mismatched_direction(client):
    """Ambulance is on South, but siren is assigned to North."""
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "North",  # MISMATCH!
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # Dual condition fails: South has vehicle but not siren; North has siren but not vehicle
    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    assert json_data["emergency_plan"] is None

    assert json_data["results"]["South"]["emergency_active"] is False
    assert json_data["results"]["North"]["emergency_active"] is False


# ===========================================================================
# SCENARIO 6: No detections returned by YOLO (blank image)
# ===========================================================================
def test_scenario_6_no_detections_graceful_fallback(client):
    """Blank black images returning 0 detections."""
    data = {
        "north": (get_blank_image_bytes(), "blank_north.jpg"),
        "east": (get_blank_image_bytes(), "blank_east.jpg"),
        "south": (get_blank_image_bytes(), "blank_south.jpg"),
        "west": (get_blank_image_bytes(), "blank_west.jpg"),
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # No crash, safe fallback bounded in [MIN_GREEN, MAX_GREEN]
    for road in ROADS:
        gt = json_data["green_times"][road]
        assert MIN_GREEN <= gt <= MAX_GREEN
    assert json_data["emergency_active"] is False


# ===========================================================================
# SCENARIO 7: Signal transitions and safety buffer invariants
# ===========================================================================
def test_scenario_7_emergency_safety_plan_invariants():
    """Verify safety plan steps, buffer duration, and conflicting direction red locks."""
    plan = build_emergency_plan(
        current_road="North",
        emergency_road="South",
        vehicle_type="Ambulance",
        vehicle_confidence=0.11,
    )
    assert plan["emergency_road"] == "South"
    assert "North" in plan["conflicting_roads"]
    assert "East" in plan["conflicting_roads"]
    assert "West" in plan["conflicting_roads"]
    assert "South" not in plan["conflicting_roads"]

    steps = plan["steps"]
    assert len(steps) == 4
    # Step 1: current active green to red
    assert steps[0]["phase"] == "current_to_red"
    # Step 2: all-red safety buffer
    assert steps[1]["phase"] == "safety_buffer"
    assert steps[1]["duration"] == SAFETY_TRANSITION_SECONDS
    # Step 3: emergency green for South
    assert steps[2]["phase"] == "emergency_green"
    assert steps[2]["road"] == "South"
    assert steps[3]["phase"] == "resume_normal"

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING 7 EMERGENCY AMBULANCE & SIGNAL PRIORITY TEST SCENARIOS")
    print("=" * 70)

    app.config["TESTING"] = True
    with app.test_client() as c:
        tests = [
            ("Scenario 1: Ambulance without siren", test_scenario_1_ambulance_without_siren),
            ("Scenario 2: Siren without emergency vehicle", test_scenario_2_siren_without_emergency_vehicle),
            ("Scenario 3: Normal traffic image without siren", test_scenario_3_normal_traffic_without_siren),
            ("Scenario 4: Correct ambulance and siren with matching direction", test_scenario_4_matching_ambulance_and_siren),
            ("Scenario 5: Wrong direction selected", test_scenario_5_mismatched_direction),
            ("Scenario 6: No detections returned by YOLO", test_scenario_6_no_detections_graceful_fallback),
        ]

        passed = 0
        for name, fn in tests:
            print(f"Running {name}...", end=" ")
            try:
                fn(c)
                print("PASSED [✓]")
                passed += 1
            except Exception as e:
                import traceback
                print(f"FAILED [✗]:")
                traceback.print_exc()

        print("Running Scenario 7: Signal transitions & safety buffer invariants...", end=" ")
        try:
            test_scenario_7_emergency_safety_plan_invariants()
            print("PASSED [✓]")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAILED [✗]:")
            traceback.print_exc()

    print("=" * 70)
    print(f"SUMMARY: {passed} / 7 SCENARIOS PASSED")
    print("=" * 70)
    if passed < 7:
        sys.exit(1)
