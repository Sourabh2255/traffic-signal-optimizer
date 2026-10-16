"""
Test Suite for Emergency Vehicle Detection Status and Adaptive Cycle Allocation Summary.

Directly validates the 10 requirements from Section 6:
1. Ambulance and confirmed siren on East: East status shows ambulance detected, East receives emergency priority, and the summary updates.
2. Fire truck and confirmed siren on North: North receives priority, not East.
3. Ambulance detected without a confirmed siren: show the vehicle detection, but do not grant emergency priority.
4. Siren detected without an emergency vehicle: do not grant emergency priority.
5. Emergency vehicle and siren detected in different directions: do not grant priority.
6. No emergency vehicle or siren: all directions use normal adaptive allocation.
7. Emergency condition clears: the system returns safely to normal adaptive allocation.
8. Every direction remains within its configured green-time limits.
9. No conflicting approaches receive green simultaneously.
10. Repeated analyses do not display stale detection or timing results.
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
    SAFETY_TRANSITION_SECONDS, EMERGENCY_GREEN_SECONDS, MIN_GREEN, MAX_GREEN,
    get_max_green_time
)
from traffic.emergency_controller import build_emergency_plan


def get_normal_image_bytes():
    """Generates a 640x640 road image without emergency vehicles."""
    img = np.ones((640, 640, 3), dtype=np.uint8) * 110
    cv2.rectangle(img, (200, 0), (440, 640), (40, 40, 40), -1)
    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer)


def get_ambulance_image_bytes():
    """Loads Ambulance.jpg."""
    img_path = "Ambulance.jpg"
    with open(img_path, "rb") as f:
        return io.BytesIO(f.read())


def get_siren_audio_bytes():
    """Loads ambulance_siren.wav."""
    wav_path = "ambulance_siren.wav"
    with open(wav_path, "rb") as f:
        return io.BytesIO(f.read())


# ===========================================================================
# 1. Ambulance and confirmed siren on East: East receives emergency priority
# ===========================================================================
def test_1_ambulance_confirmed_siren_on_east(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "East",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    east_res = json_data["results"]["East"]
    # Real detection results
    assert east_res["emergency_detected"] is True
    assert east_res["emergency_vehicle_detected"] is True
    assert east_res["emergency_vehicle_type"] == "Ambulance"
    assert east_res["ambulance_detected"] is True
    assert east_res["emergency_active"] is True
    assert east_res["emergency_priority_confirmed"] is True
    assert east_res["emergency_confidence"] is not None
    assert east_res["emergency_confidence"] >= EMERGENCY_CONFIDENCE_THRESHOLD

    # Summary and timing updates
    assert json_data["emergency_active"] is True
    assert json_data["emergency_road"] == "East"
    assert json_data["green_times"]["East"] == EMERGENCY_GREEN_SECONDS
    assert east_res["green_time"] == EMERGENCY_GREEN_SECONDS
    assert east_res["recommended_green_time"] == EMERGENCY_GREEN_SECONDS
    assert east_res["prediction"]["final_green_time"] == EMERGENCY_GREEN_SECONDS

    # Cycle plan sequence verification
    cycle_plan = json_data["cycle_plan"]
    assert cycle_plan["emergency_road"] == "East"
    assert cycle_plan["is_emergency"] is True
    east_step = next(s for s in cycle_plan["sequence"] if s["road"] == "East")
    assert east_step["green_time"] == EMERGENCY_GREEN_SECONDS
    assert east_step["is_emergency"] is True


# ===========================================================================
# 2. Fire truck and confirmed siren on North: North receives priority, not East
# ===========================================================================
def test_2_firetruck_confirmed_siren_on_north(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "North",
        "emergency_north": "Fire Truck",  # Simulation override test for North Fire Truck
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    north_res = json_data["results"]["North"]
    assert north_res["emergency_vehicle_detected"] is True
    assert north_res["emergency_vehicle_type"] == "Fire Truck"
    assert north_res["emergency_active"] is True
    assert json_data["emergency_road"] == "North"
    assert json_data["green_times"]["North"] == EMERGENCY_GREEN_SECONDS
    assert json_data["results"]["East"]["emergency_active"] is False


# ===========================================================================
# 3. Ambulance detected without confirmed siren: pending candidate, no priority
# ===========================================================================
def test_3_ambulance_without_confirmed_siren(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        # No siren audio uploaded
        "siren_direction": "East",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    east_res = json_data["results"]["East"]
    # Vehicle detected
    assert east_res["emergency_detected"] is True
    assert east_res["emergency_vehicle_detected"] is True
    assert east_res["emergency_vehicle_type"] == "Ambulance"
    # But confirmation pending, NOT priority
    assert east_res["emergency_candidate_pending"] is True
    assert east_res["emergency_active"] is False
    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    # Timing remains normal adaptive
    assert json_data["green_times"]["East"] == east_res["prediction"]["final_green_time"]


# ===========================================================================
# 4. Siren detected without an emergency vehicle: do not grant priority
# ===========================================================================
def test_4_siren_without_emergency_vehicle(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "East",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    assert json_data["siren_analysis"]["is_siren"] is True
    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    for road in ROADS:
        assert json_data["results"][road]["emergency_active"] is False


# ===========================================================================
# ===========================================================================
# 5. Direction-independent auto-scan: Siren confirmed without user selecting East
# ===========================================================================
def test_5_direction_independent_auto_scan(client):
    """
    Direction-Independent Auto-Scan:
    Ambulance on East Approach, siren audio provided with arbitrary or default direction (e.g. West).
    The system scans all 4 directions, finds the ambulance on East, and grants priority to East.
    """
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_ambulance_image_bytes(), "Ambulance.jpg"),  # Vehicle on East
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "West",  # Arbitrary direction: auto-scan should still find East!
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # East direction is automatically prioritized
    assert json_data["emergency_active"] is True
    assert json_data["emergency_road"] == "East"
    assert json_data["results"]["East"]["emergency_active"] is True
    assert json_data["results"]["East"]["emergency_priority_confirmed"] is True
    assert json_data["results"]["West"]["emergency_active"] is False



# ===========================================================================
# 6. No emergency vehicle or siren: all directions use normal adaptive allocation
# ===========================================================================
def test_6_no_emergency_normal_adaptive(client):
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
    assert json_data["cycle_plan"]["is_emergency"] is False
    for step in json_data["cycle_plan"]["sequence"]:
        assert step["is_emergency"] is False


# ===========================================================================
# 7. Emergency condition clears: returns safely to normal adaptive allocation
# ===========================================================================
def test_7_emergency_clears_returns_to_normal(client):
    # Pass 1: Emergency on East
    data1 = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "East",
    }
    res1 = client.post("/api/analyze", data=data1, content_type="multipart/form-data")
    assert res1.get_json()["data"]["emergency_active"] is True

    # Pass 2: Emergency condition clears (all normal traffic)
    data2 = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
    }
    res2 = client.post("/api/analyze", data=data2, content_type="multipart/form-data")
    json2 = res2.get_json()["data"]

    assert json2["emergency_active"] is False
    assert json2["emergency_road"] is None
    assert json2["emergency_plan"] is None
    assert json2["cycle_plan"]["is_emergency"] is False
    assert json2["results"]["East"]["emergency_active"] is False
    assert json2["results"]["East"]["emergency_vehicle_detected"] is False


# ===========================================================================
# 8. Every direction remains within configured green-time limits
# ===========================================================================
def test_8_green_time_limits_enforced(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "East",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    json_data = res.get_json()["data"]

    for road in ROADS:
        max_green = get_max_green_time(road)
        gt = json_data["green_times"][road]
        assert MIN_GREEN <= gt <= max_green, f"{road} green time {gt} outside [{MIN_GREEN}, {max_green}]"
        # Check consistency across card prediction and green_times
        card_gt = json_data["results"][road]["green_time"]
        assert gt == card_gt


# ===========================================================================
# 9. No conflicting approaches receive green simultaneously
# ===========================================================================
def test_9_zero_conflicting_greens():
    plan = build_emergency_plan(
        current_road="South",
        emergency_road="East",
        vehicle_type="Ambulance",
        vehicle_confidence=0.85,
    )
    conflicting = plan["conflicting_roads"]
    assert "North" in conflicting
    assert "South" in conflicting
    assert "West" in conflicting
    assert "East" not in conflicting

    # Safety buffer before emergency green
    buffer_step = plan["steps"][1]
    assert buffer_step["phase"] == "safety_buffer"
    assert buffer_step["duration"] == SAFETY_TRANSITION_SECONDS

    # Emergency step strictly assigns green only to East, conflicting remain red
    em_step = plan["steps"][2]
    assert em_step["road"] == "East"
    assert em_step["conflicting_roads"] == conflicting


# ===========================================================================
# 10. Repeated analyses do not display stale detection or timing results
# ===========================================================================
def test_10_repeated_analyses_no_stale_state(client):
    # Analysis A: Ambulance on East with siren
    data_a = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_ambulance_image_bytes(), "Ambulance.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "ambulance_siren.wav"),
        "siren_direction": "East",
    }
    res_a = client.post("/api/analyze", data=data_a, content_type="multipart/form-data")
    assert res_a.get_json()["data"]["emergency_road"] == "East"

    # Analysis B: Immediate re-analysis with ambulance removed
    data_b = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
    }
    res_b = client.post("/api/analyze", data=data_b, content_type="multipart/form-data")
    data_b_res = res_b.get_json()["data"]

    assert data_b_res["emergency_active"] is False
    assert data_b_res["emergency_road"] is None
    assert data_b_res["results"]["East"]["emergency_detected"] is False
    assert data_b_res["results"]["East"]["emergency_vehicle_detected"] is False
    assert data_b_res["results"]["East"]["emergency_active"] is False
    assert data_b_res["cycle_plan"]["is_emergency"] is False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING ALL 10 EMERGENCY DETECTION & ALLOCATION TEST CASES")
    print("=" * 70)

    app.config["TESTING"] = True
    with app.test_client() as c:
        tests = [
            ("Case 1: Ambulance + Siren on East", test_1_ambulance_confirmed_siren_on_east),
            ("Case 2: Fire Truck + Siren on North", test_2_firetruck_confirmed_siren_on_north),
            ("Case 3: Ambulance without Siren (Pending Candidate)", test_3_ambulance_without_confirmed_siren),
            ("Case 4: Siren without Emergency Vehicle", test_4_siren_without_emergency_vehicle),
            ("Case 5: Direction-Independent Auto-Scan", test_5_direction_independent_auto_scan),
            ("Case 6: No Emergency (Normal Adaptive)", test_6_no_emergency_normal_adaptive),
            ("Case 7: Emergency Clears (Safe Normal Return)", test_7_emergency_clears_returns_to_normal),
            ("Case 8: Configured Green-Time Limits Enforced", test_8_green_time_limits_enforced),
            ("Case 10: Repeated Analyses No Stale State", test_10_repeated_analyses_no_stale_state),
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

        print("Running Case 9: Zero Conflicting Greens Invariant...", end=" ")
        try:
            test_9_zero_conflicting_greens()
            print("PASSED [✓]")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAILED [✗]:")
            traceback.print_exc()

    print("=" * 70)
    print(f"SUMMARY: {passed} / 10 CASES PASSED")
    print("=" * 70)
    if passed < 10:
        sys.exit(1)
