"""
Comprehensive Test Suite for Direction-Independent Siren Detection.

Validates all 10 scenarios required by Section 7:
1. Siren + ambulance only in North.
2. Siren + ambulance only in East.
3. Siren + fire truck only in South.
4. Siren + emergency vehicle only in West.
5. Siren but no emergency vehicle in any direction.
6. Emergency vehicle but no confirmed siren.
7. Siren and emergency vehicles in two or more directions (Case D policy).
8. Low-confidence or ambiguous detections.
9. Repeated analyses where the emergency direction changes.
10. Emergency priority clears and normal allocation resumes.
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
from traffic.emergency_controller import resolve_emergency_priority, build_emergency_plan


def get_normal_image_bytes():
    """Generates a 640x640 traffic-like road image without emergency vehicles."""
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
# 1. Siren + ambulance only in North
# ===========================================================================
def test_1_siren_ambulance_only_in_north(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
        "emergency_north": "Ambulance",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # Priority granted to North
    assert json_data["emergency_active"] is True
    assert json_data["emergency_road"] == "North"
    north_res = json_data["results"]["North"]
    assert north_res["emergency_active"] is True
    assert north_res["emergency_priority_confirmed"] is True
    assert north_res["ambulance_detected"] is True
    assert north_res["emergency_vehicle_type"] == "Ambulance"
    assert north_res["status_color"] == "green"
    assert "emergency priority confirmed" in north_res["status_label"].lower()

    # Conflicting directions remain normal adaptive
    for r in ["East", "South", "West"]:
        assert json_data["results"][r]["emergency_active"] is False
        assert json_data["results"][r]["status_color"] == "blue"

    # Emergency plan and cycle allocation prioritize North
    assert json_data["emergency_plan"] is not None
    assert json_data["emergency_plan"]["emergency_road"] == "North"
    assert json_data["cycle_plan"]["is_emergency"] is True
    assert json_data["cycle_plan"]["emergency_road"] == "North"


# ===========================================================================
# 2. Siren + ambulance only in East
# ===========================================================================
def test_2_siren_ambulance_only_in_east(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
        "emergency_east": "Ambulance",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    assert json_data["emergency_active"] is True
    assert json_data["emergency_road"] == "East"
    east_res = json_data["results"]["East"]
    assert east_res["emergency_active"] is True
    assert east_res["emergency_priority_confirmed"] is True
    assert east_res["ambulance_detected"] is True
    assert east_res["status_color"] == "green"
    assert "emergency priority confirmed" in east_res["status_label"].lower()

    for r in ["North", "South", "West"]:
        assert json_data["results"][r]["emergency_active"] is False


# ===========================================================================
# 3. Siren + fire truck only in South
# ===========================================================================
def test_3_siren_firetruck_only_in_south(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
        "emergency_south": "Fire Truck",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    assert json_data["emergency_active"] is True
    assert json_data["emergency_road"] == "South"
    south_res = json_data["results"]["South"]
    assert south_res["emergency_active"] is True
    assert south_res["emergency_priority_confirmed"] is True
    assert south_res["fire_truck_detected"] is True
    assert south_res["emergency_vehicle_type"] == "Fire Truck"
    assert south_res["status_color"] == "green"
    assert "emergency priority confirmed" in south_res["status_label"].lower()


# ===========================================================================
# 4. Siren + emergency vehicle only in West
# ===========================================================================
def test_4_siren_emergency_vehicle_only_in_west(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
        "emergency_west": "Ambulance",
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    assert json_data["emergency_active"] is True
    assert json_data["emergency_road"] == "West"
    west_res = json_data["results"]["West"]
    assert west_res["emergency_active"] is True
    assert west_res["emergency_priority_confirmed"] is True
    assert west_res["status_color"] == "green"


# ===========================================================================
# 5. Siren but no emergency vehicle in any direction (Case B)
# ===========================================================================
def test_5_siren_no_emergency_vehicle_in_any_direction(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # Siren confirmed but no vehicle in any approach
    assert json_data["siren_analysis"]["is_siren"] is True
    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    assert json_data["emergency_plan"] is None

    # Does not falsely report ambulance
    ea = json_data["emergency_analysis"]
    assert ea["confirmed"] is False
    assert ea["emergency_detected"] is False
    assert ea["ambulance_detected"] is False
    assert ea["fire_truck_detected"] is False
    assert ea["emergency_vehicle_type"] == "None"
    assert "searching all directions" in ea["reason"].lower()

    # Direction cards show searching status
    for r in ROADS:
        card = json_data["results"][r]
        assert card["emergency_active"] is False
        assert card["emergency_priority_confirmed"] is False
        assert "searching all directions" in card["status_message"].lower()


# ===========================================================================
# 6. Emergency vehicle but no confirmed siren (Case C)
# ===========================================================================
def test_6_emergency_vehicle_without_confirmed_siren(client):
    data = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "emergency_east": "Ambulance",
        # No siren audio provided
    }
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()["data"]

    # Vehicle is detected on East
    east_res = json_data["results"]["East"]
    assert east_res["emergency_detected"] is True
    assert east_res["ambulance_detected"] is True
    assert east_res["emergency_vehicle_type"] == "Ambulance"

    # But priority is NOT granted
    assert east_res["emergency_active"] is False
    assert east_res["emergency_candidate_pending"] is True
    assert json_data["emergency_active"] is False
    assert json_data["emergency_road"] is None
    assert json_data["cycle_plan"]["is_emergency"] is False


# ===========================================================================
# 7. Siren and emergency vehicles in multiple directions (Case D)
# ===========================================================================
def test_7_siren_multiple_emergency_vehicles_policy():
    """
    Validates Case D resolution policy:
    1. If comparable confidence (< 0.20 margin), marks ambiguous to prevent deadlock.
    2. If one has >= 0.20 higher confidence, deterministically prioritizes the top candidate.
    """
    # 7A: Ambiguous scenario (North Ambulance 0.82 vs East Fire Truck 0.80)
    candidates_ambiguous = {
        "North": {"detected": True, "vehicle_type": "Ambulance", "confidence": 0.82},
        "East": {"detected": True, "vehicle_type": "Fire Truck", "confidence": 0.80},
        "South": {"detected": False, "vehicle_type": "None", "confidence": None},
        "West": {"detected": False, "vehicle_type": "None", "confidence": None},
    }
    res_amb = resolve_emergency_priority(
        siren_confirmed=True,
        road_candidates=candidates_ambiguous,
        siren_confidence=0.92,
        siren_provided=True,
    )
    assert res_amb["outcome_case"] == "CASE_D_MULTIPLE_EMERGENCY"
    assert res_amb["is_ambiguous"] is True
    assert res_amb["emergency_active"] is False
    assert res_amb["emergency_road"] is None
    assert "ambiguous" in res_amb["reason"].lower()
    assert res_amb["road_statuses"]["North"]["status_label"] == "Ambiguous Multi-Emergency"
    assert res_amb["road_statuses"]["East"]["status_label"] == "Ambiguous Multi-Emergency"

    # 7B: Decisive scenario (North Ambulance 0.95 vs East Fire Truck 0.65 -> margin 0.30 >= 0.20)
    candidates_decisive = {
        "North": {"detected": True, "vehicle_type": "Ambulance", "confidence": 0.95},
        "East": {"detected": True, "vehicle_type": "Fire Truck", "confidence": 0.65},
        "South": {"detected": False, "vehicle_type": "None", "confidence": None},
        "West": {"detected": False, "vehicle_type": "None", "confidence": None},
    }
    res_dec = resolve_emergency_priority(
        siren_confirmed=True,
        road_candidates=candidates_decisive,
        siren_confidence=0.92,
        siren_provided=True,
    )
    assert res_dec["outcome_case"] == "CASE_D_MULTIPLE_EMERGENCY"
    assert res_dec["is_ambiguous"] is False
    assert res_dec["emergency_active"] is True
    assert res_dec["emergency_road"] == "North"
    assert "prioritizing" in res_dec["reason"].lower()


# ===========================================================================
# 8. Low-confidence or ambiguous detections
# ===========================================================================
def test_8_low_confidence_detections():
    """Validates that emergency vehicle detection below threshold is ignored."""
    low_conf_candidates = {
        "North": {"detected": True, "vehicle_type": "Ambulance", "confidence": 0.10},  # < 0.15 threshold
        "East": {"detected": False, "vehicle_type": "None", "confidence": None},
        "South": {"detected": False, "vehicle_type": "None", "confidence": None},
        "West": {"detected": False, "vehicle_type": "None", "confidence": None},
    }
    res = resolve_emergency_priority(
        siren_confirmed=True,
        road_candidates=low_conf_candidates,
        siren_confidence=0.90,
        siren_provided=True,
        min_vehicle_conf=EMERGENCY_CONFIDENCE_THRESHOLD,
    )
    assert res["emergency_active"] is False
    assert res["emergency_road"] is None
    assert res["outcome_case"] == "CASE_B_SIREN_NO_VEHICLE"


# ===========================================================================
# 9. Repeated analyses where the emergency direction changes
# ===========================================================================
def test_9_repeated_analyses_direction_changes(client):
    # Step 1: North Ambulance
    data_1 = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
        "emergency_north": "Ambulance",
    }
    res_1 = client.post("/api/analyze", data=data_1, content_type="multipart/form-data")
    assert res_1.status_code == 200
    json_1 = res_1.get_json()["data"]
    assert json_1["emergency_road"] == "North"

    # Step 2: Direction changes to East Ambulance
    data_2 = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
        "emergency_east": "Ambulance",
    }
    res_2 = client.post("/api/analyze", data=data_2, content_type="multipart/form-data")
    assert res_2.status_code == 200
    json_2 = res_2.get_json()["data"]

    # North should now be inactive, East actively prioritized
    assert json_2["emergency_road"] == "East"
    assert json_2["results"]["East"]["emergency_active"] is True
    assert json_2["results"]["North"]["emergency_active"] is False


# ===========================================================================
# 10. Emergency priority clears and normal allocation resumes
# ===========================================================================
def test_10_emergency_clears_normal_allocation_resumes(client):
    # Step 1: Emergency on South
    data_em = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
        "siren_audio": (get_siren_audio_bytes(), "siren.wav"),
        "emergency_south": "Ambulance",
    }
    res_em = client.post("/api/analyze", data=data_em, content_type="multipart/form-data")
    assert res_em.get_json()["data"]["emergency_active"] is True

    # Step 2: Emergency clears
    data_normal = {
        "north": (get_normal_image_bytes(), "north.jpg"),
        "east": (get_normal_image_bytes(), "east.jpg"),
        "south": (get_normal_image_bytes(), "south.jpg"),
        "west": (get_normal_image_bytes(), "west.jpg"),
    }
    res_normal = client.post("/api/analyze", data=data_normal, content_type="multipart/form-data")
    assert res_normal.status_code == 200
    json_normal = res_normal.get_json()["data"]

    assert json_normal["emergency_active"] is False
    assert json_normal["emergency_road"] is None
    assert json_normal["emergency_plan"] is None
    assert json_normal["cycle_plan"]["is_emergency"] is False

    # All directions are bounded within configured [MIN_GREEN, MAX_GREEN]
    for r in ROADS:
        card = json_normal["results"][r]
        assert card["emergency_active"] is False
        assert card["status_label"] == "Normal Adaptive Mode"
        green = json_normal["green_times"][r]
        assert MIN_GREEN <= green <= get_max_green_time(r)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING ALL 10 DIRECTION-INDEPENDENT SIREN TEST SCENARIOS")
    print("=" * 70)

    app.config["TESTING"] = True
    with app.test_client() as c:
        tests = [
            ("Scenario 1: Siren + Ambulance only in North", test_1_siren_ambulance_only_in_north),
            ("Scenario 2: Siren + Ambulance only in East", test_2_siren_ambulance_only_in_east),
            ("Scenario 3: Siren + Fire Truck only in South", test_3_siren_firetruck_only_in_south),
            ("Scenario 4: Siren + Emergency Vehicle only in West", test_4_siren_emergency_vehicle_only_in_west),
            ("Scenario 5: Siren but no emergency vehicle in any direction", test_5_siren_no_emergency_vehicle_in_any_direction),
            ("Scenario 6: Emergency vehicle without confirmed siren", test_6_emergency_vehicle_without_confirmed_siren),
            ("Scenario 7: Siren & multiple emergency vehicles policy", lambda client: test_7_siren_multiple_emergency_vehicles_policy()),
            ("Scenario 8: Low-confidence detection filtering", lambda client: test_8_low_confidence_detections()),
            ("Scenario 9: Repeated analyses direction changes", test_9_repeated_analyses_direction_changes),
            ("Scenario 10: Emergency clears & normal allocation resumes", test_10_emergency_clears_normal_allocation_resumes),
        ]

        passed = 0
        for name, fn in tests:
            print(f"Running {name}...", end=" ", flush=True)
            try:
                fn(c)
                print("PASSED [✓]")
                passed += 1
            except Exception as e:
                import traceback
                print("FAILED [✗]:")
                traceback.print_exc()

    print("=" * 70)
    print(f"SUMMARY: {passed} / {len(tests)} SCENARIOS PASSED")
    print("=" * 70)
    if passed < len(tests):
        sys.exit(1)
