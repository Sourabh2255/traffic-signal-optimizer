"""
Emergency signal controller (Phase 11/12 & Custom YOLO Integration).

Emergency priority activates ONLY when BOTH conditions are verified:
    emergency_vehicle_detected AND siren_detected = emergency candidate

Key safety rules:
1. Detecting an ambulance/fire truck alone is NOT enough (siren must be verified).
2. Hearing a siren alone is NOT enough (emergency vehicle presence must be verified).
3. The direction must be reliably determined from the camera/road channel.
4. Conflicting directions must safely transition to RED before emergency direction receives GREEN.
5. Conflicting directions must NEVER receive GREEN simultaneously.
6. Normal adaptive green times remain strictly bounded between 15 and 60 seconds.
7. Normal operation resumes safely after the emergency condition clears.
"""

from typing import Optional, Dict, Any, List
from utils.config import (
    ROADS,
    SAFETY_TRANSITION_SECONDS,
    EMERGENCY_GREEN_SECONDS,
    EMERGENCY_CONFIDENCE_THRESHOLD,
    SIREN_CONFIDENCE_THRESHOLD,
    MIN_GREEN,
    MAX_GREEN,
)


def is_emergency_active(emergency_vehicle_type: str, siren_detected: bool) -> bool:
    """
    Core logical check for emergency priority.
    Maintained for direct backward compatibility.
    """
    vehicle_present = emergency_vehicle_type in ("Ambulance", "Fire Truck")
    return bool(vehicle_present and siren_detected)


def evaluate_emergency_candidate(
    vehicle_type: str,
    vehicle_confidence: Optional[float] = None,
    siren_detected: bool = False,
    siren_confidence: Optional[float] = None,
    siren_type: Optional[str] = None,
    road: Optional[str] = None,
    min_vehicle_conf: float = EMERGENCY_CONFIDENCE_THRESHOLD,
    min_siren_conf: float = SIREN_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluates both visual detection and acoustic siren detection with confidence thresholds.

    Rule: Emergency Vehicle Detected AND Siren Detected = Emergency Candidate

    Returns:
        Structured evaluation dict with validation reasoning and decision status.
    """
    # 1. Evaluate visual detection
    vehicle_present = vehicle_type in ("Ambulance", "Fire Truck")
    vehicle_confirmed = False
    if vehicle_present:
        if vehicle_confidence is not None:
            vehicle_confirmed = vehicle_confidence >= min_vehicle_conf
        else:
            vehicle_confirmed = True  # manual simulation override without explicit confidence score

    # 2. Evaluate acoustic siren detection
    siren_confirmed = False
    if siren_detected:
        if siren_confidence is not None:
            siren_confirmed = siren_confidence >= min_siren_conf
        else:
            siren_confirmed = True

    # 3. Apply the strict AND-condition
    is_candidate = bool(vehicle_confirmed and siren_confirmed)

    # 4. Formulate human-readable explanation matching system requirements
    if is_candidate:
        conf_str = f" ({vehicle_confidence:.0%})" if vehicle_confidence is not None else ""
        siren_conf_str = f" ({siren_confidence:.0%})" if siren_confidence is not None else ""
        siren_type_str = f" ({siren_type.replace('_', ' ')})" if siren_type else ""
        reason = (
            f"Emergency priority confirmed: {vehicle_type}{conf_str} detected with active siren{siren_type_str}{siren_conf_str}. "
            f"Safety transition granted."
        )
    elif vehicle_confirmed and not siren_confirmed:
        reason = (
            f"Emergency vehicle detected, but siren confirmation is missing. "
            f"Per protocol, normal adaptive signal timing is maintained."
        )
    elif siren_confirmed and not vehicle_confirmed:
        reason = (
            f"Siren detected, but no emergency vehicle was confirmed. "
            f"Normal adaptive signal timing is maintained."
        )
    else:
        reason = "No siren audio provided — normal adaptive mode."

    return {
        "is_emergency": is_candidate,
        "emergency_active": is_candidate,
        "candidate_status": "ACTIVE" if is_candidate else "INACTIVE",
        "vehicle_type": vehicle_type if vehicle_confirmed else "None",
        "vehicle_detected": vehicle_confirmed,
        "vehicle_confidence": vehicle_confidence,
        "siren_detected": siren_confirmed,
        "siren_confidence": siren_confidence,
        "siren_type": siren_type,
        "road": road,
        "reason": reason,
    }


def build_emergency_plan(
    current_road: str,
    emergency_road: str,
    vehicle_type: str = "Ambulance",
    vehicle_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Builds the ordered safety transition plan ensuring conflict-free signal execution.

    Safety protocol:
    1. current_to_red : Current active green light is switched to RED.
    2. safety_buffer  : All roads remain RED for SAFETY_TRANSITION_SECONDS (3s).
                        Zero simultaneous green signals allowed.
    3. emergency_green: Emergency direction receives GREEN for EMERGENCY_GREEN_SECONDS (25s).
                        All conflicting directions remain RED.
    4. resume_normal  : Cleared safely and returns to normal N -> E -> S -> W cycle,
                        with adaptive green times bounded between 15s and 60s.
    """
    if emergency_road not in ROADS:
        raise ValueError(f"Invalid emergency road '{emergency_road}'. Must be one of {ROADS}")

    conflicting_roads = [r for r in ROADS if r != emergency_road]

    return {
        "vehicle_type": vehicle_type,
        "emergency_road": emergency_road,
        "conflicting_roads": conflicting_roads,
        "safety_buffer_seconds": SAFETY_TRANSITION_SECONDS,
        "emergency_green_seconds": EMERGENCY_GREEN_SECONDS,
        "normal_green_bounds": {"min": MIN_GREEN, "max": MAX_GREEN},
        "steps": [
            {
                "phase": "current_to_red",
                "road": current_road,
                "duration": 0,
                "description": f"Switch active {current_road} signal to RED",
            },
            {
                "phase": "safety_buffer",
                "duration": SAFETY_TRANSITION_SECONDS,
                "description": f"All-red safety buffer ({SAFETY_TRANSITION_SECONDS}s). All directions RED.",
            },
            {
                "phase": "emergency_green",
                "road": emergency_road,
                "duration": EMERGENCY_GREEN_SECONDS,
                "conflicting_roads": conflicting_roads,
                "description": f"Grant GREEN to {emergency_road} for {EMERGENCY_GREEN_SECONDS}s. Conflicting directions RED.",
            },
            {
                "phase": "resume_normal",
                "duration": 0,
                "description": "Emergency cleared. Resume normal adaptive cycle sequence.",
            },
        ],
    }


def resolve_emergency_priority(
    siren_confirmed: bool,
    road_candidates: Dict[str, Dict[str, Any]],
    siren_confidence: Optional[float] = None,
    siren_type: Optional[str] = None,
    siren_provided: bool = False,
    min_vehicle_conf: float = EMERGENCY_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Direction-Independent Emergency Priority Resolution Engine.

    Evaluates omnidirectional siren classifier output against YOLO detections across
    ALL FOUR approaches (North, East, South, West) to identify which direction
    (if any) contains a confirmed emergency vehicle.

    Cases Handled:
    - Case A: Siren confirmed and an emergency vehicle is detected in exactly ONE direction.
              -> Grant priority to that direction.
    - Case B: Siren confirmed but NO emergency vehicle is detected in any direction.
              -> Maintain normal adaptive allocation; report searching all directions.
    - Case C: Emergency vehicle detected in one or more directions, but NO siren confirmed.
              -> Maintain normal adaptive allocation; mark candidate confirmation pending.
    - Case D: Siren confirmed and emergency vehicles are detected in MULTIPLE directions.
              -> Documented Disambiguation Policy:
                 1. If acoustic classifier confirms a specific siren type (e.g. 'ambulance_siren' vs 'fire_truck_siren')
                    and exactly one approach contains a matching vehicle type, prioritize that approach.
                 2. Otherwise, if one candidate has significantly higher visual confidence (margin >= 0.20),
                    prioritize the higher-confidence vehicle.
                 3. If ambiguity remains unresolved (comparable confidence, conflicting types), exclusive preemption
                    is suspended (is_ambiguous=True) to prevent intersection deadlock, preserving balanced adaptive cycle.
    - Case E: Neither siren nor emergency vehicle detected.
              -> Maintain normal adaptive allocation.
    """
    # 1. Identify all directions with confirmed emergency vehicles (Ambulance or Fire Truck)
    confirmed_candidates: List[Dict[str, Any]] = []
    all_detected_vehicles: List[Dict[str, Any]] = []

    for road in ROADS:
        r_info = road_candidates.get(road, {})
        has_vehicle = bool(r_info.get("detected", False) or r_info.get("emergency_detected", False))
        v_type = r_info.get("vehicle_type") or r_info.get("emergency_vehicle_type") or "None"
        v_conf = r_info.get("confidence") if r_info.get("confidence") is not None else r_info.get("emergency_confidence")

        # Recognize strictly emergency vehicle classes (Ambulance, Fire Truck)
        if has_vehicle and v_type in ("Ambulance", "Fire Truck"):
            vehicle_entry = {
                "road": road,
                "vehicle_type": v_type,
                "confidence": float(v_conf) if v_conf is not None else 1.0,
                "box": r_info.get("box") or r_info.get("emergency_box"),
            }
            all_detected_vehicles.append(vehicle_entry)
            if v_conf is None or float(v_conf) >= min_vehicle_conf:
                confirmed_candidates.append(vehicle_entry)

    confirmed_directions = [c["road"] for c in confirmed_candidates]

    # 2. Evaluate outcomes based on strict multi-directional logic
    emergency_active = False
    emergency_road: Optional[str] = None
    prioritized_vehicle: Optional[str] = None
    prioritized_confidence: Optional[float] = None
    is_ambiguous = False

    if siren_confirmed and len(confirmed_candidates) == 1:
        # Case A: Exactly one direction has confirmed emergency vehicle + siren
        c = confirmed_candidates[0]
        emergency_active = True
        emergency_road = c["road"]
        prioritized_vehicle = c["vehicle_type"]
        prioritized_confidence = c["confidence"]
        outcome_case = "CASE_A_SINGLE_EMERGENCY"
        conf_str = f" ({prioritized_confidence:.0%})" if prioritized_confidence is not None else ""
        siren_conf_str = f" ({siren_confidence:.0%})" if siren_confidence is not None else ""
        siren_type_str = f" ({siren_type.replace('_', ' ')})" if siren_type else ""
        reason = f"Emergency confirmed — safely prioritizing {emergency_road} direction."
        detailed_reason = f"Emergency confirmed: {prioritized_vehicle}{conf_str} detected on {emergency_road} with active siren{siren_type_str}{siren_conf_str}. Priority granted."

    elif siren_confirmed and len(confirmed_candidates) == 0:
        # Case B: Siren confirmed, but no emergency vehicle in any of the 4 directions
        emergency_active = False
        emergency_road = None
        outcome_case = "CASE_B_SIREN_NO_VEHICLE"
        reason = "Siren detected — searching all directions for emergency vehicles."
        detailed_reason = "Acoustic siren verified, but visual scans across all four directions (North, East, South, West) found no emergency vehicle. Normal adaptive mode maintained."

    elif not siren_confirmed and len(confirmed_candidates) > 0:
        # Case C: Emergency vehicle detected, but no siren confirmed
        emergency_active = False
        emergency_road = None
        outcome_case = "CASE_C_VEHICLE_NO_SIREN"
        reason = "Emergency vehicle detected, but siren confirmation is missing."
        c_desc = ", ".join(f"{c['vehicle_type']} on {c['road']}" for c in confirmed_candidates)
        detailed_reason = f"Emergency vehicle detected ({c_desc}), but siren confirmation is missing. Per safety protocol, normal adaptive mode maintained."

    elif siren_confirmed and len(confirmed_candidates) > 1:
        # Case D: Siren confirmed and multiple directions contain emergency vehicles
        outcome_case = "CASE_D_MULTIPLE_EMERGENCY"
        # Deterministic multi-tier selection policy:
        sorted_cands = sorted(confirmed_candidates, key=lambda x: x["confidence"], reverse=True)
        top1, top2 = sorted_cands[0], sorted_cands[1]

        # Tier 1: Disambiguation by acoustic siren class (if specialized model V2 identified specific siren type)
        type_matching_cands = []
        if siren_type:
            target_vehicle = "Ambulance" if "ambulance" in siren_type else ("Fire Truck" if "fire" in siren_type else None)
            if target_vehicle:
                type_matching_cands = [c for c in confirmed_candidates if c["vehicle_type"] == target_vehicle]

        if len(type_matching_cands) == 1:
            match_cand = type_matching_cands[0]
            emergency_active = True
            emergency_road = match_cand["road"]
            prioritized_vehicle = match_cand["vehicle_type"]
            prioritized_confidence = match_cand["confidence"]
            conf_str = f" ({prioritized_confidence:.0%})" if prioritized_confidence is not None else ""
            siren_conf_str = f" ({siren_confidence:.0%})" if siren_confidence is not None else ""
            reason = f"Multiple emergency vehicles detected. Disambiguated by siren type ({siren_type.replace('_', ' ')}): prioritizing {prioritized_vehicle} on {emergency_road}."
            detailed_reason = f"Multiple approaches contain emergency vehicles. Acoustic siren classification identified {siren_type.replace('_', ' ')}{siren_conf_str}, matching {prioritized_vehicle}{conf_str} on {emergency_road}. Priority granted."

        # Tier 2: Disambiguation by distinct confidence gap (margin >= 0.20)
        elif (top1["confidence"] - top2["confidence"]) >= 0.20:
            emergency_active = True
            emergency_road = top1["road"]
            prioritized_vehicle = top1["vehicle_type"]
            prioritized_confidence = top1["confidence"]
            reason = f"Multiple emergency vehicles detected. Prioritizing {top1['vehicle_type']} on {top1['road']} ({top1['confidence']:.0%} vs {top2['confidence']:.0%})."
            detailed_reason = reason

        # Tier 3: Ambiguous - Both candidates have comparable confidence; do not arbitrarily select first
        else:
            emergency_active = False
            emergency_road = None
            is_ambiguous = True
            c_desc = ", ".join(f"{c['vehicle_type']} on {c['road']} ({c['confidence']:.0%})" for c in confirmed_candidates)
            reason = f"Ambiguous multi-emergency: Confirmed emergency vehicles in multiple directions ({', '.join(confirmed_directions)}). Preserving balanced safety allocation."
            detailed_reason = f"Conflicting emergency vehicles detected on multiple approaches ({c_desc}) with comparable confidence. To prevent intersection deadlock, exclusive preemption is suspended in favor of safe adaptive queue clearing."

    else:
        # Case E: Neither siren nor emergency vehicle
        emergency_active = False
        emergency_road = None
        outcome_case = "CASE_E_NO_EMERGENCY"
        if siren_provided:
            reason = "Acoustic audio analyzed — no siren or emergency vehicle confirmed."
            detailed_reason = "Acoustic audio verified as non-emergency and no emergency vehicles detected. Normal adaptive mode active."
        else:
            reason = "No siren audio provided — normal adaptive mode."
            detailed_reason = "No siren audio provided — normal adaptive mode: signal timing optimized based on vehicular queue."

    # 3. Generate per-direction status outputs
    road_statuses: Dict[str, Dict[str, Any]] = {}
    for road in ROADS:
        road_cand = next((c for c in confirmed_candidates if c["road"] == road), None)
        any_cand = next((c for c in all_detected_vehicles if c["road"] == road), None)

        if emergency_active and road == emergency_road:
            road_statuses[road] = {
                "status_color": "green",
                "status_label": "Emergency Priority Confirmed",
                "status_message": f"Emergency confirmed — safely prioritizing {road} direction.",
                "emergency_priority_confirmed": True,
                "emergency_active": True,
            }
        elif emergency_active and road != emergency_road:
            road_statuses[road] = {
                "status_color": "blue",
                "status_label": "Normal Adaptive Mode",
                "status_message": f"Normal adaptive queue clearing — holding red during {emergency_road} emergency pass.",
                "emergency_priority_confirmed": False,
                "emergency_active": False,
            }
        else:
            # Emergency not active
            if road_cand or any_cand:
                cand_info = road_cand or any_cand
                if outcome_case == "CASE_D_MULTIPLE_EMERGENCY":
                    road_statuses[road] = {
                        "status_color": "amber",
                        "status_label": "Ambiguous Multi-Emergency",
                        "status_message": f"{cand_info['vehicle_type']} detected, but conflicting emergency in other approach. Preserving balanced cycle.",
                        "emergency_priority_confirmed": False,
                        "emergency_active": False,
                    }
                else:
                    road_statuses[road] = {
                        "status_color": "amber",
                        "status_label": "Vehicle Detected (No Siren)",
                        "status_message": "Emergency vehicle detected, but siren confirmation is missing.",
                        "emergency_priority_confirmed": False,
                        "emergency_active": False,
                    }
            elif outcome_case == "CASE_B_SIREN_NO_VEHICLE":
                road_statuses[road] = {
                    "status_color": "amber",
                    "status_label": "Siren Detected (Searching Approaches)",
                    "status_message": "Siren detected, but no emergency vehicle was confirmed — searching all directions for emergency vehicles.",
                    "emergency_priority_confirmed": False,
                    "emergency_active": False,
                }
            else:
                road_statuses[road] = {
                    "status_color": "blue",
                    "status_label": "Normal Adaptive Mode",
                    "status_message": "Normal adaptive mode: signal timing optimized based on vehicular queue.",
                    "emergency_priority_confirmed": False,
                    "emergency_active": False,
                }

    return {
        "emergency_active": emergency_active,
        "emergency_road": emergency_road,
        "vehicle_type": prioritized_vehicle or "None",
        "vehicle_confidence": prioritized_confidence,
        "confirmed_directions": confirmed_directions,
        "confirmed_candidates": confirmed_candidates,
        "all_detected_vehicles": all_detected_vehicles,
        "outcome_case": outcome_case,
        "is_ambiguous": is_ambiguous,
        "reason": reason,
        "detailed_reason": detailed_reason,
        "road_statuses": road_statuses,
        "siren_confirmed": siren_confirmed,
        "siren_confidence": siren_confidence,
        "siren_type": siren_type,
    }

