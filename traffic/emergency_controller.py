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
        reason = (
            f"Emergency priority confirmed: {vehicle_type}{conf_str} detected with active siren{siren_conf_str}. "
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
