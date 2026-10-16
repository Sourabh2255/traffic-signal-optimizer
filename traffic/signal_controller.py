"""
Software traffic signal simulator (Phase 8/9).

This module only builds the DATA for one cycle of the fixed
North -> East -> South -> West sequence, with each road's ML-predicted
green time attached. The actual countdown / auto-advance animation runs
client-side in static/js/simulation.js — Flask is stateless per request,
so a live countdown is naturally a frontend timer rather than a backend
loop (this is a standard, simple approach for a college project; a
production system would use websockets or a hardware clock instead).
"""

from utils.config import ROADS, YELLOW_LIGHT_SECONDS, get_max_green_time


def build_cycle_plan(green_times: dict, cycle_number: int = 1, emergency_road: str = None) -> dict:
    """
    Args:
        green_times: {"North": 52, "East": 20, "South": 38, "West": 15}
        cycle_number: which simulation cycle this is (multi-cycle)
        emergency_road: confirmed emergency direction, if any

    Returns:
        Structured cycle plan dictionary with sequence, yellow duration, emergency road,
        and total cycle timing.
    """
    sequence = [
        {
            "road": road,
            "green_time": green_times[road],
            "max_green_time": get_max_green_time(road),
            "yellow_time": YELLOW_LIGHT_SECONDS,
            "is_emergency": bool(emergency_road and road == emergency_road),
        }
        for road in ROADS
    ]
    total_green = sum(green_times[r] for r in ROADS)
    return {
        "cycle": cycle_number,
        "sequence": sequence,
        "yellow_duration": YELLOW_LIGHT_SECONDS,
        "emergency_road": emergency_road,
        "is_emergency": bool(emergency_road is not None),
        "total_green_duration": total_green,
        "total_cycle_duration": total_green + (len(ROADS) * YELLOW_LIGHT_SECONDS),
    }

