"""
Fixed-signal vs AI-Adaptive-signal comparison (Phase 14).

SIMULATION-BASED comparison only — not a real-world deployment result
(see project spec section 23, 36). Assumes a constant "service rate"
(vehicles that can pass per second of green light) to estimate vehicles
served and unmet demand ("waiting") for each strategy.
"""

from utils.config import ROADS, FIXED_SIGNAL_TIME, SERVICE_RATE_VEHICLES_PER_SEC


def compare_fixed_vs_adaptive(green_times: dict, counts_by_road: dict) -> dict:
    """
    Args:
        green_times: {"North": 52, "East": 20, ...} (ML-predicted, per road)
        counts_by_road: {"North": {"total": 28, ...}, ...}

    Returns a dict with per-strategy totals and an efficiency score.
    """
    fixed_total_green = FIXED_SIGNAL_TIME * len(ROADS)
    adaptive_total_green = sum(green_times.values())

    fixed_served, adaptive_served = {}, {}
    for road in ROADS:
        total = counts_by_road[road]["total"]
        fixed_served[road] = min(total, FIXED_SIGNAL_TIME * SERVICE_RATE_VEHICLES_PER_SEC)
        adaptive_served[road] = min(total, green_times[road] * SERVICE_RATE_VEHICLES_PER_SEC)

    total_vehicles = sum(c["total"] for c in counts_by_road.values()) or 1  # avoid /0

    fixed_vehicles_served = sum(fixed_served.values())
    adaptive_vehicles_served = sum(adaptive_served.values())

    fixed_waiting = sum(
        max(0, counts_by_road[r]["total"] - fixed_served[r]) for r in ROADS
    )
    adaptive_waiting = sum(
        max(0, counts_by_road[r]["total"] - adaptive_served[r]) for r in ROADS
    )

    return {
        "fixed": {
            "total_green_time": fixed_total_green,
            "vehicles_served": round(fixed_vehicles_served, 1),
            "unserved_vehicles": round(fixed_waiting, 1),
            "efficiency_score": round((fixed_vehicles_served / total_vehicles) * 100, 1),
        },
        "adaptive": {
            "total_green_time": adaptive_total_green,
            "vehicles_served": round(adaptive_vehicles_served, 1),
            "unserved_vehicles": round(adaptive_waiting, 1),
            "efficiency_score": round((adaptive_vehicles_served / total_vehicles) * 100, 1),
        },
        "total_vehicles": total_vehicles,
    }
