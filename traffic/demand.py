"""
Traffic demand score calculation (Phase 4).

Demand = weighted vehicle count + waiting-time contribution.
Weights and the waiting-time factor are configurable simulation
assumptions (utils/config.py), not measured real-world values.
"""

from utils.config import VEHICLE_WEIGHTS, WAITING_TIME_FACTOR


def calculate_demand(counts: dict, waiting_time: float = 0.0) -> float:
    """
    Args:
        counts: dict with keys car/motorcycle/bus/truck (e.g. from
            VehicleDetector.count_by_type)
        waiting_time: estimated seconds this road has been waiting
            (simulated — see Phase 15 / no physical sensor)

    Returns:
        A single numeric demand score.
    """
    vehicle_demand = sum(
        counts.get(vtype, 0) * weight
        for vtype, weight in VEHICLE_WEIGHTS.items()
    )
    waiting_contribution = waiting_time * WAITING_TIME_FACTOR
    return round(vehicle_demand + waiting_contribution, 2)
