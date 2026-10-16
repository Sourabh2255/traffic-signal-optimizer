"""
Traffic density calculation (Phase 4).

Pure Data Science / rule-based logic — no ML here. Takes vehicle counts
(from YOLO, Phase 3) and produces a simple, explainable density reading.
"""

from utils.config import DENSITY_THRESHOLDS, ROAD_CAPACITY


def classify_density(total_vehicles: int) -> str:
    """Map a raw vehicle count to a LOW/MEDIUM/HIGH/VERY HIGH tier."""
    for threshold, label in DENSITY_THRESHOLDS:
        if total_vehicles <= threshold:
            return label
    return "VERY HIGH"


def calculate_density(total_vehicles: int, capacity: int = ROAD_CAPACITY) -> dict:
    """
    Returns both the tier label and a numerical percentage.

    Density % = vehicle_count / estimated_road_capacity * 100, capped at
    100 for readability. Road capacity is a configurable *assumption*,
    not a measured value.
    """
    percentage = min(100.0, round((total_vehicles / capacity) * 100, 1))
    return {
        "level": classify_density(total_vehicles),
        "percentage": percentage,
    }
