"""
Green-time constraint helpers shared across the adaptive traffic controller.
Enforces the critical timing rule:
final_green_time = min(max_green_time, max(min_green_time, predicted_green_time))
"""

from utils.config import MIN_GREEN, MAX_GREEN, SAFE_FALLBACK_GREEN_TIME, get_max_green_time


def clamp_green_time(
    predicted_time: float,
    min_green_time: float = MIN_GREEN,
    max_green_time: float = MAX_GREEN,
) -> float:
    """
    Constrains predicted green duration to safe operational bounds.
    Rule: final_green_time = min(max_green_time, max(min_green_time, predicted_green_time))
    """
    # Guard against inverted positional arguments
    lower_bound = float(min(min_green_time, max_green_time))
    upper_bound = float(max(min_green_time, max_green_time))
    return round(float(min(upper_bound, max(lower_bound, float(predicted_time)))), 1)


def get_safe_fallback_green_time(
    road_or_max=MAX_GREEN,
    min_green_time: float = MIN_GREEN,
    max_green_time: float = None,
) -> float:
    """
    Returns a safe baseline fallback duration when prediction fails.
    Supports being called with a road name (e.g. 'East') or explicit numeric bounds.
    """
    if isinstance(road_or_max, str):
        effective_max = get_max_green_time(road_or_max) if max_green_time is None else max_green_time
    else:
        effective_max = road_or_max if max_green_time is None else max_green_time

    return clamp_green_time(
        SAFE_FALLBACK_GREEN_TIME,
        min_green_time=min_green_time,
        max_green_time=effective_max,
    )

