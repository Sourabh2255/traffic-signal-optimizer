"""
Simulation-based training dataset generator (Phase 5).

IMPORTANT / ACADEMIC HONESTY (see project spec section 17 & 36):
There is no real-world labeled traffic-signal dataset available for this
college project. Instead, we generate a dataset where the target
"optimal_green_time" is produced by a clearly documented RULE:

    demand = car*1 + motorcycle*0.5 + bus*2 + truck*2 + waiting_time*0.5
    green_time = scale(demand) clamped to [MIN_GREEN, MAX_GREEN] + small noise

This target represents a SIMULATION-BASED RECOMMENDED green time, not a
real-world-optimal timing. The Random Forest model trained on this data
(Phase 6) learns to approximate this rule from the underlying features —
which is still a legitimate ML exercise (regression, feature importance,
evaluation), but its predictions must never be presented as validated
real-world traffic engineering.
"""

import os
import numpy as np
import pandas as pd

from utils.config import (
    VEHICLE_WEIGHTS, WAITING_TIME_FACTOR, MIN_GREEN, MAX_GREEN,
    ROAD_CAPACITY, TRAFFIC_DATASET_PATH,
)

RNG_SEED = 42


def _demand_to_green_time(demand, max_possible_demand: float):
    """
    Linearly scale demand into the [MIN_GREEN, MAX_GREEN] range.
    `demand` may be a scalar or a numpy array, hence np.minimum (not the
    Python builtin min(), which fails on arrays with more than one element).
    """
    ratio = np.minimum(demand / max_possible_demand, 1.0)
    return MIN_GREEN + ratio * (MAX_GREEN - MIN_GREEN)


def generate_dataset(n_samples: int = 1200, noise_std: float = 3.0) -> pd.DataFrame:
    """Generate the simulation-based training dataset and return it as a DataFrame."""
    rng = np.random.default_rng(RNG_SEED)

    car = rng.integers(0, 35, n_samples)
    motorcycle = rng.integers(0, 20, n_samples)
    bus = rng.integers(0, 5, n_samples)
    truck = rng.integers(0, 6, n_samples)
    waiting_time = rng.integers(0, 60, n_samples)

    total = car + motorcycle + bus + truck
    density_pct = np.minimum(100.0, (total / ROAD_CAPACITY) * 100)

    demand = (
        car * VEHICLE_WEIGHTS["car"]
        + motorcycle * VEHICLE_WEIGHTS["motorcycle"]
        + bus * VEHICLE_WEIGHTS["bus"]
        + truck * VEHICLE_WEIGHTS["truck"]
        + waiting_time * WAITING_TIME_FACTOR
    )

    # Theoretical maximum demand (used to normalize demand -> green time)
    max_demand = (
        35 * VEHICLE_WEIGHTS["car"] + 20 * VEHICLE_WEIGHTS["motorcycle"]
        + 5 * VEHICLE_WEIGHTS["bus"] + 6 * VEHICLE_WEIGHTS["truck"]
        + 60 * WAITING_TIME_FACTOR
    )

    green_time = _demand_to_green_time(demand, max_demand)
    green_time = green_time + rng.normal(0, noise_std, n_samples)  # small realistic noise
    green_time = np.clip(green_time, MIN_GREEN, MAX_GREEN)

    df = pd.DataFrame({
        "car_count": car,
        "motorcycle_count": motorcycle,
        "bus_count": bus,
        "truck_count": truck,
        "total_vehicle_count": total,
        "density_percentage": density_pct.round(1),
        "demand_score": demand.round(2),
        "waiting_time": waiting_time,
        "optimal_green_time": green_time.round(1),
    })

    return df


def save_dataset(df: pd.DataFrame, path: str = TRAFFIC_DATASET_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)
    print(f"Generated {len(dataset)} rows -> {TRAFFIC_DATASET_PATH}")
    print(dataset.describe())
