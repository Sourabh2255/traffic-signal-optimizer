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


def evaluate_traffic_impact(analysis: dict) -> dict:
    """
    Constructs a comprehensive AI Traffic Impact Assessment comparing the
    unoptimized baseline situation against the AI-adaptive optimization results.

    Returns structured sections:
      - initial_situation: pre-optimization vehicle counts, types, density, queue, emergency status
      - optimization_results: recommended vs allocated green times, delta seconds & %, next signal
      - system_impact: waiting times, throughput, queue reduction, emergency clearance delay
      - provenance: explicit identification of measured vs model-estimated vs simulated metrics
    """
    if not analysis:
        return {}

    green_times = analysis.get("green_times", {})
    counts_by_road = analysis.get("counts_by_road", {})
    results = analysis.get("results", {})
    siren_analysis = analysis.get("siren_analysis", {})
    emergency_analysis = analysis.get("emergency_analysis", {})
    emergency_active = bool(analysis.get("emergency_active", False))
    emergency_road = analysis.get("emergency_road")

    comparison = compare_fixed_vs_adaptive(green_times, counts_by_road)

    # 1. Initial Traffic Situation (Pre-Optimization Baseline)
    total_vehicles = comparison["total_vehicles"]
    total_cars = sum(counts_by_road.get(r, {}).get("car", 0) for r in ROADS)
    total_motorcycles = sum(counts_by_road.get(r, {}).get("motorcycle", 0) for r in ROADS)
    total_buses = sum(counts_by_road.get(r, {}).get("bus", 0) for r in ROADS)
    total_trucks = sum(counts_by_road.get(r, {}).get("truck", 0) for r in ROADS)

    approaches_initial = {}
    for road in ROADS:
        road_res = results.get(road, {})
        counts = counts_by_road.get(road, {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0})
        density = road_res.get("density", {"level": "LOW", "percentage": 0.0})
        approaches_initial[road] = {
            "road": road,
            "total_vehicles": counts.get("total", 0),
            "counts": counts,
            "density_level": density.get("level", "LOW"),
            "density_pct": density.get("percentage", 0.0),
            "queue_level": f"{density.get('level', 'LOW')} ({counts.get('total', 0)} veh)",
            "baseline_green": FIXED_SIGNAL_TIME,
            "emergency_detected": road_res.get("emergency_detected", False),
            "emergency_vehicle_type": road_res.get("emergency_vehicle_type", "None"),
            "emergency_confidence": road_res.get("emergency_confidence"),
        }

    initial_situation = {
        "approaches": approaches_initial,
        "total_vehicles": total_vehicles,
        "total_cars": total_cars,
        "total_motorcycles": total_motorcycles,
        "total_buses": total_buses,
        "total_trucks": total_trucks,
        "baseline_cycle_green": FIXED_SIGNAL_TIME * len(ROADS),
        "baseline_cycle_total": (FIXED_SIGNAL_TIME * len(ROADS)) + (len(ROADS) * 3),
        "siren_status": {
            "provided": siren_analysis.get("provided", False),
            "detected": siren_analysis.get("is_siren", False),
            "confidence": siren_analysis.get("confidence"),
            "type": siren_analysis.get("siren_type") or ("Siren" if siren_analysis.get("is_siren") else "None"),
            "model_version": siren_analysis.get("model_version", "v1"),
        },
        "emergency_active": emergency_active,
        "emergency_road": emergency_road,
    }

    # 2. AI Optimization Results
    approaches_opt = {}
    for road in ROADS:
        road_res = results.get(road, {})
        alloc_green = green_times.get(road, FIXED_SIGNAL_TIME)
        raw_pred = road_res.get("raw_predicted_green_time") or road_res.get("prediction", {}).get("raw_prediction") or alloc_green
        delta_sec = round(alloc_green - FIXED_SIGNAL_TIME, 1)
        delta_pct = round(((alloc_green - FIXED_SIGNAL_TIME) / FIXED_SIGNAL_TIME) * 100, 1)

        if delta_sec > 0:
            status_desc = f"+{delta_sec}s Priority Extension"
        elif delta_sec < 0:
            status_desc = f"{delta_sec}s Surplus Reallocated"
        else:
            status_desc = "Baseline Maintained"

        approaches_opt[road] = {
            "road": road,
            "recommended_green": round(float(raw_pred), 1),
            "allocated_green": round(float(alloc_green), 1),
            "baseline_green": FIXED_SIGNAL_TIME,
            "delta_seconds": delta_sec,
            "delta_pct": delta_pct,
            "status_description": status_desc,
            "is_emergency_priority": bool(emergency_active and road == emergency_road),
        }

    next_signal_road = emergency_road if emergency_active else ROADS[0]

    optimization_results = {
        "approaches": approaches_opt,
        "next_signal_direction": next_signal_road,
        "adaptive_total_green": sum(green_times.values()),
        "adaptive_total_cycle": sum(green_times.values()) + (len(ROADS) * 3),
        "emergency_preemption": {
            "active": emergency_active,
            "road": emergency_road,
            "vehicle_type": emergency_analysis.get("emergency_vehicle_type") or "Ambulance",
            "reason": emergency_analysis.get("reason", "Standard adaptive operation"),
            "detailed_reason": emergency_analysis.get("detailed_reason", ""),
            "safety_buffer_seconds": 3,
            "priority_duration_seconds": 25,
        },
    }

    # 3. System Impact & Benefits (Baseline vs Optimized)
    fixed_served_total = comparison["fixed"]["vehicles_served"]
    adaptive_served_total = comparison["adaptive"]["vehicles_served"]
    fixed_unserved_total = comparison["fixed"]["unserved_vehicles"]
    adaptive_unserved_total = comparison["adaptive"]["unserved_vehicles"]

    # Queuing Delay Simulation Calculation (Webster uniform delay approximation)
    # Baseline fixed cycle: 120s green + 12s yellow = 132s
    # Served vehicles experience avg delay of (132 - 30) / 2 = 51.0s
    # Unserved vehicles wait through red + next full cycle: 51.0 + 132.0 = 183.0s
    baseline_delay_sum = sum(
        min(counts_by_road.get(r, {}).get("total", 0), FIXED_SIGNAL_TIME * SERVICE_RATE_VEHICLES_PER_SEC) * 51.0
        + max(0, counts_by_road.get(r, {}).get("total", 0) - FIXED_SIGNAL_TIME * SERVICE_RATE_VEHICLES_PER_SEC) * 183.0
        for r in ROADS
    )
    baseline_avg_delay = round(baseline_delay_sum / max(1, total_vehicles), 1)

    # Adaptive cycle: total_green + 12s yellow
    adapt_cycle_len = sum(green_times.values()) + (len(ROADS) * 3)
    adaptive_delay_sum = sum(
        min(counts_by_road.get(r, {}).get("total", 0), green_times.get(r, 30) * SERVICE_RATE_VEHICLES_PER_SEC)
        * ((adapt_cycle_len - green_times.get(r, 30)) / 2.0)
        + max(0, counts_by_road.get(r, {}).get("total", 0) - green_times.get(r, 30) * SERVICE_RATE_VEHICLES_PER_SEC)
        * (((adapt_cycle_len - green_times.get(r, 30)) / 2.0) + adapt_cycle_len)
        for r in ROADS
    )
    adaptive_avg_delay = round(adaptive_delay_sum / max(1, total_vehicles), 1)

    delay_reduction_sec = round(max(0.0, baseline_avg_delay - adaptive_avg_delay), 1)
    delay_reduction_pct = round((delay_reduction_sec / max(1.0, baseline_avg_delay)) * 100, 1)

    queue_reduction_veh = round(max(0.0, fixed_unserved_total - adaptive_unserved_total), 1)
    queue_reduction_pct = (
        round((queue_reduction_veh / fixed_unserved_total) * 100, 1) if fixed_unserved_total > 0 else 0.0
    )

    throughput_gain_veh = round(max(0.0, adaptive_served_total - fixed_served_total), 1)
    throughput_gain_pct = (
        round((throughput_gain_veh / max(1.0, fixed_served_total)) * 100, 1) if fixed_served_total > 0 else 0.0
    )

    # Emergency Delay Evaluation
    if emergency_active:
        em_baseline_delay = 45.0  # Average wait in 132s fixed round-robin cycle without preemption
        em_adaptive_delay = 3.0   # Enforced all-red safety buffer duration
        em_reduction_pct = round(((em_baseline_delay - em_adaptive_delay) / em_baseline_delay) * 100, 1)
        em_status_text = f"Reduced from ~45.0s fixed cycle wait to 3.0s all-red safety transition ({em_reduction_pct}% reduction)."
    else:
        em_baseline_delay = None
        em_adaptive_delay = None
        em_reduction_pct = None
        em_status_text = "Not available — no emergency vehicle verified in current cycle."

    system_impact = {
        "waiting_time": {
            "baseline_avg_seconds": baseline_avg_delay,
            "optimized_avg_seconds": adaptive_avg_delay,
            "reduction_seconds": delay_reduction_sec,
            "improvement_pct": delay_reduction_pct,
            "provenance": "SIMULATION ESTIMATE (Queuing Service Delay Model)",
        },
        "queue_length": {
            "baseline_unserved_vehicles": fixed_unserved_total,
            "optimized_unserved_vehicles": adaptive_unserved_total,
            "reduction_vehicles": queue_reduction_veh,
            "improvement_pct": queue_reduction_pct,
            "provenance": "SIMULATION ESTIMATE (Service Rate 0.5 veh/sec)",
        },
        "vehicles_processed": {
            "baseline_served_vehicles": fixed_served_total,
            "optimized_served_vehicles": adaptive_served_total,
            "increase_vehicles": throughput_gain_veh,
            "improvement_pct": throughput_gain_pct,
            "provenance": "SIMULATION ESTIMATE (Service Rate 0.5 veh/sec)",
        },
        "emergency_delay": {
            "baseline_delay_seconds": em_baseline_delay,
            "optimized_delay_seconds": em_adaptive_delay,
            "improvement_pct": em_reduction_pct,
            "description": em_status_text,
            "provenance": "SIMULATION MODEL (Safety Buffer Preemption Logic)",
        },
        "efficiency_score": {
            "baseline_score": comparison["fixed"]["efficiency_score"],
            "optimized_score": comparison["adaptive"]["efficiency_score"],
            "improvement_pct": round(comparison["adaptive"]["efficiency_score"] - comparison["fixed"]["efficiency_score"], 1),
            "provenance": "SIMULATION ESTIMATE (Served vs Total Ratio)",
        },
    }

    # 4. Explicit Provenance Map
    provenance = {
        "vehicle_counts": "MEASURED (COCO YOLOv8 Vehicle Detector)",
        "vehicle_classification": "MEASURED (COCO YOLOv8 Vehicle Classifier)",
        "emergency_vehicle": "MEASURED (Custom Fine-Tuned YOLOv8 Model)",
        "siren_classification": "MEASURED (Audio ML Classifier)",
        "traffic_density": "ESTIMATED (Road Capacity Model — 40 veh nominal capacity)",
        "recommended_green_time": "ESTIMATED (Random Forest ML Regression Model)",
        "allocated_green_time": "RULE-ENFORCED (Configured Operational Bounds [15s, 60s])",
        "service_throughput": "SIMULATION ESTIMATE (Constant Service Rate: 0.5 veh/sec)",
        "queuing_delay": "SIMULATION ESTIMATE (Queuing Clearance Delay Model)",
        "emergency_preemption_delay": "SIMULATION MODEL (All-Red Transition Buffer Logic)",
    }

    return {
        "initial_situation": initial_situation,
        "optimization_results": optimization_results,
        "system_impact": system_impact,
        "comparison": comparison,
        "provenance": provenance,
    }

