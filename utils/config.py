"""
Central configuration file.

Every "magic number" used across the project (green-time limits, vehicle
weights, road capacity, etc.) lives here so it is easy to find, change,
and explain during a viva/demo. Each constant is defined exactly once.

ACADEMIC HONESTY NOTE: vehicle weights, road capacity, waiting time, the
service rate, and the green-time training labels are SIMULATION
ASSUMPTIONS, not measured real-world values.
"""

import os

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Flask app settings
# ---------------------------------------------------------------------------
APP_NAME = "AI Traffic Signal Time Optimization System"
DEBUG_MODE = True  # Turn OFF before any demo recording
SECRET_KEY = "dev-secret-key-change-later"  # Fine for a local college project

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_AUDIO_EXTENSIONS = {"wav"}

MAX_UPLOAD_SIZE_MB = 25  # 4 images + up to 4 audio clips per request
MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# The four roads, in the FIXED normal sequence (spec section 3).
# ---------------------------------------------------------------------------
ROADS = ["North", "East", "South", "West"]

# ---------------------------------------------------------------------------
# YOLO / Computer Vision (Phase 3)
# ---------------------------------------------------------------------------
# Smallest pretrained YOLOv8 model (COCO). Auto-downloaded on first run.
YOLO_MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35

# COCO class names -> our vehicle categories. COCO has NO ambulance or
# fire-truck class (see emergency section below).
VEHICLE_CLASS_MAP = {
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}

# ---------------------------------------------------------------------------
# Traffic density / demand (Phase 4) — simulation assumptions
# ---------------------------------------------------------------------------
VEHICLE_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.5,
    "bus": 2.0,
    "truck": 2.0,
}

# Estimated road capacity, used only for the density PERCENTAGE.
ROAD_CAPACITY = 40

# (upper bound of total vehicle count, label)
DENSITY_THRESHOLDS = [
    (5, "LOW"),
    (15, "MEDIUM"),
    (25, "HIGH"),
    (float("inf"), "VERY HIGH"),
]

# Contribution of each simulated waiting-second to the demand score.
WAITING_TIME_FACTOR = 0.5

# ---------------------------------------------------------------------------
# Green-time ML (Phase 5-7)
# ---------------------------------------------------------------------------
MIN_GREEN = 15
MAX_GREEN = 60
FIXED_SIGNAL_TIME = 30  # baseline for Fixed vs Adaptive comparison
SAFE_FALLBACK_GREEN_TIME = FIXED_SIGNAL_TIME  # safe fallback if prediction fails
YELLOW_LIGHT_SECONDS = 3  # safe clearance phase between green transitions

# Configured maximum green time per direction (can be customized per approach)
MAX_GREEN_TIME = {
    "North": 60,
    "East": 60,
    "South": 60,
    "West": 60,
}


def get_max_green_time(road: str) -> int:
    """Return the configured maximum green time for a given direction."""
    return int(MAX_GREEN_TIME.get(road, MAX_GREEN))


TRAFFIC_MODEL_PATH = os.path.join(MODELS_FOLDER, "traffic_model.pkl")
TRAFFIC_DATASET_PATH = os.path.join(DATA_FOLDER, "traffic", "training_data.csv")
TRAFFIC_METRICS_PATH = os.path.join(DATA_FOLDER, "traffic", "model_metrics.json")

TRAFFIC_MODEL_FEATURES = [
    "car_count", "motorcycle_count", "bus_count", "truck_count",
    "total_vehicle_count", "density_percentage", "demand_score", "waiting_time",
]

# Simulated waiting time (no physical sensor): grows each analysis cycle.
WAITING_TIME_PER_CYCLE = 12
MAX_SIMULATED_WAITING_TIME = 60

# ---------------------------------------------------------------------------
# Siren / Audio ML (Phase 10-11 & Model V2 Integration)
# ---------------------------------------------------------------------------
SIREN_MODEL_PATH = os.path.join(MODELS_FOLDER, "siren_model.pkl")
SIREN_DATA_DIR = os.path.join(DATA_FOLDER, "audio")
SIREN_SAMPLE_RATE = 22050
SIREN_CLIP_SECONDS = 3
N_MFCC = 13
SIREN_CONFIDENCE_THRESHOLD = 0.55

# Siren Model V2 settings (SVM pipeline on 46 MFCC + spectral features)
SIREN_MODEL_V2_PATH = os.path.join(MODELS_FOLDER, "siren_model_v2.joblib")
SIREN_V2_SAMPLE_RATE = 22050
SIREN_V2_CLIP_SECONDS = 5.0
SIREN_V2_N_MFCC = 20
SIREN_V2_EXPECTED_FEATURES = 46
SIREN_V2_CLASSES = ["ambulance_siren", "fire_truck_siren", "negative_sound"]
SIREN_V2_CONFIDENCE_THRESHOLD = 0.50

# Active siren detector version: 'v1' or 'v2'
# Defaults to 'v1' to maintain existing detector behavior as fallback until tests pass
SIREN_ACTIVE_VERSION = "v1"


def get_siren_model_v2_path() -> str:
    """Safely adapt path to siren_model_v2.joblib if located in models or elsewhere."""
    candidates = [
        SIREN_MODEL_V2_PATH,
        os.path.join(BASE_DIR, "siren_model_v2.joblib"),
        os.path.join(os.getcwd(), "models", "siren_model_v2.joblib"),
        os.path.join(os.getcwd(), "siren_model_v2.joblib"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return SIREN_MODEL_V2_PATH


# ---------------------------------------------------------------------------
# Custom Emergency Vehicle YOLO Detector (Ambulance & Fire Truck)
# ---------------------------------------------------------------------------
EMERGENCY_MODEL_PATH = os.path.join(MODELS_FOLDER, "best_emergency_vehicle.pt")
EMERGENCY_CONFIDENCE_THRESHOLD = 0.15
EMERGENCY_IMGSZ = 1280
EMERGENCY_IOU_THRESHOLD = 0.40
EMERGENCY_CLASSES = {0: "Ambulance", 1: "Fire Truck"}

# Emergency vehicle options for UI forms:
# Defaults to Auto-Detect using the fine-tuned YOLO model, with manual simulation options
EMERGENCY_VEHICLE_TYPES = ["Auto-Detect", "None", "Ambulance", "Fire Truck"]

# ---------------------------------------------------------------------------
# Emergency safety transition (Phase 12) — simplified software model
# ---------------------------------------------------------------------------
SAFETY_TRANSITION_SECONDS = 3
EMERGENCY_GREEN_SECONDS = 25

# ---------------------------------------------------------------------------
# Fixed vs Adaptive comparison (Phase 14) — simulation only
# ---------------------------------------------------------------------------
# Assumed vehicles that clear per second of green light.
SERVICE_RATE_VEHICLES_PER_SEC = 0.5
