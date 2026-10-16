"""
Custom YOLO-based Emergency Vehicle Detector.

Dedicated detection module for identifying ambulances and fire trucks
using the trained custom YOLO model (models/best_emergency_vehicle.pt).

Key responsibilities:
- Loads the custom model once at application startup (singleton pattern).
- Detects ambulances (Class 0) and fire trucks (Class 1) in images.
- Returns bounding boxes, class names, confidence scores, and detection status.
- Handles missing model files, unreadable images, and inference errors gracefully.
- Configurable confidence threshold via utils.config.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

from utils.config import (
    EMERGENCY_MODEL_PATH,
    EMERGENCY_CONFIDENCE_THRESHOLD,
    EMERGENCY_CLASSES,
)

logger = logging.getLogger(__name__)


class EmergencyDetector:
    """
    Singleton wrapper around the custom fine-tuned YOLO emergency vehicle detector.
    Caches the model instance at the class level to avoid expensive reloads per request.
    """

    _model: Optional[YOLO] = None
    _load_attempted: bool = False
    _load_error: Optional[str] = None

    def __init__(self, model_path: str = EMERGENCY_MODEL_PATH):
        self.model_path = model_path
        if EmergencyDetector._model is None and not EmergencyDetector._load_attempted:
            self._load_model()
        self.model = EmergencyDetector._model

    @classmethod
    def is_available(cls, model_path: str = EMERGENCY_MODEL_PATH) -> bool:
        """Returns True if the custom model weights exist on disk."""
        return os.path.exists(model_path) and os.path.isfile(model_path)

    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Returns metadata about the loaded emergency model."""
        available = cls.is_available()
        loaded = cls._model is not None
        names = {}
        if loaded and cls._model:
            names = cls._model.names
        elif available:
            names = EMERGENCY_CLASSES

        return {
            "model_path": EMERGENCY_MODEL_PATH,
            "exists": available,
            "loaded": loaded,
            "classes": names,
            "confidence_threshold": EMERGENCY_CONFIDENCE_THRESHOLD,
            "error": cls._load_error,
        }

    def _load_model(self) -> None:
        """Attempts to load the custom YOLO model weights once."""
        EmergencyDetector._load_attempted = True
        if not os.path.exists(self.model_path):
            error_msg = f"Custom emergency model file not found at: {self.model_path}"
            logger.warning(error_msg)
            EmergencyDetector._load_error = error_msg
            return

        try:
            logger.info(f"Loading custom emergency YOLO model from {self.model_path}...")
            EmergencyDetector._model = YOLO(self.model_path)
            EmergencyDetector._load_error = None
            logger.info("Custom emergency YOLO model loaded successfully.")
        except Exception as e:
            error_msg = f"Failed to load custom emergency model: {e}"
            logger.exception(error_msg)
            EmergencyDetector._load_error = error_msg
            EmergencyDetector._model = None

    def detect(self, image_path: str, conf: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Runs custom emergency YOLO inference on a single image.

        Args:
            image_path: Path to the image file.
            conf: Optional confidence threshold override.

        Returns:
            List of detection dicts:
            [
                {
                    "class_name": "Ambulance",
                    "class_id": 0,
                    "confidence": 0.875,
                    "box": [x1, y1, x2, y2],
                    "is_emergency": True
                },
                ...
            ]
        """
        if self.model is None:
            if not EmergencyDetector._load_attempted:
                self._load_model()
                self.model = EmergencyDetector._model
            if self.model is None:
                logger.warning("EmergencyDetector invoked without an active model. Returning empty detections.")
                return []

        if not os.path.exists(image_path):
            logger.error(f"Image not found for emergency detection: {image_path}")
            return []

        threshold = conf if conf is not None else EMERGENCY_CONFIDENCE_THRESHOLD

        try:
            results = self.model.predict(
                source=image_path,
                conf=threshold,
                imgsz=640,
                verbose=False,
            )
        except Exception as e:
            logger.exception(f"Inference error during emergency detection on {image_path}: {e}")
            return []

        if not results:
            return []

        result = results[0]
        detections: List[Dict[str, Any]] = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = round(float(box.conf[0].item()), 3)

            # Map class ID to human-readable capitalized name
            raw_name = self.model.names.get(class_id, f"Class_{class_id}")
            if class_id == 0 or "ambulance" in str(raw_name).lower():
                display_name = "Ambulance"
            elif class_id == 1 or "fire" in str(raw_name).lower():
                display_name = "Fire Truck"
            else:
                display_name = str(raw_name).title()

            xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

            detections.append({
                "class_name": display_name,
                "class_id": class_id,
                "confidence": confidence,
                "box": xyxy,
                "is_emergency": True,
            })

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect_summary(
        self, image_path: str, conf: Optional[float] = None, road: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Returns a high-level summary of emergency vehicle presence for a given road image.

        Args:
            image_path: Path to the image file.
            conf: Confidence threshold.
            road: Road name (e.g. North, East, South, West).

        Returns:
            dict containing detection status, primary vehicle type, highest confidence,
            bounding box, road direction, and raw detections list.
        """
        detections = self.detect(image_path, conf=conf)
        ambulance_detected = any(d["class_id"] == 0 or d["class_name"] == "Ambulance" for d in detections)
        fire_truck_detected = any(d["class_id"] == 1 or d["class_name"] == "Fire Truck" for d in detections)

        if detections:
            primary = detections[0]
            return {
                "detected": True,
                "ambulance_detected": ambulance_detected,
                "fire_truck_detected": fire_truck_detected,
                "vehicle_type": primary["class_name"],
                "class_id": primary["class_id"],
                "confidence": primary["confidence"],
                "box": primary["box"],
                "road": road,
                "detections": detections,
                "model_available": self.model is not None,
                "error": None,
            }
        else:
            return {
                "detected": False,
                "ambulance_detected": False,
                "fire_truck_detected": False,
                "vehicle_type": "None",
                "class_id": None,
                "confidence": None,
                "box": None,
                "road": road,
                "detections": [],
                "model_available": self.model is not None,
                "error": EmergencyDetector._load_error if self.model is None else None,
            }
