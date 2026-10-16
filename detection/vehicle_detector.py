"""
YOLO-based vehicle detection.

This module is PURE COMPUTER VISION: given an image, it returns exactly
what YOLO detected — nothing about traffic density, demand, or green time
is calculated here. That separation matters for the project's Data
Science requirements (CV, Data Science, and ML must stay clearly
distinct — see project spec section 10).

Uses a pre-trained Ultralytics YOLOv8 model (no training from scratch).
"""

from ultralytics import YOLO

from utils.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, VEHICLE_CLASS_MAP


class VehicleDetector:
    """
    Thin wrapper around a YOLO model, filtered to vehicle classes only.

    The model is loaded ONCE and cached at the class level, because
    loading YOLO weights on every request would be slow and wasteful —
    Flask's dev server keeps the process alive between requests, so this
    cache persists naturally.
    """

    _model = None

    def __init__(self):
        if VehicleDetector._model is None:
            VehicleDetector._model = YOLO(YOLO_MODEL_PATH)
        self.model = VehicleDetector._model

    def detect(self, image_path: str) -> list:
        """
        Run YOLO on one image and return only vehicle detections.

        Returns:
            list of dicts, e.g.:
            [{"class_name": "car", "confidence": 0.91,
              "box": [x1, y1, x2, y2]}, ...]

        Raises:
            Whatever exception YOLO/OpenCV raises on a corrupt or
            unreadable image — the caller (app.py) is responsible for
            catching this and showing a friendly error message.
        """
        results = self.model.predict(
            source=image_path,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            coco_name = self.model.names[class_id]

            if coco_name not in VEHICLE_CLASS_MAP:
                continue  # ignore non-vehicle objects (person, dog, etc.)

            detections.append({
                "class_name": VEHICLE_CLASS_MAP[coco_name],
                "confidence": round(float(box.conf[0]), 3),
                "box": [round(float(c), 1) for c in box.xyxy[0].tolist()],
            })

        return detections

    @staticmethod
    def count_by_type(detections: list) -> dict:
        """
        Count detections per vehicle type, plus a total.

        Always returns all four keys (even if zero) so templates and
        later feature-engineering code don't need defensive .get() calls.
        """
        counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        for det in detections:
            counts[det["class_name"]] += 1
        counts["total"] = sum(counts.values())
        return counts
