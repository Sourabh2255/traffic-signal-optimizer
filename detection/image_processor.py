"""
Draws YOLO detection results (bounding boxes, class names, confidence
scores) onto an image using OpenCV, so the user can visually verify what
was detected.

Supports both general traffic vehicle detections (green bounding boxes)
and custom emergency vehicle detections (high-visibility red bounding boxes).
"""

import os
from typing import List, Dict, Any, Optional
import cv2

from utils.config import UPLOAD_FOLDER

# Regular traffic vehicle styling (Green accent)
VEHICLE_BOX_COLOR = (74, 222, 128)      # BGR
VEHICLE_LABEL_BG = (74, 222, 128)
VEHICLE_TEXT_COLOR = (15, 23, 42)

# Emergency vehicle styling (High-visibility Red/Crimson)
EMERGENCY_BOX_COLOR = (30, 30, 235)     # BGR: bright red
EMERGENCY_LABEL_BG = (30, 30, 235)
EMERGENCY_TEXT_COLOR = (255, 255, 255)  # white text

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_filename: str,
    emergency_detections: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Draw bounding boxes + labels on the image and save the result to uploads.

    Args:
        image_path: full path to original saved image
        detections: list of regular vehicle detection dicts from VehicleDetector
        output_filename: filename to save the annotated image as
        emergency_detections: optional list of emergency vehicle detections from EmergencyDetector

    Returns:
        The saved filename (relative to UPLOAD_FOLDER).
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    # 1. Draw regular vehicle detections (green)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), VEHICLE_BOX_COLOR, 2)

        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        label_top = max(y1 - text_h - 8, 0)
        cv2.rectangle(image, (x1, label_top), (x1 + text_w + 6, y1), VEHICLE_LABEL_BG, -1)
        cv2.putText(
            image, label, (x1 + 3, y1 - 4),
            FONT, 0.5, VEHICLE_TEXT_COLOR, 1, cv2.LINE_AA,
        )

    # 2. Draw emergency vehicle detections (red, thicker, prominent label)
    if emergency_detections:
        for em in emergency_detections:
            x1, y1, x2, y2 = [int(v) for v in em["box"]]
            label = f"[EMERGENCY] {em['class_name']} {em['confidence']:.2f}"

            # Thicker box for emergency vehicle
            cv2.rectangle(image, (x1, y1), (x2, y2), EMERGENCY_BOX_COLOR, 3)

            (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.55, 2)
            label_top = max(y1 - text_h - 10, 0)
            cv2.rectangle(image, (x1, label_top), (x1 + text_w + 8, y1), EMERGENCY_LABEL_BG, -1)
            cv2.putText(
                image, label, (x1 + 4, y1 - 5),
                FONT, 0.55, EMERGENCY_TEXT_COLOR, 2, cv2.LINE_AA,
            )

    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    return output_filename
