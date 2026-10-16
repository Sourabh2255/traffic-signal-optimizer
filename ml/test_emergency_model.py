#!/usr/bin/env python3
"""
Visual Prediction Testing Script for Trained Emergency Vehicle YOLO Model.

Loads the best trained weights (runs/detect/emergency_vehicle/weights/best.pt)
and runs inference on selected unseen test-split images, saving annotated
detections with bounding boxes, class names, and confidence scores to
runs/detect/emergency_vehicle_predictions/.
"""

import argparse
import sys
from pathlib import Path
from typing import List

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Error: Ultralytics is required. Install it with 'pip install ultralytics'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run visual test predictions using trained emergency vehicle YOLO model."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/emergency_vehicle/weights/best.pt",
        help="Path to trained model weights (default: runs/detect/emergency_vehicle/weights/best.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/emergency_yolo/test/images",
        help="Path to test images directory or specific image file (default: data/emergency_yolo/test/images).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/detect/emergency_vehicle_predictions",
        help="Directory to save prediction visual outputs (default: runs/detect/emergency_vehicle_predictions).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Inference image size (default: 416).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=15,
        help="Maximum number of test images to annotate and save (default: 15).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)

    print("=" * 78)
    print("EMERGENCY VEHICLE DETECTION: VISUAL PREDICTION TESTING")
    print("=" * 78)

    if not weights_path.is_file():
        print(f"[ERROR] Trained weights file not found at: {weights_path.resolve()}", file=sys.stderr)
        print("Please train the model first using 'ml/train_emergency_model.py'.", file=sys.stderr)
        sys.exit(1)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[ERROR] Source images path not found: {source_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Collect images
    if source_path.is_file():
        image_files = [source_path]
    else:
        # Collect image files from directory
        all_images = sorted(list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")))
        if not all_images:
            print(f"[ERROR] No images found in: {source_path.resolve()}", file=sys.stderr)
            sys.exit(1)

        # Select a sample of images (prioritizing test images)
        image_files = all_images[: args.max_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model Weights       : {weights_path.resolve()}")
    print(f"[INFO] Source Directory    : {source_path.resolve()}")
    print(f"[INFO] Output Directory    : {output_dir.resolve()}")
    print(f"[INFO] Confidence Threshold: {args.conf}")
    print(f"[INFO] Evaluating {len(image_files)} sample test image(s)...")

    model = YOLO(str(weights_path))

    total_detections = 0
    detections_by_class = {0: 0, 1: 0}
    names = model.names

    for idx, img_p in enumerate(image_files, 1):
        results = model.predict(
            source=str(img_p),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=True,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes
        box_count = len(boxes)
        total_detections += box_count

        det_summary = []
        for box in boxes:
            cid = int(box.cls[0].item())
            cconf = float(box.conf[0].item())
            cname = names.get(cid, str(cid))
            detections_by_class[cid] = detections_by_class.get(cid, 0) + 1
            det_summary.append(f"{cname} ({cconf:.2f})")

        det_str = ", ".join(det_summary) if det_summary else "No emergency vehicles detected"
        print(f"  [{idx}/{len(image_files)}] {img_p.name[:40]:<40} -> {det_str}")

    print("\n" + "=" * 78)
    print("TEST PREDICTION SUMMARY")
    print("=" * 78)
    print(f"Total Test Images Processed : {len(image_files)}")
    print(f"Total Detections Found      : {total_detections}")
    for cid, count in detections_by_class.items():
        print(f"  - Class {cid} ({names.get(cid, 'unknown')}): {count} detections")
    print(f"Annotated test prediction images saved to: {output_dir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
