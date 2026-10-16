#!/usr/bin/env python3
"""
Custom YOLO Training Script for Emergency Vehicle Detection (Ambulance & Fire Truck).

Trains a lightweight YOLO model (preferred: yolo11n.pt) on the prepared
data/emergency_yolo/ dataset with two classes:
    Class 0: ambulance
    Class 1: fire_truck

Designed for resource-conscious CPU execution on local hardware:
- Default parameters: imgsz=416, batch=2, workers=0, cache=False, epochs=15.
- Output directory: runs/detect/emergency_vehicle/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML is required. Please install it with 'pip install pyyaml'.")

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    sys.exit("Error: Ultralytics is required. Please install it with 'pip install ultralytics'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Custom YOLO Model for Ambulance and Fire Truck Detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/emergency_yolo/data.yaml",
        help="Path to dataset YAML file (default: data/emergency_yolo/data.yaml).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Pretrained weights to fine-tune from (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size (default: 2 for CPU/low RAM safety).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Input image size (default: 416).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0 for Windows/CPU safety).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device: 'cpu' or '0' (default: cpu).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save experiment runs (default: runs/detect).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="emergency_vehicle",
        help="Experiment name (default: emergency_vehicle).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing experiment folder if specified.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to train on (default: 1.0 for full dataset).",
    )

    return parser.parse_args()


def verify_dataset(yaml_path: Path) -> Path:
    """Verifies that the dataset configuration exists and paths resolve."""
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Loaded dataset configuration from: {yaml_path}")
    print(f"[INFO] Classes: {cfg.get('names')}")
    print(f"[INFO] Number of classes (nc): {cfg.get('nc')}")

    # Set datasets_dir to the current workspace to avoid fallback to old locations
    project_root = Path.cwd().resolve()
    ultralytics.settings.update({"datasets_dir": str(project_root)})

    # Ensure path in yaml is set properly
    yaml_dir = yaml_path.parent.resolve()
    if "path" not in cfg or not cfg["path"]:
        cfg["path"] = str(yaml_dir).replace("\\", "/")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)

    return yaml_path.resolve()


def verify_weights(weights_name: str) -> str:
    """Ensures pretrained weights are available."""
    weights_path = Path(weights_name)
    if weights_path.is_file():
        print(f"[INFO] Found local pretrained weights: {weights_path.resolve()} ({weights_path.stat().st_size:,} bytes)")
        return str(weights_path)

    print(f"[INFO] Local weights '{weights_name}' not found. Ultralytics will attempt to download them.")
    return weights_name


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("CUSTOM YOLO TRAINING: AMBULANCE & FIRE-TRUCK DETECTION")
    print("=" * 78)

    yaml_path = Path(args.data)
    verified_yaml = verify_dataset(yaml_path)
    weights_file = verify_weights(args.weights)

    print("\n[TRAINING CONFIGURATION]")
    print(f"  - Pretrained Weights : {weights_file}")
    print(f"  - Dataset Config     : {verified_yaml}")
    print(f"  - Epochs             : {args.epochs}")
    print(f"  - Batch Size         : {args.batch}")
    print(f"  - Image Size (imgsz) : {args.imgsz}")
    print(f"  - Workers            : {args.workers}")
    print(f"  - Device             : {args.device}")
    print(f"  - Seed               : {args.seed}")
    print(f"  - Early Stopping     : patience={args.patience}")
    print(f"  - Destination        : {Path(args.project) / args.name}")
    print("=" * 78)

    try:
        model = YOLO(weights_file)
        print(f"[INFO] Successfully loaded model architecture from: {weights_file}")

        # Start training
        results = model.train(
            data=str(verified_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            patience=args.patience,
            seed=args.seed,
            cache=False,
            fraction=args.fraction,
            verbose=True,
            plots=True,
        )

        output_dir = Path(args.project) / args.name
        best_weights = output_dir / "weights" / "best.pt"
        last_weights = output_dir / "weights" / "last.pt"

        print("\n" + "=" * 78)
        print("TRAINING PROCESS COMPLETED")
        print("=" * 78)
        print(f"Experiment saved to: {output_dir.resolve()}")
        if best_weights.is_file():
            print(f"[SUCCESS] Best weights saved at: {best_weights.resolve()} ({best_weights.stat().st_size:,} bytes)")
        else:
            print(f"[WARNING] Best weights not found at expected path: {best_weights}")

        if last_weights.is_file():
            print(f"[INFO] Last weights saved at: {last_weights.resolve()} ({last_weights.stat().st_size:,} bytes)")

    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
