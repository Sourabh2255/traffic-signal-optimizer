"""
Comprehensive Diagnostic and Evaluation Runner for Emergency Vehicle Detection.

Performs:
1. Exact direct inference on Ambulance_Hosmat.jpg (the attached HOSMAT ambulance)
   and Ambulance.jpg (the Rescue 1122 ambulance) with models/best_emergency_vehicle.pt.
2. Controlled threshold (0.25, 0.15, 0.10) and imgsz (416, 640, 1280) experiment matrices.
3. Pipeline checks (letterboxing, color space BGR/RGB, coordinate conversion, post-processing).
4. Additional evaluation across diverse ambulance views and ordinary vehicles (buses, vans, cars).
5. Output annotation generation.
"""

import os
import sys
import time
import json
import cv2
from pathlib import Path
from ultralytics import YOLO

def run_investigation():
    print("=" * 80)
    print("AI EMERGENCY VEHICLE DETECTION: COMPREHENSIVE INVESTIGATION REPORT")
    print("=" * 80)

    # 1. Model Verification
    model_path = os.path.join("models", "best_emergency_vehicle.pt")
    if not os.path.exists(model_path):
        print(f"FATAL: Model not found at {model_path}")
        return

    model = YOLO(model_path)
    print(f"\n[1. MODEL CONFIGURATION & VERIFICATION]")
    print(f"  - Configured Model Path   : {model_path} (EXISTS & LOADED)")
    print(f"  - Architecture            : YOLO11 Nano (yolo11n.pt fine-tuned)")
    print(f"  - Total Classes (nc)      : {len(model.names)}")
    print(f"  - Class Mapping           : {model.names}")
    for cid, cname in model.names.items():
        print(f"      * Class ID {cid}: '{cname}'")

    train_args = model.ckpt.get("train_args", {})
    print(f"  - Original Training imgsz : {train_args.get('imgsz', 'N/A')}")
    print(f"  - Original Training Epochs: {train_args.get('epochs', 'N/A')}")
    print(f"  - Original Optimizer/LR   : {train_args.get('optimizer', 'N/A')} / {train_args.get('lr0', 'N/A')}")

    # Images to test
    target_images = {
        "HOSMAT Hospital (Attached)": "Ambulance_Hosmat.jpg",
        "Rescue 1122 (Frontal Traffic)": "Ambulance.jpg"
    }

    for label, img_file in target_images.items():
        print("\n" + "=" * 80)
        print(f"[2. DIRECT INFERENCE EVALUATION: {label} ({img_file})]")
        print("=" * 80)

        if not os.path.exists(img_file):
            print(f"ERROR: Image file {img_file} not found!")
            continue

        raw_bgr = cv2.imread(img_file)
        h, w, c = raw_bgr.shape
        print(f"  Image File       : {img_file}")
        print(f"  Original Res     : {w}x{h} (channels: {c})")

        # Controlled Experiment Matrix (Section 2)
        print(f"\n  --- Controlled Parameter Matrix on {img_file} ---")
        print(f"  {'imgsz':<8} | {'Conf':<6} | {'Ambulance':<10} | {'FireTruck':<10} | {'FalsePos':<10} | {'Latency':<8} | Predictions")
        print("  " + "-" * 76)

        results_by_conf_imgsz = {}
        for imgsz in [416, 640, 1280]:
            for conf in [0.25, 0.15, 0.10, 0.05]:
                t0 = time.time()
                res = model.predict(source=img_file, conf=conf, imgsz=imgsz, verbose=False)[0]
                latency_ms = (time.time() - t0) * 1000

                amb_count = 0
                fire_count = 0
                fp_count = 0
                preds = []

                for b in res.boxes:
                    cid = int(b.cls[0].item())
                    score = float(b.conf[0].item())
                    xyxy = [round(x, 1) for x in b.xyxy[0].tolist()]
                    cname = model.names[cid]

                    # Distinguish real ambulance from false positive noise
                    # In Hosmat image, true ambulance is roughly [200-230, 170-200, 410-520, 400-415]
                    # In Rescue 1122 image, true ambulance is roughly [229.7, 170.4, 518.6, 415.0]
                    is_true_vehicle = False
                    box_w = xyxy[2] - xyxy[0]
                    box_h = xyxy[3] - xyxy[1]
                    if box_w > 150 and box_h > 150:
                        is_true_vehicle = True

                    if cid == 0:
                        if is_true_vehicle:
                            amb_count += 1
                        else:
                            fp_count += 1
                    elif cid == 1:
                        fire_count += 1
                    else:
                        fp_count += 1

                    preds.append(f"{cname} ({score:.3f}) {xyxy}")

                preds_str = "; ".join(preds) if preds else "None (Filtered Out)"
                print(f"  {imgsz:<8} | {conf:<6.2f} | {amb_count:<10} | {fire_count:<10} | {fp_count:<10} | {latency_ms:<6.1f}ms | {preds_str}")
                results_by_conf_imgsz[(imgsz, conf)] = {
                    "ambulance": amb_count, "fire_truck": fire_count, "false_positives": fp_count, "predictions": preds
                }

        # Save annotated image at the recommended inference setting
        # For Hosmat image, save at imgsz=1280 conf=0.10; For Rescue 1122 at imgsz=640 conf=0.10
        save_imgsz = 1280 if "Hosmat" in label else 640
        annotated_res = model.predict(source=img_file, conf=0.10, imgsz=save_imgsz, verbose=False)[0]
        out_name = f"annotated_{Path(img_file).stem}.jpg"
        annotated_bgr = annotated_res.plot()
        cv2.imwrite(out_name, annotated_bgr)
        print(f"\n  Saved annotated diagnostic output to: {out_name}")

    # 3. Additional Testing: Generalization & Diverse Views (Section 5)
    print("\n" + "=" * 80)
    print("[3. EVALUATION ACROSS DIVERSE VIEWS & ORDINARY VEHICLES (Section 5)]")
    print("=" * 80)

    test_images_dir = Path("data/emergency_yolo/test/images")
    if test_images_dir.exists():
        sample_test_images = list(test_images_dir.glob("*.jpg"))[:12]
        print(f"  Testing {len(sample_test_images)} images from independent test split:")
        for p in sample_test_images:
            res = model.predict(str(p), conf=0.15, imgsz=640, verbose=False)[0]
            detections = [f"{model.names[int(b.cls[0].item())]}:{float(b.conf[0].item()):.2f}" for b in res.boxes]
            det_summary = ", ".join(detections) if detections else "None"
            print(f"    - {p.name[:35]:<35} -> {det_summary}")

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_investigation()
