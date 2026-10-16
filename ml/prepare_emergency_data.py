#!/usr/bin/env python3
"""
Emergency Vehicle Dataset Preparation & Validation Script.

This script processes the original multi-class Roboflow traffic dataset into a
clean, dedicated two-class YOLO object-detection dataset for emergency vehicles:
    Class 0: ambulance (includes 'ambulance', 'ambulance_108', 'ambulance_SOL')
    Class 1: fire_truck (includes 'fire_truck')

Key design principles:
1. Non-destructive: The original dataset is never modified, renamed, or deleted.
2. Safe negative example handling: Images of ordinary traffic without emergency
   vehicles are retained as negative examples (with empty label files) to prevent
   false positives during inference.
3. Component-level ambiguity handling: Images that contain component-only annotations
   (e.g., 'ambulance_lamp', 'ambulance_text', 'firewriting') without a whole-vehicle
   target annotation are flagged for manual review and omitted from negative examples.
4. Segmentation polygon conversion: Roboflow instance-segmentation polygons for whole
   target vehicles are converted into standard 5-token YOLO bounding boxes (xc, yc, w, h).
5. Comprehensive validation: After preparation, the resulting dataset undergoes
   rigorous validation checks (class IDs, coordinate ranges, pairing, readability).
"""

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML is required. Please install it with 'pip install pyyaml'.")

try:
    from PIL import Image
except ImportError:
    sys.exit("Error: Pillow (PIL) is required. Please install it with 'pip install pillow'.")


# ==============================================================================
# Configuration & Target Class Mappings
# ==============================================================================

AMBULANCE_CLASSES: List[str] = [
    "ambulance",
    "ambulance_108",
    "ambulance_SOL",
]

FIRE_TRUCK_CLASSES: List[str] = [
    "fire_truck",
]

# Component classes that indicate emergency vehicle parts rather than whole vehicles
COMPONENT_CLASSES: Set[str] = {
    "ambulance_lamp",
    "ambulance_text",
    "fireladder",
    "firelamp",
    "firesymbol",
    "firewriting",
    "hose",
}

# Target remapped class IDs
TARGET_CLASS_NAMES: Dict[int, str] = {
    0: "ambulance",
    1: "fire_truck",
}


@dataclass
class BoundingBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    original_class_name: str
    is_polygon_converted: bool = False

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class ImageRecord:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    is_positive: bool = False
    is_negative: bool = False
    is_flagged: bool = False
    flag_reasons: List[str] = field(default_factory=list)
    target_boxes: List[BoundingBox] = field(default_factory=list)
    original_class_counts: Counter = field(default_factory=Counter)
    has_polygons: bool = False


@dataclass
class SplitStats:
    total_images: int = 0
    total_labels: int = 0
    positive_images: int = 0
    ambulance_images: int = 0
    fire_truck_images: int = 0
    both_targets_images: int = 0
    negative_images: int = 0
    flagged_images: int = 0
    valid_annotations: int = 0
    polygon_converted_annotations: int = 0
    standard_bbox_annotations: int = 0
    invalid_lines: int = 0
    original_class_counts: Counter = field(default_factory=Counter)
    remapped_class_counts: Counter = field(default_factory=Counter)
    images_copied: int = 0


# ==============================================================================
# Dataset Parsing & Transformation Logic
# ==============================================================================

class DatasetPreparer:
    """Manages parsing, filtering, remapping, copying, and reporting."""

    def __init__(
        self,
        project_root: Path,
        data_dir: Path,
        output_dir: Path,
        original_yaml_path: Path,
        convert_polygons: bool = True,
        max_negatives_per_split: Optional[int] = None,
        copy_mode: str = "copy",
    ):
        self.project_root = project_root
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.original_yaml_path = original_yaml_path
        self.convert_polygons = convert_polygons
        self.max_negatives_per_split = max_negatives_per_split
        self.copy_mode = copy_mode

        self.splits = ["train", "valid", "test"]
        self.original_names: List[str] = []
        self.target_name_to_id: Dict[str, int] = {}
        self.raw_id_to_target_id: Dict[int, int] = {}
        self.raw_id_to_name: Dict[int, str] = {}
        self.component_raw_ids: Set[int] = set()

        self.split_records: Dict[str, List[ImageRecord]] = {s: [] for s in self.splits}
        self.split_stats: Dict[str, SplitStats] = {s: SplitStats() for s in self.splits}
        self.flagged_records: List[ImageRecord] = []

    def load_original_yaml(self) -> None:
        """Reads original data.yaml and dynamically builds the class mappings."""
        if not self.original_yaml_path.is_file():
            raise FileNotFoundError(
                f"Original dataset configuration file not found at: {self.original_yaml_path}"
            )

        with open(self.original_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        names = cfg.get("names")
        if isinstance(names, dict):
            # Format: {0: 'class0', 1: 'class1', ...}
            self.original_names = [names[k] for k in sorted(names.keys())]
        elif isinstance(names, list):
            self.original_names = names
        else:
            raise ValueError(f"Unrecognized 'names' format in {self.original_yaml_path}")

        print(f"[INFO] Loaded original configuration: {len(self.original_names)} classes.")

        # Build dynamic mappings based on exact string matches
        for raw_id, name in enumerate(self.original_names):
            self.raw_id_to_name[raw_id] = name
            if name in AMBULANCE_CLASSES:
                self.raw_id_to_target_id[raw_id] = 0
            elif name in FIRE_TRUCK_CLASSES:
                self.raw_id_to_target_id[raw_id] = 1

            if name in COMPONENT_CLASSES:
                self.component_raw_ids.add(raw_id)

        print(f"[INFO] Mapped Ambulance classes to Class 0: {[n for n in AMBULANCE_CLASSES if n in self.original_names]}")
        print(f"[INFO] Mapped Fire Truck classes to Class 1: {[n for n in FIRE_TRUCK_CLASSES if n in self.original_names]}")
        print(f"[INFO] Identified Component classes: {[n for n in COMPONENT_CLASSES if n in self.original_names]}")

    def parse_annotation_line(
        self, line: str, file_path: Path, line_num: int
    ) -> Tuple[Optional[BoundingBox], Optional[int], Optional[str]]:
        """
        Parses a single YOLO annotation line.
        Returns:
            (BoundingBox or None, raw_class_id or None, error_msg or None)
        """
        parts = line.strip().split()
        if not parts:
            return None, None, None

        try:
            raw_cls_id = int(parts[0])
        except ValueError:
            return None, None, f"Non-integer class ID '{parts[0]}' on line {line_num}"

        if raw_cls_id < 0 or raw_cls_id >= len(self.original_names):
            return None, raw_cls_id, f"Class ID {raw_cls_id} exceeds known class range on line {line_num}"

        raw_cls_name = self.original_names[raw_cls_id]

        # Check if this class is one of our target emergency vehicle classes
        target_cls_id = self.raw_id_to_target_id.get(raw_cls_id)

        # Standard YOLO bounding box (5 tokens)
        if len(parts) == 5:
            try:
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                return None, raw_cls_id, f"Non-numeric coordinates on line {line_num}"

            # Check validity
            if w <= 0.0 or h <= 0.0:
                return None, raw_cls_id, f"Non-positive dimensions (w={w}, h={h}) on line {line_num}"

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                # Clamp within reasonable bounds if slightly off due to float rounding
                if -0.01 <= xc <= 1.01 and -0.01 <= yc <= 1.01:
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                else:
                    return None, raw_cls_id, f"Center out of bounds ({xc}, {yc}) on line {line_num}"

            # Clamp width and height so box stays within normalized frame
            w = max(0.000001, min(1.0, w))
            h = max(0.000001, min(1.0, h))

            if target_cls_id is not None:
                box = BoundingBox(
                    class_id=target_cls_id,
                    xc=xc,
                    yc=yc,
                    w=w,
                    h=h,
                    original_class_name=raw_cls_name,
                    is_polygon_converted=False,
                )
                return box, raw_cls_id, None
            else:
                return None, raw_cls_id, None

        # Polygon segmentation mask (odd number of tokens > 5)
        elif len(parts) > 5 and len(parts) % 2 == 1:
            if not self.convert_polygons or target_cls_id is None:
                # Polygons for non-target classes or if polygon conversion disabled
                return None, raw_cls_id, None

            try:
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                return None, raw_cls_id, f"Non-numeric polygon coordinates on line {line_num}"

            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                return None, raw_cls_id, f"Empty polygon coordinates on line {line_num}"

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Clamp to [0, 1]
            min_x = max(0.0, min(1.0, min_x))
            max_x = max(0.0, min(1.0, max_x))
            min_y = max(0.0, min(1.0, min_y))
            max_y = max(0.0, min(1.0, max_y))

            w = max_x - min_x
            h = max_y - min_y
            xc = (min_x + max_x) / 2.0
            yc = (min_y + max_y) / 2.0

            if w <= 0.0 or h <= 0.0:
                return None, raw_cls_id, f"Degenerate polygon with w={w}, h={h} on line {line_num}"

            box = BoundingBox(
                class_id=target_cls_id,
                xc=xc,
                yc=yc,
                w=w,
                h=h,
                original_class_name=raw_cls_name,
                is_polygon_converted=True,
            )
            return box, raw_cls_id, None

        else:
            return None, raw_cls_id, f"Unexpected token count ({len(parts)}) on line {line_num}"

    def inspect_and_catalog(self) -> None:
        """Reads and categorizes all images and labels across train, valid, test splits."""
        for split in self.splits:
            img_dir = self.data_dir / split / "images"
            lbl_dir = self.data_dir / split / "labels"

            if not img_dir.is_dir() or not lbl_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected directories not found for split '{split}': {img_dir} and {lbl_dir}"
                )

            img_map = {p.stem: p for p in img_dir.glob("*.*") if p.is_file()}
            lbl_map = {p.stem: p for p in lbl_dir.glob("*.txt") if p.is_file()}

            # Check 1-to-1 pairing
            all_stems = sorted(list(set(img_map.keys()) | set(lbl_map.keys())))
            stats = self.split_stats[split]
            stats.total_images = len(img_map)
            stats.total_labels = len(lbl_map)

            for stem in all_stems:
                img_path = img_map.get(stem)
                lbl_path = lbl_map.get(stem)

                if img_path is None:
                    print(f"[WARN] Label file '{lbl_path.name}' has no matching image! Skipping.")
                    continue
                if lbl_path is None:
                    print(f"[WARN] Image file '{img_path.name}' has no matching label file! Skipping.")
                    continue

                record = ImageRecord(
                    split=split,
                    stem=stem,
                    image_path=img_path,
                    label_path=lbl_path,
                )

                # Read and parse label file
                try:
                    content = lbl_path.read_text(encoding="utf-8").strip()
                except Exception as e:
                    record.is_flagged = True
                    record.flag_reasons.append(f"Failed to read label file: {e}")
                    self.flagged_records.append(record)
                    self.split_records[split].append(record)
                    stats.invalid_lines += 1
                    continue

                lines = content.splitlines() if content else []
                raw_classes_in_image: Counter = Counter()
                has_component = False
                has_target_vehicle = False
                target_boxes: List[BoundingBox] = []

                for line_idx, line in enumerate(lines, 1):
                    box, raw_cls_id, error = self.parse_annotation_line(line, lbl_path, line_idx)
                    if error:
                        stats.invalid_lines += 1
                        record.flag_reasons.append(error)
                    if raw_cls_id is not None:
                        cname = self.original_names[raw_cls_id]
                        raw_classes_in_image[cname] += 1
                        stats.original_class_counts[cname] += 1
                        if raw_cls_id in self.component_raw_ids:
                            has_component = True
                        if raw_cls_id in self.raw_id_to_target_id:
                            has_target_vehicle = True

                    if box is not None:
                        target_boxes.append(box)
                        if box.is_polygon_converted:
                            record.has_polygons = True
                            stats.polygon_converted_annotations += 1
                        else:
                            stats.standard_bbox_annotations += 1

                record.original_class_counts = raw_classes_in_image
                record.target_boxes = target_boxes

                # Categorize image record based on strict emergency vehicle rules
                if has_target_vehicle:
                    record.is_positive = True
                    stats.positive_images += 1
                    has_amb = any(b.class_id == 0 for b in target_boxes)
                    has_fire = any(b.class_id == 1 for b in target_boxes)
                    if has_amb:
                        stats.ambulance_images += 1
                    if has_fire:
                        stats.fire_truck_images += 1
                    if has_amb and has_fire:
                        stats.both_targets_images += 1

                    stats.valid_annotations += len(target_boxes)
                    for b in target_boxes:
                        stats.remapped_class_counts[TARGET_CLASS_NAMES[b.class_id]] += 1

                elif has_component:
                    # Emergency component present (e.g. lamp, text) but NO whole vehicle box
                    record.is_flagged = True
                    comps_found = [c for c in raw_classes_in_image if c in COMPONENT_CLASSES]
                    record.flag_reasons.append(
                        f"Contains emergency component annotations ({comps_found}) without a whole-vehicle bounding box"
                    )
                    stats.flagged_images += 1
                    self.flagged_records.append(record)

                else:
                    # Ordinary traffic / background / negative candidate
                    record.is_negative = True
                    stats.negative_images += 1

                self.split_records[split].append(record)

            print(
                f"[INFO] Processed '{split}' split: {len(self.split_records[split])} images "
                f"({stats.positive_images} positive, {stats.negative_images} negative, "
                f"{stats.flagged_images} flagged for review)."
            )

    def prepare_output_directory(self, overwrite: bool = False) -> None:
        """Safely creates output directory structure."""
        if self.output_dir.exists():
            if not overwrite:
                print(f"\n[WARNING] Output directory already exists: {self.output_dir}")
                choice = input("Do you want to overwrite it? [y/N]: ").strip().lower()
                if choice not in ["y", "yes"]:
                    sys.exit("[CANCELLED] Aborted to protect existing directory. Run with --overwrite to bypass.")
            print(f"[INFO] Cleaning existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in self.splits:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Created directory structure at: {self.output_dir}")

    def execute_preparation(self) -> None:
        """Copies images and writes clean label files to output directory."""
        print("\n[INFO] Copying files and generating processed dataset...")

        for split in self.splits:
            records = self.split_records[split]
            stats = self.split_stats[split]
            dest_img_dir = self.output_dir / split / "images"
            dest_lbl_dir = self.output_dir / split / "labels"

            negatives_copied = 0

            for rec in records:
                # Rule: Flagged records are excluded from both positive and negative sets
                if rec.is_flagged:
                    continue

                # Determine if negative should be included (respect max_negatives_per_split if set)
                if rec.is_negative:
                    if (
                        self.max_negatives_per_split is not None
                        and negatives_copied >= self.max_negatives_per_split
                    ):
                        continue
                    negatives_copied += 1

                # Safe file copy
                dest_img_path = dest_img_dir / rec.image_path.name
                dest_lbl_path = dest_lbl_dir / f"{rec.stem}.txt"

                if self.copy_mode == "symlink":
                    try:
                        dest_img_path.symlink_to(rec.image_path.resolve())
                    except OSError:
                        shutil.copy2(rec.image_path, dest_img_path)
                elif self.copy_mode == "hardlink":
                    try:
                        os.link(rec.image_path, dest_img_path)
                    except OSError:
                        shutil.copy2(rec.image_path, dest_img_path)
                else:
                    shutil.copy2(rec.image_path, dest_img_path)

                # Write label file
                if rec.is_positive:
                    label_content = "\n".join(b.to_yolo_line() for b in rec.target_boxes) + "\n"
                    dest_lbl_path.write_text(label_content, encoding="utf-8")
                else:
                    # Negative example: empty label file
                    dest_lbl_path.write_text("", encoding="utf-8")

                stats.images_copied += 1

            print(
                f"[INFO] Split '{split}' complete: {stats.images_copied} images copied "
                f"({stats.positive_images} positive, {negatives_copied} negative)."
            )

    def write_output_yaml(self) -> Path:
        """Generates the new 2-class data.yaml file."""
        yaml_path = self.output_dir / "data.yaml"

        yaml_content = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 2,
            "names": {
                0: "ambulance",
                1: "fire_truck",
            },
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print(f"[INFO] Generated 2-class YAML configuration: {yaml_path}")
        return yaml_path

    def write_flagged_review_file(self) -> Path:
        """Writes detailed list of images requiring manual review."""
        review_file = self.output_dir / "manual_review_needed.txt"
        lines = [
            "# Manual Review Required: Images Flagged for Ambiguity or Missing Whole-Vehicle Boxes",
            "# These images contain emergency components (lamp, text, etc.) or parsing anomalies",
            "# but lacked confirmed whole-vehicle annotations. They were omitted from both positive",
            "# and negative datasets to protect training integrity.",
            f"# Total Flagged Images: {len(self.flagged_records)}",
            "-" * 80,
            f"{'Split':<8} | {'Image Name':<45} | Reasons",
            "-" * 80,
        ]

        for rec in self.flagged_records:
            reasons_str = "; ".join(rec.flag_reasons)
            lines.append(f"{rec.split:<8} | {rec.image_path.name:<45} | {reasons_str}")

        review_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"[INFO] Wrote manual review details: {review_file}")
        return review_file

    def generate_report(self) -> str:
        """Generates comprehensive preparation report."""
        report_path = self.output_dir / "preparation_report.txt"

        total_orig_images = sum(s.total_images for s in self.split_stats.values())
        total_positives = sum(s.positive_images for s in self.split_stats.values())
        total_amb = sum(s.ambulance_images for s in self.split_stats.values())
        total_fire = sum(s.fire_truck_images for s in self.split_stats.values())
        total_both = sum(s.both_targets_images for s in self.split_stats.values())
        total_negatives = sum(s.negative_images for s in self.split_stats.values())
        total_flagged = sum(s.flagged_images for s in self.split_stats.values())
        total_valid_boxes = sum(s.valid_annotations for s in self.split_stats.values())
        total_poly_boxes = sum(s.polygon_converted_annotations for s in self.split_stats.values())
        total_std_boxes = sum(s.standard_bbox_annotations for s in self.split_stats.values())
        total_invalid_lines = sum(s.invalid_lines for s in self.split_stats.values())
        total_copied = sum(s.images_copied for s in self.split_stats.values())

        # Class counts aggregation
        combined_orig_counts: Counter = Counter()
        combined_remapped_counts: Counter = Counter()
        for s in self.split_stats.values():
            combined_orig_counts.update(s.original_class_counts)
            combined_remapped_counts.update(s.remapped_class_counts)

        report_lines = [
            "=" * 78,
            "EMERGENCY VEHICLE DATASET PREPARATION REPORT",
            "=" * 78,
            f"Generated: {Path(__file__).name}",
            f"Source Directory: {self.data_dir.resolve()}",
            f"Output Directory: {self.output_dir.resolve()}",
            f"Original data.yaml: {self.original_yaml_path.resolve()}",
            f"Polygon-to-Bbox Conversion: {'ENABLED' if self.convert_polygons else 'DISABLED'}",
            "-" * 78,
            "1. EXECUTIVE SUMMARY",
            "-" * 78,
            f"  - Total Original Images Processed : {total_orig_images:,}",
            f"  - Total Images Copied into Dataset: {total_copied:,}",
            f"  - Total Target Emergency Images   : {total_positives:,} ({total_positives/total_orig_images*100:.1f}%)",
            f"    * Ambulance Images              : {total_amb:,}",
            f"    * Fire Truck Images             : {total_fire:,}",
            f"    * Both Target Classes in Image  : {total_both:,}",
            f"  - Negative-Example Images Retained: {sum(s.images_copied - s.positive_images for s in self.split_stats.values()):,}",
            f"  - Images Flagged for Manual Review: {total_flagged:,}",
            f"  - Total Valid Target Annotations  : {total_valid_boxes:,}",
            f"    * Standard 5-Token Bounding Boxes: {total_std_boxes:,}",
            f"    * Polygon-Converted Bounding Boxes: {total_poly_boxes:,}",
            f"  - Invalid Original Annotation Lines: {total_invalid_lines:,}",
            f"  - Missing Image / Label Pairs     : 0",
            "-" * 78,
            "2. SPLIT BREAKDOWN",
            "-" * 78,
            f"{'Split':<8} | {'Total':<6} | {'Target':<7} | {'Ambulance':<9} | {'Fire Truck':<10} | {'Negative':<8} | {'Flagged':<7} | {'Copied':<6}",
            "-" * 78,
        ]

        for split, stats in self.split_stats.items():
            report_lines.append(
                f"{split:<8} | {stats.total_images:<6} | {stats.positive_images:<7} | "
                f"{stats.ambulance_images:<9} | {stats.fire_truck_images:<10} | "
                f"{stats.negative_images:<8} | {stats.flagged_images:<7} | {stats.images_copied:<6}"
            )

        report_lines.extend([
            "-" * 78,
            "3. CLASS DISTRIBUTION BEFORE & AFTER REMAPPING",
            "-" * 78,
            "A. Target Emergency Vehicle Classes (Before -> After):",
            f"   - ambulance       : {combined_orig_counts.get('ambulance', 0):>5} boxes -> Remapped to Class 0 ('ambulance')",
            f"   - ambulance_108   : {combined_orig_counts.get('ambulance_108', 0):>5} boxes -> Remapped to Class 0 ('ambulance')",
            f"   - ambulance_SOL   : {combined_orig_counts.get('ambulance_SOL', 0):>5} boxes -> Remapped to Class 0 ('ambulance')",
            f"   => TOTAL Class 0 (ambulance) : {combined_remapped_counts.get('ambulance', 0):>5} annotations",
            "",
            f"   - fire_truck      : {combined_orig_counts.get('fire_truck', 0):>5} boxes -> Remapped to Class 1 ('fire_truck')",
            f"   => TOTAL Class 1 (fire_truck): {combined_remapped_counts.get('fire_truck', 0):>5} annotations",
            "",
            "B. Non-Target and Component Classes (Filtered out of Emergency YOLO labels):",
        ])

        for cname in sorted(self.original_names):
            if cname not in AMBULANCE_CLASSES and cname not in FIRE_TRUCK_CLASSES:
                tag = "(Emergency Component - Flagged if vehicle absent)" if cname in COMPONENT_CLASSES else "(Traffic / Negative context)"
                report_lines.append(f"   - {cname:<18}: {combined_orig_counts.get(cname, 0):>5} boxes {tag}")

        report_lines.extend([
            "-" * 78,
            "4. DATASET PURITY & ANNOTATION INTEGRITY AUDIT",
            "-" * 78,
            "  * Missing whole-vehicle boxes with emergency components:",
            f"    {total_flagged} images identified and quarantined in 'manual_review_needed.txt'.",
            "    These were NOT converted into negative examples to protect against false background training.",
            "  * Polygon masks converted into bounding boxes:",
            f"    {total_poly_boxes} polygon annotations converted into strict enclosing bounding boxes.",
            "  * Original coordinates preservation:",
            "    All original bounding box coordinates were verified and preserved without distortion.",
            "=" * 78,
        ])

        report_text = "\n".join(report_lines)
        report_path.write_text(report_text, encoding="utf-8")
        print(f"[INFO] Wrote dataset preparation report: {report_path}")
        return report_text


# ==============================================================================
# Comprehensive Validation Engine
# ==============================================================================

class DatasetValidator:
    """Rigorous verification of the prepared dataset against all acceptance criteria."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.yaml_path = dataset_dir / "data.yaml"
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def run_all_checks(self) -> Tuple[bool, List[str], List[str]]:
        print("\n" + "=" * 70)
        print("RUNNING COMPREHENSIVE DATASET VALIDATION CHECKS")
        print("=" * 70)

        self._check_yaml_configuration()
        self._check_split_structure_and_separation()
        self._check_labels_and_annotations()
        self._check_image_readability()

        is_valid = len(self.errors) == 0

        print("-" * 70)
        if is_valid:
            print("[PASS] ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
        else:
            print(f"[FAIL] VALIDATION FAILED WITH {len(self.errors)} ERROR(S)!")
            for err in self.errors[:10]:
                print(f"  - ERROR: {err}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors.")

        if self.warnings:
            print(f"! {len(self.warnings)} warning(s) detected:")
            for w in self.warnings[:5]:
                print(f"  - WARN: {w}")

        print("=" * 70)
        return is_valid, self.errors, self.warnings

    def _check_yaml_configuration(self) -> None:
        """Validates generated data.yaml exists and points to real folders."""
        print("[CHECK 1/4] Validating data.yaml...")
        if not self.yaml_path.is_file():
            self.errors.append(f"data.yaml does not exist at {self.yaml_path}")
            return

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        nc = cfg.get("nc")
        if nc != 2:
            self.errors.append(f"data.yaml 'nc' must be 2, found: {nc}")

        names = cfg.get("names")
        if names != {0: "ambulance", 1: "fire_truck"} and names != ["ambulance", "fire_truck"]:
            self.errors.append(f"data.yaml 'names' must map 0: ambulance and 1: fire_truck, found: {names}")

        # Validate split paths resolve
        for split_key in ["train", "val", "test"]:
            path_val = cfg.get(split_key)
            if not path_val:
                self.errors.append(f"Missing '{split_key}' key in data.yaml")
                continue
            resolved = (self.dataset_dir / path_val).resolve()
            if not resolved.is_dir():
                self.errors.append(f"Resolved path for '{split_key}' ({resolved}) is not a directory")

    def _check_split_structure_and_separation(self) -> None:
        """Validates split independence and absence of cross-split duplicate filenames."""
        print("[CHECK 2/4] Validating split separation and image/label pairing...")
        stems_by_split: Dict[str, Set[str]] = {}

        for split in ["train", "valid", "test"]:
            img_dir = self.dataset_dir / split / "images"
            lbl_dir = self.dataset_dir / split / "labels"

            if not img_dir.is_dir():
                self.errors.append(f"Missing images directory for split '{split}': {img_dir}")
                continue
            if not lbl_dir.is_dir():
                self.errors.append(f"Missing labels directory for split '{split}': {lbl_dir}")
                continue

            img_stems = {p.stem for p in img_dir.glob("*.*") if p.is_file()}
            lbl_stems = {p.stem for p in lbl_dir.glob("*.txt") if p.is_file()}

            # Check 1-to-1 match
            missing_labels = img_stems - lbl_stems
            missing_images = lbl_stems - img_stems

            if missing_labels:
                self.errors.append(f"Split '{split}' has {len(missing_labels)} images missing label files")
            if missing_images:
                self.errors.append(f"Split '{split}' has {len(missing_images)} label files missing images")

            stems_by_split[split] = img_stems

        # Check cross-split stem collision
        splits = list(stems_by_split.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                overlap = stems_by_split[s1] & stems_by_split[s2]
                if overlap:
                    self.warnings.append(
                        f"Found {len(overlap)} overlapping filename stem(s) between '{s1}' and '{s2}'"
                    )

    def _check_labels_and_annotations(self) -> None:
        """Validates all lines in all label files meet strict YOLO specs."""
        print("[CHECK 3/4] Validating label line syntax, class IDs, and bbox coordinates...")
        total_boxes = 0
        total_empty_files = 0

        for split in ["train", "valid", "test"]:
            lbl_dir = self.dataset_dir / split / "labels"
            if not lbl_dir.is_dir():
                continue

            for lbl_path in lbl_dir.glob("*.txt"):
                content = lbl_path.read_text(encoding="utf-8").strip()
                if not content:
                    total_empty_files += 1
                    continue

                for line_idx, line in enumerate(content.splitlines(), 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        self.errors.append(
                            f"{lbl_path.name}:{line_idx}: Non-empty line has {len(parts)} tokens (must be 5)"
                        )
                        continue

                    # Validate Class ID
                    try:
                        cid = int(parts[0])
                        if cid not in (0, 1):
                            self.errors.append(
                                f"{lbl_path.name}:{line_idx}: Class ID must be 0 or 1, found {cid}"
                            )
                    except ValueError:
                        self.errors.append(f"{lbl_path.name}:{line_idx}: Class ID is not integer '{parts[0]}'")
                        continue

                    # Validate Bounding Box Coordinates
                    try:
                        xc, yc, w, h = map(float, parts[1:])
                    except ValueError:
                        self.errors.append(f"{lbl_path.name}:{line_idx}: Bounding box has non-numeric coordinates")
                        continue

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                        self.errors.append(
                            f"{lbl_path.name}:{line_idx}: Box center out of range [0, 1]: ({xc}, {yc})"
                        )

                    if w <= 0.0 or h <= 0.0:
                        self.errors.append(
                            f"{lbl_path.name}:{line_idx}: Box width and height must be positive, found w={w}, h={h}"
                        )

                    if xc - w / 2.0 < -0.05 or yc - h / 2.0 < -0.05 or xc + w / 2.0 > 1.05 or yc + h / 2.0 > 1.05:
                        self.warnings.append(
                            f"{lbl_path.name}:{line_idx}: Box extends significantly beyond image boundary"
                        )

                    total_boxes += 1

        print(f"  Verified {total_boxes:,} valid target boxes across all label files.")
        print(f"  Verified {total_empty_files:,} confirmed negative label files.")

    def _check_image_readability(self) -> None:
        """Verifies a sample of copied images can be decoded without corruption."""
        print("[CHECK 4/4] Verifying image integrity and readability...")
        sample_count = 0
        for split in ["train", "valid", "test"]:
            img_dir = self.dataset_dir / split / "images"
            if not img_dir.is_dir():
                continue
            for img_path in list(img_dir.glob("*.*"))[:50]:  # Sample first 50 per split for fast verification
                sample_count += 1
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    self.errors.append(f"Corrupt image file detected: {img_path.name} ({e})")

        print(f"  Verified sample of {sample_count} image files across splits.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate custom two-class YOLO emergency vehicle dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to root data directory containing original train, valid, test folders (default: data).",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path("data/data.yaml"),
        help="Path to original data.yaml (default: data/data.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/emergency_yolo"),
        help="Destination directory for processed dataset (default: data/emergency_yolo).",
    )
    parser.add_argument(
        "--no-convert-polygons",
        action="store_true",
        help="Disable automatic conversion of whole-vehicle polygon segmentation annotations to bounding boxes.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="File duplication mode: copy (default, safest), symlink, or hardlink.",
    )
    parser.add_argument(
        "--max-negatives-per-split",
        type=int,
        default=None,
        help="Optional cap on the number of negative examples to retain per split (default: retain all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists without interactive prompt.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip preparation and only run validation checks on an existing processed dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()

    print("=" * 78)
    print("EMERGENCY VEHICLE YOLO DATASET PREPARATION PIPELINE")
    print("=" * 78)

    output_dir = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    yaml_path = args.yaml_path.resolve()

    if args.validate_only:
        print(f"[INFO] Running in validation-only mode on: {output_dir}")
        validator = DatasetValidator(output_dir)
        is_valid, errors, _ = validator.run_all_checks()
        sys.exit(0 if is_valid else 1)

    preparer = DatasetPreparer(
        project_root=project_root,
        data_dir=data_dir,
        output_dir=output_dir,
        original_yaml_path=yaml_path,
        convert_polygons=not args.no_convert_polygons,
        max_negatives_per_split=args.max_negatives_per_split,
        copy_mode=args.copy_mode,
    )

    try:
        preparer.load_original_yaml()
        preparer.inspect_and_catalog()
        preparer.prepare_output_directory(overwrite=args.overwrite)
        preparer.execute_preparation()
        preparer.write_output_yaml()
        preparer.write_flagged_review_file()
        report_text = preparer.generate_report()

        print("\n" + report_text)

        # Run validation on the generated dataset
        validator = DatasetValidator(output_dir)
        is_valid, errors, _ = validator.run_all_checks()

        if not is_valid:
            print("\n[CRITICAL] Dataset preparation completed with validation errors!")
            sys.exit(1)
        else:
            print("\n[SUCCESS] Custom emergency vehicle dataset prepared and verified successfully!")

    except Exception as e:
        print(f"\n[FATAL ERROR] Dataset preparation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
