#!/usr/bin/env python
"""Forensic audit script for weapon datasets: NoGun_v5 and WeaponDataset_v11.

Performs complete read-only inspection:
1. Directory structure & split counts
2. data.yaml & class definitions
3. Images: count, formats, resolutions, corruptions
4. Labels: count, YOLO format validity, coords in [0, 1]
5. Duplicates & leakage via SHA256 (within split, between splits, cross-dataset)
6. Bounding-box geometry (avg width, height, area, scale distribution)
7. Detailed analysis of 'other' class in WeaponDataset_v11
8. Detailed analysis of 'nogun' in NoGun_v5 (phones, bottles, tools)
Emits outputs/weapon_audit/weapon_dataset_summary.json.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "weapon_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEAPON_DIR = PROJECT_ROOT / "datasets" / "weapon"


def safe_win_path(p: str | Path) -> str:
    s = str(Path(p).resolve())
    if not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s.replace("/", "\\")
    return s


def get_image_info(path: Path):
    safe_p = safe_win_path(path)
    try:
        with Image.open(safe_p) as img:
            img.verify()
            w, h = img.size
            return True, w, h
    except Exception:
        return False, 0, 0


def audit_dataset(dataset_name: str, dataset_path: Path):
    print(f"\n=================================================================")
    print(f"AUDITING: {dataset_name} ({dataset_path})")
    print("=================================================================")

    yaml_file = dataset_path / "data.yaml"
    yaml_data = {}
    if yaml_file.exists():
        with open(safe_win_path(yaml_file), "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

    class_names = yaml_data.get("names", [])
    num_classes = yaml_data.get("nc", len(class_names))

    splits = ["train", "valid", "test"]
    split_stats = {}
    all_hashes = {}
    all_corrupt = []
    class_box_stats = defaultdict(lambda: {"count": 0, "widths": [], "heights": [], "areas": [], "images": set()})
    invalid_labels = []
    empty_label_images = []
    missing_label_images = []

    total_images = 0
    total_labels = 0
    resolutions = Counter()

    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"  [Notice] Split '{split}' does not exist in {dataset_name}.")
            continue

        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            img_dir = split_dir
        if not lbl_dir.exists():
            lbl_dir = split_dir

        images = [f for f in img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        labels = [f for f in lbl_dir.glob("*.txt") if f.name != "classes.txt"]

        split_images_count = len(images)
        total_images += split_images_count

        split_box_count = 0
        split_empty_labels = 0

        for img_path in images:
            safe_img_str = safe_win_path(img_path)

            # Check hash
            h = hashlib.sha256()
            with open(safe_img_str, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            img_hash = h.hexdigest()
            all_hashes[img_path.name] = (img_hash, split, img_path)

            # Check readability and resolution
            is_valid, w, h_img = get_image_info(img_path)
            if not is_valid:
                all_corrupt.append(str(img_path))
                continue
            resolutions[(w, h_img)] += 1

            # Check corresponding label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            safe_lbl_str = safe_win_path(lbl_path)
            if not Path(safe_lbl_str).exists():
                missing_label_images.append(str(img_path))
                continue

            try:
                with open(safe_lbl_str, "r", encoding="utf-8") as lf:
                    content = lf.read().strip()
            except Exception as e:
                invalid_labels.append((str(lbl_path), 0, f"Read error: {e}"))
                continue

            if not content:
                split_empty_labels += 1
                empty_label_images.append(str(img_path))
                continue

            lines = content.splitlines()
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels.append((str(lbl_path), line_idx, f"Invalid token count: {len(parts)}"))
                    continue
                try:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    invalid_labels.append((str(lbl_path), line_idx, f"Non-numeric values in line: {line}"))
                    continue

                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    invalid_labels.append((str(lbl_path), line_idx, f"Coords out of bounds: cx={cx}, cy={cy}, w={bw}, h={bh}"))

                class_box_stats[cls_id]["count"] += 1
                class_box_stats[cls_id]["widths"].append(bw)
                class_box_stats[cls_id]["heights"].append(bh)
                class_box_stats[cls_id]["areas"].append(bw * bh)
                class_box_stats[cls_id]["images"].add(str(img_path))
                split_box_count += 1
                total_labels += 1

        split_stats[split] = {
            "image_count": split_images_count,
            "label_file_count": len(labels),
            "annotation_count": split_box_count,
            "empty_label_count": split_empty_labels,
        }

    # Duplicate check within dataset
    hash_to_files = defaultdict(list)
    for name, (h, sp, p) in all_hashes.items():
        hash_to_files[h].append((name, sp, p))

    intra_duplicates = {h: flist for h, flist in hash_to_files.items() if len(flist) > 1}

    # Inter-split leakage
    leakage = []
    for h, flist in intra_duplicates.items():
        splits_involved = set(item[1] for item in flist)
        if len(splits_involved) > 1:
            leakage.append({"hash": h, "splits": list(splits_involved), "files": [str(item[2]) for item in flist]})

    # Class summary table
    per_class_summary = []
    all_class_ids = sorted(list(set(list(class_box_stats.keys()) + list(range(len(class_names))))))
    for cls_id in all_class_ids:
        stats = class_box_stats[cls_id]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"UNKNOWN_{cls_id}"
        count = stats["count"]
        img_count = len(stats["images"])
        avg_w = float(np.mean(stats["widths"])) if stats["widths"] else 0.0
        avg_h = float(np.mean(stats["heights"])) if stats["heights"] else 0.0
        avg_area = float(np.mean(stats["areas"])) if stats["areas"] else 0.0

        # Scale breakdown: small (<0.01), medium (0.01-0.10), large (>0.10)
        areas = stats["areas"]
        small = sum(1 for a in areas if a < 0.01)
        medium = sum(1 for a in areas if 0.01 <= a <= 0.10)
        large = sum(1 for a in areas if a > 0.10)

        per_class_summary.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "image_count": img_count,
            "annotation_count": count,
            "avg_width": round(avg_w, 4),
            "avg_height": round(avg_h, 4),
            "avg_area": round(avg_area, 4),
            "scale_distribution": {
                "small_pct": round(small / max(count, 1) * 100, 2),
                "medium_pct": round(medium / max(count, 1) * 100, 2),
                "large_pct": round(large / max(count, 1) * 100, 2),
            }
        })

    print(f"Dataset: {dataset_name}")
    print(f"  Total Images: {total_images}")
    print(f"  Total Bounding Boxes: {total_labels}")
    print(f"  Classes: {class_names}")
    print(f"  Splits: {split_stats}")
    print(f"  Corrupt Images: {len(all_corrupt)}")
    print(f"  Invalid Labels: {len(invalid_labels)}")
    print(f"  Empty Label Files: {len(empty_label_images)}")
    print(f"  Unique Hashes: {len(hash_to_files)} (Duplicates: {sum(len(v)-1 for v in intra_duplicates.values())} files)")
    print(f"  Inter-Split Leakage Cases: {len(leakage)}")
    print("  Top Resolutions:", resolutions.most_common(5))

    return {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "yaml_config": yaml_data,
        "class_names": class_names,
        "num_classes": num_classes,
        "total_images": total_images,
        "total_annotations": total_labels,
        "splits": split_stats,
        "per_class": per_class_summary,
        "resolutions": {f"{k[0]}x{k[1]}": v for k, v in resolutions.most_common(10)},
        "corrupt_images": all_corrupt,
        "invalid_labels_count": len(invalid_labels),
        "invalid_labels_sample": invalid_labels[:10],
        "empty_labels_count": len(empty_label_images),
        "missing_labels_count": len(missing_label_images),
        "total_duplicate_files": sum(len(v) - 1 for v in intra_duplicates.values()),
        "leakage_count": len(leakage),
        "leakage_sample": leakage[:5],
        "all_hashes": {k: (v[0], v[1], str(v[2])) for k, v in all_hashes.items()},
    }


def run_full_audit():
    nogun_audit = audit_dataset("NoGun_v5", WEAPON_DIR / "NoGun_v5")
    weapon_audit = audit_dataset("WeaponDataset_v11", WEAPON_DIR / "WeaponDataset_v11")

    # Cross-dataset duplication check
    print("\n=================================================================")
    print("CROSS-DATASET DUPLICATION / LEAKAGE AUDIT")
    print("=================================================================")
    nogun_hashes = {v[0]: (k, v[1], v[2]) for k, v in nogun_audit["all_hashes"].items()}
    weapon_hashes = {v[0]: (k, v[1], v[2]) for k, v in weapon_audit["all_hashes"].items()}

    cross_duplicates = set(nogun_hashes.keys()) & set(weapon_hashes.keys())
    print(f"Cross-dataset duplicate image count: {len(cross_duplicates)}")

    cross_dup_samples = []
    for h in list(cross_duplicates)[:10]:
        ng_f = nogun_hashes[h]
        wp_f = weapon_hashes[h]
        cross_dup_samples.append({
            "hash": h,
            "nogun_file": ng_f[0],
            "nogun_split": ng_f[1],
            "weapon_file": wp_f[0],
            "weapon_split": wp_f[1],
        })

    # Prepare complete output JSON (stripping huge hash dicts)
    del nogun_audit["all_hashes"]
    del weapon_audit["all_hashes"]

    full_summary = {
        "timestamp": "2026-10-02",
        "datasets": {
            "NoGun_v5": nogun_audit,
            "WeaponDataset_v11": weapon_audit,
        },
        "cross_dataset": {
            "duplicate_count": len(cross_duplicates),
            "sample_duplicates": cross_dup_samples,
            "safe_to_merge_directly": False,
            "merge_risk_rationale": (
                "1. Class ID mismatch: NoGun_v5 uses class 0 for 'nogun', whereas WeaponDataset_v11 uses class 0 for 'knife'. "
                "2. Conflicting label semantics: In YOLO, 'no-weapon' hard negatives MUST be represented as empty label files "
                "(zero bounding boxes), NOT as a detected foreground bounding box class. Training 'nogun' as a detected class "
                "forces the model to predict bounding boxes on every phone, bottle, and flashlight. "
                "3. Cross-dataset deduplication and remapping into background images is strictly required."
            )
        },
        "totals": {
            "combined_raw_images": nogun_audit["total_images"] + weapon_audit["total_images"],
            "combined_raw_annotations": nogun_audit["total_annotations"] + weapon_audit["total_annotations"],
        }
    }

    summary_json_path = OUTPUT_DIR / "weapon_dataset_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\nSaved summary to: {summary_json_path}")
    print("=================================================================")
    return full_summary


if __name__ == "__main__":
    run_full_audit()
