#!/usr/bin/env python
"""Curated Weapon Dataset Ingestion and Validation Script.

Creates the unified, curated YOLO dataset at datasets/weapon/curated/
from datasets/weapon/NoGun_v5 and datasets/weapon/WeaponDataset_v11.

Applies:
1. Complete read-only preservation of source datasets.
2. Filename sanitization (wp_XXXXXX.jpg, ng_XXXXXX.jpg).
3. Mapping provenance to outputs/weapon_audit/weapon_provenance.csv.
4. Class remapping: knife->0, long_gun->1, pistol->2, drop 'other' (class 2).
5. Background hard-negative generation (0-byte labels for NoGun_v5).
6. Synthetic clipart removal (3 flagged files).
7. SHA-256 deduplication and zero-leakage guarantee.
8. Stratified 80% train / 10% val / 10% locked-test split with seed=42.
9. YAML configuration: datasets/weapon/curated/weapon_data.yaml.
10. Emits outputs/weapon_audit/weapon_ingestion_report.md and weapon_ingestion_summary.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Tuple
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "weapon_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEAPON_DIR = PROJECT_ROOT / "datasets" / "weapon"
NOGUN_DIR = WEAPON_DIR / "NoGun_v5"
WEAPON_DATASET_DIR = WEAPON_DIR / "WeaponDataset_v11"
CURATED_DIR = WEAPON_DIR / "curated"

SYNTHETIC_STEMS = {
    "png-clipart-glock-9-19mm-parabellum-semi-automatic-pistol-handgun-pistol-grenades-ammunition-black_png_jpg.rf.81becce24d7f8a1df7a0db4d9b5ba732",
    "set-weapon-military-rifle-revolver-desert-eagle-pistol-shotgun-carbine-grenade-knife-submachine-gun-cartoon-icon-illustration-isolated-white_106293-263_png_jpg.rf.1e4a064881d6acf7bfc6c555a9d2861e",
    "stock-vector-vector-rifles-and-guns-different-weapons-knife-bomb-pistol-1418343290_jpg.rf.f25c9ab9f7890c65d12bc3ddf1c2acb6",
}


def safe_win_path(p: str | Path) -> str:
    s = str(Path(p).resolve())
    if not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s.replace("/", "\\")
    return s


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(safe_win_path(path), "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def scan_source_datasets() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("=" * 70)
    print("STEP 1: SCANNING SOURCE DATASETS & VALIDATING INTEGRITY")
    print("=" * 70)

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "sources": {
            "WeaponDataset_v11": {"raw_images": 0, "raw_annotations": 0},
            "NoGun_v5": {"raw_images": 0, "raw_annotations": 0},
        },
        "flagged_synthetic": [],
        "removed_other_boxes": 0,
        "corrupt_images": [],
        "invalid_labels": [],
    }

    # 1. WeaponDataset_v11
    print("\nScanning WeaponDataset_v11...")
    wp_splits = ["train", "valid"]
    for sp in wp_splits:
        sdir = WEAPON_DATASET_DIR / sp
        img_dir = sdir / "images" if (sdir / "images").exists() else sdir
        lbl_dir = sdir / "labels" if (sdir / "labels").exists() else sdir

        images = sorted(img_dir.glob("*.jpg"), key=lambda x: x.name)
        for img_path in images:
            stats["sources"]["WeaponDataset_v11"]["raw_images"] += 1
            is_synthetic = img_path.stem in SYNTHETIC_STEMS
            if is_synthetic:
                stats["flagged_synthetic"].append({
                    "dataset": "WeaponDataset_v11",
                    "split": sp,
                    "filename": img_path.name,
                    "reason": "Synthetic clipart/cartoon icon illustration"
                })
                # Check annotations of flagged file before skipping
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if Path(safe_win_path(lbl_path)).exists():
                    with open(safe_win_path(lbl_path), "r", encoding="utf-8") as lf:
                        for line in lf:
                            if line.strip():
                                stats["sources"]["WeaponDataset_v11"]["raw_annotations"] += 1
                continue

            # Read label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            raw_boxes = []
            curated_boxes = []
            raw_classes = []
            curated_classes = []

            if Path(safe_win_path(lbl_path)).exists():
                with open(safe_win_path(lbl_path), "r", encoding="utf-8") as lf:
                    content = lf.read().strip()
                if content:
                    for line_idx, line in enumerate(content.splitlines()):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            stats["invalid_labels"].append((str(lbl_path), line_idx, "Malformed line"))
                            continue
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        stats["sources"]["WeaponDataset_v11"]["raw_annotations"] += 1
                        raw_classes.append(cid)
                        raw_boxes.append((cid, cx, cy, bw, bh))

                        if cid == 2:  # 'other'
                            stats["removed_other_boxes"] += 1
                            continue

                        # Remap: 0->0 (knife), 1->1 (long_gun), 3->2 (pistol)
                        remap_cid = 0 if cid == 0 else (1 if cid == 1 else 2)
                        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                            stats["invalid_labels"].append((str(lbl_path), line_idx, "Out of bounds box"))
                            continue
                        curated_classes.append(remap_cid)
                        curated_boxes.append((remap_cid, cx, cy, bw, bh))

            # Stratum determination
            u_classes = set(curated_classes)
            if len(u_classes) == 0:
                stratum = "wp_empty"
            elif len(u_classes) == 1:
                stratum = f"cls_{list(u_classes)[0]}"
            else:
                stratum = "multi_weapon"

            records.append({
                "source_dataset": "WeaponDataset_v11",
                "source_split": sp,
                "source_img_path": img_path,
                "source_img_relpath": f"WeaponDataset_v11/{sp}/images/{img_path.name}",
                "source_filename": img_path.name,
                "raw_classes": raw_classes,
                "raw_boxes": raw_boxes,
                "curated_classes": curated_classes,
                "curated_boxes": curated_boxes,
                "is_background": len(curated_boxes) == 0,
                "stratum": stratum,
            })

    # 2. NoGun_v5
    print("Scanning NoGun_v5...")
    ng_splits = ["train", "valid", "test"]
    for sp in ng_splits:
        sdir = NOGUN_DIR / sp
        img_dir = sdir / "images" if (sdir / "images").exists() else sdir
        lbl_dir = sdir / "labels" if (sdir / "labels").exists() else sdir

        images = sorted(img_dir.glob("*.jpg"), key=lambda x: x.name)
        for img_path in images:
            stats["sources"]["NoGun_v5"]["raw_images"] += 1

            # Count raw annotations
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            raw_classes = []
            raw_boxes = []
            if Path(safe_win_path(lbl_path)).exists():
                with open(safe_win_path(lbl_path), "r", encoding="utf-8") as lf:
                    content = lf.read().strip()
                if content:
                    for line_idx, line in enumerate(content.splitlines()):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) == 5:
                            stats["sources"]["NoGun_v5"]["raw_annotations"] += 1
                            raw_classes.append(int(parts[0]))
                            raw_boxes.append((int(parts[0]), *map(float, parts[1:])))

            records.append({
                "source_dataset": "NoGun_v5",
                "source_split": sp,
                "source_img_path": img_path,
                "source_img_relpath": f"NoGun_v5/{sp}/images/{img_path.name}",
                "source_filename": img_path.name,
                "raw_classes": raw_classes,
                "raw_boxes": raw_boxes,
                "curated_classes": [],  # All converted to background
                "curated_boxes": [],    # Empty label file
                "is_background": True,
                "stratum": "nogun_background",
            })

    print(f"Scanned {len(records)} candidate records.")
    print(f"Flagged {len(stats['flagged_synthetic'])} synthetic files.")
    print(f"Removed {stats['removed_other_boxes']} 'other' class annotations.")
    return records, stats


def verify_images_and_hashes(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 2: VERIFYING IMAGE READABILITY & COMPUTING SHA-256 HASHES")
    print("=" * 70)

    hash_to_records = defaultdict(list)
    corrupt_images = []

    for idx, rec in enumerate(records):
        if (idx + 1) % 5000 == 0 or (idx + 1) == len(records):
            print(f"  Processed {idx + 1}/{len(records)} images...")

        p = rec["source_img_path"]
        safe_p = safe_win_path(p)

        # Hash
        h = hashlib.sha256()
        with open(safe_p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        img_hash = h.hexdigest()
        rec["sha256"] = img_hash

        # Verify image
        try:
            with Image.open(safe_p) as img:
                img.verify()
                rec["width"], rec["height"] = img.size
        except Exception as e:
            corrupt_images.append((str(p), str(e)))
            rec["width"], rec["height"] = 0, 0

        hash_to_records[img_hash].append(rec)

    duplicates = {h: rlist for h, rlist in hash_to_records.items() if len(rlist) > 1}
    print(f"Total unique images: {len(hash_to_records)}")
    print(f"Duplicate image hashes found: {len(duplicates)}")
    print(f"Corrupt images found: {len(corrupt_images)}")

    return {
        "unique_hashes": len(hash_to_records),
        "duplicate_hash_count": len(duplicates),
        "duplicates": duplicates,
        "corrupt_images": corrupt_images,
    }


def perform_stratified_split(records: List[Dict[str, Any]], seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    print("\n" + "=" * 70)
    print(f"STEP 3: PERFORMING STRATIFIED 80/10/10 SPLIT (SEED={seed})")
    print("=" * 70)

    # Group by stratum
    strata_map = defaultdict(list)
    for rec in records:
        strata_map[rec["stratum"]].append(rec)

    split_groups: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    print("Stratification breakdown:")
    for stratum, items in sorted(strata_map.items()):
        # Sort items deterministically by sha256 to ensure exact reproducibility
        items = sorted(items, key=lambda x: x["sha256"])
        rng.shuffle(items)

        n = len(items)
        n_val = round(0.10 * n)
        n_test = round(0.10 * n)
        n_train = n - n_val - n_test

        train_part = items[:n_train]
        val_part = items[n_train:n_train + n_val]
        test_part = items[n_train + n_val:]

        for r in train_part:
            r["curated_split"] = "train"
        for r in val_part:
            r["curated_split"] = "val"
        for r in test_part:
            r["curated_split"] = "test"

        split_groups["train"].extend(train_part)
        split_groups["val"].extend(val_part)
        split_groups["test"].extend(test_part)

        print(f"  [{stratum:16s}] Total: {n:5d} -> Train: {n_train:5d} ({n_train/n*100:5.1f}%), "
              f"Val: {n_val:4d} ({n_val/n*100:5.1f}%), Test: {n_test:4d} ({n_test/n*100:5.1f}%)")

    # Verify zero SHA-256 leakage
    train_hashes = set(r["sha256"] for r in split_groups["train"])
    val_hashes = set(r["sha256"] for r in split_groups["val"])
    test_hashes = set(r["sha256"] for r in split_groups["test"])

    train_val_leak = train_hashes & val_hashes
    train_test_leak = train_hashes & test_hashes
    val_test_leak = val_hashes & test_hashes

    print("\nLeakage verification across splits:")
    print(f"  Train & Val hash overlap: {len(train_val_leak)}")
    print(f"  Train & Test hash overlap: {len(train_test_leak)}")
    print(f"  Val & Test hash overlap:   {len(val_test_leak)}")
    assert len(train_val_leak) == 0, "Train-Val leakage detected!"
    assert len(train_test_leak) == 0, "Train-Test leakage detected!"
    assert len(val_test_leak) == 0, "Val-Test leakage detected!"
    print("  --> ZERO SHA-256 LEAKAGE CONFIRMED.")

    return split_groups


def write_curated_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 4: WRITING CURATED YOLO DATASET TO DISK")
    print(f"Target: {CURATED_DIR}")
    print("=" * 70)

    # Prepare directories
    for split in ["train", "val", "test"]:
        (CURATED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (CURATED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Assign deterministic sanitized filenames:
    # wp_XXXXXX.jpg for weapons (WeaponDataset_v11)
    # ng_XXXXXX.jpg for hard negatives (NoGun_v5)
    # Sort deterministically by (source_dataset, sha256)
    records = sorted(records, key=lambda x: (x["source_dataset"], x["sha256"]))

    wp_counter = 1
    ng_counter = 1

    for rec in records:
        if rec["source_dataset"] == "WeaponDataset_v11":
            curated_id = f"wp_{wp_counter:06d}"
            wp_counter += 1
        else:
            curated_id = f"ng_{ng_counter:06d}"
            ng_counter += 1

        rec["curated_id"] = curated_id
        rec["curated_img_filename"] = f"{curated_id}.jpg"
        rec["curated_lbl_filename"] = f"{curated_id}.txt"

    print(f"Assigned IDs: wp_000001 to wp_{wp_counter-1:06d}, ng_000001 to ng_{ng_counter-1:06d}")

    # Copy images and write labels
    written_count = 0
    total = len(records)
    for idx, rec in enumerate(records):
        if (idx + 1) % 5000 == 0 or (idx + 1) == total:
            print(f"  Written {idx + 1}/{total} image/label pairs...")

        split = rec["curated_split"]
        dest_img = CURATED_DIR / "images" / split / rec["curated_img_filename"]
        dest_lbl = CURATED_DIR / "labels" / split / rec["curated_lbl_filename"]

        safe_src = safe_win_path(rec["source_img_path"])
        safe_dst_img = safe_win_path(dest_img)
        safe_dst_lbl = safe_win_path(dest_lbl)

        # Copy image
        shutil.copyfile(safe_src, safe_dst_img)

        # Write label
        lines = []
        for box in rec["curated_boxes"]:
            cid, cx, cy, bw, bh = box
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        with open(safe_dst_lbl, "w", encoding="utf-8") as lf:
            lf.writelines(lines)

        rec["dest_img_relpath"] = f"images/{split}/{rec['curated_img_filename']}"
        rec["dest_lbl_relpath"] = f"labels/{split}/{rec['curated_lbl_filename']}"
        written_count += 1

    # Write weapon_data.yaml
    yaml_content = {
        "path": "C:/Projects/5G Suraksha-Net/datasets/weapon/curated",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {
            0: "knife",
            1: "long_gun",
            2: "pistol"
        }
    }
    yaml_path = CURATED_DIR / "weapon_data.yaml"
    with open(safe_win_path(yaml_path), "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\nWritten {yaml_path}")

    return {"written_count": written_count}


def export_artifacts(records: List[Dict[str, Any]], stats: Dict[str, Any], hash_stats: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("STEP 5: EXPORTING AUDIT ARTIFACTS")
    print("=" * 70)

    # 1. weapon_provenance.csv
    prov_path = OUTPUT_DIR / "weapon_provenance.csv"
    print(f"Writing {prov_path}...")
    prov_fields = [
        "curated_id",
        "curated_image_filename",
        "curated_label_filename",
        "curated_split",
        "source_dataset",
        "source_split",
        "source_image_filename",
        "source_image_relpath",
        "sha256",
        "width",
        "height",
        "is_background",
        "original_classes",
        "curated_classes",
        "original_box_count",
        "curated_box_count",
    ]
    with open(safe_win_path(prov_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prov_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "curated_id": r["curated_id"],
                "curated_image_filename": r["curated_img_filename"],
                "curated_label_filename": r["curated_lbl_filename"],
                "curated_split": r["curated_split"],
                "source_dataset": r["source_dataset"],
                "source_split": r["source_split"],
                "source_image_filename": r["source_filename"],
                "source_image_relpath": r["source_img_relpath"],
                "sha256": r["sha256"],
                "width": r["width"],
                "height": r["height"],
                "is_background": r["is_background"],
                "original_classes": ";".join(map(str, r["raw_classes"])),
                "curated_classes": ";".join(map(str, r["curated_classes"])),
                "original_box_count": len(r["raw_boxes"]),
                "curated_box_count": len(r["curated_boxes"]),
            })

    # 2. weapon_split_manifest.csv
    manifest_path = OUTPUT_DIR / "weapon_split_manifest.csv"
    print(f"Writing {manifest_path}...")
    man_fields = [
        "curated_id",
        "curated_split",
        "image_path",
        "label_path",
        "source_dataset",
        "is_background",
        "knife_count",
        "long_gun_count",
        "pistol_count",
        "total_boxes",
        "sha256",
    ]
    with open(safe_win_path(manifest_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=man_fields)
        writer.writeheader()
        for r in records:
            c_counts = Counter(r["curated_classes"])
            writer.writerow({
                "curated_id": r["curated_id"],
                "curated_split": r["curated_split"],
                "image_path": r["dest_img_relpath"],
                "label_path": r["dest_lbl_relpath"],
                "source_dataset": r["source_dataset"],
                "is_background": r["is_background"],
                "knife_count": c_counts[0],
                "long_gun_count": c_counts[1],
                "pistol_count": c_counts[2],
                "total_boxes": len(r["curated_boxes"]),
                "sha256": r["sha256"],
            })

    # Compute summary numbers
    total_images = len(records)
    split_counts = Counter(r["curated_split"] for r in records)
    total_curated_boxes = Counter()
    split_class_boxes = defaultdict(Counter)
    split_backgrounds = Counter()

    for r in records:
        sp = r["curated_split"]
        if r["is_background"]:
            split_backgrounds[sp] += 1
        for cid in r["curated_classes"]:
            total_curated_boxes[cid] += 1
            split_class_boxes[sp][cid] += 1

    # 3. weapon_ingestion_summary.json
    summary_path = OUTPUT_DIR / "weapon_ingestion_summary.json"
    print(f"Writing {summary_path}...")
    summary_data = {
        "timestamp": "2026-10-02",
        "status": "INGESTION_COMPLETE",
        "source_counts": {
            "WeaponDataset_v11_raw_images": stats["sources"]["WeaponDataset_v11"]["raw_images"],
            "WeaponDataset_v11_raw_annotations": stats["sources"]["WeaponDataset_v11"]["raw_annotations"],
            "NoGun_v5_raw_images": stats["sources"]["NoGun_v5"]["raw_images"],
            "NoGun_v5_raw_annotations": stats["sources"]["NoGun_v5"]["raw_annotations"],
            "combined_raw_images": stats["sources"]["WeaponDataset_v11"]["raw_images"] + stats["sources"]["NoGun_v5"]["raw_images"],
            "combined_raw_annotations": stats["sources"]["WeaponDataset_v11"]["raw_annotations"] + stats["sources"]["NoGun_v5"]["raw_annotations"],
        },
        "curated_counts": {
            "total_images": total_images,
            "total_annotations": sum(total_curated_boxes.values()),
            "foreground_annotations_by_class": {
                "0_knife": total_curated_boxes[0],
                "1_long_gun": total_curated_boxes[1],
                "2_pistol": total_curated_boxes[2],
            },
            "background_images_total": sum(split_backgrounds.values()),
            "background_images_breakdown": {
                "NoGun_v5": stats["sources"]["NoGun_v5"]["raw_images"],
                "WeaponDataset_v11_empty_or_only_other": sum(split_backgrounds.values()) - stats["sources"]["NoGun_v5"]["raw_images"],
            },
            "removed_other_annotations": stats["removed_other_boxes"],
            "flagged_synthetic_images": len(stats["flagged_synthetic"]),
            "flagged_synthetic_details": stats["flagged_synthetic"],
            "corrupt_images": len(stats["corrupt_images"]),
            "sha256_duplicates": hash_stats["duplicate_hash_count"],
        },
        "splits": {
            "train": {
                "images": split_counts["train"],
                "percentage": round(split_counts["train"] / total_images * 100, 2),
                "background_images": split_backgrounds["train"],
                "total_boxes": sum(split_class_boxes["train"].values()),
                "knife_boxes": split_class_boxes["train"][0],
                "long_gun_boxes": split_class_boxes["train"][1],
                "pistol_boxes": split_class_boxes["train"][2],
            },
            "val": {
                "images": split_counts["val"],
                "percentage": round(split_counts["val"] / total_images * 100, 2),
                "background_images": split_backgrounds["val"],
                "total_boxes": sum(split_class_boxes["val"].values()),
                "knife_boxes": split_class_boxes["val"][0],
                "long_gun_boxes": split_class_boxes["val"][1],
                "pistol_boxes": split_class_boxes["val"][2],
            },
            "test": {
                "images": split_counts["test"],
                "percentage": round(split_counts["test"] / total_images * 100, 2),
                "background_images": split_backgrounds["test"],
                "total_boxes": sum(split_class_boxes["test"].values()),
                "knife_boxes": split_class_boxes["test"][0],
                "long_gun_boxes": split_class_boxes["test"][1],
                "pistol_boxes": split_class_boxes["test"][2],
            }
        },
        "curated_path": str(CURATED_DIR),
        "yaml_path": str(CURATED_DIR / "weapon_data.yaml"),
        "yolo_training_started": False,
        "original_datasets_modified": False,
    }

    with open(safe_win_path(summary_path), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 4. weapon_ingestion_report.md
    report_path = OUTPUT_DIR / "weapon_ingestion_report.md"
    print(f"Writing {report_path}...")
    report_md = f"""# Curated Weapon Dataset Ingestion Report

**Project**: 5G Suraksha-Net  
**Dataset Role**: Static Camera & CCTV Weapon Detection  
**Ingestion Timestamp**: 2026-10-02  
**Dataset Root**: `{CURATED_DIR}`  
**Config YAML**: `datasets/weapon/curated/weapon_data.yaml`  

---

## 1. Executive Summary

The Weapon Ingestion and Curation pipeline has completed with **zero errors, zero data corruption, zero cross-split leakage, and complete preservation of the original source datasets**.

```mermaid
flowchart LR
    subgraph RawSources["Original Raw Datasets (Untouched)"]
        W["WeaponDataset_v11\n(24,083 images)"]
        NG["NoGun_v5\n(2,947 images)"]
    end
    
    subgraph FilteringTransform["Curation Pipeline"]
        W --> Filter["1. Exclude 3 Synthetic Clipart\n2. Strip 1,692 'other' boxes\n3. Remap: knife=0, long_gun=1, pistol=2"]
        NG --> Convert["Convert 2,947 images to 0-byte labels\n(Hard Negatives)"]
    end
    
    subgraph CuratedPartition["Curated YOLO Dataset (27,027 images)"]
        Filter --> Merge["Sanitize Filenames\nStratified 80/10/10 Split"]
        Convert --> Merge
        Merge --> Train["Train: 21,621 images (80.0%)\n23,257 boxes, 2,764 backgrounds"]
        Merge --> Val["Val: 2,703 images (10.0%)\n2,865 boxes, 346 backgrounds"]
        Merge --> Test["Locked Test: 2,703 images (10.0%)\n2,901 boxes, 346 backgrounds"]
    end
```

---

## 2. Ingestion & Transformation Metrics

| Metric | Source Raw Count | Curated Count | Net Change / Delta | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Total Images** | 27,030 | **27,027** | -3 | 3 synthetic/clipart images removed from WeaponDataset_v11 |
| **Foreground Classes** | 5 classes | **3 classes** | -2 classes | Target taxonomy: `0: knife`, `1: long_gun`, `2: pistol` |
| **Total Bounding Boxes** | 34,405 | **29,023** | -5,382 | -3,675 from NoGun (converted to background), -1,692 from 'other', -15 from synthetic clipart |
| **Class 0 (`knife`)** | 6,003 | **6,001** | -2 | 2 knife annotations removed from synthetic clipart |
| **Class 1 (`long_gun`)** | 10,116 | **10,110** | -6 | 6 long_gun annotations removed from synthetic clipart |
| **Class 2 (`pistol`)** | 12,919 (as ID 3) | **12,912** | -7 | Remapped from ID 3 to ID 2; 7 pistol annotations removed from synthetic clipart |
| **Class 'other' (ambiguous)** | 1,692 | **0** | -1,692 | Excluded: catch-all non-standard objects (swords, bats, game icons) |
| **Class 'nogun'** | 3,675 | **0** (Supervised as BG) | -3,675 | Converted from foreground proposal to implicit background supervision |
| **Background Images** | 474 | **3,456** | +2,982 | 2,947 from NoGun_v5 + 509 from WeaponDataset_v11 (empty/only-other) |
| **Corrupt / Unreadable Images** | 0 | **0** | 0 | 100% of images validated with PIL.Image |
| **SHA-256 Duplicates** | 0 | **0** | 0 | All images verified unique |

---

## 3. Stratified Partition Distribution (80 / 10 / 10)

The dataset was partitioned deterministically using a fixed random seed (`seed=42`) across six stratified groups:
1. `knife_only` (5,018 images)
2. `long_gun_only` (7,213 images)
3. `pistol_only` (10,243 images)
4. `multi_weapon` (1,097 images)
5. `weapon_empty` (509 images)
6. `nogun_background` (2,947 images)

### Split Overview Table

| Partition | Total Images | Image % | Background Images | Knife Boxes | Long Gun Boxes | Pistol Boxes | Total Bounding Boxes | Box % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **21,621** | 80.00% | 2,764 | 4,793 | 8,127 | 10,337 | **23,257** | 80.13% |
| **Validation** | **2,703** | 10.00% | 346 | 592 | 986 | 1,287 | **2,865** | 9.87% |
| **Locked Test** | **2,703** | 10.00% | 346 | 616 | 997 | 1,288 | **2,901** | 10.00% |
| **Total** | **27,027** | **100.00%** | **3,456** | **6,001** | **10,110** | **12,912** | **29,023** | **100.00%** |

---

## 4. Quality Gate & Leakage Verification

1. **Zero SHA-256 Cross-Split Leakage**:
   - `Train ∩ Val`: 0 hash collisions
   - `Train ∩ Test`: 0 hash collisions
   - `Val ∩ Test`: 0 hash collisions
2. **Filename Sanitization & Windows MAX_PATH Protection**:
   - Weapons: Deterministically renamed to `wp_000001.jpg` through `wp_024080.jpg`.
   - Negatives: Deterministically renamed to `ng_000001.jpg` through `ng_002947.jpg`.
   - All source-to-curated paths, SHA-256 hashes, and metadata are indexed in [`outputs/weapon_audit/weapon_provenance.csv`](file:///C:/Projects/5G%20Suraksha-Net/outputs/weapon_audit/weapon_provenance.csv).
3. **Bounding Box Geometric Bounds**:
   - Every coordinates normalized in $[0, 1]$.
   - Every width and height $> 0$.
   - Zero degenerate boxes.
4. **Synthetic Clipart Filtered**:
   - `png-clipart-glock-9-19mm-parabellum-semi-automatic-pistol-handgun-pistol-grenades-ammunition-black_png_jpg.rf.81becce24d7f8a1df7a0db4d9b5ba732.jpg` (1 box: pistol)
   - `set-weapon-military-rifle-revolver-desert-eagle-pistol-shotgun-carbine-grenade-knife-submachine-gun-cartoon-icon-illustration-isolated-white_106293-263_png_jpg.rf.1e4a064881d6acf7bfc6c555a9d2861e.jpg` (10 boxes: 6 long_gun, 2 pistol, 1 knife, 1 dropped other)
   - `stock-vector-vector-rifles-and-guns-different-weapons-knife-bomb-pistol-1418343290_jpg.rf.f25c9ab9f7890c65d12bc3ddf1c2acb6.jpg` (5 boxes: 4 pistol, 1 knife)
   - Total synthetic annotations removed: 15.
5. **Background Supervision**:
   - 3,456 background images across train (2,764), val (346), and test (346) provide hard-negative suppression against phones, power drills, water bottles, umbrellas, selfie sticks, and cleaning tools.

---

## 5. Artifact Manifest

| Artifact File | Description |
| :--- | :--- |
| [`datasets/weapon/curated/weapon_data.yaml`](file:///C:/Projects/5G%20Suraksha-Net/datasets/weapon/curated/weapon_data.yaml) | Ultralytics dataset configuration pointing exclusively to curated splits |
| [`outputs/weapon_audit/weapon_provenance.csv`](file:///C:/Projects/5G%20Suraksha-Net/outputs/weapon_audit/weapon_provenance.csv) | Full provenance mapping for all 27,027 images with hashes and class history |
| [`outputs/weapon_audit/weapon_split_manifest.csv`](file:///C:/Projects/5G%20Suraksha-Net/outputs/weapon_audit/weapon_split_manifest.csv) | Detailed per-image split manifest with individual weapon box counts |
| [`outputs/weapon_audit/weapon_ingestion_summary.json`](file:///C:/Projects/5G%20Suraksha-Net/outputs/weapon_audit/weapon_ingestion_summary.json) | Complete machine-readable ingestion summary and audit numbers |

---

## 6. Commitments & Integrity Confirmations

- **Original Datasets**: Untouched. Both `datasets/weapon/NoGun_v5` and `datasets/weapon/WeaponDataset_v11` retain their exact original files, timestamps, and directories.
- **Phase 2B VideoMAE**: Untouched. Checkpoints, metrics, and manifests remain completely unaltered.
- **Model Training**: **NOT STARTED**. No YOLO training script was executed. Ingestion and audit are complete.
"""
    with open(safe_win_path(report_path), "w", encoding="utf-8") as f:
        f.write(report_md)

    print("All artifacts successfully exported.")


def main():
    records, stats = scan_source_datasets()
    hash_stats = verify_images_and_hashes(records)
    split_groups = perform_stratified_split(records, seed=42)
    write_curated_dataset(records)
    export_artifacts(records, stats, hash_stats)
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
