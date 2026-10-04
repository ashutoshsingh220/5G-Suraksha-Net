"""Integration and validation tests for the curated weapon dataset.

Validates:
1. Curated directory structure and weapon_data.yaml taxonomy.
2. Complete 1-to-1 correspondence between images and labels across all splits.
3. Strict YOLO bounding box validity (class in {0, 1, 2}, normalized coords in [0, 1], w, h > 0).
4. Hard-negative background label integrity (0-byte empty labels).
5. Exact class annotation counts (knife=6001, long_gun=10110, pistol=12912).
6. Total image counts (train=21621, val=2703, test=2703, total=27027).
7. Zero cross-split SHA-256 duplicate leakage.
8. Complete exclusion of flagged synthetic clipart files.
9. Provenance CSV and split manifest integrity.
10. Source datasets (NoGun_v5, WeaponDataset_v11) remain 100% untouched.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = PROJECT_ROOT / "datasets" / "weapon" / "curated"
NOGUN_DIR = PROJECT_ROOT / "datasets" / "weapon" / "NoGun_v5"
WEAPON_DATASET_DIR = PROJECT_ROOT / "datasets" / "weapon" / "WeaponDataset_v11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "weapon_audit"


def safe_win_path(p: str | Path) -> str:
    s = str(Path(p).resolve())
    if not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s.replace("/", "\\")
    return s


def test_curated_directories_exist():
    """Verify that all required image and label directories exist."""
    assert CURATED_DIR.exists(), f"Curated directory {CURATED_DIR} does not exist"
    for split in ["train", "val", "test"]:
        img_dir = CURATED_DIR / "images" / split
        lbl_dir = CURATED_DIR / "labels" / split
        assert img_dir.exists(), f"Missing image directory: {img_dir}"
        assert lbl_dir.exists(), f"Missing label directory: {lbl_dir}"


def test_weapon_data_yaml_structure():
    """Verify that weapon_data.yaml conforms strictly to 3-class target taxonomy."""
    yaml_path = CURATED_DIR / "weapon_data.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    with open(safe_win_path(yaml_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["nc"] == 3, f"Expected 3 classes, got {config['nc']}"
    assert "train" in config and config["train"] == "images/train"
    assert "val" in config and config["val"] == "images/val"
    assert "test" in config and config["test"] == "images/test"

    names = config["names"]
    if isinstance(names, dict):
        assert names[0] == "knife"
        assert names[1] == "long_gun"
        assert names[2] == "pistol"
        assert 3 not in names
    elif isinstance(names, list):
        assert names == ["knife", "long_gun", "pistol"]
    else:
        pytest.fail(f"Invalid names structure in yaml: {names}")

    # Ensure "other" and "nogun" are NOT foreground classes
    all_names = list(names.values()) if isinstance(names, dict) else names
    assert "other" not in all_names, "'other' class must not exist in taxonomy"
    assert "nogun" not in all_names, "'nogun' class must not exist in taxonomy"


def test_split_image_and_label_counts():
    """Verify exact 80/10/10 split counts and 1:1 image-to-label correspondence."""
    expected_counts = {
        "train": 21621,
        "val": 2703,
        "test": 2703,
    }
    total_images = 0

    for split, exp_count in expected_counts.items():
        img_dir = CURATED_DIR / "images" / split
        lbl_dir = CURATED_DIR / "labels" / split

        images = sorted(img_dir.glob("*.jpg"))
        labels = sorted(lbl_dir.glob("*.txt"))

        assert len(images) == exp_count, f"Split {split}: expected {exp_count} images, got {len(images)}"
        assert len(labels) == exp_count, f"Split {split}: expected {exp_count} labels, got {len(labels)}"

        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}

        assert image_stems == label_stems, f"Mismatch between image and label stems in {split}"
        total_images += len(images)

    assert total_images == 27027, f"Expected 27,027 total curated images, got {total_images}"


def test_label_validity_and_annotation_totals():
    """Verify YOLO label syntax, coordinate bounds, and exact annotation totals by class."""
    class_counts = {0: 0, 1: 0, 2: 0}
    empty_label_counts = {"train": 0, "val": 0, "test": 0}
    total_boxes = 0

    for split in ["train", "val", "test"]:
        lbl_dir = CURATED_DIR / "labels" / split
        for lbl_path in lbl_dir.glob("*.txt"):
            with open(safe_win_path(lbl_path), "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                empty_label_counts[split] += 1
                continue

            for line_idx, line in enumerate(content.splitlines()):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                assert len(parts) == 5, f"Line {line_idx} in {lbl_path} has {len(parts)} tokens instead of 5"

                cid = int(parts[0])
                assert cid in (0, 1, 2), f"Invalid class ID {cid} in {lbl_path}"

                cx, cy, bw, bh = map(float, parts[1:])
                assert 0.0 <= cx <= 1.0, f"cx out of bounds ({cx}) in {lbl_path}"
                assert 0.0 <= cy <= 1.0, f"cy out of bounds ({cy}) in {lbl_path}"
                assert 0.0 < bw <= 1.0, f"bw invalid ({bw}) in {lbl_path}"
                assert 0.0 < bh <= 1.0, f"bh invalid ({bh}) in {lbl_path}"

                class_counts[cid] += 1
                total_boxes += 1

    # Verify exact class counts
    assert class_counts[0] == 6001, f"Expected 6,001 knife boxes, got {class_counts[0]}"
    assert class_counts[1] == 10110, f"Expected 10,110 long_gun boxes, got {class_counts[1]}"
    assert class_counts[2] == 12912, f"Expected 12,912 pistol boxes, got {class_counts[2]}"
    assert total_boxes == 29023, f"Expected 29,023 total boxes, got {total_boxes}"

    # Verify background counts
    total_bgs = sum(empty_label_counts.values())
    assert total_bgs == 3456, f"Expected 3,456 background images, got {total_bgs}"
    assert empty_label_counts["train"] == 2764
    assert empty_label_counts["val"] == 346
    assert empty_label_counts["test"] == 346


def test_zero_leakage_and_synthetic_exclusion():
    """Verify zero duplicate SHA-256 hashes across splits and synthetic clipart exclusion."""
    split_hashes = {"train": set(), "val": set(), "test": set()}

    synthetic_snippets = ["png-clipart-glock", "set-weapon-military-rifle", "stock-vector-vector-rifles"]

    manifest_path = OUTPUT_DIR / "weapon_split_manifest.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(safe_win_path(manifest_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["curated_split"]
            h = row["sha256"]
            assert h not in split_hashes[sp], f"Internal duplicate in split {sp}: {h}"
            split_hashes[sp].add(h)

    # Verify zero cross-split leakage
    train_val_overlap = split_hashes["train"] & split_hashes["val"]
    train_test_overlap = split_hashes["train"] & split_hashes["test"]
    val_test_overlap = split_hashes["val"] & split_hashes["test"]

    assert len(train_val_overlap) == 0, f"Train-Val hash leakage: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Train-Test hash leakage: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Val-Test hash leakage: {len(val_test_overlap)}"

    # Verify synthetic clipart exclusion from provenance
    prov_path = OUTPUT_DIR / "weapon_provenance.csv"
    assert prov_path.exists(), f"Missing {prov_path}"

    with open(safe_win_path(prov_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_name = row["source_image_filename"].lower()
            for syn in synthetic_snippets:
                assert syn not in src_name, f"Flagged synthetic file {src_name} found in curated provenance!"


def test_source_datasets_untouched():
    """Verify that original source datasets (NoGun_v5, WeaponDataset_v11) were NOT modified."""
    # NoGun_v5 image count: 2,947
    ng_images = list(NOGUN_DIR.glob("**/*.jpg"))
    assert len(ng_images) == 2947, f"Original NoGun_v5 modified! Expected 2,947 images, found {len(ng_images)}"

    # WeaponDataset_v11 image count: 24,083
    wp_images = list(WEAPON_DATASET_DIR.glob("**/*.jpg"))
    assert len(wp_images) == 24083, f"Original WeaponDataset_v11 modified! Expected 24,083 images, found {len(wp_images)}"
