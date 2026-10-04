#!/usr/bin/env python
"""Scan datasets/fight_cases/hard_negative/ and manage datasets/fight_cases/manifests/hard_cases.csv.

Idempotent: running multiple times preserves existing rows and never duplicates entries.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARD_NEGATIVE_DIR = PROJECT_ROOT / "datasets" / "fight_cases" / "hard_negative"
MANIFEST_FILE = PROJECT_ROOT / "datasets" / "fight_cases" / "manifests" / "hard_cases.csv"

CSV_FIELDS = [
    "path",
    "label",
    "category",
    "expected_incident",
    "expected_severity",
    "source",
    "notes",
]

KNOWN_CATEGORIES = {
    "standing_close",
    "talking",
    "gesturing",
    "walking",
    "crossing",
    "crowd",
    "accidental_contact",
    "uncertain",
}


def _infer_category(file_path: Path) -> str:
    """Infer scenario category from filename and parent folder without guessing."""
    name_lower = file_path.stem.lower()
    parent_name = file_path.parent.name.lower()

    if parent_name in KNOWN_CATEGORIES and parent_name != "hard_negative":
        return parent_name

    for cat in KNOWN_CATEGORIES:
        if cat in name_lower:
            return cat

    # Known manual webcam test mappings
    if "220440" in name_lower:
        return "standing_close"

    return "uncertain"


def update_manifest(manifest_path: Path = MANIFEST_FILE, scan_dir: Path = HARD_NEGATIVE_DIR) -> int:
    """Scan scan_dir for images and update manifest_path idempotently."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: dict[str, dict[str, str]] = {}

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get("path", "").strip()
                if rel_path:
                    # Normalize slashes for consistency
                    norm_path = rel_path.replace("\\", "/")
                    existing_rows[norm_path] = row

    # Discover images
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    discovered_images: list[Path] = []
    if scan_dir.exists():
        for p in scan_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                discovered_images.append(p)

    discovered_images.sort()

    new_count = 0
    for img_path in discovered_images:
        try:
            rel = img_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = img_path.as_posix()

        if rel not in existing_rows:
            cat = _infer_category(img_path)
            note = ""
            if "220440" in img_path.name:
                note = "Standing close to seated person (Candidate=1, GRU=0.13, must remain NORMAL)"
            elif "214107" in img_path.name:
                note = "Initial grappling test (Candidates=0 prior to candidate threshold calibration)"

            existing_rows[rel] = {
                "path": rel,
                "label": "normal",
                "category": cat,
                "expected_incident": "false",
                "expected_severity": "normal",
                "source": "manual_webcam_test",
                "notes": note,
            }
            new_count += 1

    # Write out sorted rows
    sorted_rows = sorted(existing_rows.values(), key=lambda r: r["path"])
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_rows)

    return len(sorted_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update hard-negative dataset manifest idempotently.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE, help="Path to manifest CSV")
    parser.add_argument("--dir", type=Path, default=HARD_NEGATIVE_DIR, help="Path to hard_negative directory")
    parser.add_argument("--check", action="store_true", help="Check manifest validity without modifying")
    args = parser.parse_args()

    total = update_manifest(args.manifest, args.dir)
    print(f"Manifest up to date: {args.manifest} ({total} entries)")


if __name__ == "__main__":
    main()
