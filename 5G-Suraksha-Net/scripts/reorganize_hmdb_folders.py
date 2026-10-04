#!/usr/bin/env python
"""Safe reorganization script for HMDB51 action folders scattered at project root.

Moves ONLY the 15 verified HMDB51 action-class folders:
  brush_hair, cartwheel, catch, chew, clap, climb, hit, hug, kick,
  punch, shake_hands, sit, stand, talk, walk
from project root into:
  datasets/action/HMBD/<class_name>/

Performs pre-move and post-move integrity audits.
Preserves all Phase 2B model checkpoints, manifests, outputs, and configs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HMBD_TARGET_DIR = PROJECT_ROOT / "datasets" / "action" / "HMBD"

HMDB_CLASSES = [
    "brush_hair",
    "cartwheel",
    "catch",
    "chew",
    "clap",
    "climb",
    "hit",
    "hug",
    "kick",
    "punch",
    "shake_hands",
    "sit",
    "stand",
    "talk",
    "walk",
]


def reorganize():
    print("=================================================================")
    print("HMDB51 ROOT REORGANIZATION")
    print("=================================================================")
    assert HMBD_TARGET_DIR.exists(), f"Target directory not found: {HMBD_TARGET_DIR}"

    # 1. Audit root folders
    folders_to_move = []
    for cls in HMDB_CLASSES:
        src = PROJECT_ROOT / cls
        if src.exists() and src.is_dir():
            files = list(src.iterdir())
            assert all(f.suffix == ".avi" for f in files if f.is_file()), f"Unexpected non-avi file in {src}"
            folders_to_move.append((cls, src, files))
        else:
            print(f"  [Notice] Folder {cls} not found at root (already moved or missing).")

    print(f"Identified {len(folders_to_move)} HMDB51 folders at project root:")
    for cls, src, files in folders_to_move:
        print(f"  - {cls} ({len(files)} files: {[f.name for f in files]})")

    # 2. Perform safe moves
    moved = []
    for cls, src, files in folders_to_move:
        dest = HMBD_TARGET_DIR / cls
        if dest.exists():
            print(f"  Destination {dest} already exists. Merging files...")
            for f in files:
                target_file = dest / f.name
                if not target_file.exists():
                    shutil.move(str(f), str(target_file))
            # Remove empty source dir
            if not list(src.iterdir()):
                src.rmdir()
        else:
            shutil.move(str(src), str(dest))
        moved.append((cls, dest))
        print(f"  Moved: {cls} -> {dest}")

    # 3. Post-move verification
    print("\n--- Verifying Post-Move State ---")
    remaining_at_root = [cls for cls in HMDB_CLASSES if (PROJECT_ROOT / cls).exists()]
    assert len(remaining_at_root) == 0, f"Error: Folders still at root: {remaining_at_root}"
    print("  --> [OK] Zero HMDB51 action folders remain at project root.")

    for cls, dest in moved:
        assert dest.exists(), f"Destination missing: {dest}"
        files_in_dest = list(dest.glob("*.avi"))
        assert len(files_in_dest) > 0, f"No AVI files in destination: {dest}"
    print(f"  --> [OK] All {len(moved)} moved folders verified at {HMBD_TARGET_DIR}.")

    # 4. Manifest Integrity Check
    print("\n--- Checking Phase 2B Manifests Integrity ---")
    exp_dir = PROJECT_ROOT / "outputs" / "phase2b" / "experiment_001"
    import pandas as pd
    for m in ["train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]:
        mf_path = exp_dir / m
        if mf_path.exists():
            df = pd.read_csv(mf_path)
            sample_paths = df["video_path"].head(20).tolist()
            missing = [p for p in sample_paths if not Path(p).exists()]
            assert len(missing) == 0, f"Manifest {m} paths missing: {missing}"
            print(f"  --> [OK] {m} ({len(df)} entries) resolves all sampled file paths cleanly.")

    print("=================================================================")
    print("REORGANIZATION AND INTEGRITY CHECKS COMPLETED SUCCESSFULLY")
    print("=================================================================")


if __name__ == "__main__":
    reorganize()
