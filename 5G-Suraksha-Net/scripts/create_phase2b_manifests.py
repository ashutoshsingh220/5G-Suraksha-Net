#!/usr/bin/env python
"""Create machine-verified, source-grouped train/val/test splits for Phase 2B.

Enforces strict leakage prevention:
- RWF-2000: Grouped by video_id / group_id. Official train split preserved;
            validation split cleanly divided into val & test by group_id.
- RLVS, Hockey, Movies: Clean source-grouped train/val/test splits.
- HMDB51: Grouped strictly by movie/source prefix so no scene/actor is ever
          shared between train and validation/test.
- Base duplicate clips (identified by SHA-256) are strictly excluded.
- Emits train_manifest.csv, val_manifest.csv, test_manifest.csv, and split_summary.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "outputs" / "phase2b" / "experiment_001"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)


def safe_win_path(path_str: str) -> str:
    """Format path with \\?\\ if needed for Windows long path compatibility."""
    p = Path(path_str).resolve()
    s = str(p)
    if len(s) > 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s.replace("/", "\\")
    return s


def load_base_manifest(manifest_path: Path, dataset_name: str):
    """Load base manifest, filter duplicates, and format records."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])

    # Identify SHA duplicates
    sha_seen = set()
    records = []
    dup_count = 0

    for e in entries:
        sha = e.get("sha256")
        if sha in sha_seen:
            dup_count += 1
            continue
        sha_seen.add(sha)

        label_str = e.get("label", "NON_FIGHT")
        binary_label = 1 if label_str == "FIGHT" else 0
        path_str = safe_win_path(e.get("path"))

        records.append({
            "clip_id": e.get("clip_id"),
            "video_path": path_str,
            "dataset_source": dataset_name,
            "source_class": "Fight" if binary_label == 1 else "NonFight",
            "label": binary_label,
            "label_name": "FIGHT" if binary_label == 1 else "NORMAL",
            "group_id": e.get("group_id") or e.get("video_id") or e.get("clip_id"),
            "original_split": e.get("split", "train"),
            "duration_sec": e.get("duration_sec", 0.0),
            "frame_count": e.get("frame_count", 0),
            "fps": e.get("fps", 30.0),
        })

    print(f"Loaded {dataset_name}: {len(records)} usable records ({dup_count} duplicates excluded)")
    return records


def split_records_by_group(records, train_ratio=0.70, val_ratio=0.15, seed=SEED):
    """Deterministically partition records into train/val/test by group_id with ZERO overlap."""
    rng = random.Random(seed)
    records_by_group = defaultdict(list)
    for r in records:
        records_by_group[r["group_id"]].append(r)

    # Stratify groups by whether they contain any FIGHT clips
    fight_groups = []
    normal_only_groups = []
    for gid, recs in records_by_group.items():
        if any(r["label"] == 1 for r in recs):
            fight_groups.append(gid)
        else:
            normal_only_groups.append(gid)

    rng.shuffle(fight_groups)
    rng.shuffle(normal_only_groups)

    def partition(groups):
        n = len(groups)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        return set(groups[:n_tr]), set(groups[n_tr:n_tr + n_va]), set(groups[n_tr + n_va:])

    tr_f, va_f, te_f = partition(fight_groups)
    tr_n, va_n, te_n = partition(normal_only_groups)

    train_groups = tr_f | tr_n
    val_groups = va_f | va_n
    test_groups = te_f | te_n

    train_recs = [r for gid in train_groups for r in records_by_group[gid]]
    val_recs = [r for gid in val_groups for r in records_by_group[gid]]
    test_recs = [r for gid in test_groups for r in records_by_group[gid]]

    return train_recs, val_recs, test_recs


def process_hmdb51():
    """Extract records from extracted HMDB51 classes grouped by movie prefix."""
    hmdb_dir = PROJECT_ROOT / "datasets" / "processed" / "hmdb51_selected"
    classes = {
        # Positives
        "punch": (1, "FIGHT"),
        "kick": (1, "FIGHT"),
        # Hard Negatives
        "stand": (0, "NORMAL"),
        "sit": (0, "NORMAL"),
        "talk": (0, "NORMAL"),
        "walk": (0, "NORMAL"),
        "hug": (0, "NORMAL"),
        "shake_hands": (0, "NORMAL"),
        "kiss": (0, "NORMAL"),
        "wave": (0, "NORMAL"),
        "clap": (0, "NORMAL"),
    }

    records = []
    idx = 0
    for cls_name, (bin_label, lbl_name) in classes.items():
        cls_dir = hmdb_dir / cls_name
        if not cls_dir.exists():
            continue
        for avi in sorted(cls_dir.glob("*.avi")):
            name = avi.name
            target = f"_{cls_name}_"
            if target in name:
                movie_pfx = name.split(target)[0]
            else:
                movie_pfx = name.rsplit("_", 5)[0]

            records.append({
                "clip_id": f"hmdb51_{idx:05d}",
                "video_path": safe_win_path(str(avi)),
                "dataset_source": "HMDB51",
                "source_class": cls_name,
                "label": bin_label,
                "label_name": lbl_name,
                "group_id": f"hmdb_{movie_pfx}",
                "original_split": "none",
                "duration_sec": 3.0,
                "frame_count": 90,
                "fps": 30.0,
            })
            idx += 1

    print(f"Loaded HMDB51: {len(records)} clips across {len(set(r['group_id'] for r in records))} movie groups")
    return records


def build_manifests():
    # 1. Load base datasets
    rwf_recs = load_base_manifest(PROJECT_ROOT / "datasets" / "manifests" / "rwf2000_v1.json", "RWF-2000")
    rlvs_recs = load_base_manifest(PROJECT_ROOT / "datasets" / "manifests" / "rlvs_v1.json", "RLVS")
    hockey_recs = load_base_manifest(PROJECT_ROOT / "datasets" / "manifests" / "hockey_fight_v1.json", "Hockey Fight")
    movies_recs = load_base_manifest(PROJECT_ROOT / "datasets" / "manifests" / "movies_fight_v1.json", "Movies Fight")
    hmdb_recs = process_hmdb51()

    # 2. Partition RWF-2000: train keeps original train; val partition is split 50/50 by group_id into val and test
    rwf_train = [r for r in rwf_recs if r["original_split"] == "train"]
    rwf_val_pool = [r for r in rwf_recs if r["original_split"] == "val"]
    rwf_val_groups = list(set(r["group_id"] for r in rwf_val_pool))
    rng = random.Random(SEED)
    rng.shuffle(rwf_val_groups)
    val_cut = len(rwf_val_groups) // 2
    rwf_val_set = set(rwf_val_groups[:val_cut])
    rwf_test_set = set(rwf_val_groups[val_cut:])

    rwf_val = [r for r in rwf_val_pool if r["group_id"] in rwf_val_set]
    rwf_test = [r for r in rwf_val_pool if r["group_id"] in rwf_test_set]

    # 3. RLVS, Hockey, Movies use their existing clean splits
    rlvs_train = [r for r in rlvs_recs if r["original_split"] == "train"]
    rlvs_val = [r for r in rlvs_recs if r["original_split"] == "val"]
    rlvs_test = [r for r in rlvs_recs if r["original_split"] == "test"]

    hockey_train = [r for r in hockey_recs if r["original_split"] == "train"]
    hockey_val = [r for r in hockey_recs if r["original_split"] == "val"]
    hockey_test = [r for r in hockey_recs if r["original_split"] == "test"]

    movies_train = [r for r in movies_recs if r["original_split"] == "train"]
    movies_val = [r for r in movies_recs if r["original_split"] == "val"]
    movies_test = [r for r in movies_recs if r["original_split"] == "test"]

    # 4. HMDB51 split by movie prefix (group_id)
    hmdb_train, hmdb_val, hmdb_test = split_records_by_group(hmdb_recs, train_ratio=0.70, val_ratio=0.15, seed=SEED)

    # 5. Combine into master train, val, test
    train_all = rwf_train + rlvs_train + hockey_train + movies_train + hmdb_train
    val_all = rwf_val + rlvs_val + hockey_val + movies_val + hmdb_val
    test_all = rwf_test + rlvs_test + hockey_test + movies_test + hmdb_test

    # Tag records with final split
    for r in train_all:
        r["split"] = "train"
    for r in val_all:
        r["split"] = "val"
    for r in test_all:
        r["split"] = "test"

    # Verify ZERO group overlap between train, val, and test
    train_groups = set(r["group_id"] for r in train_all)
    val_groups = set(r["group_id"] for r in val_all)
    test_groups = set(r["group_id"] for r in test_all)

    train_val_overlap = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    assert len(train_val_overlap) == 0, f"LEAKAGE DETECTED between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"LEAKAGE DETECTED between train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"LEAKAGE DETECTED between val and test: {val_test_overlap}"

    print(f"VERIFIED: Zero group leakage across train/val/test splits!")

    # Write CSVs
    fieldnames = ["clip_id", "video_path", "dataset_source", "source_class", "label", "label_name", "group_id", "split"]

    def write_csv(path: Path, recs):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in recs:
                writer.writerow(r)

    train_csv = EXP_DIR / "train_manifest.csv"
    val_csv = EXP_DIR / "val_manifest.csv"
    test_csv = EXP_DIR / "test_manifest.csv"

    write_csv(train_csv, train_all)
    write_csv(val_csv, val_all)
    write_csv(test_csv, test_all)

    # Compute split statistics
    def get_stats(recs):
        lbl_counts = Counter(r["label_name"] for r in recs)
        src_counts = Counter(r["dataset_source"] for r in recs)
        cls_counts = Counter(r["source_class"] for r in recs)
        return {
            "total": len(recs),
            "fight": lbl_counts["FIGHT"],
            "normal": lbl_counts["NORMAL"],
            "fight_ratio": round(lbl_counts["FIGHT"] / max(len(recs), 1), 4),
            "by_source": dict(src_counts),
            "by_class": dict(cls_counts),
        }

    summary = {
        "train": get_stats(train_all),
        "val": get_stats(val_all),
        "test": get_stats(test_all),
        "overall": {
            "total_clips": len(train_all) + len(val_all) + len(test_all),
            "train_percent": round(len(train_all) / (len(train_all) + len(val_all) + len(test_all)) * 100, 2),
            "val_percent": round(len(val_all) / (len(train_all) + len(val_all) + len(test_all)) * 100, 2),
            "test_percent": round(len(test_all) / (len(train_all) + len(val_all) + len(test_all)) * 100, 2),
            "leakage_check": "PASSED - ZERO GROUP OVERLAP",
        }
    }

    summary_json = EXP_DIR / "split_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=================================================================")
    print("PHASE 2B MANIFEST CREATION COMPLETED")
    print("=================================================================")
    print(f"Train: {summary['train']['total']} clips (FIGHT: {summary['train']['fight']}, NORMAL: {summary['train']['normal']})")
    print(f"Val:   {summary['val']['total']} clips (FIGHT: {summary['val']['fight']}, NORMAL: {summary['val']['normal']})")
    print(f"Test:  {summary['test']['total']} clips (FIGHT: {summary['test']['fight']}, NORMAL: {summary['test']['normal']})")
    print(f"Total: {summary['overall']['total_clips']} clips")
    print(f"Saved: {train_csv}")
    print(f"Saved: {val_csv}")
    print(f"Saved: {test_csv}")
    print(f"Saved: {summary_json}")
    print("=================================================================")


if __name__ == "__main__":
    build_manifests()
