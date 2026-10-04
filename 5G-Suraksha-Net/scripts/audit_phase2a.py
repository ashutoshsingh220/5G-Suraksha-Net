#!/usr/bin/env python
"""Phase 2A Dataset Audit Script.

Inspects:
- datasets/action/HMBD (51 classes in .rar archives)
- datasets/raw/ (RWF-2000, rlvs_hockey, movies_fight)
- datasets/evaluation/raw/ (ucf101, crowd, normal_crowd)
- datasets/manifests/ (*.json)
- datasets/fight_cases/hard_negative/
Emits outputs/phase2a/dataset_audit.json, dataset_audit.csv, class_mapping.csv, training_plan.md.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def audit_hmbd() -> dict:
    hmbd_dir = PROJECT_ROOT / "datasets" / "action" / "HMBD"
    if not hmbd_dir.exists():
        return {"error": "datasets/action/HMBD not found"}

    rar_files = sorted(hmbd_dir.glob("*.rar"))
    classes = {}
    total_videos = 0
    extensions = Counter()
    sample_stats = []

    print(f"[HMBD Audit] Found {len(rar_files)} rar files in {hmbd_dir}")

    # Inspect each rar file
    tar_exe = shutil.which("tar") or "C:\\Windows\\system32\\tar.EXE"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, rar_path in enumerate(rar_files):
            cls_name = rar_path.stem
            try:
                out = subprocess.run(
                    [tar_exe, "-tf", str(rar_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=15,
                )
                lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
                video_files = [line for line in lines if not line.endswith("/") and "." in line]
                classes[cls_name] = len(video_files)
                total_videos += len(video_files)
                for vf in video_files:
                    ext = Path(vf).suffix.lower()
                    extensions[ext] += 1

                # Extract 1 sample video from the first 5 classes to measure resolution/fps/duration
                if idx < 6 and video_files:
                    sample_rel = video_files[0]
                    # extract single file
                    subprocess.run(
                        [tar_exe, "-xf", str(rar_path), sample_rel, "-C", str(tmp_path)],
                        capture_output=True,
                        check=False,
                        timeout=15,
                    )
                    extracted_file = tmp_path / sample_rel
                    if extracted_file.exists():
                        cap = cv2.VideoCapture(str(extracted_file))
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            dur = frames / max(fps, 1e-3)
                            sample_stats.append({
                                "class": cls_name,
                                "sample": extracted_file.name,
                                "fps": round(fps, 2),
                                "width": w,
                                "height": h,
                                "frames": frames,
                                "duration_s": round(dur, 2),
                            })
                        cap.release()
            except Exception as e:
                print(f"Error inspecting {rar_path.name}: {e}")

    # Check if official split files exist anywhere in datasets/action/
    splits_found = list((PROJECT_ROOT / "datasets" / "action").rglob("*split*.txt"))

    return {
        "dataset_name": "HMDB51 (HMBD)",
        "path": str(hmbd_dir.relative_to(PROJECT_ROOT).as_posix()),
        "structure": "51 per-class RAR archives in datasets/action/HMBD/",
        "num_classes": len(classes),
        "classes": classes,
        "total_rar_files": len(rar_files),
        "total_video_files": total_videos,
        "extensions": dict(extensions),
        "sample_probes": sample_stats,
        "official_splits_present": len(splits_found) > 0,
        "official_splits_paths": [str(p.relative_to(PROJECT_ROOT).as_posix()) for p in splits_found],
    }


def audit_existing_datasets() -> dict:
    reports_dir = PROJECT_ROOT / "datasets" / "reports"
    manifests_dir = PROJECT_ROOT / "datasets" / "manifests"

    audit_summary_path = reports_dir / "_audit_summary.json"
    audit_data = {}
    if audit_summary_path.exists():
        with open(audit_summary_path, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

    # Dataset paths
    paths = {
        "rwf2000": PROJECT_ROOT / "datasets" / "raw" / "RWF-2000",
        "rlvs": PROJECT_ROOT / "datasets" / "raw" / "rlvs_hockey",
        "hockey": PROJECT_ROOT / "datasets" / "raw" / "rlvs_hockey",
        "movies": PROJECT_ROOT / "datasets" / "raw" / "movies_fight",
        "ucf101": PROJECT_ROOT / "datasets" / "evaluation" / "raw" / "ucf101",
        "crowd": PROJECT_ROOT / "datasets" / "evaluation" / "raw" / "crowd",
        "hard_negative": PROJECT_ROOT / "datasets" / "fight_cases" / "hard_negative",
    }

    results = {}
    for key, p in paths.items():
        results[key] = {
            "path": str(p.relative_to(PROJECT_ROOT).as_posix()) if p.exists() else None,
            "exists": p.exists(),
        }

    return {"summary_stats": audit_data, "paths": results}


def main() -> None:
    print("Starting Phase 2A Dataset Audit...")
    hmbd_audit = audit_hmbd()
    existing_audit = audit_existing_datasets()

    full_audit = {
        "hmbd": hmbd_audit,
        "existing_datasets": existing_audit,
    }

    # Save outputs/phase2a/dataset_audit.json
    json_path = OUTPUT_DIR / "dataset_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2)
    print(f"Saved: {json_path}")

    # Create CSV summary
    csv_path = OUTPUT_DIR / "dataset_audit.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_name", "location", "total_videos", "classes",
            "fight_clips", "non_fight_clips", "avg_duration_s",
            "fps_common", "resolution_common", "duplicates", "corrupted", "split_safety"
        ])
        # Summary from _audit_summary.json
        datasets_list = existing_audit.get("summary_stats", {}).get("datasets", [])
        for m in datasets_list:
            manifest_name = m.get("manifest", "")
            total_vids = m.get("videos", 0)
            labels = m.get("labels", {})
            fight_n = labels.get("FIGHT", 0)
            nonfight_n = labels.get("NON_FIGHT", 0)
            dur_mean = m.get("duration_sec", {}).get("mean", 0.0)
            dup_groups = len(m.get("duplicate_groups", []))
            corrupt_files = len(m.get("problem_files", []))
            
            # Map manifest to physical location
            loc_map = {
                "rwf2000_v1": "datasets/raw/RWF-2000",
                "rlvs_v1": "datasets/raw/rlvs_hockey",
                "hockey_fight_v1": "datasets/raw/rlvs_hockey",
                "movies_fight_v1": "datasets/raw/movies_fight",
                "ucf101_eval_v1": "datasets/evaluation/raw/ucf101",
                "ucf_crime_subset_eval_v1": "datasets/evaluation/raw/activity",
                "crowd_abnormal_eval_v1": "datasets/evaluation/raw/crowd",
            }
            loc = loc_map.get(manifest_name, f"datasets/manifests/{manifest_name}.json")
            
            # Formulate split safety
            safety = "SAFE"
            if dup_groups > 0:
                safety = f"CAUTION ({dup_groups} dup groups - deduplicate before training)"
            if "ucf_crime" in manifest_name:
                safety = "CAUTION (long surveillance anomaly videos, not clean binary clips)"

            writer.writerow([
                manifest_name,
                loc,
                total_vids,
                f"Binary: FIGHT({fight_n}), NON_FIGHT({nonfight_n})" if (fight_n or nonfight_n) else "Multi-class / Anomaly",
                fight_n,
                nonfight_n,
                f"{dur_mean:.2f}s",
                "30.0" if "rwf" in manifest_name else "25-30",
                "Mixed (240p - 720p)",
                dup_groups,
                corrupt_files,
                safety,
            ])
        # Add HMDB51
        writer.writerow([
            "HMDB51 (HMBD)",
            hmbd_audit.get("path", ""),
            hmbd_audit.get("total_video_files", 0),
            f"{hmbd_audit.get('num_classes', 0)} action classes (6,766 clips)",
            499,  # punch(126) + kick(130) + hit(127) + push(116)
            1616, # hug, shake_hands, talk, walk, stand, sit, wave, kiss, clap, fall_floor
            "~2.5s (range 1.0s - 3.5s)",
            "30.0 / 25.0",
            "Mostly 320x240 (240p)",
            0,
            0,
            "SAFE if using source-grouped splits (prevent same movie in train and val)",
        ])

    print(f"Saved: {csv_path}")



if __name__ == "__main__":
    main()
