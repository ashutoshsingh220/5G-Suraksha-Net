"""Stage 1: Reproduce and Diagnose Low-Resolution Detection Bottleneck.

Analyzes the exact clips identified in Phase 3 as stage1_detection failures.
Records per-clip:
- video_id / clip_id
- resolution, source_fps
- ground_truth / proposed label (clip-level only)
- frames sampled
- detected person counts (total, min, max, mean per frame)
- frame counts & percentages with 0, 1, and >=2 detections
- detection confidence statistics (min, median, mean, max) across all person detections (conf >= 0.05)
- person bbox dimensions (min, median, max width, height, area in original pixels)
- tracking behavior (total tracks formed, track length stats)
- candidate pairing possibility (whether >=2 tracks ever satisfy proximity IoU or distance)
- temporal window formation (whether any 32-frame sequence is emitted)
- Failure stage categorization: DETECTION vs TRACKING vs PAIRING vs TEMPORAL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from suraksha.config import PROJECT_ROOT, load_config
from suraksha.detection.tracker import MultiObjectTracker
from suraksha.fight.candidate import FightCandidateDetector
from suraksha.fight.recognizer import FightRecognizer
from suraksha.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


def run_stage1_diagnostics(
    clips: list[dict],
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.50,
    imgsz: int = 640,
    max_frames: int = 90,
    device: str = "cuda",
) -> list[dict]:
    cfg = load_config()
    cfg.detection.conf_threshold = conf_thresh
    cfg.detection.iou_threshold = iou_thresh
    cfg.detection.imgsz = imgsz

    from ultralytics import YOLO

    model = YOLO(cfg.detection.weights)
    tracker = MultiObjectTracker(cfg.detection, cfg.tracking, device=device)

    results = []

    for idx, c in enumerate(clips):
        clip_id = c["clip_id"]
        vpath = ROOT / c["path"]
        source_class = c.get("source_class", "unknown")
        ground_truth = c.get("ground_truth", 1)  # All are fight/combat clips
        manifest = c.get("manifest", "")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            log.error("Failed to open %s", vpath)
            continue

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 30.0

        tracker.reset()
        candidate_detector = FightCandidateDetector(cfg.fight.candidate, (orig_h, orig_w))
        recognizer = FightRecognizer(cfg.fight, (orig_h, orig_w), device=device)

        frame_idx = 0
        frames_sampled = 0
        frames_0_det = 0
        frames_1_det = 0
        frames_ge2_det = 0

        all_confs: list[float] = []
        all_raw_confs_gt05: list[float] = []
        all_widths: list[float] = []
        all_heights: list[float] = []
        all_areas: list[float] = []

        tracks_seen = set()
        candidates_formed = 0
        windows_formed = 0
        windows_scored_total = 0
        verified_fights = 0
        person_counts_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frames_sampled >= max_frames):
                break

            ts = frames_sampled / source_fps

            # 1. Raw detection statistics (to see what YOLO predicts at low confidence >= 0.05)
            # Ultralytics model call with conf=0.05 to capture full distribution
            raw_res = model.predict(
                frame,
                conf=0.05,
                iou=iou_thresh,
                imgsz=imgsz,
                classes=[0],
                verbose=False,
                device=0 if device == "cuda" else "cpu",
            )
            for r in raw_res:
                if r.boxes is not None and len(r.boxes) > 0:
                    c_vals = r.boxes.conf.cpu().numpy().tolist()
                    all_raw_confs_gt05.extend(c_vals)

            # 2. Pipeline tracker update with active production config
            tracks = tracker.update(frame)
            n_persons = len(tracks)
            person_counts_per_frame.append(n_persons)

            if n_persons == 0:
                frames_0_det += 1
            elif n_persons == 1:
                frames_1_det += 1
            else:
                frames_ge2_det += 1

            for t in tracks:
                tracks_seen.add(t.track_id)
                all_confs.append(float(t.confidence))
                bw = float(t.bbox_xyxy[2] - t.bbox_xyxy[0])
                bh = float(t.bbox_xyxy[3] - t.bbox_xyxy[1])
                all_widths.append(bw)
                all_heights.append(bh)
                all_areas.append(bw * bh)

            # 3. Pipeline candidate detection & temporal windows via recognizer
            verified = recognizer.update(frame, tracks, ts, fps=source_fps)
            if verified:
                verified_fights += len(verified)

            for key, af in recognizer._active.items():
                if len(af.candidate.features) >= cfg.fight.temporal.window_frames:
                    windows_formed += 1
                if len(af.window_scores) > 0:
                    windows_scored_total += len(af.window_scores)

            # check candidate count from recognizer.detector
            active_cands = [c for c in recognizer.detector._pairs.values() if c.engaged_frames >= cfg.fight.candidate.proximity_min_frames]
            if active_cands:
                candidates_formed += len(active_cands)

            frames_sampled += 1

        cap.release()

        # Compute summary statistics
        pct_0 = round(100.0 * frames_0_det / max(1, frames_sampled), 2)
        pct_1 = round(100.0 * frames_1_det / max(1, frames_sampled), 2)
        pct_ge2 = round(100.0 * frames_ge2_det / max(1, frames_sampled), 2)

        # Diagnose bottleneck category
        max_p = max(person_counts_per_frame) if person_counts_per_frame else 0
        if max_p < 2:
            failure_category = "DETECTION_FAILURE"
            failure_detail = f"Never detected >=2 persons simultaneously (max={max_p}, frames with >=2: 0%)"
        elif len(tracks_seen) < 2:
            failure_category = "TRACKING_FAILURE"
            failure_detail = f"Fewer than 2 unique tracks formed (tracks={len(tracks_seen)})"
        elif candidates_formed == 0:
            failure_category = "PAIRING_FAILURE"
            failure_detail = f"Tracks never satisfied proximity IoU/distance or motion energy"
        elif windows_formed == 0:
            failure_category = "TRACKING_CONTINUITY_FAILURE"
            failure_detail = f"Candidate formed but tracks fragmented before 32-frame temporal window"
        else:
            failure_category = "TEMPORAL_CLASSIFICATION_FAILURE"
            failure_detail = f"Temporal window formed ({windows_formed} windows) but classifier failed verification"

        rec = {
            "clip_id": clip_id,
            "manifest": manifest,
            "source_class": source_class,
            "ground_truth_label": "FIGHT" if ground_truth == 1 else "NON_FIGHT",
            "label_scope": "clip-level only (no frame-level bounding-box ground truth available)",
            "resolution": f"{orig_w}x{orig_h}",
            "source_fps": round(source_fps, 2),
            "frames_sampled": frames_sampled,
            "unique_tracks_count": len(tracks_seen),
            "max_simultaneous_persons": max_p,
            "mean_persons_per_frame": round(float(np.mean(person_counts_per_frame)), 3) if person_counts_per_frame else 0.0,
            "frames_zero_detections": frames_0_det,
            "frames_one_detection": frames_1_det,
            "frames_ge2_detections": frames_ge2_det,
            "pct_frames_zero_detections": pct_0,
            "pct_frames_one_detection": pct_1,
            "pct_frames_ge2_detections": pct_ge2,
            "active_conf_threshold": conf_thresh,
            "active_imgsz": imgsz,
            "confidence_stats": {
                "count": len(all_confs),
                "min": round(float(np.min(all_confs)), 3) if all_confs else None,
                "median": round(float(np.median(all_confs)), 3) if all_confs else None,
                "mean": round(float(np.mean(all_confs)), 3) if all_confs else None,
                "max": round(float(np.max(all_confs)), 3) if all_confs else None,
            },
            "raw_confs_gt05_stats": {
                "count": len(all_raw_confs_gt05),
                "min": round(float(np.min(all_raw_confs_gt05)), 3) if all_raw_confs_gt05 else None,
                "median": round(float(np.median(all_raw_confs_gt05)), 3) if all_raw_confs_gt05 else None,
                "mean": round(float(np.mean(all_raw_confs_gt05)), 3) if all_raw_confs_gt05 else None,
                "max": round(float(np.max(all_raw_confs_gt05)), 3) if all_raw_confs_gt05 else None,
                "count_between_010_and_035": sum(1 for c in all_raw_confs_gt05 if 0.10 <= c < conf_thresh),
            },
            "bbox_stats": {
                "min_width_px": round(float(np.min(all_widths)), 1) if all_widths else None,
                "median_width_px": round(float(np.median(all_widths)), 1) if all_widths else None,
                "max_width_px": round(float(np.max(all_widths)), 1) if all_widths else None,
                "min_height_px": round(float(np.min(all_heights)), 1) if all_heights else None,
                "median_height_px": round(float(np.median(all_heights)), 1) if all_heights else None,
                "max_height_px": round(float(np.max(all_heights)), 1) if all_heights else None,
                "min_area_px": round(float(np.min(all_areas)), 1) if all_areas else None,
                "median_area_px": round(float(np.median(all_areas)), 1) if all_areas else None,
                "max_area_px": round(float(np.max(all_areas)), 1) if all_areas else None,
            },
            "candidate_pairing_possible": candidates_formed > 0,
            "candidates_formed_count": candidates_formed,
            "temporal_model_receives_sequence": windows_formed > 0,
            "temporal_windows_count": windows_formed,
            "failure_category": failure_category,
            "failure_detail": failure_detail,
        }
        results.append(rec)
        print(f"[{idx+1:2d}/{len(clips)}] {clip_id} ({source_class}) {rec['resolution']} | >=2 frames: {pct_ge2}% | raw_box>=0.10: {rec['raw_confs_gt05_stats']['count_between_010_and_035']} | Category: {failure_category}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", default="datasets/reports/stage1_low_res_diagnostics.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=90)
    args = ap.parse_args()
    setup_logging("INFO")

    # Load the 30 stage1 failure clips from confusion_analysis_v2.json
    with open("datasets/reports/confusion_analysis_v2.json") as f:
        conf_data = json.load(f)

    stage1_items = [x for x in conf_data["false_negatives"] if x.get("failure_stage") == "stage1_detection"]

    # Manifest map to resolve video paths
    manifest_map = {}
    for m_name in ["ucf_crime_subset_eval_v1", "rwf2000_v1", "rlvs_v1"]:
        with open(f"datasets/manifests/{m_name}.json") as f:
            m = json.load(f)
            for e in m["entries"]:
                manifest_map[e["clip_id"]] = e

    clips_to_diagnose = []
    for item in stage1_items:
        cid = item["clip_id"]
        if cid in manifest_map:
            e = manifest_map[cid]
            clips_to_diagnose.append({
                "clip_id": cid,
                "path": e["path"],
                "source_class": item["source_class"],
                "ground_truth": item["ground_truth"],
                "manifest": item["manifest"],
            })

    print(f"\n========================================================")
    print(f"STAGE 1: RUNNING DIAGNOSTICS ON {len(clips_to_diagnose)} LOW-RES CLIPS")
    print(f"========================================================\n")

    t0 = time.time()
    results = run_stage1_diagnostics(
        clips_to_diagnose,
        conf_thresh=0.35,
        iou_thresh=0.50,
        imgsz=640,
        max_frames=args.max_frames,
        device=args.device,
    )
    elapsed = time.time() - t0

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "STAGE 1 - Low-Resolution Detection Diagnostics",
            "date": "2026-09-30",
            "hardware": "RTX 4050 Laptop GPU",
            "clips_evaluated": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "results": results,
        }, f, indent=2)

    print(f"\nDiagnostic results saved to {out_p} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
