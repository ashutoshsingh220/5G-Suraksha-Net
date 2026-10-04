"""Diagnostic analysis script for datasets/videos/external/cctv_test_01.mp4."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from suraksha.config import load_config
from suraksha.detection.tracker import MultiObjectTracker
from suraksha.crowd.analyzer import CrowdAnalyzer
from suraksha.fight.recognizer import FightRecognizer
from suraksha.capture.stream import StreamCapture


def main():
    cfg = load_config()
    cfg.capture.rtsp_url = "datasets/videos/external/cctv_test_01.mp4"
    cfg.capture.pace = "fast"

    cap = StreamCapture(cfg.capture)
    first = True
    tracker = None
    crowd_analyzer = None
    fight_recognizer = None

    total_frames = 0
    frames_with_ge1 = 0
    frames_with_ge2 = 0
    person_counts = []
    unique_tracks = set()
    confidences = []
    active_candidate_pairs = set()
    temporal_scores = []
    verified_fights = []

    t0 = time.time()
    for ev in cap.frames():
        total_frames += 1
        h, w = ev.frame.shape[:2]
        if first:
            tracker = MultiObjectTracker(cfg.detection, cfg.tracking, device="cuda")
            crowd_analyzer = CrowdAnalyzer(cfg.crowd, cfg.camera_id, (h, w))
            fight_recognizer = FightRecognizer(cfg.fight, (h, w), device="cuda")
            first = False

        tracks = tracker.update(ev.frame)
        crowd = crowd_analyzer.update(ev.frame, tracks, ev.timestamp)
        verified = fight_recognizer.update(ev.frame, tracks, ev.timestamp, fps=ev.source_fps)

        n_p = len(tracks)
        person_counts.append(n_p)
        if n_p >= 1:
            frames_with_ge1 += 1
        if n_p >= 2:
            frames_with_ge2 += 1

        for t in tracks:
            unique_tracks.add(t.track_id)
            confidences.append(t.confidence)

        for af in fight_recognizer._active.values():
            active_candidate_pairs.add(af.candidate.track_ids)
            if af.window_scores:
                temporal_scores.append(float(af.window_scores[-1]))

        for vf in verified:
            verified_fights.append({
                "track_ids": list(vf.track_ids),
                "confidence": float(vf.confidence),
                "duration": float(vf.end_time - vf.start_time),
            })

    elapsed = time.time() - t0
    cap.release()

    res = {
        "video": cfg.capture.rtsp_url,
        "total_frames": total_frames,
        "elapsed_sec": round(elapsed, 2),
        "processing_fps": round(total_frames / max(elapsed, 0.001), 2),
        "resolution": f"{w}x{h}",
        "frames_with_ge1_person": frames_with_ge1,
        "pct_frames_with_ge1": round(frames_with_ge1 / max(total_frames, 1) * 100, 2),
        "frames_with_ge2_persons": frames_with_ge2,
        "pct_frames_with_ge2": round(frames_with_ge2 / max(total_frames, 1) * 100, 2),
        "max_simultaneous_persons": max(person_counts) if person_counts else 0,
        "mean_persons_per_frame": round(float(np.mean(person_counts)), 2) if person_counts else 0.0,
        "unique_track_ids_count": len(unique_tracks),
        "mean_detection_confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0,
        "min_detection_confidence": round(float(np.min(confidences)), 3) if confidences else 0.0,
        "max_detection_confidence": round(float(np.max(confidences)), 3) if confidences else 0.0,
        "active_candidate_pairs_count": len(active_candidate_pairs),
        "temporal_windows_scored": len(temporal_scores),
        "temporal_scores_mean": round(float(np.mean(temporal_scores)), 3) if temporal_scores else None,
        "temporal_scores_max": round(float(np.max(temporal_scores)), 3) if temporal_scores else None,
        "verified_fights_count": len(verified_fights),
        "verified_fights": verified_fights,
    }

    out_file = Path("outputs/diagnostics/cctv_test_01_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 60)
    print("CCTV_TEST_01 EMPIRICAL PIPELINE MEASUREMENT")
    print("=" * 60)
    print(f"Total frames processed:          {res['total_frames']} ({res['resolution']})")
    print(f"Processing time / FPS:           {res['elapsed_sec']}s ({res['processing_fps']} FPS on GPU)")
    print(f"Frames with >=1 person:          {res['frames_with_ge1_person']} ({res['pct_frames_with_ge1']}%)")
    print(f"Frames with >=2 people:          {res['frames_with_ge2_persons']} ({res['pct_frames_with_ge2']}%)")
    print(f"Max simultaneous persons:        {res['max_simultaneous_persons']}")
    print(f"Unique track IDs:                {res['unique_track_ids_count']}")
    print(f"Detection confidence (min/mean/max): {res['min_detection_confidence']} / {res['mean_detection_confidence']} / {res['max_detection_confidence']}")
    print(f"Active candidate pairs:          {res['active_candidate_pairs_count']}")
    print(f"Temporal windows scored:         {res['temporal_windows_scored']}")
    print(f"Temporal scores (mean/max):      {res['temporal_scores_mean']} / {res['temporal_scores_max']}")
    print(f"Verified fights detected:        {res['verified_fights_count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
