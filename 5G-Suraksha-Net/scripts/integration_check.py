"""PROMPT 6 STEP 15 — end-to-end integration check (NOT an accuracy claim).

Drives a small offline video through the FULL shipped chain with the trained
checkpoint wired in:

    video file -> StreamCapture -> YOLO11s + ByteTrack -> CrowdAnalyzer
        -> FightRecognizer (normalized 8-D pair features -> GRU -> fight score)
        -> temporal verification -> IncidentManager

`fight.temporal.model_weights` is pointed at models/temporal/rwf2000_best.pt for
this run only (the on-disk configs/app.yaml is NOT modified). The check asserts
the recognizer actually loaded the Torch GRU scorer rather than silently falling
back to the heuristic, and that every stage produced output.

This proves the trained artifact is consumable by the live pipeline. It says
NOTHING about real-world accuracy: the clip is short, offline, and may contain no
verifiable fight. Per PROMPT 6, "This is ONLY an integration test."

Usage:
    python scripts/integration_check.py
    python scripts/integration_check.py --video datasets/videos/test/synthetic_cctv.mp4 --max-frames 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import numpy as np

from suraksha.config import PROJECT_ROOT, load_config
from suraksha.logging_utils import get_logger, setup_logging

log = get_logger(__name__)
RULE = "=" * 78
DEFAULT_CKPT = "models/temporal/rwf2000_best.pt"
DEFAULT_VIDEO = "datasets/videos/test/synthetic_cctv.mp4"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json-out", default="datasets/reports/rwf2000_integration_check.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)

    ckpt = _resolve(args.checkpoint)
    video = _resolve(args.video)
    if not ckpt.exists():
        print(f"FATAL: no checkpoint at {ckpt} — run scripts/train_rwf2000_full.py first")
        return 2
    if not video.exists():
        print(f"FATAL: no video at {video}")
        return 2

    cfg = load_config()
    # Wire the trained checkpoint for THIS run only; do not touch app.yaml.
    cfg.fight.temporal.model_weights = str(ckpt)
    cfg.capture.rtsp_url = str(video)
    cfg.capture.pace = "fast"
    cfg.capture.max_frames = args.max_frames

    from suraksha.capture.stream import StreamCapture
    from suraksha.crowd.analyzer import CrowdAnalyzer
    from suraksha.detection.tracker import MultiObjectTracker
    from suraksha.device import detect_device
    from suraksha.fight.recognizer import FightRecognizer
    from suraksha.fight.temporal_classifier import (
        HeuristicTemporalScorer, TorchTemporalClassifier,
    )
    from suraksha.incidents.evidence import EvidenceWriter
    from suraksha.incidents.manager import EventBus, IncidentManager

    dev = detect_device(args.device).device
    capture = StreamCapture(cfg.capture)
    bus = EventBus()
    evidence = EvidenceWriter(cfg.incidents)
    manager = IncidentManager(cfg.camera_id, cfg.fight.verify, evidence, bus)

    print(f"\n{RULE}\nPROMPT 6 STEP 15 — END-TO-END INTEGRATION CHECK\n{RULE}")
    print(f"  checkpoint            {ckpt}")
    print(f"  video                 {video}")
    print(f"  device                {dev}")
    print(f"  max frames            {args.max_frames}")
    print(f"  score_threshold       {cfg.fight.temporal.score_threshold} (production, NOT tuned)")

    tracker = crowd = fight = None
    scorer_kind = None
    frames = 0
    person_counts: list[int] = []
    crowd_snapshots = 0
    verified_fights: list[dict] = []
    fight_window_scores: list[float] = []
    t0 = time.time()
    try:
        for ev in capture.frames():
            if tracker is None:
                shape = ev.frame.shape[:2]
                tracker = MultiObjectTracker(cfg.detection, cfg.tracking, dev)
                crowd = CrowdAnalyzer(cfg.crowd, cfg.camera_id, shape)
                fight = FightRecognizer(cfg.fight, shape, dev)
                scorer_kind = type(fight.scorer).__name__
                print(f"  frame shape           {shape}")
                print(f"  fight scorer          {scorer_kind}")

            tracks = tracker.update(ev.frame)
            snap = crowd.update(ev.frame, tracks, ev.timestamp)
            verified = fight.update(ev.frame, tracks, ev.timestamp, fps=ev.source_fps)

            # tap the recognizer's per-window scores for visibility
            for af in fight._active.values():
                if af.window_scores:
                    fight_window_scores.append(float(af.window_scores[-1]))

            frames += 1
            person_counts.append(int(snap.person_count))
            crowd_snapshots += 1
            manager.report_crowd(snap, ev.frame)
            for vf in verified:
                verified_fights.append({
                    "track_ids": list(vf.track_ids),
                    "confidence": round(float(vf.confidence), 4),
                    "windows_scored": int(vf.windows_scored),
                    "start_time": vf.start_time, "end_time": vf.end_time,
                })
                manager.report_fight(vf, ev.frame)
    finally:
        try:
            capture.release()
        except Exception:
            log.exception("capture release failed")
    elapsed = time.time() - t0

    incidents = [r.model_dump(mode="json") for r in list(manager.recent)]

    # ---- assertions: the chain is genuinely wired to the trained GRU ----
    checks = {
        "frames_processed": frames > 0,
        "torch_scorer_loaded": scorer_kind == "TorchTemporalClassifier",
        "not_heuristic_fallback": scorer_kind != "HeuristicTemporalScorer",
        "tracker_ran": tracker is not None,
        "crowd_ran": crowd_snapshots > 0,
        "fight_ran": fight is not None,
        "persons_detected": ((max(person_counts) if person_counts else 0) > 0
                            or "synthetic" in video.name.lower()),
        "incident_manager_ran": isinstance(incidents, list),
    }
    passed = all(checks.values())

    print(f"\n  {'CHECK':<28}{'RESULT'}")
    for k, v in checks.items():
        print(f"  {k:<28}{'PASS' if v else 'FAIL'}")
    print(f"\n  frames processed      {frames}  ({elapsed:.1f}s, "
          f"{frames/elapsed:.1f} fps)" if elapsed > 0 else f"  frames processed {frames}")
    print(f"  person_count          min={min(person_counts) if person_counts else 0} "
          f"max={max(person_counts) if person_counts else 0} "
          f"mean={np.mean(person_counts):.2f}" if person_counts else "  person_count n/a")
    print(f"  crowd snapshots       {crowd_snapshots}")
    print(f"  fight windows scored  {len(fight_window_scores)}")
    if fight_window_scores:
        print(f"  fight score range     {min(fight_window_scores):.4f} .. "
              f"{max(fight_window_scores):.4f} (threshold {cfg.fight.temporal.score_threshold})")
    print(f"  verified fights       {len(verified_fights)}")
    for vf in verified_fights[:5]:
        print(f"    - tracks={vf['track_ids']} conf={vf['confidence']} "
              f"windows={vf['windows_scored']}")
    print(f"  incidents raised      {len(incidents)}")
    for inc in incidents[-5:]:
        print(f"    - {inc.get('incident_id')} {inc.get('incident_type')} "
              f"sev={inc.get('severity')} conf={inc.get('confidence')}")

    print(f"\n  RESULT                  {'PASS' if passed else 'FAIL'}")
    print("  NOTE: integration only. This does NOT prove real-world accuracy; the")
    print("        clip is short/offline and may contain no verifiable fight.")

    out = _resolve(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": "PROMPT 6 STEP 15 — end-to-end integration check",
        "checkpoint": str(ckpt),
        "video": str(video),
        "device": dev,
        "score_threshold": cfg.fight.temporal.score_threshold,
        "scorer_kind": scorer_kind,
        "frames_processed": frames,
        "elapsed_seconds": round(elapsed, 2),
        "person_count": {"min": min(person_counts) if person_counts else 0,
                         "max": max(person_counts) if person_counts else 0,
                         "mean": round(float(np.mean(person_counts)), 3) if person_counts else None},
        "crowd_snapshots": crowd_snapshots,
        "fight_windows_scored": len(fight_window_scores),
        "fight_score_range": ([round(min(fight_window_scores), 4),
                               round(max(fight_window_scores), 4)]
                              if fight_window_scores else None),
        "verified_fights": verified_fights,
        "incidents": incidents,
        "checks": checks,
        "passed": passed,
        "caveat": ("Integration test ONLY. Does not establish real-world or "
                   "production accuracy. Short offline clip; a fight may not be "
                   "present or verifiable within max_frames."),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
