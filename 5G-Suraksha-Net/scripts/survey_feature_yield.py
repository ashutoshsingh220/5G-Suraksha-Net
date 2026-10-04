"""Measure what the GRU's 8-dim pair features actually yield on RWF-2000.

WHY THIS EXISTS
---------------
`datasets/reports/_audit_summary.md` predicts 15 windows per RWF-2000 clip
(30 000 total) from *frame counts*. That prediction assumes one window per
32-frame slice of the video. The GRU does not consume video slices — it consumes
per-frame features of an ENGAGED PERSON PAIR, so a window exists only where two
tracked people overlap by at least `candidate.proximity_iou` for long enough.

The two numbers are unrelated, and only measurement tells you the real one. This
script extracts and caches features for a deterministic sample and reports:

  * windows per video, and how many videos yield ZERO windows
  * the FIGHT vs NON_FIGHT window ratio (video-level balance does NOT imply
    window-level balance — pair engagement is itself correlated with fighting)
  * how often `flow >= candidate.motion_energy_threshold` (the inference gate)
  * detection coverage: frames with 0 / 1 / >=2 tracked persons
  * wall-clock per video -> a full-corpus extraction estimate

Features are cached to `datasets/processed/features/<manifest>/` as compressed
.npz (~1 KiB per window). No frame images are ever written. Re-running skips
videos already cached, so a survey and a full extraction share the work.

Usage:
    python scripts/survey_feature_yield.py --per-cell 20
    python scripts/survey_feature_yield.py --per-cell 20 --cells train:FIGHT val:NON_FIGHT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from suraksha.config import load_config
from suraksha.data.manifest import load_manifest
from suraksha.detection.tracker import MultiObjectTracker
from suraksha.fight.candidate import FightCandidateDetector
from suraksha.logging_utils import setup_logging
from suraksha.training.features import (
    PairFeatureExtractor, load_features, save_features,
)

DEFAULT_CACHE = ROOT / "datasets" / "processed" / "features"


def cache_path(cache_root: Path, manifest: str, clip_id: str) -> Path:
    return cache_root / manifest / f"{clip_id}.npz"


def select(entries, per_cell: int, cells: list[tuple[str, str]]) -> list:
    """Deterministic sample: first `per_cell` clip_ids of each (split, label)."""
    picked = []
    for split, label in cells:
        pool = sorted((e for e in entries if e.split == split and e.label == label),
                      key=lambda e: e.clip_id)
        picked.extend(pool[:per_cell])
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="rwf2000_v1")
    ap.add_argument("--per-cell", type=int, default=20,
                    help="videos per (split,label) cell")
    ap.add_argument("--cells", nargs="*", default=["train:FIGHT", "train:NON_FIGHT",
                                                   "val:FIGHT", "val:NON_FIGHT"],
                    help="split:label cells to sample")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--frame-width", type=int, default=640)
    ap.add_argument("--frame-height", type=int, default=360)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--no-cache-reuse", action="store_true")
    args = ap.parse_args()

    setup_logging("WARNING")
    cfg = load_config()
    m = load_manifest(args.manifest)
    cells = [tuple(c.split(":")) for c in args.cells]
    sample = select(m.entries, args.per_cell, cells)
    if not sample:
        print("no entries matched the requested cells")
        return 1

    cache_root = Path(args.cache_dir)
    detector = FightCandidateDetector(cfg.fight.candidate,
                                      (args.frame_height, args.frame_width))
    extractor = None
    tracker = None

    rows = []
    todo = []
    for e in sample:
        cp = cache_path(cache_root, args.manifest, e.clip_id)
        if cp.exists() and not args.no_cache_reuse:
            rows.append((e, load_features(cp), 0.0, True))
        else:
            todo.append((e, cp))

    if todo:
        print(f"extracting {len(todo)} videos ({len(rows)} reused from cache) ...")
        tracker = MultiObjectTracker(cfg.detection, cfg.tracking, device=args.device)
        extractor = PairFeatureExtractor(
            tracker, detector,
            window_frames=cfg.fight.temporal.window_frames,
            stride_frames=cfg.fight.temporal.stride_frames,
            motion_energy_threshold=cfg.fight.candidate.motion_energy_threshold,
        )
    for e, cp in todo:
        t0 = time.time()
        vf = extractor.extract(e.clip_id, ROOT / e.path, e.label, e.split,
                               e.fps or 0.0, e.effective_frames or 0)
        dt = time.time() - t0
        vf.path = e.path
        save_features(vf, cp)
        rows.append((e, vf, dt, False))

    # ---- aggregate ----
    per_cell = defaultdict(lambda: dict(videos=0, windows=0, gate=0, zero=0,
                                        seconds=0.0, notes=Counter(),
                                        tracks=Counter(), flows=[]))
    for e, vf, dt, cached in rows:
        c = per_cell[(vf.split, vf.label)]
        c["videos"] += 1
        c["windows"] += vf.n_windows
        c["gate"] += int(vf.gate_passed.sum()) if vf.n_windows else 0
        c["zero"] += 1 if vf.n_windows == 0 else 0
        c["seconds"] += dt
        c["notes"][vf.note or "ok"] += 1
        if len(vf.tracks_per_frame):
            tp = vf.tracks_per_frame
            c["tracks"]["frames"] += int(len(tp))
            c["tracks"]["zero_person"] += int((tp == 0).sum())
            c["tracks"]["one_person"] += int((tp == 1).sum())
            c["tracks"]["two_plus"] += int((tp >= 2).sum())
        if vf.n_windows:
            c["flows"].append(vf.mean_flow)

    W = cfg.fight.temporal.window_frames
    S = cfg.fight.temporal.stride_frames
    thr = cfg.fight.candidate.motion_energy_threshold
    iou = cfg.fight.candidate.proximity_iou

    print("\n" + "=" * 96)
    print(f"FEATURE YIELD SURVEY — {args.manifest}   window={W} stride={S} "
          f"proximity_iou={iou} motion_energy_threshold={thr}")
    print(f"detector frame shape = ({args.frame_height}, {args.frame_width})   "
          f"sample = {len(rows)} videos")
    print("=" * 96)
    hdr = (f"{'cell':22s} {'vids':>5s} {'windows':>8s} {'win/vid':>8s} {'zero-yield':>11s} "
           f"{'gate>=thr':>10s} {'flow mean':>10s} {'s/vid':>7s}")
    print(hdr)
    print("-" * 96)
    tot = dict(videos=0, windows=0, gate=0, zero=0, seconds=0.0)
    for key in sorted(per_cell):
        c = per_cell[key]
        fl = np.concatenate(c["flows"]) if c["flows"] else np.array([0.0])
        print(f"{key[0] + '/' + key[1]:22s} {c['videos']:5d} {c['windows']:8d} "
              f"{c['windows'] / c['videos']:8.2f} "
              f"{c['zero']:5d} ({100 * c['zero'] / c['videos']:4.1f}%) "
              f"{c['gate']:6d} ({100 * c['gate'] / max(c['windows'], 1):4.1f}%) "
              f"{fl.mean():10.3f} {c['seconds'] / c['videos']:7.2f}")
        for k in tot:
            tot[k] += c[k]

    print("-" * 96)
    f_by = defaultdict(int)
    for key, c in per_cell.items():
        f_by[key[1]] += c["windows"]
    ratio = (f_by["FIGHT"] / f_by["NON_FIGHT"]) if f_by.get("NON_FIGHT") else float("inf")
    print(f"TOTAL videos={tot['videos']}  windows={tot['windows']}  "
          f"zero-yield={tot['zero']} ({100 * tot['zero'] / tot['videos']:.1f}%)  "
          f"gate passed={tot['gate']} ({100 * tot['gate'] / max(tot['windows'], 1):.2f}%)")
    print(f"WINDOW-LEVEL CLASS RATIO  FIGHT {f_by['FIGHT']} : NON_FIGHT "
          f"{f_by['NON_FIGHT']}  = {ratio:.2f} : 1")
    print("   (video-level is 1000:1000 — engagement gating makes windows imbalanced)")

    tr = Counter()
    for c in per_cell.values():
        tr.update(c["tracks"])
    if tr["frames"]:
        print(f"DETECTION COVERAGE  frames={tr['frames']}  0 persons="
              f"{100 * tr['zero_person'] / tr['frames']:.1f}%  1 person="
              f"{100 * tr['one_person'] / tr['frames']:.1f}%  >=2 persons="
              f"{100 * tr['two_plus'] / tr['frames']:.1f}%")
        print("   (a window needs >=2 overlapping tracks, so 0/1-person frames "
              "can never produce one)")

    extracted = [dt for _, _, dt, cached in rows if not cached]
    if extracted:
        per_video = float(np.mean(extracted))
        print(f"\nTIMING  {per_video:.2f} s/video measured on {len(extracted)} videos")
        print(f"   full RWF-2000 (2000 videos) extraction estimate: "
              f"{per_video * 2000 / 60:.1f} min single-process")
        mean_frames = float(np.mean([e.effective_frames or 150 for e, _, _, c in rows if not c]))
        print(f"   -> {per_video * 1000 / max(mean_frames, 1):.1f} ms per frame "
              f"(YOLO+ByteTrack+flow at {args.frame_width}x{args.frame_height})")

    if tot["windows"]:
        need = 2000 * (tot["windows"] / tot["videos"])
        print(f"\nPROJECTED CORPUS  {tot['windows'] / tot['videos']:.2f} windows/video "
              f"-> ~{int(need)} windows for all 2000 RWF-2000 videos "
              f"(audit predicted 30 000 from frame counts)")

    if args.json_out:
        out = {
            "manifest": args.manifest, "window_frames": W, "stride_frames": S,
            "proximity_iou": iou, "motion_energy_threshold": thr,
            "frame_shape": [args.frame_height, args.frame_width],
            "sample_videos": len(rows),
            "per_cell": {f"{k[0]}/{k[1]}": {kk: (vv if not isinstance(vv, Counter) else dict(vv))
                                            for kk, vv in v.items() if kk != "flows"}
                         for k, v in per_cell.items()},
            "windows_by_label": dict(f_by), "window_class_ratio": ratio,
            "detection_coverage": dict(tr), "totals": {k: v for k, v in tot.items()},
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
