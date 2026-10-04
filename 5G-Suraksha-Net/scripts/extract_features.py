"""Build the 8-dim pair-feature cache for RWF-2000 (no training happens here).

The GRU consumes per-frame person-pair kinematics, not pixels, so the cache is
produced by running the SAME tracker + candidate detector the live pipeline uses
over each clip. Windows are stored one compressed .npz per clip under
datasets/processed/features/<manifest>/ — about 1 KiB per 32-frame window, and
never a single frame image on disk.

This is the expensive stage: ~4.5 s per 150-frame clip at 640x360, so all 2000
RWF-2000 clips take roughly 150 minutes single-process. Already-cached clips are
skipped, so the run is resumable and shares work with survey_feature_yield.py.

Examples:
    # deterministic 80-clip sample (20 per split/label cell)
    python scripts/extract_features.py --per-cell 20

    # the full official train+val split (long; ~150 min)
    python scripts/extract_features.py --all

    # specific clips, re-extracted even if cached
    python scripts/extract_features.py --clip-ids rwf2000_00000 rwf2000_01601 --force
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from suraksha.config import PROJECT_ROOT, load_config, load_training_config
from suraksha.data.manifest import load_manifest
from suraksha.logging_utils import setup_logging
from suraksha.training.extract import build_cache, default_cells, select_cells


def pick_entries(entries, args, tcfg):
    if args.clip_ids:
        wanted = set(args.clip_ids)
        picked = [e for e in entries if e.clip_id in wanted]
        missing = wanted - {e.clip_id for e in picked}
        if missing:
            print(f"warning: {len(missing)} clip id(s) not in manifest: "
                  f"{sorted(missing)[:5]}")
        return picked

    labels = args.labels or list(tcfg.dataset.labels)
    splits = args.splits or [tcfg.dataset.train_split, tcfg.dataset.val_split]
    cells = [(s, l) for s in splits for l in labels]

    if args.all:
        per_cell = 10 ** 9
    elif args.per_cell:
        per_cell = args.per_cell
    else:
        per_cell = tcfg.dataset.sample.survey_videos_per_cell

    picked = select_cells(entries, per_cell, cells)
    if args.limit:
        picked = picked[: args.limit]
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="", help="path to configs/training.yaml")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--all", action="store_true",
                    help="extract every clip in the selected splits (long)")
    ap.add_argument("--per-cell", type=int, default=0,
                    help="clips per (split,label) cell")
    ap.add_argument("--limit", type=int, default=0, help="hard cap on clip count")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--clip-ids", nargs="*", default=None)
    ap.add_argument("--device", default="")
    ap.add_argument("--frame-width", type=int, default=0)
    ap.add_argument("--frame-height", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-extract cached clips")
    ap.add_argument("--shard-index", type=int, default=0, help="0-based shard index for parallel workers")
    ap.add_argument("--num-shards", type=int, default=1, help="total number of parallel shards")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    tcfg = load_training_config(args.config or None)
    cfg = load_config()

    manifest = args.manifest or tcfg.dataset.manifest
    cache_dir = Path(args.cache_dir or tcfg.dataset.feature_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    device = args.device or tcfg.dataset.extraction.device
    fw = args.frame_width or tcfg.dataset.extraction.frame_width
    fh = args.frame_height or tcfg.dataset.extraction.frame_height

    entries = load_manifest(manifest).entries
    picked = pick_entries(entries, args, tcfg)
    if not picked:
        print("no manifest entries matched the selection")
        return 1

    trainable = set(tcfg.dataset.labels)
    rejected = [e for e in picked if e.label not in trainable]
    picked = [e for e in picked if e.label in trainable]
    if rejected:
        print(f"skipping {len(rejected)} clip(s) with non-trainable labels: "
              f"{Counter(e.label for e in rejected)}")

    if args.num_shards > 1:
        picked = [e for i, e in enumerate(picked) if i % args.num_shards == args.shard_index]
        print(f"sharding: shard {args.shard_index}/{args.num_shards} -> {len(picked)} clips")

    print(f"manifest={manifest}  clips={len(picked)}  cache={cache_dir}")
    print(f"frame shape=({fh},{fw})  device={device}  "
          f"window={cfg.fight.temporal.window_frames}  stride={cfg.fight.temporal.stride_frames}")

    report = build_cache(
        picked, cache_dir,
        manifest=manifest, project_root=PROJECT_ROOT, app_config=cfg,
        frame_shape=(fh, fw), device=device, force=args.force,
        max_frames=args.max_frames or tcfg.dataset.extraction.max_frames_per_clip,
    )

    s = report.summary()
    print("\n" + "=" * 78)
    print("FEATURE CACHE BUILD")
    print("=" * 78)
    for k in ("requested", "built", "reused", "failed", "windows",
              "zero_yield_clips", "zero_yield_pct", "seconds", "seconds_per_clip"):
        print(f"  {k:<20}{s[k]}")

    by_cell = defaultdict(lambda: dict(clips=0, windows=0, zero=0, gate=0))
    notes = Counter()
    for r in report.results:
        c = by_cell[(r.split, r.label)]
        c["clips"] += 1
        c["windows"] += r.n_windows
        c["zero"] += 1 if r.zero_yield else 0
        c["gate"] += r.gate_passed
        notes[r.note or "ok"] += 1

    print(f"\n  {'cell':22s}{'clips':>7s}{'windows':>9s}{'win/clip':>10s}{'zero-yield':>12s}")
    print("  " + "-" * 60)
    for key in sorted(by_cell):
        c = by_cell[key]
        print(f"  {key[0] + '/' + key[1]:22s}{c['clips']:7d}{c['windows']:9d}"
              f"{c['windows'] / max(c['clips'], 1):10.2f}"
              f"{c['zero']:7d} ({100 * c['zero'] / max(c['clips'], 1):4.1f}%)")

    if notes:
        print("\n  notes: " + ", ".join(f"{k}={v}" for k, v in notes.most_common()))
    if report.built:
        per_clip = report.seconds / report.built
        remaining = len(entries) - report.reused - report.built
        print(f"\n  measured {per_clip:.2f} s/clip -> "
              f"~{per_clip * max(remaining, 0) / 60:.0f} min for the remaining "
              f"{max(remaining, 0)} clips")

    if args.json_out:
        out = {
            "manifest": manifest, "cache_dir": str(cache_dir),
            "frame_shape": [fh, fw], "summary": s,
            "per_cell": {f"{k[0]}/{k[1]}": v for k, v in by_cell.items()},
            "notes": dict(notes),
            "clips": [r.__dict__ for r in report.results],
        }
        dest = Path(args.json_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\n  wrote {dest}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
