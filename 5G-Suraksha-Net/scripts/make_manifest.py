#!/usr/bin/env python
"""Build a dataset manifest from a folder of video clips.

Usage:
  python scripts/make_manifest.py --name fight_v1 --task fight_temporal \
      --clips-dir datasets/raw/fight --label fight --source sparsh_cctv

Assumes a flat directory of .mp4/.avi/.mkv clips. Labels can be fixed
(--label) or taken from parent folder names (fight/ vs no_fight/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def probe(path: Path) -> tuple[int | None, float | None, tuple[int, int] | None]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    cap.release()
    return n, fps, (w, h) if w and h else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", default="fight_temporal", choices=["fight_temporal", "crowd_density"])
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--label", default=None, help="fixed label; omit to use parent folder name")
    ap.add_argument("--source", default="sparsh_cctv")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    from suraksha.data.manifest import ClipEntry, Manifest, file_sha256, save_manifest, validate_manifest

    root = Path(args.clips_dir)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        print(f"No video clips found under {root}")
        return 1

    m = Manifest(name=args.name, task=args.task)
    for i, f in enumerate(files):
        label = args.label or f.parent.name
        n, fps, res = probe(f)
        try:
            rel = f.relative_to(Path.cwd())
        except ValueError:
            rel = f
        m.add(ClipEntry(
            clip_id=f"{args.name}_{i:05d}",
            path=str(rel).replace("\\", "/"),
            label=label,
            split=args.split,
            num_frames=n,
            fps=round(fps, 3) if fps else None,
            resolution=res,
            source=args.source,
            sha256=file_sha256(f),
        ))
        print(f"  + {m.entries[-1].clip_id} [{label}] {f.name}")

    problems = validate_manifest(m)
    if problems:
        print("Manifest problems:")
        for p in problems:
            print(" -", p)
    path = save_manifest(m)
    print(f"Manifest: {path}")
    print(f"Labels: {m.by_label()}  Splits: {m.split_counts()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
