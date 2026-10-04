#!/usr/bin/env python
"""Generate a small synthetic CCTV-style test video fixture (no downloads).

Creates moving person-like blobs + timestamp overlay on a noisy background so
the full pipeline (decode -> YOLO -> ByteTrack -> crowd -> fight -> API) can be
exercised offline. YOLO will not classify the blobs as persons — the fixture
validates plumbing and structured output, NOT detection accuracy.

Usage:
  python scripts/make_test_video.py
  python scripts/make_test_video.py --out datasets/videos/test/synthetic_cctv.mp4 \
      --seconds 6 --fps 15 --width 640 --height 360
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "datasets" / "videos" / "test" / "synthetic_cctv.mp4"


def generate(out: Path, seconds: float, fps: int, width: int, height: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out}")

    rng = np.random.default_rng(42)
    n_frames = int(seconds * fps)
    # static background: dark "street" with perspective lines
    bg = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(bg, (0, int(height * 0.55)), (width, height), (60, 60, 60), -1)
    cv2.line(bg, (0, int(height * 0.55)), (width, int(height * 0.55)), (90, 90, 90), 1)

    for i in range(n_frames):
        t = i / fps
        frame = bg.copy()
        # sensor noise
        frame = cv2.add(frame, rng.integers(0, 12, frame.shape, dtype=np.uint8))

        # 4 moving "person" blobs (torso + head), crossing and converging
        for k in range(4):
            speed = 30 + 15 * k
            x = int((speed * t + k * width / 4.5) % (width + 60)) - 30
            if k % 2 == 1:
                x = width - x
            y = int(height * 0.55 + 20 * math.sin(t * 1.5 + k)) + k * 12
            color = (70 + 40 * k, 90, 200 - 30 * k)
            cv2.rectangle(frame, (x, y), (x + 26, y + 90), color, -1)      # torso
            cv2.circle(frame, (x + 13, y - 10), 11, color, -1)            # head

        # two blobs that "clash" in the middle for half the clip (fight-like motion)
        if seconds * 0.4 <= t <= seconds * 0.8:
            cx = width // 2
            jitter = int(10 * math.sin(t * 25))
            cv2.rectangle(frame, (cx - 40 + jitter, 200), (cx - 10 + jitter, 300), (0, 0, 220), -1)
            cv2.rectangle(frame, (cx + 10 - jitter, 200), (cx + 40 - jitter, 300), (0, 0, 220), -1)

        # CCTV-style overlay
        cv2.putText(frame, "CAM_TEST  SYNTHETIC FIXTURE", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"2026-09-27 12:00:{int(t):02d}", (8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(frame)

    writer.release()
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({n_frames} frames, {fps} fps, {width}x{height}, {size_kb:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()
    generate(Path(args.out), args.seconds, args.fps, args.width, args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
