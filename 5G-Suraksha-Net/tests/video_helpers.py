"""Video helpers shared by dataset tests: tiny generated mp4s."""
from pathlib import Path

import cv2
import numpy as np


def make_tiny_video(path: Path, n_frames: int = 24, size: tuple[int, int] = (64, 48),
                    fps: float = 12.0, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(n_frames):
        w.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()
    return path


def make_corrupt_video(path: Path) -> Path:
    """A file with a video extension but garbage content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is definitely not a video file" * 100)
    return path
