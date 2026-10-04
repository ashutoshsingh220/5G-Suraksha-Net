"""Annotated video export for 5G Suraksha-Net."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from suraksha.logging_utils import get_logger

log = get_logger(__name__)


class AnnotatedVideoWriter:
    """Writes annotated frames to an MP4 video file using OpenCV's VideoWriter."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],  # (width, height)
        fourcc: str = "mp4v",
    ):
        self.output_path = Path(output_path).resolve()
        self.fps = float(fps) if fps and fps > 0 else 15.0
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.fourcc_str = fourcc
        self.frames_written = 0

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc_str)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc_code,
            self.fps,
            self.frame_size,
        )

        if not self._writer.isOpened():
            log.error("Failed to open VideoWriter for %s", self.output_path)
            raise RuntimeError(f"Could not open VideoWriter at {self.output_path}")

        log.info(
            "AnnotatedVideoWriter initialized: %s (%dx%d @ %.1f fps, codec=%s)",
            self.output_path,
            self.frame_size[0],
            self.frame_size[1],
            self.fps,
            self.fourcc_str,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write an annotated frame. Automatically resizes if shape differs from target size."""
        if self._writer is None or not self._writer.isOpened():
            return

        h, w = frame.shape[:2]
        target_w, target_h = self.frame_size

        if (w, h) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        self._writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        """Close and release the video writer."""
        if self._writer is not None:
            if self._writer.isOpened():
                self._writer.release()
                log.info(
                    "AnnotatedVideoWriter released: %s (%d frames written)",
                    self.output_path,
                    self.frames_written,
                )
            self._writer = None

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
