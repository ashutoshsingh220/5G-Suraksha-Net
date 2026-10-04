import os
import socket
import time
from typing import Optional, Tuple, Union
from urllib.parse import urlparse
import cv2
import numpy as np


def probe_rtsp_reachability(url: str, timeout: float = 1.0) -> Tuple[bool, str]:
    """
    Fast TCP socket probe to verify if the RTSP camera host and port are reachable.
    Avoids OpenCV FFmpeg's 30s-120s blocking interrupt callback on offline IP addresses.
    Returns (is_reachable: bool, error_message: str).
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or 554
        if not host:
            return False, f"Invalid RTSP URL: No hostname found in '{url}'"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            err = sock.connect_ex((host, port))
            if err != 0:
                return False, f"Host {host}:{port} unreachable (socket error {err})"
            return True, ""
    except Exception as e:
        return False, f"Network probe exception: {e}"


class StreamCapture:
    """Robust video capture wrapper for RTSP CCTV streams and video files."""

    def __init__(
        self,
        source: Union[str, int],
        name: Optional[str] = None,
        target_fps: Optional[float] = None,
    ):
        self.source = source
        self.target_fps = target_fps
        self.frame_interval = (
            (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0
        )
        self.last_frame_time = 0.0
        self.is_rtsp = isinstance(source, str) and (
            source.startswith("rtsp://") or source.startswith("rtsps://")
        )
        self.is_live = self.is_rtsp or isinstance(source, int)
        self.name = name or (f"Sparsh CCTV ({source})" if self.is_rtsp else str(source))

        if self.is_rtsp:
            # Fast TCP reachability probe (<=1.0s) to prevent OpenCV FFmpeg from hanging for 30s-120s
            is_reachable, probe_err = probe_rtsp_reachability(str(self.source), timeout=1.0)
            if not is_reachable:
                raise RuntimeError(
                    f"RTSP camera offline or unreachable: {probe_err}\n"
                    f"Target URL: {self.source}\n"
                    f"Verify camera power, network connectivity, and IP address."
                )

            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|timeout;3000000|stimeout;3000000|fflags;nobuffer|flags;low_delay"
            )
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {self.source}\n"
                f"For RTSP: verify network connection, IP, credentials, and camera power.\n"
                f"For files: verify the file path exists and is readable."
            )

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 15.0 if target_fps else 30.0
        self.total_frames = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else -1
        )
        self.frame_idx = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next video frame with optional 15 FPS clock pacing."""
        if self.frame_interval > 0:
            elapsed = time.time() - self.last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
            self.last_frame_time = time.time()

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
            return True, frame

        # If live stream, attempt a single reconnect attempt with backoff
        now = time.time()
        if self.is_rtsp and (now - getattr(self, "_last_reconnect_time", 0.0) > 3.0):
            self._last_reconnect_time = now
            print("[StreamCapture] RTSP feed interrupted, attempting reconnection...")
            try:
                self.cap.release()
                time.sleep(0.5)
                reachable, _ = probe_rtsp_reachability(str(self.source), timeout=1.0)
                if reachable:
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    if self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            self.frame_idx += 1
                            return True, frame
            except Exception as e:
                print(f"[StreamCapture] Reconnection failed: {e}")

        return False, None

    def release(self):
        """Release underlying OpenCV VideoCapture."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
