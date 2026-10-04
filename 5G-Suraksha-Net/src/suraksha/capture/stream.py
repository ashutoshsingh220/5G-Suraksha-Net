"""Frame capture from RTSP streams, local video files, or webcams.

All three source kinds yield the same FrameEvent interface, so the downstream
pipeline (YOLO -> ByteTrack -> crowd/fight) is source-agnostic.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from suraksha.config import CaptureConfig, PROJECT_ROOT
from suraksha.logging_utils import get_logger

log = get_logger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}
_STREAM_PROTOCOLS = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def sanitize_url(url: str) -> str:
    """Mask credentials in network stream URLs (e.g. rtsp://user:pass@host -> rtsp://user:***@host)."""
    try:
        raw = str(url).strip()
        parts = urlsplit(raw)
        if parts.password:
            user = parts.username or ""
            port = f":{parts.port}" if parts.port else ""
            host = parts.hostname or ""
            netloc = f"{user}:***@{host}{port}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        pass
    return str(url)


@dataclass
class FrameEvent:
    frame: np.ndarray
    frame_idx: int
    timestamp: float          # wall-clock time of capture (time.time())
    source_fps: float         # fps reported by the stream


class StreamCapture:
    """Wraps cv2.VideoCapture for RTSP streams, local video files, or webcams.

    - rtsp/rtmp/http/udp/tcp URL -> network stream, reconnects on dropout
    - numeric string/int         -> webcam index
    - file path                  -> local video; generator ENDS at EOF (no reconnect)
    Pace modes (cfg.pace): throttle | realtime | fast.
    """

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx = 0
        self._attempts = 0
        self.source_kind = self._resolve_source_kind(cfg)
        self.resolved_url = self._resolve_url(cfg, self.source_kind)
        self._switch_lock = threading.Lock()
        self._pending_switch: dict | None = None

    @staticmethod
    def _normalize_kind(kind: str | None, url: str) -> str:
        if kind:
            k = str(kind).strip().lower()
            if k in ("rtsp", "stream", "rtsps", "rtmp", "http", "https"):
                return "stream"
            if k in ("webcam", "camera", "cam"):
                return "webcam"
            if k in ("file", "video", "mp4"):
                return "file"
        return StreamCapture._classify(url)

    def switch_source(
        self,
        url: str,
        source_kind: str | None = None,
        pace: str | None = None,
    ) -> bool:
        """Switch video capture source dynamically on the fly without restarting pipeline."""
        with self._switch_lock:
            new_kind = self._normalize_kind(source_kind, url)
            new_pace = pace or self.cfg.pace
            self.resolved_url = url
            self.source_kind = new_kind
            self.cfg.pace = new_pace
            self._frame_idx = 0
            self._attempts = 0
            self._pending_switch = {
                "url": url,
                "source_kind": new_kind,
                "pace": new_pace,
            }
        return True

    @staticmethod
    def _classify(url: str) -> str:
        u = str(url).strip()
        if u.lower().startswith(_STREAM_PROTOCOLS):
            return "stream"
        if u.isdigit():
            return "webcam"
        return "file"

    @classmethod
    def _resolve_source_kind(cls, cfg: CaptureConfig) -> str:
        u = str(cfg.rtsp_url).strip()
        if u.lower().endswith((".mp4", ".avi", ".mkv", ".mov")) or Path(u).is_file():
            return "file"
        if cfg.source_type:
            st = cfg.source_type.strip().lower()
            if st in ("rtsp", "stream", "rtsps", "rtmp", "http", "https"):
                return "stream"
            if st in ("webcam", "camera", "cam"):
                return "webcam"
            if st in ("file", "video", "mp4"):
                return "file"
        return cls._classify(cfg.rtsp_url)

    @staticmethod
    def _resolve_url(cfg: CaptureConfig, kind: str) -> str:
        u = str(cfg.rtsp_url).strip()
        if kind == "webcam":
            if u.isdigit():
                return u
            return str(cfg.camera_index)
        if kind == "stream":
            if not u.lower().startswith(_STREAM_PROTOCOLS):
                return f"rtsp://{u}"
        return u

    def _open(self) -> bool:
        url = self.resolved_url
        kind = self._normalize_kind(self.source_kind, url)
        self.source_kind = kind

        if kind == "webcam":
            try:
                cam_idx = int(url)
            except (ValueError, TypeError):
                cam_idx = self.cfg.camera_index
            # Windows: prefer DirectShow; fallback to default backend (CAP_ANY)
            self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
            if self._cap.isOpened():
                try:
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low-latency webcam
                except Exception:
                    pass
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        elif kind == "stream":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|analyzeduration;500000|probesize;500000"
            self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low-latency RTSP
            except Exception:
                pass
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        else:  # file
            path = Path(url)
            if not path.is_absolute() and not path.exists():
                path = (PROJECT_ROOT / path).resolve()
            if not path.exists():
                log.error("Video file not found: %s", path)
                return False
            if path.suffix.lower() not in _VIDEO_EXTS:
                log.warning("Unusual video extension '%s' — attempting to open anyway", path.suffix)
            self._cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)

        ok = self._cap.isOpened()
        if ok:
            log.info("Opened %s source: %s", kind, sanitize_url(url))
            self._attempts = 0
        else:
            log.warning("Failed to open %s source: %s (attempt %d)", kind, sanitize_url(url), self._attempts + 1)
        return ok

    def _connect(self) -> bool:
        while True:
            if self._pending_switch is not None:
                return True
            if self._open():
                self._attempts = 0
                return True
            if self.source_kind == "file":
                return False  # a missing/broken file never reconnects
            self._attempts += 1
            limit = self.cfg.max_reconnect_attempts if self.cfg.max_reconnect_attempts >= 0 else (2 if self.source_kind == "webcam" else 3)
            if 0 <= limit <= self._attempts:
                return False
            time.sleep(min(1.0, self.cfg.reconnect_delay_s))

    def frames(self) -> Iterator[FrameEvent]:
        """Generator of frames.

        - stream/webcam: infinite, reconnects transparently
        - file: ends at EOF (StopIteration)
        Pacing per cfg.pace: throttle (cap at target_fps) | realtime (sleep to
        source fps, simulating a live camera) | fast (no pacing, benchmarking).
        """
        self._connect()
        src_fps = (self._cap.get(cv2.CAP_PROP_FPS) if self._cap else None) or float(self.cfg.target_fps)
        pace = self.cfg.pace
        if pace == "throttle":
            min_interval = 1.0 / max(self.cfg.target_fps, 1)
        elif pace == "realtime":
            min_interval = 1.0 / max(src_fps, 1.0)
        else:  # fast
            min_interval = 0.0
        last_emit = 0.0
        emitted = 0

        while True:
            # Handle dynamic hot-swap of video capture source on the fly
            if self._pending_switch is not None:
                with self._switch_lock:
                    req = self._pending_switch
                    self._pending_switch = None
                if req and req.get("url"):
                    new_url = req["url"]
                    new_kind = self._normalize_kind(req.get("source_kind"), new_url)
                    new_pace = req.get("pace") or self.cfg.pace
                    log.info("HOT_SWAPPING_CAPTURE_SOURCE kind=%s url=%s pace=%s", new_kind, sanitize_url(new_url), new_pace)
                    if self._cap:
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                    self.resolved_url = new_url
                    self.source_kind = new_kind
                    self.cfg.pace = new_pace
                    self._frame_idx = 0
                    self._attempts = 0
                    if not self._connect():
                        log.warning("Failed opening hot-swapped source %s", sanitize_url(new_url))
                        continue
                    src_fps = self._cap.get(cv2.CAP_PROP_FPS) or float(self.cfg.target_fps)
                    pace = new_pace
                    if pace == "throttle":
                        min_interval = 1.0 / max(self.cfg.target_fps, 1)
                    elif pace == "realtime":
                        min_interval = 1.0 / max(src_fps, 1.0)
                    else:
                        min_interval = 0.0
                    last_emit = time.time()
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.3)
                continue

            ok, frame = self._cap.read()  # type: ignore[union-attr]
            if not ok or frame is None:
                if self.source_kind == "file":
                    # Loop video file seamlessly for continuous demonstration & live analysis
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        # Re-open file capture cleanly from beginning
                        if self._cap:
                            try:
                                self._cap.release()
                            except Exception:
                                pass
                        self._cap = cv2.VideoCapture(self.resolved_url)
                        if self._cap and self._cap.isOpened():
                            ok, frame = self._cap.read()
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                else:
                    log.warning("Stream read failure — reconnecting")
                    if self._cap:
                        self._cap.release()
                    if not self._connect():
                        raise ConnectionError("Stream reconnect failed permanently")
                    continue

            # Auto-downscale high-res frames to maximum 640/720 dimension for fluid 25-30 FPS inference
            if frame is not None:
                fh, fw = frame.shape[:2]
                max_dim = 640
                if max(fw, fh) > max_dim:
                    scale = max_dim / float(max(fw, fh))
                    nw, nh = int(fw * scale), int(fh * scale)
                    nw = nw if nw % 2 == 0 else nw - 1
                    nh = nh if nh % 2 == 0 else nh - 1
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

            self._frame_idx += 1
            now = time.time()
            if min_interval > 0.0:
                wait = min_interval - (now - last_emit)
                if wait > 0:
                    if pace == "realtime":
                        time.sleep(wait)   # simulate live camera timing
                        now = time.time()
                    else:
                        continue           # throttle: drop frames above target_fps
            last_emit = now

            emitted += 1
            yield FrameEvent(
                frame=frame,
                frame_idx=self._frame_idx,
                timestamp=now,
                source_fps=src_fps,
            )
            if self.cfg.max_frames and emitted >= self.cfg.max_frames:
                log.info("Reached max_frames=%d — stopping capture", self.cfg.max_frames)
                return

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "StreamCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
