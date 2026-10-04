"""Evidence capture: annotated snapshots + short video clips."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from suraksha.config import IncidentsConfig, PROJECT_ROOT
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ClipRecordingSession:
    """State for an ongoing clip recording session that captures post-event frames."""
    incident_id: str
    start_time: float
    end_time: float
    target_end_time: float
    frames: list[np.ndarray] = field(default_factory=list)
    callback: Callable[[str, str], None] | None = None
    completed: bool = False


class EvidenceWriter:
    """Keeps a rolling frame buffer so clips include seconds BEFORE and AFTER an incident."""

    def __init__(self, cfg: IncidentsConfig):
        self.cfg = cfg
        maxlen = cfg.clip_fps * (cfg.clip_seconds_before + cfg.clip_seconds_after + 2)
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._active_sessions: dict[str, ClipRecordingSession] = {}
        self._ffmpeg = shutil.which("ffmpeg")
        self._lock = threading.Lock()
        self._pending_futures: list[Future] = []
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="evidence_writer")
            if getattr(cfg, "async_write", False)
            else None
        )
        for d in (cfg.snapshot_dir, cfg.clip_dir, cfg.report_dir):
            Path(d if Path(d).is_absolute() else PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    def push_frame(self, frame: np.ndarray, now: float) -> list[str]:
        """Push frame to rolling buffer and active post-event recording sessions.
        
        Returns a list of incident IDs whose recording sessions completed on this frame.
        """
        ready_sessions: list[ClipRecordingSession] = []
        with self._lock:
            self._buffer.append((now, frame))
            for inc_id, sess in list(self._active_sessions.items()):
                if sess.completed:
                    continue
                sess.frames.append(frame.copy())
                if now >= sess.target_end_time - 1e-3:
                    sess.completed = True
                    ready_sessions.append(sess)
                    del self._active_sessions[inc_id]

        completed_ids: list[str] = []
        for sess in ready_sessions:
            if self._executor is not None:
                fut = self._executor.submit(self._write_session_clip, sess)
                self._pending_futures.append(fut)
                completed_ids.append(sess.incident_id)
            else:
                clip = self._write_session_clip(sess)
                if clip:
                    completed_ids.append(sess.incident_id)
        return completed_ids

    def wait_pending(self, timeout: float = 10.0) -> None:
        """Wait for any asynchronous clip encodings to complete."""
        if not self._pending_futures:
            return
        t0 = time.time()
        for fut in list(self._pending_futures):
            rem = max(0.1, timeout - (time.time() - t0))
            try:
                fut.result(timeout=rem)
            except Exception as e:
                log.warning("Evidence clip future failed: %s", e)
        self._pending_futures.clear()

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def save_snapshot(
        self, frame: np.ndarray, incident_id: str, annotations: list[tuple] | None = None
    ) -> str:
        """annotations: list of (bbox_xyxy, label, color_bgr)."""
        img = frame.copy()
        for bbox, label, color in annotations or []:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out = self._resolve(self.cfg.snapshot_dir) / f"{incident_id}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(out)

    def start_clip_session(
        self,
        incident_id: str,
        start_time: float,
        end_time: float,
        callback: Callable[[str, str], None] | None = None,
    ) -> ClipRecordingSession:
        """Start accumulating frames for an incident clip covering [start_time - before, end_time + after]."""
        t0 = start_time - self.cfg.clip_seconds_before
        target_end = end_time + self.cfg.clip_seconds_after

        with self._lock:
            pre_frames = [f.copy() for (t, f) in self._buffer if t0 <= t <= end_time]
            session = ClipRecordingSession(
                incident_id=incident_id,
                start_time=start_time,
                end_time=end_time,
                target_end_time=target_end,
                frames=pre_frames,
                callback=callback,
            )
            self._active_sessions[incident_id] = session
            log.info(
                "EVIDENCE_RECORDING_STARTED incident=%s pre_frames=%d target_end=%.2f",
                incident_id, len(pre_frames), target_end,
            )
            return session

    def _write_session_clip(self, sess: ClipRecordingSession) -> str | None:
        frames = sess.frames
        incident_id = sess.incident_id
        if len(frames) < 5:
            log.warning("Not enough frames for session clip %s (%d)", incident_id, len(frames))
            return None

        out = self._resolve(self.cfg.clip_dir) / f"{incident_id}.mp4"
        tmp = out.with_suffix(".tmp.mp4")
        h, w = frames[0].shape[:2]

        # Try native Windows Media Foundation H.264 encoder for direct HTML5 browser compatibility
        h264_written = False
        try:
            writer = cv2.VideoWriter(str(out), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), self.cfg.clip_fps, (w, h))
            if writer.isOpened():
                for f in frames:
                    writer.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
                writer.release()
                if out.is_file() and out.stat().st_size > 1000:
                    h264_written = True
        except Exception as e:
            log.debug("MSMF H264 encode attempt: %s", e)

        if not h264_written:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.cfg.clip_fps, (w, h))
            for f in frames:
                writer.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
            writer.release()

            if self._ffmpeg:
                try:
                    subprocess.run(
                        [self._ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp),
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                        check=True, timeout=60,
                    )
                    tmp.unlink(missing_ok=True)
                except Exception as e:
                    log.warning("ffmpeg re-encode failed (%s); keeping mp4v file", e)
                    tmp.replace(out)
            else:
                tmp.replace(out)

        clip_path = str(out)
        log.info(
            "EVIDENCE_RECORDING_FINALIZED incident=%s total_frames=%d path=%s",
            incident_id, len(frames), clip_path,
        )
        if sess.callback:
            try:
                sess.callback(incident_id, clip_path)
            except Exception:
                log.exception("Error executing session completion callback for %s", incident_id)
        return clip_path

    def finalize_all(self) -> list[str]:
        """Finalize any pending clip sessions immediately (e.g. at stream end)."""
        with self._lock:
            pending = list(self._active_sessions.values())
            self._active_sessions.clear()

        completed: list[str] = []
        for sess in pending:
            sess.completed = True
            clip = self._write_session_clip(sess)
            if clip:
                completed.append(sess.incident_id)

        if self._executor is not None:
            self.wait_pending(timeout=30.0)
            self._executor.shutdown(wait=True)
            self._executor = None

        return completed

    def save_clip(self, incident_id: str, start_time: float, end_time: float) -> str | None:
        """Write buffered frames from [start - before, end + after] to mp4 synchronously."""
        t0 = start_time - self.cfg.clip_seconds_before
        t1 = end_time + self.cfg.clip_seconds_after
        with self._lock:
            frames = [f.copy() for (t, f) in self._buffer if t0 <= t <= t1]
        sess = ClipRecordingSession(
            incident_id=incident_id,
            start_time=start_time,
            end_time=end_time,
            target_end_time=t1,
            frames=frames,
        )
        return self._write_session_clip(sess)
