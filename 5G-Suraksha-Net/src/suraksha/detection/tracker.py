"""ByteTrack multi-object tracking via Ultralytics' built-in tracker."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from suraksha.config import DetectionConfig, TrackingConfig
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class TrackedPerson:
    track_id: int
    bbox_xyxy: np.ndarray   # float32 [x1, y1, x2, y2]
    confidence: float

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])


class MultiObjectTracker:
    """Runs YOLO11s + ByteTrack in one call (ultralytics track API).

    Using the integrated tracker keeps detection and association in sync and
    avoids a second inference pass.
    """

    def __init__(self, det_cfg: DetectionConfig, trk_cfg: TrackingConfig, device: str = "auto"):
        from ultralytics import YOLO

        self.det_cfg = det_cfg
        self.trk_cfg = trk_cfg
        self.model = YOLO(det_cfg.weights)
        self.device_arg: int | str | None = (
            0 if device == "cuda" else ("cpu" if device == "cpu" else None)
        )
        log.info("Tracker ready (tracker=%s)", trk_cfg.tracker_config)

    def reset(self) -> None:
        """Drop ByteTrack state so the next update() starts a fresh sequence.

        `persist=True` keeps track IDs and Kalman filters alive between calls,
        which is exactly right for one continuous stream and exactly wrong when
        the same tracker instance is reused across separate video files: clip B
        would inherit clip A's IDs and motion estimates. Offline feature
        extraction calls this between clips so its output does not depend on the
        order clips happen to be processed in.
        """
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        n = 0
        for t in trackers:
            reset = getattr(t, "reset", None)
            if callable(reset):
                reset()
                n += 1
        if n:
            log.debug("ByteTrack state reset on %d tracker(s)", n)
        return None

    def update(self, frame: np.ndarray) -> list[TrackedPerson]:
        kwargs: dict = dict(
            conf=self.det_cfg.conf_threshold,
            iou=self.det_cfg.iou_threshold,
            imgsz=self.det_cfg.imgsz,
            classes=[self.det_cfg.person_class_id],
            tracker=self.trk_cfg.tracker_config,
            persist=self.trk_cfg.persist,
            verbose=False,
        )
        if self.device_arg is not None:
            kwargs["device"] = self.device_arg
        results = self.model.track(frame, **kwargs)

        tracks: list[TrackedPerson] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.cpu().numpy()
            for tid, box, c in zip(ids, xyxy, conf):
                tracks.append(TrackedPerson(track_id=int(tid), bbox_xyxy=box, confidence=float(c)))
        return tracks
