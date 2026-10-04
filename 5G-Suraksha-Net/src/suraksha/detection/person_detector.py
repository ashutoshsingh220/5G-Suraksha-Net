"""YOLO11s person detection wrapper (Ultralytics)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from suraksha.config import DetectionConfig
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class PersonDetection:
    bbox_xyxy: np.ndarray   # float32 [x1, y1, x2, y2]
    confidence: float


class PersonDetector:
    """Detects persons only (COCO class 0) with YOLO11s."""

    def __init__(self, cfg: DetectionConfig, device: str = "auto"):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = "cuda:0" if device == "cuda" else ("cpu" if device == "cpu" else None)
        log.info("Loading YOLO weights: %s (device=%s)", cfg.weights, device)
        self.model = YOLO(cfg.weights)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        kwargs: dict = dict(
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            classes=[self.cfg.person_class_id],
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(frame, **kwargs)

        detections: list[PersonDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                detections.append(
                    PersonDetection(
                        bbox_xyxy=box.xyxy[0].cpu().numpy().astype(np.float32),
                        confidence=float(box.conf[0]),
                    )
                )
        return detections
