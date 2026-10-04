import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from config import FIRE_MODEL_PATH, FIRE_CLASSES, FIRE_COLORS, DEFAULT_FIRE_CONF


@dataclass
class FireDetection:
    """Individual fire or smoke bounding box detection."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    label: str
    color: Tuple[int, int, int]


@dataclass
class FireResult:
    """Detection result from Fire and Smoke Detector."""

    has_fire: bool = False
    has_smoke: bool = False
    fire_confidence: float = 0.0
    smoke_confidence: float = 0.0
    detections: List[FireDetection] = field(default_factory=list)

    @property
    def is_fire(self) -> bool:
        """Backward compatibility: True if fire is detected."""
        return self.has_fire

    @property
    def fire_probability(self) -> float:
        """Backward compatibility: Max confidence for fire."""
        return self.fire_confidence

    @property
    def non_fire_probability(self) -> float:
        """Backward compatibility: Approximate non-fire probability."""
        return max(0.0, 1.0 - self.fire_confidence)

    @property
    def predicted_label(self) -> str:
        """Human-readable hazard summary."""
        if self.has_fire and self.has_smoke:
            return "Fire & Smoke"
        if self.has_fire:
            return "Fire"
        if self.has_smoke:
            return "Smoke"
        return "Normal"

    @property
    def confidence(self) -> float:
        """Highest hazard confidence score."""
        return max(self.fire_confidence, self.smoke_confidence)


class FireDetector:
    """Ultralytics YOLO-based Fire & Smoke Detector."""

    def __init__(
        self,
        model_path: str = FIRE_MODEL_PATH,
        conf_thres: float = DEFAULT_FIRE_CONF,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.conf_thres = conf_thres

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Fire classifier weights not found at: {self.model_path}"
            )

        self.model = YOLO(self.model_path)

    def set_confidence(self, conf: float):
        """Update detection confidence threshold."""
        self.conf_thres = conf

    def detect(self, frame_bgr: np.ndarray) -> FireResult:
        """
        Runs fire and smoke detection on a BGR video frame.

        Returns:
            FireResult containing detected bounding boxes, status flags, and confidence scores.
        """
        results = self.model(
            frame_bgr, conf=self.conf_thres, device=self.device, verbose=False
        )
        detections: List[FireDetection] = []

        has_fire = False
        has_smoke = False
        max_fire_conf = 0.0
        max_smoke_conf = 0.0

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                label = self.model.names.get(cls_id, f"class_{cls_id}").lower()
                color = FIRE_COLORS.get(label, (150, 150, 150))

                if "fire" in label:
                    has_fire = True
                    if conf > max_fire_conf:
                        max_fire_conf = conf
                elif "smoke" in label:
                    has_smoke = True
                    if conf > max_smoke_conf:
                        max_smoke_conf = conf

                detections.append(
                    FireDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        label=label.capitalize(),
                        color=color,
                    )
                )

        return FireResult(
            has_fire=has_fire,
            has_smoke=has_smoke,
            fire_confidence=max_fire_conf,
            smoke_confidence=max_smoke_conf,
            detections=detections,
        )
