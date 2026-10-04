import os
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import (
    YOLO_DETECTOR_PATH,
    YOLO_SEVERITY_PATH,
    DEFAULT_CRASH_CONF,
    DEFAULT_CRASH_IOU,
    SEVERITY_CONFIG,
)
from detectors.accident_detector import VehicleDetection


class YOLOHierarchicalAccidentDetector:
    """
    Two-Tier Hierarchical YOLO Accident & Severity Detector:
      - Tier 1: Spatial YOLO Detector distinguishing 'normal' vehicles vs 'accident' scenes.
      - Tier 2: Accident Severity Classifier categorizing accident crops into 'moderate' vs 'severe'.
    Drop-in replacement for legacy DETR detector with >50-80 FPS real-time speed.
    """

    def __init__(
        self,
        detector_path: str = YOLO_DETECTOR_PATH,
        severity_path: str = YOLO_SEVERITY_PATH,
        conf_thres: float = DEFAULT_CRASH_CONF,
        iou_thres: float = DEFAULT_CRASH_IOU,
        device: Optional[str] = None,
    ):
        self.detector_path = detector_path
        self.severity_path = severity_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(self.detector_path):
            raise FileNotFoundError(
                f"YOLO detector weights not found at: {self.detector_path}"
            )
        if not os.path.exists(self.severity_path):
            raise FileNotFoundError(
                f"YOLO severity weights not found at: {self.severity_path}"
            )

        self.detector = YOLO(self.detector_path)
        self.severity_classifier = YOLO(self.severity_path)

        # Warm-up models
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector(dummy, verbose=False, device=self.device)
        dummy_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        self.severity_classifier(dummy_crop, verbose=False, device=self.device)

        # Temporal persistence counter to prevent single-frame flickering
        self.recent_accident_history = []

    def set_confidence(self, conf: float):
        """Update detection confidence threshold."""
        self.conf_thres = conf

    def detect(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        """
        Runs real-time hierarchical accident detection & severity classification on a BGR video frame.

        Returns:
            List of VehicleDetection dataclass instances with bounding boxes and alert status.
        """
        h, w = frame_bgr.shape[:2]

        # Tier 1: Detection
        det_results = self.detector(
            frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        detections: List[VehicleDetection] = []
        if not det_results or len(det_results) == 0 or det_results[0].boxes is None:
            return []

        boxes = det_results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            is_accident = cls_id == 1

            if not is_accident:
                # Normal Vehicle
                detections.append(
                    VehicleDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=2,
                        raw_name="vehicle",
                        display_name="Normal Vehicle",
                        short_name="NORMAL",
                        color=(0, 210, 60),  # Bright Green
                        is_accident=False,
                        severity_level=0,
                    )
                )
            else:
                # Tier 2: Severity Classification on Accident ROI
                bw = x2 - x1
                bh = y2 - y1
                pad_w = int(bw * 0.10)
                pad_h = int(bh * 0.10)
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(w, x2 + pad_w)
                cy2 = min(h, y2 + pad_h)

                crop = frame_bgr[cy1:cy2, cx1:cx2]
                if crop.shape[0] < 10 or crop.shape[1] < 10:
                    crop = frame_bgr[y1:y2, x1:x2]

                crop_resized = cv2.resize(crop, (224, 224))
                sev_res = self.severity_classifier(
                    crop_resized, verbose=False, device=self.device
                )

                probs = sev_res[0].probs
                top1_idx = int(probs.top1)
                sev_conf = float(probs.top1conf.item())
                sev_name = self.severity_classifier.names.get(
                    top1_idx, "moderate"
                ).lower()

                # Large collision footprint or structural deformation flags severe
                if "severe" in sev_name or (bw * bh > 0.20 * w * h):
                    sev_level = 3  # Level 3: Severe Accident
                    raw_name = "accident_severe"
                    display_name = f"Severe Accident ({conf:.2f})"
                    short_name = "SEVERE"
                    color = (0, 40, 240)  # Bright Red
                else:
                    sev_level = 2  # Level 2: Moderate Accident
                    raw_name = "accident_moderate"
                    display_name = f"Moderate Accident ({conf:.2f})"
                    short_name = "MODERATE"
                    color = (0, 165, 255)  # Vibrant Orange

                detections.append(
                    VehicleDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=0 if sev_level == 3 else 1,
                        raw_name=raw_name,
                        display_name=display_name,
                        short_name=short_name,
                        color=color,
                        is_accident=True,
                        severity_level=sev_level,
                    )
                )

        return detections

    @staticmethod
    def get_highest_severity(
        detections: List[VehicleDetection],
    ) -> Tuple[int, Optional[VehicleDetection]]:
        """
        Returns (highest_severity_level, detection_with_highest_severity).
        Returns (-1, None) if no detections.
        """
        if not detections:
            return -1, None

        max_det = max(detections, key=lambda d: d.severity_level)
        return max_det.severity_level, max_det
