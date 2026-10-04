import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acc_det.config import (
    TIER1_BEST_DETECTOR,
    TIER2_BEST_SEVERITY,
    TARGET_FPS,
    FRAME_INTERVAL_MS,
)


@dataclass
class AccidentDetection:
    """Represents a single detected vehicle or accident with severity."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    is_accident: bool
    label: str  # 'normal', 'moderate_accident', or 'severe_accident'
    display_name: str
    severity_level: int  # 0: normal, 2: moderate, 3: severe
    severity_confidence: float  # Confidence from Tier 2 severity model
    color: Tuple[int, int, int]  # BGR color for HUD display


@dataclass
class PipelineResult:
    """Result of full hierarchical inference on a single frame."""

    frame_idx: int
    detections: List[AccidentDetection] = field(default_factory=list)
    has_accident: bool = False
    highest_severity: int = 0
    highest_detection: Optional[AccidentDetection] = None
    inference_time_ms: float = 0.0
    tier1_time_ms: float = 0.0
    tier2_time_ms: float = 0.0


class HierarchicalAccidentPipeline:
    """
    Two-Tier Hierarchical Accident & Severity Detection Pipeline:
      - Tier 1: YOLO Spatial Detector (distinguishes 'normal' vs 'accident')
      - Tier 2: Accident Severity Classifier (classifies accident crops into 'moderate' vs 'severe')
    Optimized for real-time 15 FPS camera feeds.
    """

    def __init__(
        self,
        detector_path: str = TIER1_BEST_DETECTOR,
        severity_path: str = TIER2_BEST_SEVERITY,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

        if not os.path.exists(detector_path):
            raise FileNotFoundError(
                f"Tier 1 detector weights not found at: {detector_path}"
            )
        if not os.path.exists(severity_path):
            raise FileNotFoundError(
                f"Tier 2 severity weights not found at: {severity_path}"
            )

        print(f"[HierarchicalPipeline] Loading Tier 1 Detector from: {detector_path}")
        self.detector = YOLO(detector_path)

        print(
            f"[HierarchicalPipeline] Loading Tier 2 Severity Classifier from: {severity_path}"
        )
        self.severity_classifier = YOLO(severity_path)

        # Warm-up models
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector(dummy, verbose=False, device=self.device)
        dummy_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        self.severity_classifier(dummy_crop, verbose=False, device=self.device)
        print("[HierarchicalPipeline] Models warmed up on device:", self.device)

    def set_confidence(self, conf: float):
        """Update detection confidence threshold."""
        self.conf_thres = conf

    def process_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0
    ) -> PipelineResult:
        """
        Executes hierarchical two-tier inference on a video frame.
        """
        start_t = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        # ---------------- Tier 1: Spatial Detection ----------------
        t1_start = time.perf_counter()
        det_results = self.detector(
            frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )
        t1_duration = (time.perf_counter() - t1_start) * 1000.0

        detections: List[AccidentDetection] = []
        has_accident = False
        highest_severity = 0
        highest_det = None
        t2_duration = 0.0

        if det_results and len(det_results) > 0 and det_results[0].boxes is not None:
            boxes = det_results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                # Clamp coords
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 1, min(w, x2))
                y2 = max(y1 + 1, min(h, y2))

                # Check if detector identified accident (class 1)
                is_accident = cls_id == 1

                if not is_accident:
                    # Normal vehicle / normal traffic
                    det = AccidentDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        is_accident=False,
                        label="normal",
                        display_name="Normal Vehicle",
                        severity_level=0,
                        severity_confidence=conf,
                        color=(0, 210, 60),  # Bright Green
                    )
                    detections.append(det)
                else:
                    # ---------------- Tier 2: Severity Classification ----------------
                    has_accident = True
                    t2_start = time.perf_counter()

                    # Crop accident region with 10% context padding
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
                    t2_duration += (time.perf_counter() - t2_start) * 1000.0

                    # Parse severity result
                    probs = sev_res[0].probs
                    top1_idx = int(probs.top1)
                    sev_conf = float(probs.top1conf.item())
                    sev_name = self.severity_classifier.names.get(
                        top1_idx, "moderate"
                    ).lower()

                    # Contextual promotion heuristic:
                    # Very large collision footprint relative to frame or multiple overlapping vehicles
                    if "severe" in sev_name or (bw * bh > 0.20 * w * h):
                        severity_level = 3  # Severe
                        display_name = f"Severe Accident ({sev_conf:.0%})"
                        label_name = "severe_accident"
                        color = (0, 40, 240)  # Bright Red
                    else:
                        severity_level = 2  # Moderate
                        display_name = f"Moderate Accident ({sev_conf:.0%})"
                        label_name = "moderate_accident"
                        color = (0, 165, 255)  # Vibrant Orange

                    if severity_level > highest_severity:
                        highest_severity = severity_level

                    det = AccidentDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        is_accident=True,
                        label=label_name,
                        display_name=display_name,
                        severity_level=severity_level,
                        severity_confidence=sev_conf,
                        color=color,
                    )
                    detections.append(det)

                    if (
                        highest_det is None
                        or det.severity_level > highest_det.severity_level
                    ):
                        highest_det = det

        total_duration = (time.perf_counter() - start_t) * 1000.0

        return PipelineResult(
            frame_idx=frame_idx,
            detections=detections,
            has_accident=has_accident,
            highest_severity=highest_severity,
            highest_detection=highest_det,
            inference_time_ms=total_duration,
            tier1_time_ms=t1_duration,
            tier2_time_ms=t2_duration,
        )
