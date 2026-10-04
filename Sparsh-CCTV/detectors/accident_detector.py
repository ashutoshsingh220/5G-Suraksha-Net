import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
import torchvision.ops as ops
from transformers import AutoModelForObjectDetection

from config import (
    CRASH_MODEL_DIR,
    CRASH_MODEL_PATH,
    SEVERITY_CONFIG,
    DEFAULT_CRASH_CONF,
    DEFAULT_CRASH_IOU,
)


@dataclass
class VehicleDetection:
    """Represents a single detected vehicle or accident with severity."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int
    raw_name: str
    display_name: str
    short_name: str
    color: Tuple[int, int, int]
    is_accident: bool
    severity_level: int


class AccidentDetector:
    """DETR ResNet-50 SafeTensors Accident & Vehicle Object Detector with CUDA FP16 support."""

    def __init__(
        self,
        model_dir: str = CRASH_MODEL_DIR,
        conf_thres: float = DEFAULT_CRASH_CONF,
        iou_thres: float = DEFAULT_CRASH_IOU,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (800, 800),
    ):
        self.model_dir = model_dir
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Check weights exist
        weights_file = os.path.join(self.model_dir, "model.safetensors")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"Crash classifier weights not found at: {weights_file}"
            )

        # Load DETR model locally without internet calls
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_dir, local_files_only=True
        )

        # Use FP16 on CUDA for 2x real-time speedup
        self.is_cuda = self.device.type == "cuda"
        self.dtype = torch.float16 if self.is_cuda else torch.float32

        if self.is_cuda:
            self.model = self.model.to(self.device).half()
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # ImageNet normalization tensors pre-allocated on device
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=self.dtype
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=self.dtype
        ).view(1, 3, 1, 1)

    def set_confidence(self, conf: float):
        """Update detection confidence threshold."""
        self.conf_thres = conf

    def detect(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        """
        Runs real-time accident and vehicle detection on a BGR video frame.

        Returns:
            List of VehicleDetection dataclass instances with bounding boxes and alert status.
        """
        h, w = frame_bgr.shape[:2]

        # 1. Fast GPU-accelerated Preprocessing
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype)
            / 255.0
        )
        tensor = (tensor - self.mean) / self.std

        # 2. Forward inference with PyTorch inference mode
        with torch.inference_mode():
            outputs = self.model(tensor)

        logits = outputs.logits[0]  # Shape: (100, 4)
        pred_boxes = outputs.pred_boxes[
            0
        ]  # Shape: (100, 4) in [cx, cy, bw, bh] normalized

        # 3. Softmax over the 3 foreground classes (class 3 is background/no-object in DETR)
        probs = logits.softmax(-1)[:, :3]
        scores, labels = probs.max(-1)

        keep = scores >= self.conf_thres
        if not keep.any():
            return []

        kept_boxes = pred_boxes[keep]
        kept_scores = scores[keep]
        kept_labels = labels[keep]

        # 4. Vectorized conversion from [cx, cy, bw, bh] to pixel coordinates [x1, y1, x2, y2]
        cx = kept_boxes[:, 0] * w
        cy = kept_boxes[:, 1] * h
        bw = kept_boxes[:, 2] * w
        bh = kept_boxes[:, 3] * h

        x1 = (cx - 0.5 * bw).clamp(0, w - 1)
        y1 = (cy - 0.5 * bh).clamp(0, h - 1)
        x2 = (cx + 0.5 * bw).clamp(0, w - 1)
        y2 = (cy + 0.5 * bh).clamp(0, h - 1)

        pixel_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # 5. Non-Maximum Suppression to remove duplicate queries
        nms_indices = ops.nms(pixel_boxes, kept_scores, self.iou_thres)

        detections: List[VehicleDetection] = []
        for idx in nms_indices:
            cls_id = int(kept_labels[idx].item())
            conf_val = float(kept_scores[idx].item())
            box_coords = [int(v.item()) for v in pixel_boxes[idx]]

            cfg = SEVERITY_CONFIG.get(
                cls_id,
                {
                    "raw_name": f"class_{cls_id}",
                    "display_name": f"Class {cls_id}",
                    "short_name": f"C{cls_id}",
                    "color": (200, 200, 200),
                    "is_accident": False,
                    "level": 0,
                },
            )

            detections.append(
                VehicleDetection(
                    bbox=(box_coords[0], box_coords[1], box_coords[2], box_coords[3]),
                    confidence=conf_val,
                    class_id=cls_id,
                    raw_name=cfg["raw_name"],
                    display_name=cfg["display_name"],
                    short_name=cfg["short_name"],
                    color=cfg["color"],
                    is_accident=cfg["is_accident"],
                    severity_level=cfg["level"],
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
