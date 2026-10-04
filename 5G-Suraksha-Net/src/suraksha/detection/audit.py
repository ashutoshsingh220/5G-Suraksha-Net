"""Read-Only False-Positive Diagnostic Auditor for YOLO11s Weapon Detector.

Logs all weapon detections and automatically persists full annotated frames,
bounding box crops, and JSON metadata when confidence >= 0.60.
Preserves model weights, thresholds, and inference logic untouched.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from suraksha.config import PROJECT_ROOT
from suraksha.detection.weapon import RawWeaponDetection, WeaponEvent
from suraksha.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs" / "weapon_training" / "manual_validation" / "false_positive_audit"


class FalsePositiveAuditor:
    """Diagnostic tool that logs detections and preserves high-confidence artifacts for forensic review."""

    def __init__(
        self,
        output_dir: Path | str = DEFAULT_AUDIT_DIR,
        high_conf_thresh: float = 0.60,
        source_name: str = "webcam:0",
        save_high_confidence: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.high_conf_thresh = high_conf_thresh
        self.source_name = source_name
        self.save_high_confidence = save_high_confidence

        self.frames_dir = self.output_dir / "frames"
        self.crops_dir = self.output_dir / "crops"
        self.meta_dir = self.output_dir / "metadata"

        if self.save_high_confidence:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / "audit_log.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.total_frames_processed = 0
        self.total_detections = 0
        self.detections_by_class: dict[str, int] = {"knife": 0, "long_gun": 0, "pistol": 0}
        self.high_conf_detections: list[dict[str, Any]] = []
        self.all_confidences: list[float] = []
        self.leg_long_gun_reproduced = False
        self.start_time = time.time() if "time" in globals() else 0.0

    def record_detections(
        self,
        frame: np.ndarray,
        detections: list[RawWeaponDetection],
        frame_idx: int,
        timestamp: float,
        annotated_frame: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Record and log detections from current frame."""
        self.total_frames_processed += 1
        h, w = frame.shape[:2]
        iso_ts = datetime.now(timezone.utc).astimezone().isoformat()

        saved_records = []

        for det in detections:
            self.total_detections += 1
            if isinstance(det, dict):
                cname = str(det.get("cls_name", ""))
                conf = float(det.get("conf", 0.0))
                bbox_raw = det.get("box", [0.0, 0.0, 0.0, 0.0])
                cls_id = int(det.get("cls_id", 0))
            else:
                cname = det.class_name
                conf = float(det.confidence)
                bbox_raw = det.bbox_xyxy
                cls_id = int(det.cls_id)

            self.detections_by_class[cname] = self.detections_by_class.get(cname, 0) + 1
            self.all_confidences.append(conf)

            x1, y1, x2, y2 = [float(v) for v in bbox_raw]
            bbox_list = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            aspect_ratio = round(bh / bw, 2)
            cy = (y1 + y2) / 2.0
            vertical_pos = "lower_half" if cy > (h / 2.0) else "upper_half"

            # Check if this resembles the leg/long_gun false positive
            is_suspected_leg = (cname == "long_gun" and vertical_pos == "lower_half" and aspect_ratio > 1.8)
            if is_suspected_leg and conf >= 0.50:
                self.leg_long_gun_reproduced = True

            record = {
                "timestamp_iso": iso_ts,
                "timestamp_unix": round(timestamp, 4),
                "frame_idx": frame_idx,
                "source": self.source_name,
                "frame_resolution": [w, h],
                "class_name": cname,
                "cls_id": cls_id,
                "confidence": round(conf, 4),
                "bbox_xyxy": bbox_list,
                "bbox_aspect_ratio_h_w": aspect_ratio,
                "location_in_frame": vertical_pos,
                "suspected_leg_long_gun": is_suspected_leg,
            }

            # 1. Console & JSONL Logging for ANY weapon detection
            log_line = (
                f"[AUDIT] Frame #{frame_idx} (t={timestamp:.2f}s) | "
                f"Class: {cname.upper()} | Conf: {conf:.4f} | "
                f"BBox: {bbox_list} | AR: {aspect_ratio} | Loc: {vertical_pos} | Source: {self.source_name}"
            )
            print(log_line)
            log.info(log_line)

            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                log.warning("Failed to write to audit log: %s", e)

            # 2. When confidence >= 0.60: Save full frame, crop, and metadata
            if self.save_high_confidence and conf >= self.high_conf_thresh:
                prefix = f"frame_{frame_idx:06d}_{cname}_{int(conf * 100):02d}"

                # A. Save full annotated frame
                full_img = annotated_frame.copy() if annotated_frame is not None else frame.copy()
                if annotated_frame is None:
                    # Draw alert box & label
                    color = (0, 0, 255) if cname in ("pistol", "long_gun") else (0, 165, 255)
                    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(full_img, (ix1, iy1), (ix2, iy2), color, 3)
                    lbl = f"[HIGH-CONF AUDIT] {cname.upper()} {conf:.2f}"
                    cv2.rectangle(full_img, (ix1, max(0, iy1 - 25)), (ix1 + len(lbl) * 11, iy1), color, -1)
                    cv2.putText(full_img, lbl, (ix1 + 3, iy1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                frame_path = self.frames_dir / f"{prefix}.jpg"
                cv2.imwrite(str(frame_path), full_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # B. Save cropped image with 15px margin
                pad = 15
                cx1 = max(0, int(x1) - pad)
                cy1 = max(0, int(y1) - pad)
                cx2 = min(w, int(x2) + pad)
                cy2 = min(h, int(y2) + pad)
                crop_img = frame[cy1:cy2, cx1:cx2]

                crop_path = self.crops_dir / f"crop_{prefix}.jpg"
                if crop_img.size > 0:
                    cv2.imwrite(str(crop_path), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # C. Save JSON metadata
                meta_record = dict(record)
                meta_record["full_image_path"] = str(frame_path.relative_to(PROJECT_ROOT))
                meta_record["crop_image_path"] = str(crop_path.relative_to(PROJECT_ROOT))
                meta_path = self.meta_dir / f"meta_{prefix}.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_record, f, indent=2)

                print(f"  --> [HIGH-CONF SAVED] Full frame: {frame_path.name} | Crop: {crop_path.name}")
                self.high_conf_detections.append(meta_record)
                saved_records.append(meta_record)

        return saved_records

    def finalize(self) -> dict[str, Any]:
        """Produce final summary of diagnostic audit."""
        # Confidence distribution breakdown
        c_25_39 = sum(1 for c in self.all_confidences if 0.25 <= c < 0.40)
        c_40_59 = sum(1 for c in self.all_confidences if 0.40 <= c < 0.60)
        c_60_79 = sum(1 for c in self.all_confidences if 0.60 <= c < 0.80)
        c_80_10 = sum(1 for c in self.all_confidences if c >= 0.80)

        summary = {
            "source": self.source_name,
            "total_frames_processed": self.total_frames_processed,
            "total_weapon_detections": self.total_detections,
            "detections_by_class": self.detections_by_class,
            "high_confidence_count": len(self.high_conf_detections),
            "high_confidence_threshold": self.high_conf_thresh,
            "confidence_distribution": {
                "0.25 - 0.39 (low)": c_25_39,
                "0.40 - 0.59 (moderate)": c_40_59,
                "0.60 - 0.79 (high)": c_60_79,
                "0.80 - 1.00 (critical)": c_80_10,
            },
            "mean_confidence": round(float(np.mean(self.all_confidences)), 4) if self.all_confidences else 0.0,
            "max_confidence": round(float(np.max(self.all_confidences)), 4) if self.all_confidences else 0.0,
            "leg_long_gun_reproduced": self.leg_long_gun_reproduced,
            "high_confidence_detections": self.high_conf_detections,
            "audit_directory": str(self.output_dir.relative_to(PROJECT_ROOT)),
        }

        summary_file = self.output_dir / "audit_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("FALSE-POSITIVE DIAGNOSTIC AUDIT SUMMARY")
        print("=" * 70)
        print(f"Total Frames Processed:   {self.total_frames_processed}")
        print(f"Total Weapon Detections:  {self.total_detections}")
        print(f"By Class:                 knife={self.detections_by_class['knife']}, long_gun={self.detections_by_class['long_gun']}, pistol={self.detections_by_class['pistol']}")
        print(f"High-Conf (>= {self.high_conf_thresh:.2f}):    {len(self.high_conf_detections)}")
        print(f"Confidence Distribution:  0.25-0.39: {c_25_39} | 0.40-0.59: {c_40_59} | 0.60-0.79: {c_60_79} | >=0.80: {c_80_10}")
        print(f"Leg/Long-Gun Reproduced:  {self.leg_long_gun_reproduced}")
        print(f"Audit Summary File:       {summary_file}")
        print("=" * 70 + "\n")

        return summary
