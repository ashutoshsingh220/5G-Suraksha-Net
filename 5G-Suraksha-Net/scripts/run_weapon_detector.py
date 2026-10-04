#!/usr/bin/env python
"""Standalone Manual Live Weapon Detector Tool.

Evaluates and visualizes raw predictions from the trained YOLO11s model:
outputs/weapon_training/best.pt

Supports 4 sources:
1. Video file (MP4, AVI, MKV, etc.)
2. Laptop webcam (cv2.CAP_DSHOW on Windows)
3. Raspberry Pi C270 RTSP (low-latency TCP, auto-reconnecting)
4. Generic / Sparsh RTSP camera

Live HUD Overlay:
- Rolling FPS
- Inference latency (ms)
- Source identifier
- Total weapon count in current frame
- Color-coded bounding boxes per class (knife, long_gun, pistol)
- Class name & confidence score percentage

Usage:
    # 1. MP4 video file
    python scripts/run_weapon_detector.py --source file --path datasets/videos/external/cctv_test_01.mp4 --save-video outputs/weapon_training/manual_validation/annotated_cctv.mp4

    # 2. Laptop webcam
    python scripts/run_weapon_detector.py --source webcam --camera-index 0 --conf 0.25

    # 3. Raspberry Pi C270 RTSP
    python scripts/run_weapon_detector.py --source rtsp --path rtsp://10.254.18.48:8554/drone

    # 4. Generic / Sparsh RTSP
    python scripts/run_weapon_detector.py --source rtsp --path rtsp://user:pass@192.168.1.100:554/live
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from suraksha.detection.audit import FalsePositiveAuditor

DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "weapon_training" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "weapon_training" / "manual_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors in BGR
CLASS_COLORS = {
    0: (255, 200, 0),    # Knife: Bright Cyan/Yellow (BGR: Blue=0, Green=200, Red=255 -> Amber/Orange)
    1: (0, 69, 255),     # Long Gun: Red-Orange (BGR)
    2: (255, 0, 180),    # Pistol: Magenta/Purple (BGR)
}
CLASS_NAMES = {
    0: "knife",
    1: "long_gun",
    2: "pistol",
}


def safe_win_path(p: str | Path) -> str:
    s = str(Path(p).resolve())
    if not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s.replace("/", "\\")
    return s


class WeaponDetectorViewer:
    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "0",
    ):
        self.weights_path = Path(weights_path)
        assert self.weights_path.exists(), f"Model weights not found: {self.weights_path}"
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device if torch.cuda.is_available() else "cpu"

        print(f"Loading YOLO11s weapon detector from: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        print(f"Model initialized on device: {self.device}")

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            imgsz=640,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for b, c, cl in zip(xyxy, conf, cls_ids):
                    detections.append({
                        "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "conf": float(c),
                        "cls_id": int(cl),
                        "class_name": CLASS_NAMES.get(int(cl), f"weapon_{cl}"),
                    })

        return detections, latency_ms

    def render_hud(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float,
        latency_ms: float,
        source_name: str,
        frame_idx: int,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # 1. Draw Bounding Boxes & Labels
        num_weapons = len(detections)
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cid = det["cls_id"]
            cname = det["class_name"]
            score = det["conf"]
            color = CLASS_COLORS.get(cid, (0, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{cname.upper()} {score*100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bg_y1 = max(0, y1 - th - baseline - 4)
            bg_y2 = y1
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 8, bg_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, bg_y2 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Top HUD Banner
        hud_height = 42
        cv2.rectangle(annotated, (0, 0), (w, hud_height), (20, 20, 20), -1)
        cv2.line(annotated, (0, hud_height), (w, hud_height), (80, 80, 80), 1)

        # HUD Left: Model & Source
        source_text = f"5G Suraksha-Net | YOLO11s | {source_name}"
        cv2.putText(annotated, source_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # HUD Center: Performance
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        perf_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | VRAM: {vram_mb:.0f}MB"
        (ptw, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(annotated, perf_text, (w // 2 - ptw // 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)

        # HUD Right: Status Badge
        if num_weapons > 0:
            badge_text = f"! WEAPON DETECTED: {num_weapons} !"
            badge_color = (0, 0, 255)  # Bright Red Alert
            text_color = (255, 255, 255)
        else:
            badge_text = "SURVEILLANCE SECURE"
            badge_color = (40, 160, 40)  # Green
            text_color = (255, 255, 255)

        (btw, bth), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        badge_x1 = w - btw - 20
        cv2.rectangle(annotated, (badge_x1, 6), (w - 10, hud_height - 6), badge_color, -1)
        cv2.putText(annotated, badge_text, (badge_x1 + 5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 2, cv2.LINE_AA)

        return annotated


def create_capture(source_type: str, path: str, camera_index: int) -> Tuple[cv2.VideoCapture, str]:
    if source_type == "webcam":
        print(f"Connecting to webcam index {camera_index} (DirectShow)...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("DirectShow failed; falling back to CAP_ANY...")
            cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        source_name = f"Webcam ({camera_index})"
    elif source_type == "rtsp":
        print(f"Connecting to RTSP stream: {path} (low-latency TCP)...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;500000|probesize;500000"
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        # Label source cleanly
        if "10.254.18.48" in path:
            source_name = "Raspberry Pi C270 RTSP"
        elif "sparsh" in path.lower():
            source_name = "Sparsh CCTV RTSP"
        else:
            source_name = f"RTSP Stream ({path[:25]}...)"
    else:  # file
        print(f"Opening video file: {path}...")
        p = Path(path)
        assert p.exists(), f"Video file not found: {p}"
        cap = cv2.VideoCapture(str(p), cv2.CAP_ANY)
        source_name = f"File: {p.name}"

    return cap, source_name


def run_detector(
    source_type: str,
    path: str = "",
    camera_index: int = 0,
    weights_path: Path = DEFAULT_WEIGHTS,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    display: bool = True,
    save_video_path: Optional[str] = None,
    save_json_path: Optional[str] = None,
    save_snapshots_dir: Optional[str] = None,
    max_frames: int = 0,
    save_high_confidence: bool = False,
) -> Dict[str, Any]:
    print("=" * 70)
    print(f"STARTING WEAPON DETECTOR | SOURCE: {source_type.upper()}")
    print("=" * 70)

    viewer = WeaponDetectorViewer(
        weights_path=weights_path,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
    )

    cap, source_name = create_capture(source_type, path, camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open capture source: {source_type} ({path})")
        err_res = {
            "source": source_name,
            "source_type": source_type,
            "success": False,
            "error": "Failed to open capture source",
        }
        if save_json_path:
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(err_res, f, indent=2)
        return err_res

    auditor = None
    if save_high_confidence:
        auditor = FalsePositiveAuditor(source_name=source_name, save_high_confidence=True)
        print(f"[AUDIT] False-positive diagnostic auditor enabled for source '{source_name}'.")
        print(f"[AUDIT] Detections will be logged to {auditor.log_file}")
        print(f"[AUDIT] High-confidence (>= 0.60) artifacts will be saved to {auditor.output_dir}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Source opened successfully:")
    print(f"  Name: {source_name}")
    print(f"  Native Resolution: {frame_width}x{frame_height}")
    print(f"  Reported FPS: {source_fps:.1f}")

    # Video writer setup if requested
    writer = None
    if save_video_path:
        save_p = Path(save_video_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_p), fourcc, max(15.0, min(source_fps, 30.0)), (frame_width, frame_height))
        print(f"Saving annotated video to: {save_p}")

    snapshot_p = None
    snapshot_count = 0
    if save_snapshots_dir:
        snapshot_p = Path(save_snapshots_dir)
        snapshot_p.mkdir(parents=True, exist_ok=True)

    window_name = f"5G Suraksha-Net — YOLO11s Weapon Detector [{source_name}]"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, max(640, frame_width)), min(720, max(360, frame_height)))

    frame_idx = 0
    total_detections_count = 0
    detections_by_class = {0: 0, 1: 0, 2: 0}
    fps_rolling = 0.0
    latencies = []
    t_start = time.perf_counter()

    print("\nRunning inference. Press 'q' or ESC in window to exit...\n")
    try:
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                if source_type == "file":
                    print("\nReached end of video file.")
                    break
                else:
                    print("Stream frame drop, attempting reconnect...")
                    time.sleep(0.5)
                    continue

            frame_idx += 1

            # Detect
            dets, latency_ms = viewer.detect(frame)
            latencies.append(latency_ms)

            for d in dets:
                detections_by_class[d["cls_id"]] += 1
                total_detections_count += 1

            # Compute rolling FPS
            dt = time.perf_counter() - t_frame_start
            inst_fps = 1.0 / max(dt, 1e-4)
            fps_rolling = 0.9 * fps_rolling + 0.1 * inst_fps if fps_rolling > 0 else inst_fps

            # Render HUD
            annotated_frame = viewer.render_hud(
                frame=frame,
                detections=dets,
                fps=fps_rolling,
                latency_ms=latency_ms,
                source_name=source_name,
                frame_idx=frame_idx,
            )

            if writer is not None:
                writer.write(annotated_frame)

            if auditor is not None and len(dets) > 0:
                auditor.record_detections(
                    frame=frame,
                    detections=dets,
                    frame_idx=frame_idx,
                    timestamp=time.time(),
                    annotated_frame=annotated_frame,
                )

            if snapshot_p is not None:
                should_save = False
                if len(dets) > 0 and snapshot_count < 10:
                    should_save = True
                    snapshot_count += 1
                elif frame_idx in (1, 25, 50, 100) and snapshot_count < 10:
                    should_save = True
                    snapshot_count += 1
                if should_save:
                    snap_file = snapshot_p / f"frame_{frame_idx:04d}_dets_{len(dets)}.jpg"
                    cv2.imwrite(str(snap_file), annotated_frame)

            if display:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("\nUser requested exit.")
                    break

            if max_frames > 0 and frame_idx >= max_frames:
                print(f"\nReached max frames limit: {max_frames}")
                break

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx:5d} | FPS: {fps_rolling:5.1f} | Latency: {latency_ms:5.1f}ms | Detections: {len(dets)}")

    finally:
        total_time = time.perf_counter() - t_start
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        if auditor is not None:
            auditor.finalize()

    avg_fps = round(frame_idx / max(total_time, 1e-4), 1)
    avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0
    p95_latency = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

    print("\n" + "=" * 70)
    print("SESSION SUMMARY:")
    print(f"  Source:             {source_name}")
    print(f"  Resolution:         {frame_width}x{frame_height}")
    print(f"  Frames Processed:   {frame_idx}")
    print(f"  Elapsed Time:       {total_time:.2f}s")
    print(f"  Average FPS:        {avg_fps}")
    print(f"  Mean Latency:       {avg_latency} ms")
    print(f"  P95 Latency:        {p95_latency} ms")
    print(f"  Peak VRAM:          {vram_mb} MB")
    print(f"  Total Detections:   {total_detections_count}")
    print(f"  By Class:           knife={detections_by_class[0]}, long_gun={detections_by_class[1]}, pistol={detections_by_class[2]}")
    print("=" * 70)

    res = {
        "source": source_name,
        "source_type": source_type,
        "success": True,
        "resolution": f"{frame_width}x{frame_height}",
        "frames_processed": frame_idx,
        "elapsed_sec": round(total_time, 2),
        "avg_fps": avg_fps,
        "mean_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "peak_vram_mb": vram_mb,
        "total_detections": total_detections_count,
        "detections_by_class": {
            "knife": detections_by_class[0],
            "long_gun": detections_by_class[1],
            "pistol": detections_by_class[2],
        },
        "saved_video": save_video_path,
    }

    if save_json_path:
        json_p = Path(save_json_path)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Session summary saved to JSON: {json_p}")

    return res


def main():
    parser = argparse.ArgumentParser(description="5G Suraksha-Net — Real-Time Weapon Detector Viewer")
    parser.add_argument(
        "--source",
        choices=["file", "webcam", "rtsp"],
        required=True,
        help="Input source type: file | webcam | rtsp",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to video file or RTSP stream URL",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam camera index (default: 0)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=True,
        help="Show live annotated video window (default: True)",
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Run headless without opening a GUI window",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default="",
        help="Optional path to save annotated output video (.mp4)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save summary metrics (.json)",
    )
    parser.add_argument(
        "--save-snapshots",
        type=str,
        default="",
        help="Optional directory path to save sample annotated snapshots",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = unlimited)",
    )
    parser.add_argument(
        "--save-high-confidence",
        action="store_true",
        help="enable read-only false-positive auditor (saves frame/crop/metadata for conf >= 0.60)",
    )

    args = parser.parse_args()

    run_detector(
        source_type=args.source,
        path=args.path,
        camera_index=args.camera_index,
        weights_path=Path(args.weights),
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        display=args.display,
        save_video_path=args.save_video if args.save_video else None,
        save_json_path=args.save_json if args.save_json else None,
        save_snapshots_dir=args.save_snapshots if args.save_snapshots else None,
        max_frames=args.max_frames,
        save_high_confidence=args.save_high_confidence,
    )


if __name__ == "__main__":
    main()
