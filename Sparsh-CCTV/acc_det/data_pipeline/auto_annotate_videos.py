import os
import sys
import glob
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acc_det.config import ROOT_DIR, ACC_DET_DIR, TIER1_BEST_DETECTOR, DEVICE

VIDEO_DERIVED_DIR = os.path.join(ACC_DET_DIR, "dataset", "video_derived")
IMAGES_DIR = os.path.join(VIDEO_DERIVED_DIR, "images")
LABELS_DIR = os.path.join(VIDEO_DERIVED_DIR, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


def convert_xyxy_to_yolo(xyxy, img_w, img_h):
    """Converts pixel [x1, y1, x2, y2] to normalized [xc, yc, nw, nh]."""
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(img_w - 1), float(x1)))
    y1 = max(0.0, min(float(img_h - 1), float(y1)))
    x2 = max(x1 + 1.0, min(float(img_w), float(x2)))
    y2 = max(y1 + 1.0, min(float(img_h), float(y2)))

    bw = x2 - x1
    bh = y2 - y1
    if bw < 5.0 or bh < 5.0:
        return None

    xc = (x1 + bw / 2.0) / float(img_w)
    yc = (y1 + bh / 2.0) / float(img_h)
    nw = bw / float(img_w)
    nh = bh / float(img_h)

    xc = max(0.001, min(0.999, xc))
    yc = max(0.001, min(0.999, yc))
    nw = max(0.005, min(0.999, nw))
    nh = max(0.005, min(0.999, nh))

    return xc, yc, nw, nh


def auto_annotate_all(
    accident_dir: str,
    no_accident_dir: str,
    stride: int = 18,
    max_frames_per_video: int = 35,
    conf_thres: float = 0.38,
):
    print("=" * 65)
    print("   AUTOMATED SURVEILLANCE VIDEO ANNOTATION PIPELINE   ")
    print("=" * 65)
    print(f"[*] Teacher Model   : {TIER1_BEST_DETECTOR}")
    print(f"[*] Target Stride   : 1 frame every {stride} frames (~1 fps)")
    print(f"[*] Max Frames/Vid  : {max_frames_per_video}")
    print(f"[*] Confidence Gate : {conf_thres}")
    print(f"[*] Output Directory: {VIDEO_DERIVED_DIR}")
    print("=" * 65)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    teacher = YOLO(TIER1_BEST_DETECTOR)

    acc_videos = sorted(glob.glob(os.path.join(accident_dir, "*.mp4")))
    noacc_videos = sorted(glob.glob(os.path.join(no_accident_dir, "*.mp4")))

    print(f"[+] Found {len(acc_videos)} Accident Videos")
    print(f"[+] Found {len(noacc_videos)} No-Accident Videos")

    total_images_saved = 0
    total_accident_boxes = 0
    total_normal_boxes = 0
    total_bg_frames = 0

    # ---------------- 1. Process 57 Accident Videos ----------------
    print(
        "\n[Phase 1/2] Processing Accident Videos (Temporal Incident Localization)..."
    )
    for vid_path in tqdm(acc_videos, desc="Accident Videos"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # First pass: Sample frames and scan for temporal incident window
        sampled_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or len(sampled_frames) >= max_frames_per_video:
                break
            if frame_idx % stride == 0:
                sampled_frames.append((frame_idx, frame))
            frame_idx += 1
        cap.release()

        if not sampled_frames:
            continue

        # Run teacher detector on sampled frames
        frame_dets = []
        for f_idx, frame in sampled_frames:
            results = teacher(frame, conf=conf_thres, device=device, verbose=False)
            has_acc = False
            boxes = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    cls_id = int(b.cls[0].item())
                    score = float(b.conf[0].item())
                    xyxy = b.xyxy[0].tolist()
                    boxes.append((cls_id, score, xyxy))
                    if cls_id == 1 and score >= conf_thres:
                        has_acc = True
            frame_dets.append(
                {"f_idx": f_idx, "frame": frame, "boxes": boxes, "has_acc": has_acc}
            )

        # Temporal persistence filter: find accident cluster
        acc_indices = [i for i, d in enumerate(frame_dets) if d["has_acc"]]
        # Active incident window spans from first detected accident to last + 1
        start_incident_idx = min(acc_indices) if acc_indices else -1
        end_incident_idx = max(acc_indices) if acc_indices else -1

        for i, d in enumerate(frame_dets):
            frame = d["frame"]
            f_idx = d["f_idx"]
            is_in_incident_window = (
                start_incident_idx >= 0
                and start_incident_idx <= i <= end_incident_idx + 1
            )

            # Generate YOLO annotations
            yolo_lines = []
            img_saved_name = f"cctv_acc_{vid_name}_f{f_idx:05d}.jpg"
            lbl_saved_name = f"cctv_acc_{vid_name}_f{f_idx:05d}.txt"

            for cls_id, score, xyxy in d["boxes"]:
                coords = convert_xyxy_to_yolo(xyxy, w, h)
                if not coords:
                    continue
                xc, yc, nw, nh = coords

                if is_in_incident_window:
                    # In incident window: retain both accident and normal vehicles
                    if cls_id == 1 and score >= conf_thres:
                        yolo_lines.append(f"1 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                        total_accident_boxes += 1
                    elif cls_id == 0 and score >= (conf_thres - 0.05):
                        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                        total_normal_boxes += 1
                else:
                    # Before or after incident window: all vehicles are normal traffic
                    if score >= (conf_thres - 0.05):
                        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                        total_normal_boxes += 1

            # Save 640x640 letterbox/resized image
            resized_frame = cv2.resize(frame, (640, 640))
            img_out = os.path.join(IMAGES_DIR, img_saved_name)
            lbl_out = os.path.join(LABELS_DIR, lbl_saved_name)

            cv2.imwrite(img_out, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(lbl_out, "w") as f:
                f.writelines(yolo_lines)

            total_images_saved += 1
            if len(yolo_lines) == 0:
                total_bg_frames += 1

    # ---------------- 2. Process 73 No-Accident Videos ----------------
    print(
        "\n[Phase 2/2] Processing No-Accident Videos (Zero Leakage Vehicle & Negative Harvest)..."
    )
    for vid_path in tqdm(noacc_videos, desc="No-Accident Videos"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        saved_vid_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret or saved_vid_frames >= max_frames_per_video:
                break

            if frame_idx % (stride + 2) == 0:
                results = teacher(
                    frame, conf=(conf_thres - 0.05), device=device, verbose=False
                )
                yolo_lines = []

                if results and len(results) > 0 and results[0].boxes is not None:
                    for b in results[0].boxes:
                        score = float(b.conf[0].item())
                        xyxy = b.xyxy[0].tolist()
                        coords = convert_xyxy_to_yolo(xyxy, w, h)
                        if coords and score >= (conf_thres - 0.05):
                            xc, yc, nw, nh = coords
                            # In no-accident videos, ALL detected vehicles are strictly class 0 (normal)
                            # Zero accident leakage guarantee!
                            yolo_lines.append(
                                f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
                            )
                            total_normal_boxes += 1

                img_saved_name = f"cctv_noacc_{vid_name}_f{frame_idx:05d}.jpg"
                lbl_saved_name = f"cctv_noacc_{vid_name}_f{frame_idx:05d}.txt"

                resized_frame = cv2.resize(frame, (640, 640))
                img_out = os.path.join(IMAGES_DIR, img_saved_name)
                lbl_out = os.path.join(LABELS_DIR, lbl_saved_name)

                cv2.imwrite(img_out, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                with open(lbl_out, "w") as f:
                    f.writelines(yolo_lines)

                total_images_saved += 1
                if len(yolo_lines) == 0:
                    total_bg_frames += 1
                saved_vid_frames += 1

            frame_idx += 1
        cap.release()

    print("\n" + "=" * 65)
    print("   AUTO-ANNOTATION COMPLETED SUCCESSFULLY   ")
    print("=" * 65)
    print(f"Total Frames Auto-Annotated : {total_images_saved}")
    print(f"Accident Bounding Boxes     : {total_accident_boxes}")
    print(f"Normal Vehicle Boxes        : {total_normal_boxes}")
    print(
        f"Pure Background Negatives   : {total_bg_frames} ({total_bg_frames / max(1, total_images_saved):.1%})"
    )
    print("=" * 65)

    return {
        "images": total_images_saved,
        "accidents": total_accident_boxes,
        "normal": total_normal_boxes,
        "background": total_bg_frames,
    }


def main():
    acc_dir = os.path.join(ROOT_DIR, "data", "videos", "accident")
    noacc_dir = os.path.join(ROOT_DIR, "data", "videos", "no accident")
    auto_annotate_all(
        accident_dir=acc_dir,
        no_accident_dir=noacc_dir,
        stride=18,
        max_frames_per_video=35,
        conf_thres=0.38,
    )


if __name__ == "__main__":
    main()
