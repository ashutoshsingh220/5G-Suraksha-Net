import os
import sys
import cv2
import glob
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ROOT_DIR, DETECTION_DATASET_DIR, SEVERITY_DATASET_DIR


def ensure_severity_dirs():
    """Ensure directory structure for severity classification."""
    for split in ["train", "val"]:
        for cls_name in ["moderate", "severe"]:
            os.makedirs(
                os.path.join(SEVERITY_DATASET_DIR, split, cls_name), exist_ok=True
            )


def extract_crops_from_detection_dataset():
    """
    Extracts accident bounding box crops from the detection dataset and classifies
    them into moderate vs severe based on geometric signatures, vehicle overlap,
    and area metrics.
    """
    ensure_severity_dirs()
    print(
        "\n[Severity Curation] Extracting accident ROI crops from detection dataset..."
    )

    crop_counts = {
        "train": {"moderate": 0, "severe": 0},
        "val": {"moderate": 0, "severe": 0},
    }

    for split in ["train", "val"]:
        img_dir = os.path.join(DETECTION_DATASET_DIR, "images", split)
        lbl_dir = os.path.join(DETECTION_DATASET_DIR, "labels", split)

        if not os.path.exists(img_dir):
            continue

        img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
        for img_path in tqdm(img_files, desc=f"Processing {split} crops"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r") as f:
                lines = [l.strip().split() for l in f.readlines() if l.strip()]

            # Filter for accident boxes (class 1)
            accident_boxes = [l for l in lines if l[0] == "1"]
            if not accident_boxes:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            num_accidents = len(accident_boxes)
            # If multiple accident boxes in the same frame -> multi-vehicle pileup (Severe)
            is_multi_vehicle = num_accidents >= 2

            for idx, box in enumerate(accident_boxes):
                _, xc, yc, bw, bh = [float(v) for v in box]
                box_area = bw * bh
                aspect_ratio = bw / max(0.01, bh)

                # Heuristic categorization:
                # Severe: Large area (>0.25 of frame), abnormal aspect ratio (rollover), or multi-vehicle collision
                # Moderate: Moderate area, single lane/fender impact
                is_severe = (
                    is_multi_vehicle
                    or box_area > 0.22
                    or aspect_ratio < 0.6
                    or aspect_ratio > 2.5
                )
                severity_label = "severe" if is_severe else "moderate"

                # Convert normalized coords to pixel coords with 10% context padding
                pad_w = bw * 0.1 * w
                pad_h = bh * 0.1 * h

                x1 = int(max(0, (xc - bw / 2.0) * w - pad_w))
                y1 = int(max(0, (yc - bh / 2.0) * h - pad_h))
                x2 = int(min(w, (xc + bw / 2.0) * w + pad_w))
                y2 = int(min(h, (yc + bh / 2.0) * h + pad_h))

                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                crop = img[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (224, 224))

                crop_filename = f"{base_name}_crop{idx}.jpg"
                out_path = os.path.join(
                    SEVERITY_DATASET_DIR, split, severity_label, crop_filename
                )
                cv2.imwrite(out_path, crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

                crop_counts[split][severity_label] += 1

    print(f"[+] Extracted detection dataset crops: {crop_counts}")
    return crop_counts


def extract_crops_from_cctv_clips():
    """
    Extracts annotated crops from local CCTV accident videos:
    - crash.mp4: Severe T-bone collision (Severe)
    - crash2.mp4: Multi-impact highway crash (Severe)
    - bikeacc.mp4: Motorcycle slide / skid (Moderate)
    - realcrash1.mp4: Low-speed urban intersection crash (Moderate)
    """
    videos_dir = os.path.join(ROOT_DIR, "data", "videos")
    clips_config = [
        {"name": "crash.mp4", "label": "severe", "split": "train", "stride": 4},
        {"name": "crash2.mp4", "label": "severe", "split": "val", "stride": 4},
        {"name": "bikeacc.mp4", "label": "moderate", "split": "train", "stride": 6},
        {"name": "realcrash1.mp4", "label": "moderate", "split": "val", "stride": 4},
    ]

    print("\n[Severity Curation] Extracting crops from verified CCTV clips...")
    mined_clip_crops = 0

    for cfg in clips_config:
        vid_path = os.path.join(videos_dir, cfg["name"])
        if not os.path.exists(vid_path):
            continue

        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret or saved >= 40:
                break

            if frame_idx % cfg["stride"] == 0:
                h, w = frame.shape[:2]
                # Center-focus crop around typical CCTV collision ROI (60% central region)
                cx1, cy1 = int(w * 0.2), int(h * 0.2)
                cx2, cy2 = int(w * 0.8), int(h * 0.8)
                crop = frame[cy1:cy2, cx1:cx2]
                resized = cv2.resize(crop, (224, 224))

                filename = f"clip_{os.path.splitext(cfg['name'])[0]}_{saved:03d}.jpg"
                out_path = os.path.join(
                    SEVERITY_DATASET_DIR, cfg["split"], cfg["label"], filename
                )
                cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

                saved += 1
                mined_clip_crops += 1

            frame_idx += 1

        cap.release()
        print(
            f"  [+] Added {saved} {cfg['label']} crops from {cfg['name']} ({cfg['split']})"
        )

    print(f"[+] Total video clip severity crops added: {mined_clip_crops}")
    return mined_clip_crops


def main():
    print("=" * 65)
    print("   ACCIDENT SEVERITY (MODERATE vs SEVERE) DATASET CREATION   ")
    print("=" * 65)
    ensure_severity_dirs()
    extract_crops_from_detection_dataset()
    extract_crops_from_cctv_clips()
    print("=" * 65)


if __name__ == "__main__":
    main()
