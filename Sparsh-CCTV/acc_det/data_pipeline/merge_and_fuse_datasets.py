import os
import sys
import glob
import shutil
import random
import cv2
from pathlib import Path
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acc_det.config import (
    ACC_DET_DIR,
    DETECTION_DATASET_DIR,
    SEVERITY_DATASET_DIR,
    DETECTION_CLASSES,
)

VIDEO_DERIVED_DIR = os.path.join(ACC_DET_DIR, "dataset", "video_derived")
VD_IMAGES_DIR = os.path.join(VIDEO_DERIVED_DIR, "images")
VD_LABELS_DIR = os.path.join(VIDEO_DERIVED_DIR, "labels")


def fuse_datasets():
    print("=" * 65)
    print("   DATASET FUSION & STRATIFIED TRAIN/VAL/TEST MERGE   ")
    print("=" * 65)

    all_vd_images = sorted(glob.glob(os.path.join(VD_IMAGES_DIR, "*.jpg")))
    print(f"[*] Found {len(all_vd_images)} newly mined video-derived images to merge.")

    random.seed(42)
    # Shuffle for stratified split
    shuffled_imgs = list(all_vd_images)
    random.shuffle(shuffled_imgs)

    n_total = len(shuffled_imgs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    n_test = n_total - n_train - n_val

    splits_map = {
        "train": shuffled_imgs[:n_train],
        "val": shuffled_imgs[n_train : n_train + n_val],
        "test": shuffled_imgs[n_train + n_val :],
    }

    print(f"[*] Merging Splits: Train={n_train}, Val={n_val}, Test={n_test}")

    new_severity_crops = {"train": 0, "val": 0}

    for split_name, img_list in splits_map.items():
        dst_img_dir = os.path.join(DETECTION_DATASET_DIR, "images", split_name)
        dst_lbl_dir = os.path.join(DETECTION_DATASET_DIR, "labels", split_name)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for img_path in tqdm(img_list, desc=f"Merging {split_name}"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(VD_LABELS_DIR, f"{base_name}.txt")

            # Copy image
            shutil.copy2(img_path, os.path.join(dst_img_dir, f"{base_name}.jpg"))

            # Copy label
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, f"{base_name}.txt"))

                # Also extract accident crops for severity dataset if in train or val
                if split_name in ("train", "val"):
                    with open(lbl_path, "r") as f:
                        lines = [l.strip().split() for l in f.readlines() if l.strip()]
                    accident_boxes = [l for l in lines if l[0] == "1"]

                    if accident_boxes:
                        img = cv2.imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            for c_idx, box in enumerate(accident_boxes):
                                _, xc, yc, bw, bh = [float(v) for v in box]
                                # Heuristic severity classification for crop
                                is_severe = len(accident_boxes) >= 2 or (bw * bh > 0.18)
                                sev_label = "severe" if is_severe else "moderate"

                                pad_w = int(bw * 0.10 * w)
                                pad_h = int(bh * 0.10 * h)
                                x1 = max(0, int((xc - bw / 2.0) * w) - pad_w)
                                y1 = max(0, int((yc - bh / 2.0) * h) - pad_h)
                                x2 = min(w, int((xc + bw / 2.0) * w) + pad_w)
                                y2 = min(h, int((yc + bh / 2.0) * h) + pad_h)

                                if x2 - x1 >= 15 and y2 - y1 >= 15:
                                    crop = img[y1:y2, x1:x2]
                                    crop_res = cv2.resize(crop, (224, 224))
                                    crop_out = os.path.join(
                                        SEVERITY_DATASET_DIR,
                                        split_name,
                                        sev_label,
                                        f"vd_{base_name}_c{c_idx}.jpg",
                                    )
                                    cv2.imwrite(
                                        crop_out,
                                        crop_res,
                                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                                    )
                                    new_severity_crops[split_name] += 1

    # Count final dataset totals
    train_total = len(
        glob.glob(os.path.join(DETECTION_DATASET_DIR, "images", "train", "*.jpg"))
    )
    val_total = len(
        glob.glob(os.path.join(DETECTION_DATASET_DIR, "images", "val", "*.jpg"))
    )
    test_total = len(
        glob.glob(os.path.join(DETECTION_DATASET_DIR, "images", "test", "*.jpg"))
    )

    print("\n" + "=" * 65)
    print("   DATASET FUSION COMPLETED SUCCESSFULLY!   ")
    print("=" * 65)
    print(f"Total Fused Train Images : {train_total}")
    print(f"Total Fused Val Images   : {val_total}")
    print(f"Total Fused Test Images  : {test_total}")
    print(f"Grand Total Detection Set: {train_total + val_total + test_total} images")
    print(f"New Severity Crops Added : {new_severity_crops}")
    print("=" * 65)


if __name__ == "__main__":
    fuse_datasets()
