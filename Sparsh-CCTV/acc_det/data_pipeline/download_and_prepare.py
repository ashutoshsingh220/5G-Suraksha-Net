import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import polars as pl
from huggingface_hub import hf_hub_download

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    DETECTION_DATASET_DIR,
    HF_DATASET_CCTV,
    HF_DATASET_TRAFFIC,
    DETECTION_CLASSES,
)


def ensure_dirs():
    """Create directory structure for YOLO detection dataset."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DETECTION_DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DETECTION_DATASET_DIR, "labels", split), exist_ok=True)


def convert_coco_to_yolo_bbox(bbox, img_w=640.0, img_h=640.0):
    """
    Converts [x_min, y_min, w, h] to YOLO normalized [x_c, y_c, nw, nh].
    Returns None if box is invalid or out of bounds.
    """
    if len(bbox) != 4:
        return None
    x_min, y_min, w, h = bbox
    if w <= 1.0 or h <= 1.0:
        return None

    x_c = (x_min + w / 2.0) / img_w
    y_c = (y_min + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    # Clamp to [0, 1]
    x_c = max(0.001, min(0.999, x_c))
    y_c = max(0.001, min(0.999, y_c))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))

    return x_c, y_c, nw, nh


def process_cctv_dataset(max_train=2500, max_val=350, max_test=350):
    """
    Downloads and extracts 'justjuu/traffic-accident-cctv-object-detection'.
    Classes in dataset:
      0: accident     -> mapped to class 1 ('accident')
      1: non_accident -> mapped to class 0 ('normal')
    """
    print(f"\n[1/2] Processing CCTV Accident Dataset ({HF_DATASET_CCTV})...")
    files_map = {
        "train": [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ],
        "val": ["data/validation-00000-of-00001.parquet"],
        "test": ["data/test-00000-of-00001.parquet"],
    }
    split_limits = {"train": max_train, "val": max_val, "test": max_test}

    total_saved = {"train": 0, "val": 0, "test": 0}

    for split, parquet_files in files_map.items():
        limit = split_limits[split]
        for pfile in parquet_files:
            if total_saved[split] >= limit:
                break
            print(f"Downloading {pfile} for split: {split}...")
            local_path = hf_hub_download(
                repo_id=HF_DATASET_CCTV, filename=pfile, repo_type="dataset"
            )
            df = pl.read_parquet(local_path, columns=["image", "objects"])
            print(f"Loaded {len(df)} records. Extracting to {split}...")

            for i in tqdm(range(len(df)), desc=f"Extracting {split}"):
                if total_saved[split] >= limit:
                    break
                row = df[i]
                img_data = row["image"][0]
                img_bytes = img_data.get("bytes")
                if not img_bytes:
                    continue

                img_filename = f"cctv_{split}_{total_saved[split]:05d}.jpg"
                lbl_filename = f"cctv_{split}_{total_saved[split]:05d}.txt"

                img_path = os.path.join(
                    DETECTION_DATASET_DIR, "images", split, img_filename
                )
                lbl_path = os.path.join(
                    DETECTION_DATASET_DIR, "labels", split, lbl_filename
                )

                # Write image
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # Parse bboxes
                objs = row["objects"][0]
                bboxes = objs.get("bbox", [])
                categories = objs.get("category", [])

                lines = []
                for bbox, cat in zip(bboxes, categories):
                    # In justjuu: 0 = accident, 1 = non_accident (normal)
                    # Unified: 0 = normal, 1 = accident
                    unified_cls = 1 if cat == 0 else 0
                    yolo_box = convert_coco_to_yolo_bbox(bbox, 640.0, 640.0)
                    if yolo_box:
                        x_c, y_c, nw, nh = yolo_box
                        lines.append(
                            f"{unified_cls} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}\n"
                        )

                # Write label file (empty if no valid boxes, serving as negative background)
                with open(lbl_path, "w") as f:
                    f.writelines(lines)

                total_saved[split] += 1

    print(f"[+] CCTV Dataset complete: {total_saved}")
    return total_saved


def process_traffic_dataset(max_images=3000):
    """
    Downloads and extracts 'hiennguyen9874/traffic-accident-detection'.
    Contains accident images and negative background samples.
    Category 0 = accident -> mapped to class 1 ('accident').
    Splits into 80% train, 10% val, 10% test.
    """
    print(f"\n[2/2] Processing Traffic Accident Dataset ({HF_DATASET_TRAFFIC})...")
    local_path = hf_hub_download(
        repo_id=HF_DATASET_TRAFFIC,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )

    df = pl.read_parquet(
        local_path, columns=["image", "objects", "width", "height", "is_accident"]
    )
    total_len = min(len(df), max_images)
    print(
        f"Loaded {len(df)} records. Extracting up to {total_len} samples with 80/10/10 split..."
    )

    counts = {"train": 0, "val": 0, "test": 0}

    for i in tqdm(range(total_len), desc="Extracting Traffic Dataset"):
        # Deterministic 80/10/10 split based on index
        idx_mod = i % 10
        if idx_mod < 8:
            split = "train"
        elif idx_mod == 8:
            split = "val"
        else:
            split = "test"

        row = df[i]
        img_data = row["image"][0]
        img_bytes = img_data.get("bytes")
        if not img_bytes:
            continue

        img_w = float(row["width"][0]) if row["width"][0] else 640.0
        img_h = float(row["height"][0]) if row["height"][0] else 640.0

        img_filename = f"traffic_{counts[split]:05d}.jpg"
        lbl_filename = f"traffic_{counts[split]:05d}.txt"

        img_path = os.path.join(DETECTION_DATASET_DIR, "images", split, img_filename)
        lbl_path = os.path.join(DETECTION_DATASET_DIR, "labels", split, lbl_filename)

        with open(img_path, "wb") as f:
            f.write(img_bytes)

        objs = row["objects"][0]
        bboxes = objs.get("bbox", [])
        categories = objs.get("category", [])

        lines = []
        if bboxes is not None:
            for bbox, cat in zip(bboxes, categories):
                unified_cls = 1  # accident
                yolo_box = convert_coco_to_yolo_bbox(bbox, img_w, img_h)
                if yolo_box:
                    x_c, y_c, nw, nh = yolo_box
                    lines.append(
                        f"{unified_cls} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}\n"
                    )

        with open(lbl_path, "w") as f:
            f.writelines(lines)

        counts[split] += 1

    print(f"[+] Traffic Dataset complete: {counts}")
    return counts


def create_data_yaml():
    """Generates data.yaml for Ultralytics YOLO training."""
    yaml_path = os.path.join(DETECTION_DATASET_DIR, "data.yaml")
    data = {
        "path": Path(DETECTION_DATASET_DIR).resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": DETECTION_CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"\n[+] Created YOLO dataset YAML: {yaml_path}")
    print(f"    Classes: {DETECTION_CLASSES}")


def main():
    print("=" * 65)
    print("   SURVEILLANCE ACCIDENT DETECTION - DATASET INGESTION   ")
    print("=" * 65)
    ensure_dirs()
    cctv_counts = process_cctv_dataset(max_train=2200, max_val=320, max_test=320)
    traffic_counts = process_traffic_dataset(max_images=2800)
    create_data_yaml()

    total_train = cctv_counts["train"] + traffic_counts["train"]
    total_val = cctv_counts["val"] + traffic_counts["val"]
    total_test = cctv_counts["test"] + traffic_counts["test"]

    print("\n" + "=" * 65)
    print(f"   DATASET PREPARATION COMPLETED SUCCESSFULLY!   ")
    print(f"   Train Images: {total_train}")
    print(f"   Val Images  : {total_val}")
    print(f"   Test Images : {total_test}")
    print(f"   Total Images: {total_train + total_val + total_test}")
    print("=" * 65)


if __name__ == "__main__":
    main()
