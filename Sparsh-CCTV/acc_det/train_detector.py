import os
import sys
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from acc_det.config import (
    DETECTION_DATASET_DIR,
    WEIGHTS_DIR,
    TIER1_BEST_DETECTOR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tier 1 YOLO Road CCTV Accident Detector"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Base model checkpoint (default: yolo11s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMG_SIZE, help=f"Image size (default: {IMG_SIZE})"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="GPU device ID or 'cpu' (default: 0)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--name", type=str, default="accident_detector", help="Experiment name"
    )
    return parser.parse_args()


def train():
    args = parse_args()
    yaml_path = os.path.join(DETECTION_DATASET_DIR, "data.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"data.yaml not found at: {yaml_path}. Run download_and_prepare.py first!"
        )

    print("=" * 65)
    print("   TIER 1: YOLO SURVEILLANCE ACCIDENT DETECTOR TRAINING   ")
    print("=" * 65)
    print(f"[*] Base Model       : {args.model}")
    print(f"[*] Dataset Config   : {yaml_path}")
    print(f"[*] Target Device    : {args.device}")
    print(f"[*] Image Resolution : {args.imgsz}x{args.imgsz}")
    print(f"[*] Batch Size       : {args.batch}")
    print(f"[*] Max Epochs       : {args.epochs}")
    print(f"[*] Early Stop (Pat) : {PATIENCE}")
    print("=" * 65)

    model = YOLO(args.model)

    # Train model with CCTV surveillance hyperparameter optimizations
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=PATIENCE,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        amp=True,  # Mixed Precision FP16 on CUDA
        mosaic=1.0,  # Essential for object scale variation
        mixup=0.1,  # Teaches model overlapping visual features
        fliplr=0.5,  # Horizontal flip
        flipud=0.0,  # Keep ground plane intact
        hsv_h=0.015,  # Hue variation
        hsv_s=0.7,  # Saturation variation for day/night
        hsv_v=0.4,  # Brightness variation for streetlights & shadows
        name=args.name,
        project=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "train"
        ),
    )

    # Retrieve best model weight path
    train_dir = results.save_dir
    best_weight_path = os.path.join(train_dir, "weights", "best.pt")

    if os.path.exists(best_weight_path):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        shutil.copy2(best_weight_path, TIER1_BEST_DETECTOR)
        print("\n" + "=" * 65)
        print(f"[+] Training completed successfully!")
        print(f"[+] Best weights saved to: {TIER1_BEST_DETECTOR}")
        print("=" * 65)
    else:
        print(f"[!] Warning: Best weights not found at {best_weight_path}")

    return results


if __name__ == "__main__":
    train()
