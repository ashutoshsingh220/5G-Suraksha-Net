import os
import sys
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from acc_det.config import SEVERITY_DATASET_DIR, WEIGHTS_DIR, TIER2_BEST_SEVERITY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tier 2 Accident Severity Classifier (Moderate vs Severe)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-cls.pt",
        help="Base classification checkpoint (default: yolo11n-cls.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=224, help="Crop image size (default: 224)"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="GPU device ID or 'cpu' (default: 0)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--name", type=str, default="severity_classifier", help="Experiment name"
    )
    return parser.parse_args()


def train():
    args = parse_args()
    data_dir = os.path.abspath(SEVERITY_DATASET_DIR)

    if not os.path.exists(os.path.join(data_dir, "train")):
        raise FileNotFoundError(
            f"Severity dataset not found at: {data_dir}. Run prepare_severity_data.py first!"
        )

    print("=" * 65)
    print("   TIER 2: ACCIDENT SEVERITY CLASSIFIER (MODERATE vs SEVERE)   ")
    print("=" * 65)
    print(f"[*] Base Model       : {args.model}")
    print(f"[*] Dataset Dir      : {data_dir}")
    print(f"[*] Target Device    : {args.device}")
    print(f"[*] Crop Resolution  : {args.imgsz}x{args.imgsz}")
    print(f"[*] Batch Size       : {args.batch}")
    print(f"[*] Max Epochs       : {args.epochs}")
    print("=" * 65)

    model = YOLO(args.model)

    results = model.train(
        data=data_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        amp=True,
        name=args.name,
        project=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "train_cls"
        ),
    )

    train_dir = results.save_dir
    best_weight_path = os.path.join(train_dir, "weights", "best.pt")

    if os.path.exists(best_weight_path):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        shutil.copy2(best_weight_path, TIER2_BEST_SEVERITY)
        print("\n" + "=" * 65)
        print(f"[+] Severity training completed successfully!")
        print(f"[+] Best severity weights saved to: {TIER2_BEST_SEVERITY}")
        print("=" * 65)
    else:
        print(f"[!] Warning: Best weights not found at {best_weight_path}")

    return results


if __name__ == "__main__":
    train()
