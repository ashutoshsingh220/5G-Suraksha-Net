import os
import sys
from ultralytics import YOLO

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acc_det.config import (
    DETECTION_DATASET_DIR,
    SEVERITY_DATASET_DIR,
    TIER1_BEST_DETECTOR,
    TIER2_BEST_SEVERITY,
)


def evaluate_models():
    print("=" * 65)
    print("   HIERARCHICAL ACCIDENT SYSTEM - MODEL EVALUATION   ")
    print("=" * 65)

    # 1. Tier 1 Detection Evaluation on Test Split
    print("\n[1/2] Evaluating Tier 1 Spatial Detector on Unseen Test Split...")
    yaml_path = os.path.join(DETECTION_DATASET_DIR, "data.yaml")
    detector = YOLO(TIER1_BEST_DETECTOR)
    det_metrics = detector.val(data=yaml_path, split="test", verbose=True)

    print("\n--- Tier 1 Detector Test Metrics ---")
    print(f"Overall mAP@50       : {det_metrics.box.map50:.4f}")
    print(f"Overall mAP@50-95    : {det_metrics.box.map:.4f}")
    print(f"Overall Precision    : {det_metrics.box.mp:.4f}")
    print(f"Overall Recall       : {det_metrics.box.mr:.4f}")

    # Class-specific metrics
    if hasattr(det_metrics.box, "maps") and len(det_metrics.box.maps) >= 2:
        print(f"  Class 0 (Normal)   mAP@50: {det_metrics.box.maps[0]:.4f}")
        print(f"  Class 1 (Accident) mAP@50: {det_metrics.box.maps[1]:.4f}")

    # 2. Tier 2 Severity Evaluation on Validation Split
    print("\n[2/2] Evaluating Tier 2 Severity Classifier on Validation Split...")
    severity_classifier = YOLO(TIER2_BEST_SEVERITY)
    sev_metrics = severity_classifier.val(
        data=os.path.abspath(SEVERITY_DATASET_DIR), split="val", verbose=True
    )

    print("\n--- Tier 2 Severity Test Metrics ---")
    print(f"Top-1 Accuracy       : {sev_metrics.top1:.4f}")
    print(f"Top-5 Accuracy       : {sev_metrics.top5:.4f}")

    print("\n" + "=" * 65)
    print("   MODEL EVALUATION COMPLETED   ")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_models()
