import os
import sys
import glob

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acc_det.config import ACC_DET_DIR

VIDEO_DERIVED_DIR = os.path.join(ACC_DET_DIR, "dataset", "video_derived")
IMAGES_DIR = os.path.join(VIDEO_DERIVED_DIR, "images")
LABELS_DIR = os.path.join(VIDEO_DERIVED_DIR, "labels")


def validate_all_annotations():
    print("=" * 65)
    print("   AUTOMATED ANNOTATION QUALITY & INTEGRITY VALIDATOR   ")
    print("=" * 65)

    img_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    lbl_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))

    print(f"[*] Found {len(img_files)} images and {len(lbl_files)} label files.")
    if len(img_files) == 0:
        print(
            "[!] Error: No auto-annotated images found. Run auto_annotate_videos.py first!"
        )
        return False

    errors = []
    warnings = []

    total_boxes = 0
    class_counts = {0: 0, 1: 0}
    empty_label_files = 0
    noacc_accident_leakage = 0

    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LABELS_DIR, f"{base_name}.txt")

        if not os.path.exists(lbl_path):
            errors.append(f"Missing label file for: {base_name}")
            continue

        with open(lbl_path, "r") as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]

        if not lines:
            empty_label_files += 1
            continue

        is_noacc_video = "cctv_noacc_" in base_name

        for line_idx, parts in enumerate(lines):
            if len(parts) != 5:
                errors.append(
                    f"{base_name}.txt line {line_idx}: Expected 5 fields, got {len(parts)}"
                )
                continue

            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                nw = float(parts[3])
                nh = float(parts[4])
            except ValueError as e:
                errors.append(f"{base_name}.txt line {line_idx}: Parse error: {e}")
                continue

            if cls_id not in (0, 1):
                errors.append(
                    f"{base_name}.txt line {line_idx}: Invalid class {cls_id}"
                )

            # Zero leakage assertion
            if is_noacc_video and cls_id == 1:
                noacc_accident_leakage += 1
                errors.append(
                    f"CRITICAL: Accident label leaked into no-accident file: {base_name}"
                )

            # Boundary checks
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                errors.append(
                    f"{base_name}.txt line {line_idx}: Center out of bounds: xc={xc}, yc={yc}"
                )
            if not (0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                errors.append(
                    f"{base_name}.txt line {line_idx}: Dimensions out of bounds: nw={nw}, nh={nh}"
                )

            # Area & aspect checks
            area = nw * nh
            if area < 0.0002:
                warnings.append(
                    f"{base_name}.txt line {line_idx}: Very small bounding box: area={area:.5f}"
                )

            ratio = nw / max(0.001, nh)
            if ratio < 0.10 or ratio > 8.0:
                warnings.append(
                    f"{base_name}.txt line {line_idx}: Extreme aspect ratio: {ratio:.2f}"
                )

            total_boxes += 1
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    print("\n" + "-" * 40)
    print("   ANNOTATION VALIDATION AUDIT REPORT   ")
    print("-" * 40)
    print(f"Total Images Audited       : {len(img_files)}")
    print(f"Total Bounding Boxes       : {total_boxes}")
    print(f"  - Normal Vehicles (0)    : {class_counts[0]}")
    print(f"  - Accident Instances (1) : {class_counts[1]}")
    print(
        f"Pure Background Negatives  : {empty_label_files} ({empty_label_files / len(img_files):.1%})"
    )
    print(f"No-Accident Label Leakage  : {noacc_accident_leakage} (Must be 0)")
    print(f"Integrity Errors Found     : {len(errors)}")
    print(f"Minor Geometric Warnings   : {len(warnings)}")
    print("-" * 40)

    if errors:
        print("\n[!] Validation Failed with Errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        return False

    print(
        "\n[+] SUCCESS: All annotations passed 100% of mathematical & category integrity checks!"
    )
    return True


if __name__ == "__main__":
    validate_all_annotations()
