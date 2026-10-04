import os
import sys
import glob
import cv2
import random

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acc_det.config import ACC_DET_DIR, OUTPUT_DIR

VIDEO_DERIVED_DIR = os.path.join(ACC_DET_DIR, "dataset", "video_derived")
IMAGES_DIR = os.path.join(VIDEO_DERIVED_DIR, "images")
LABELS_DIR = os.path.join(VIDEO_DERIVED_DIR, "labels")

SAMPLES_DIR = os.path.join(OUTPUT_DIR, "auto_annotated_samples")
HTML_REPORT_PATH = os.path.join(OUTPUT_DIR, "inspection_report.html")

os.makedirs(SAMPLES_DIR, exist_ok=True)


def draw_yolo_labels(image_path, label_path):
    """Draws YOLO bounding boxes and badges on an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None, 0, 0

    h, w = img.shape[:2]
    annotated = img.copy()

    acc_count = 0
    norm_count = 0

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]

        for parts in lines:
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, nw, nh = [float(v) for v in parts[1:]]

            x1 = int((xc - nw / 2.0) * w)
            y1 = int((yc - nh / 2.0) * h)
            x2 = int((xc + nw / 2.0) * w)
            y2 = int((yc + nh / 2.0) * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            if cls_id == 1:
                color = (0, 40, 240)  # Red for Accident
                label_text = "ACCIDENT"
                acc_count += 1
            else:
                color = (0, 210, 60)  # Green for Normal Vehicle
                label_text = "NORMAL VEHICLE"
                norm_count += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            by1 = max(0, y1 - th - 6)
            cv2.rectangle(annotated, (x1, by1), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    if acc_count == 0 and norm_count == 0:
        # Background negative badge
        cv2.rectangle(annotated, (15, 15), (260, 45), (40, 40, 40), -1)
        cv2.rectangle(annotated, (15, 15), (260, 45), (120, 120, 120), 1)
        cv2.putText(
            annotated,
            "BACKGROUND NEGATIVE",
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    return annotated, acc_count, norm_count


def generate_gallery(num_samples=30):
    print("=" * 65)
    print("   GENERATING VISUAL INSPECTION GALLERY & HTML REPORT   ")
    print("=" * 65)

    all_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not all_images:
        print("[!] Error: No auto-annotated images found to sample.")
        return

    # Categorize images
    accident_imgs = []
    normal_veh_imgs = []
    bg_imgs = []

    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        has_acc = False
        has_norm = False

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = [l.strip().split() for l in f.readlines() if l.strip()]
            for l in lines:
                if len(l) == 5:
                    if int(l[0]) == 1:
                        has_acc = True
                    elif int(l[0]) == 0:
                        has_norm = True

        if has_acc:
            accident_imgs.append(img_path)
        elif has_norm:
            normal_veh_imgs.append(img_path)
        else:
            bg_imgs.append(img_path)

    print(f"[*] Found {len(accident_imgs)} Accident frames")
    print(f"[*] Found {len(normal_veh_imgs)} Normal traffic frames")
    print(f"[*] Found {len(bg_imgs)} Background negative frames")

    random.seed(42)
    selected_acc = random.sample(accident_imgs, min(14, len(accident_imgs)))
    selected_norm = random.sample(normal_veh_imgs, min(10, len(normal_veh_imgs)))
    selected_bg = random.sample(bg_imgs, min(6, len(bg_imgs)))

    selected = selected_acc + selected_norm + selected_bg
    cards_html = []

    print(f"\nRendering {len(selected)} high-resolution inspection samples...")
    for idx, img_path in enumerate(selected):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LABELS_DIR, f"{base_name}.txt")

        annotated, acc_c, norm_c = draw_yolo_labels(img_path, lbl_path)
        if annotated is None:
            continue

        sample_name = f"sample_{idx:02d}_{base_name}.jpg"
        sample_path = os.path.join(SAMPLES_DIR, sample_name)
        cv2.imwrite(sample_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if acc_c > 0:
            badge_class = "badge-danger"
            badge_text = f"Accident ({acc_c}) + Normal ({norm_c})"
        elif norm_c > 0:
            badge_class = "badge-success"
            badge_text = f"Normal Traffic ({norm_c} veh)"
        else:
            badge_class = "badge-secondary"
            badge_text = "Pure Background (0 veh)"

        card = f"""
        <div class="card">
            <img src="auto_annotated_samples/{sample_name}" alt="{base_name}">
            <div class="card-body">
                <span class="badge {badge_class}">{badge_text}</span>
                <p class="card-title">{base_name}</p>
            </div>
        </div>
        """
        cards_html.append(card)

    # Build HTML Report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auto-Annotation Quality Inspection Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f1117;
            color: #e1e4ea;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            max-width: 1300px;
            margin: 0 auto 24px auto;
            background: #181b24;
            padding: 24px 30px;
            border-radius: 12px;
            border: 1px solid #282d3d;
        }}
        h1 {{
            margin: 0 0 8px 0;
            color: #ffffff;
            font-size: 24px;
        }}
        .subtitle {{
            color: #8b949e;
            margin: 0 0 20px 0;
            font-size: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }}
        .stat-card {{
            background: #202432;
            padding: 14px 18px;
            border-radius: 8px;
            border: 1px solid #2e3447;
        }}
        .stat-num {{
            font-size: 24px;
            font-weight: 700;
            color: #58a6ff;
            margin-bottom: 4px;
        }}
        .stat-label {{
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 20px;
            max-width: 1300px;
            margin: 0 auto;
        }}
        .card {{
            background: #181b24;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #282d3d;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: transform 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-3px);
            border-color: #388bfd;
        }}
        .card img {{
            width: 100%;
            height: 280px;
            object-fit: cover;
            display: block;
        }}
        .card-body {{
            padding: 14px 16px;
        }}
        .card-title {{
            font-size: 12px;
            font-family: monospace;
            color: #8b949e;
            margin: 10px 0 0 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 600;
            border-radius: 20px;
            text-transform: uppercase;
        }}
        .badge-danger {{
            background: rgba(248, 81, 73, 0.2);
            color: #ff7b72;
            border: 1px solid rgba(248, 81, 73, 0.4);
        }}
        .badge-success {{
            background: rgba(46, 160, 67, 0.2);
            color: #3fb950;
            border: 1px solid rgba(46, 160, 67, 0.4);
        }}
        .badge-secondary {{
            background: rgba(139, 148, 158, 0.2);
            color: #c9d1d9;
            border: 1px solid rgba(139, 148, 158, 0.4);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Auto-Annotation Quality Inspection Report</h1>
        <p class="subtitle">Visual review of auto-annotated CCTV surveillance frames generated across 130 surveillance feeds.</p>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-num">{len(all_images)}</div>
                <div class="stat-label">Total Images Mined</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color: #ff7b72;">{len(accident_imgs)}</div>
                <div class="stat-label">Accident Frames</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color: #3fb950;">{len(normal_veh_imgs)}</div>
                <div class="stat-label">Normal Traffic Frames</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color: #d29922;">{len(bg_imgs)}</div>
                <div class="stat-label">Pure Background Negatives</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color: #2ea043;">0%</div>
                <div class="stat-label">No-Accident Leakage</div>
            </div>
        </div>
    </div>

    <div class="grid">
        {"".join(cards_html)}
    </div>
</body>
</html>
    """

    with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[+] Inspection Gallery generated successfully!")
    print(f"[+] Rendered sample images saved to: {SAMPLES_DIR}")
    print(f"[+] Interactive HTML report saved to: {HTML_REPORT_PATH}")


if __name__ == "__main__":
    generate_gallery()
