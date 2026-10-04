import os
import sys
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ROOT_DIR, DETECTION_DATASET_DIR


def mine_negative_frames(max_frames_per_video=60, stride=45):
    """
    Extracts normal traffic CCTV frames from local video recordings
    and adds them as negative background samples (0 accident bboxes)
    to teach YOLO to reject normal road traffic, parked cars, and streetlights.
    """
    videos_dir = os.path.join(ROOT_DIR, "data", "videos")
    normal_videos = ["v3.mp4", "v1.mp4", "v10.mp4", "v15.mp4", "v20.mp4"]

    existing_videos = [
        v for v in normal_videos if os.path.exists(os.path.join(videos_dir, v))
    ]
    if not existing_videos:
        print("[Warning] No normal traffic test videos found for mining.")
        return 0

    train_img_dir = os.path.join(DETECTION_DATASET_DIR, "images", "train")
    train_lbl_dir = os.path.join(DETECTION_DATASET_DIR, "labels", "train")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)

    print(
        f"\n[Hard Negative Mining] Extracting normal CCTV frames from {len(existing_videos)} videos..."
    )
    mined_count = 0

    for vid_name in existing_videos:
        vid_path = os.path.join(videos_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret or extracted >= max_frames_per_video:
                break

            if frame_idx % stride == 0:
                # Resize to 640x640 standard YOLO input
                resized = cv2.resize(frame, (640, 640))
                img_name = (
                    f"cctv_neg_{os.path.splitext(vid_name)[0]}_{extracted:04d}.jpg"
                )
                lbl_name = (
                    f"cctv_neg_{os.path.splitext(vid_name)[0]}_{extracted:04d}.txt"
                )

                cv2.imwrite(
                    os.path.join(train_img_dir, img_name),
                    resized,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                # Create empty label file (background image)
                with open(os.path.join(train_lbl_dir, lbl_name), "w") as f:
                    pass

                extracted += 1
                mined_count += 1

            frame_idx += 1

        cap.release()
        print(f"  [+] Mined {extracted} background frames from {vid_name}")

    print(
        f"[+] Hard Negative Mining complete: {mined_count} total background frames added to training set."
    )
    return mined_count


if __name__ == "__main__":
    mine_negative_frames()
