import os
from typing import Dict, Tuple

# Base Directories
ACC_DET_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ACC_DET_DIR)

DATASET_DIR = os.path.join(ACC_DET_DIR, "dataset")
DETECTION_DATASET_DIR = os.path.join(DATASET_DIR, "detection")
SEVERITY_DATASET_DIR = os.path.join(DATASET_DIR, "severity")

WEIGHTS_DIR = os.path.join(ACC_DET_DIR, "weights")
OUTPUT_DIR = os.path.join(ACC_DET_DIR, "output")
RUNS_DIR = os.path.join(ACC_DET_DIR, "runs")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Datasets from Hugging Face
HF_DATASET_CCTV = "justjuu/traffic-accident-cctv-object-detection"
HF_DATASET_TRAFFIC = "hiennguyen9874/traffic-accident-detection"

# Pretrained Weights
HF_PRETRAINED_SURVEILLANCE_YOLO = "Enos-123/traffic-accident-detection-yolo11x"
HF_PRETRAINED_FILENAME = "weights/epoch61.pt"

# Model Checkpoints
TIER1_BEST_DETECTOR = os.path.join(WEIGHTS_DIR, "best_detector.pt")
TIER2_BEST_SEVERITY = os.path.join(WEIGHTS_DIR, "best_severity.pt")

# Tier 1 Detection Classes (Spatial Detection)
DETECTION_CLASSES = {0: "normal", 1: "accident"}

# Tier 2 Severity Classes (Accident Classification)
SEVERITY_CLASSES = {0: "moderate", 1: "severe"}

# Video & CCTV Stream Camera Settings
TARGET_FPS = 15.0
FRAME_INTERVAL_MS = 1000.0 / TARGET_FPS  # 66.66 ms

# Default Training Parameters
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 35
PATIENCE = 10
WORKERS = 4
DEVICE = "cuda:0"
