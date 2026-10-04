# 5G Suraksha-Net — Crowd Monitoring & Fight Detection Module

Real-time crowd monitoring and **temporal** fight detection from a live Sparsh
CCTV/RTSP stream. This repository contains ONLY this module; fire detection,
accident detection, ambulance coordination and emergency orchestration live in
teammates' modules and integrate through the documented JSON/REST/WebSocket
contract (see REST/WebSocket schema contracts in `src/suraksha/incidents/schemas.py`).

## Pipeline

```
Sparsh CCTV RTSP
      ↓  (capture/stream.py — reconnecting frame grabber, throttled to target FPS)
YOLO11s person detection + ByteTrack multi-object tracking
      ↓  (detection/ — Ultralytics integrated track API)
Crowd monitoring                       Fight detection
├── person count                       ├── candidate pairs (kinematics + IoU
├── zone density (polygon zones)       │   / distance fallback + optical flow)
├── crowd growth (sliding-window slope)├── TEMPORAL recognition over 32-frame
└── movement/panic anomaly             │   windows (trained PyTorch GRU classifier;
    (flow entropy + track speeds)      │   rwf2000_v2_best.pt active in app.yaml)
      ↓                                └── incident verification (persistence +
      ↓                                    consecutive-window rule → NEVER a
      ↓                                    single-frame decision)
Incident manager (dedupe/cooldown) → Incident JSON + snapshot + short clip
      ↓
FastAPI REST + WebSocket  →  teammate's orchestration system
```

## Why PyTorch for the temporal classifier

Ultralytics YOLO11 and ByteTrack already run on PyTorch. Using PyTorch for the
temporal fight classifier keeps a single framework: one CUDA context, shared
device management, no dual-framework memory cost, and features flow naturally
from the tracking stack into the classifier. TensorFlow is not used anywhere
in this module.

## Setup

```bash
cd "D:\projects\5G Suraksha-Net"

# 1. virtualenv (already created as .venv if you followed the guided setup)
python -m venv .venv
.venv\Scripts\activate

# 2. CUDA torch FIRST (RTX 4050 → cu121 build), then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .          # makes `suraksha` importable

# 3. environment self-test (records GPU/CUDA info to logs/environment.log)
python scripts\check_env.py

# 4. one-time: YOLO11s weights (~19 MB)
python scripts\download_weights.py

# 5. configure
copy .env.example .env    # set SURAKSHA_RTSP_URL to the Sparsh stream
```

CPU fallback works automatically — if CUDA is unavailable the pipeline runs
on CPU (slower; lower `target_fps` in `configs/app.yaml`).

## One-Click IMC Demo (Drone / Raspberry Pi Camera)

To launch the live IMC demonstration with the Raspberry Pi 4 + Logitech C270 camera:

```batch
:: Option A: Double-click from Windows Explorer or run from CMD:
START_IMC_DEMO.bat

:: Option B: From PowerShell:
.\scripts\start_imc_demo.ps1
```

To stop the demo and release resources cleanly:
```batch
STOP_IMC_DEMO.bat
```
For complete network, hardware, and operational details, refer to the root project documentation.

## Multi-Source Camera Execution Modes

Suraksha-Net supports 4 distinct video input modes converging into the identical, shared AI pipeline:

### 1. Laptop Webcam (Stand-Alone / Offline — No Pi Required)
Runs purely on the Windows laptop using the integrated or USB webcam:
```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --source webcam --camera-index 0 --display
```
*(Default camera index is 0; omit `--camera-index` or pass a custom integer index as needed).*

### 2. Drone-Mounted Raspberry Pi 4 RTSP Stream
Runs the live end-to-end drone camera path (Logitech C270 -> Pi -> MediaMTX -> Windows laptop):
```cmd
START_IMC_DEMO.bat
```
*Or via pipeline CLI:*
```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --source rtsp --path "rtsp://10.254.18.48:8554/drone" --display
```

### 3. Generic CCTV / Sparsh Camera RTSP Stream
Connects to an on-premise CCTV or Sparsh IP camera over RTSP:
```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --source rtsp --path "rtsp://<CAMERA_IP>:554/live" --display
```

### 4. Recorded CCTV Video File (MP4)
Runs offline analysis or repeatable benchmarks on recorded CCTV footage:
```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --source file --path datasets\videos\test\synthetic_cctv.mp4 --display
```

---

## Run (Additional Options)

```bash
# Full pipeline + REST/WebSocket API (http://localhost:8100/docs)
python scripts\run_api.py

# Offline benchmark: decode+process as fast as possible (no pacing)
python scripts\run_pipeline.py --source file --path clip.mp4 --fast

# Export annotated video without opening GUI window (headless):
python scripts\run_pipeline.py --source file --path clip.mp4 --fast --save-video outputs\annotated.mp4

# Run test suite (180 tests)
python -m pytest
```

`--preview` opens an OpenCV window rendering YOLO person bounding boxes, ByteTrack IDs, active candidate interactions, verified fight alerts, and a diagnostic HUD overlay.
`--save-video <PATH>` writes the annotated frames directly to an MP4 video file (`mp4v` codec) with zero duplicate inference passes.
`--fast` = benchmark (no pacing), `--realtime` = pace at source FPS,
default = throttle to `capture.target_fps`. `--max-frames N` limits a run.
A run ends automatically at video EOF and prints a latency/FPS/GPU summary.

### API surface (integration contract)

| Endpoint | Purpose |
|---|---|
| `GET /health` | module status, device, FPS |
| `GET /crowd/status` | latest counts / zone densities / growth / movement |
| `GET /incidents?incident_type=fight` | recent incidents (JSON) |
| `GET /incidents/{id}` | incident detail |
| `GET /incidents/{id}/snapshot` | annotated JPEG |
| `GET /incidents/{id}/clip` | short MP4 (±5 s around incident) |
| `WS /ws/incidents` | live push of every incident JSON |

Incident JSON schema: `src/suraksha/incidents/schemas.py` (Pydantic, versioned).

## Configuration

- `configs/app.yaml` — all pipeline tuning (thresholds, windows, zones file, clip lengths)
- `configs/zones.yaml` — crowd density zones per camera (normalized polygons)
- `.env` — machine-specific values (RTSP URL, device, ports). See `.env.example`
- No paths are hardcoded in source; everything resolves through `suraksha.config`

## Dataset manifests

Training data is referenced (never embedded) via JSON manifests in
`datasets/manifests/`. Full ingestion (probe → sha256 → validate → video-level
split → leakage check → class stats → reports):

```bash
python scripts\ingest_dataset.py --root datasets\raw\rwf2000 ^
    --dataset-name rwf2000 --manifest-name rwf2000_v1 --group A ^
    --license "research-use-only (RWF-2000 terms)" --source kaggle_mirror
```

Group B (independent hard-negative evaluation, never mixed with training):
see evaluation split configurations in `configs/data.yaml`
(labels, splits, 16/32-frame sequences, preprocessing, augmentation).
Legacy simple builder: `scripts/make_manifest.py`.

## Repository layout

```
configs/    YAML configuration (app, zones)
datasets/   raw/processed data (gitignored) + manifests/
logs/       suraksha.log (rotating), environment.log (env records)
models/     weights: yolo11s.pt, trained fight classifier checkpoints
outputs/    snapshots/ clips/ incidents/ (evidence, gitignored)
scripts/    check_env, run_pipeline, run_api, download_weights, make_manifest
src/suraksha/
  api/          FastAPI + WebSocket
  capture/      RTSP frame grabber
  crowd/        count, zone density, growth, movement anomaly
  data/         dataset manifest system
  detection/    YOLO11s person detector + ByteTrack tracker
  fight/        candidate detection, temporal classifier, recognizer
  incidents/    schemas (integration contract), evidence, manager+event bus
  pipeline.py   orchestrator
tests/      pytest suite (config, crowd, fight, schemas, manifests)
```

## Current status / verified state

- Fight recognition uses the **retrained PyTorch GRU classifier**
  (`models/temporal/rwf2000_v2_best.pt`, 16,321 params), active by default in
  `configs/app.yaml` with dual candidate proximity gating (`proximity_distance: 1.5`
  body-widths) and extended ByteTrack tracking (`track_buffer: 60`).
  Operating threshold is calibrated to `0.55` (85.2% precision on RWF-2000 val,
  92.3% precision on CCTV benchmarks).
- Heuristic scorer remains as safe fallback if `model_weights` is unconfigured.
- `ffmpeg` is not on PATH on this machine; clips are written with OpenCV's
  mp4v codec. Install ffmpeg for H.264 clips (auto-detected when present).
- Zone polygons in `configs/zones.yaml` are placeholders — calibrate them to
  the real Sparsh camera view.
- 149 unit and integration tests in test suite, 100% passing.
