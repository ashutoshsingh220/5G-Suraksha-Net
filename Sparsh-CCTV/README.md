# Sparsh CCTV Real-Time Accident Severity & Fire Detection System

An AI surveillance system for real-time traffic monitoring, vehicle accident severity classification, and fire/hazard detection using deep learning on CCTV video streams and recorded video files.

---

## Models Integrated

1. **Traffic Accident & Vehicle Detector** (`models/crash_classifier/model.safetensors`):
   - Architecture: **Hugging Face DETR (DEtection TRansformer with ResNet-50 backbone)**
   - Format: SafeTensors with local offline `config.json` & `preprocessor_config.json`
   - Real-time CUDA FP16 inference for CCTV camera feeds and recordings
   - Classes detected:
     - `accident` (Traffic Accident - Red badge, Level 3 alert)
     - `vehicle` (Normal Vehicle - Green badge, Level 0 normal)

2. **Fire & Smoke Hazard Detector** (`models/fire_classifier/fire_classifier.pt`):
   - Architecture: **Ultralytics YOLO** Object Detector
   - Classes: `0: fire`, `1: other`, `2: smoke`
   - Real-time bounding box detection of flames and smoke plumes with high-priority emergency alerts and border flashes.

---

## Features

- **Multi-Source Support**:
  - Video files (`.mp4`, `.avi`, `.mkv`, etc.)
  - Sparsh CCTV RTSP live stream (`rtsp://admin:admin123@192.168.128.10:554/...`)
  - Webcams (source index `0`, `1`, etc.)
- **Surveillance Camera HUD**:
  - Live FPS counter & timestamp.
  - Active vehicle counter & categorized incident breakdown.
  - Color-coded bounding boxes with confidence scores.
  - Automatic flashing alert banners for severe accidents and fires.
- **Incident Logging & Snapshots**:
  - Automatically captures high-resolution evidence snapshots (`output/snapshots/`) when severe accidents or fires occur (with cooldown to prevent spam).
  - Logs structured event data to `output/incident_log.csv`.
- **Interactive Controls**:
  - `Space`: Pause / Resume playback
  - `S`: Save manual snapshot to `output/snapshots/`
  - `H`: Toggle HUD overlay on/off
  - `Q` or `ESC`: Quit cleanly
- **Video Recording**:
  - Optional `--save` flag to export processed videos with full HUD overlay.

---

## Quick Start & Usage

### 1. Run on a Video File
```powershell
.venv/Scripts/python.exe main.py --source data/videos/v3.mp4
```

### 2. Save Processed Video to Disk
```powershell
.venv/Scripts/python.exe main.py --source data/videos/v3.mp4 --save output/recordings/v3_annotated.mp4
```

### 3. Connect to Sparsh CCTV RTSP Stream
```powershell
.venv/Scripts/python.exe main.py --rtsp
```

### 4. Adjust Detection Thresholds
```powershell
.venv/Scripts/python.exe main.py --source data/videos/v3.mp4 --conf 0.40 --fire-conf 0.70
```

### 5. Run Headless (No GUI Window)
```powershell
.venv/Scripts/python.exe --source data/videos/v3.mp4 --no-display --save
```

---

## Automated Tests

Run the test suite to verify model loading, synthetic inference, visualizer rendering, and video capture:
```powershell
.venv/Scripts/python.exe tests/test_pipeline.py
```

---

## Project Structure

```
SparshCCTV_Agentic_System/
├── config.py                  # System configurations, RTSP credentials, thresholds, colors
├── main.py                    # Main CLI application entry point
├── stream_capture.py          # Video stream capture handler (RTSP & video files)
├── visualizer.py              # CCTV HUD overlay and incident rendering engine
├── incident_logger.py         # Snapshot capturer and CSV incident logging
├── api/                       # FastAPI gateway package
│   ├── app.py                 # FastAPI application factory & middleware
│   ├── dashboard.py           # Web surveillance dashboard HTML/JS
│   ├── dependencies.py        # Dependency injection providers
│   ├── routes_surveillance.py # Stream, control, frame, and snapshot endpoints
│   ├── routes_settings.py     # Dynamic runtime settings CRUD endpoints
│   ├── routes_incidents.py    # Incidents, dispatches, and snapshot serving
│   ├── routes_emergency.py    # Geocoding and 3km responder facilities endpoints
│   ├── routes_system.py       # Health checks and system telemetry
│   └── schemas.py             # Pydantic request and response models
├── surveillance_engine.py     # Real-time stream coordinator & settings listener
├── services/                  # Emergency response & dispatch microservices
│   ├── alert_dispatcher.py    # Multi-channel alert dispatcher (SMS, Email, Log)
│   ├── email_service.py       # Asynchronous SMTP alert email service
│   ├── emergency_registry.py  # 3km nearby hospital, police, fire finder
│   └── geo_service.py         # Address geocoding and Google Maps resolution
├── detectors/
│   ├── accident_detector.py   # DETR ResNet-50 accident detector (legacy)
│   ├── fire_detector.py       # Ultralytics YOLO fire & smoke detector
│   └── yolo_hierarchical_detector.py # 15 FPS YOLO hierarchical accident classifier
├── models/                    # Model weights directory
├── output/                    # Snapshots, recordings, and audit logs
└── tests/                     # Unit and integration test suites
    ├── test_api.py            # FastAPI endpoints test suite
    ├── test_dynamic_settings.py # Dynamic runtime reconfiguration tests
    ├── test_email_service.py  # Email service tests
    ├── test_emergency_dispatch.py # Geocoding & emergency routing tests
    └── test_pipeline.py       # Detector and visualizer tests
```

---

## FastAPI Server & Web Dashboard

### 1. Launch FastAPI Server
```powershell
python main.py
```
Or with custom port and host:
```powershell
python main.py --host 0.0.0.0 --port 8000
```
Or run directly via Uvicorn:
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Access Web Interfaces & Endpoints
- **Web Dashboard**: `http://localhost:8000/` (Live MJPEG video player, real-time telemetry cards, stream controls, live dynamic settings editor, incident logs)
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

### 3. API Endpoints Overview

| Category | Method | Path | Description |
|---|---|---|---|
| **Surveillance** | `POST` | `/api/surveillance/start` | Start background surveillance stream |
| | `POST` | `/api/surveillance/stop` | Stop surveillance stream |
| | `POST` | `/api/surveillance/pause` | Pause processing |
| | `POST` | `/api/surveillance/resume` | Resume processing |
| | `GET` | `/api/surveillance/status` | Real-time stream telemetry & active alerts |
| | `GET` | `/api/surveillance/stream` | Continuous multipart MJPEG video stream |
| | `GET` | `/api/surveillance/frame` | Latest single frame as JPEG |
| | `GET` | `/api/surveillance/detections` | Structured JSON detections of current frame |
| | `POST` | `/api/surveillance/snapshot` | Capture and save manual snapshot |
| | `POST` | `/api/surveillance/trigger-alert`| Manually trigger incident alert & emergency dispatch |
| **Settings** | `GET` | `/api/settings` | View all active system settings |
| | `PATCH` | `/api/settings` | Dynamically update any setting at runtime without restart |
| | `GET / PATCH` | `/api/settings/camera` | Camera ID, RTSP URL, IP, port, channel, FPS |
| | `GET / PATCH` | `/api/settings/detectors` | Crash confidence, fire confidence, backend |
| | `GET / PATCH` | `/api/settings/alerts` | Alert cooldown, search radius, max facilities |
| | `GET / PATCH` | `/api/settings/email` | SMTP credentials and department email routing |
| | `GET / PATCH` | `/api/settings/twilio` | Twilio SMS credentials and emergency phone |
| **Incidents** | `GET` | `/api/incidents` | Query incident history from CSV log |
| | `GET` | `/api/dispatches` | Query emergency alert dispatches from JSON |
| | `GET` | `/api/snapshots` | List captured incident snapshot files |
| | `GET` | `/api/snapshots/{filename}` | Download / view specific snapshot image |
| **Emergency** | `GET` | `/api/emergency/location` | Resolved camera location & coordinates |
| | `GET` | `/api/emergency/facilities` | Nearby hospitals, police, fire within 3km |
| | `POST` | `/api/emergency/resolve-location` | Geocode custom address string |
| | `POST` | `/api/emergency/test-email` | Send test alert email |
| | `POST` | `/api/emergency/test-sms` | Send test emergency SMS |
| **System** | `GET` | `/api/system/health` | Health check & uptime |
| | `GET` | `/api/system/info` | GPU, CUDA, model weights status |

