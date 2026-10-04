# 5G Suraksha-Net & Sparsh CCTV
## Dual-Tier Autonomous AI Surveillance & Emergency Response Ecosystem

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![YOLO11](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![5G SA Slicing](https://img.shields.io/badge/5G_SA-Network_Slicing-E1140A)](https://www.3gpp.org/)
[![MAVLink](https://img.shields.io/badge/MAVLink-Pixhawk_Telemetry-1E88E5)](https://mavlink.io/)
[![Hardware](https://img.shields.io/badge/Hardware-Sparsh_CCTV_RTSP-4CAF50)](https://sparshsecuritech.com/)

An integrated, enterprise-grade surveillance infrastructure combining **5G-connected Tactical Aerial UAVs (Drones)** and **Municipal Ground CCTV Networks**. The ecosystem delivers multi-tier situational awareness, multi-hazard AI detection, automated evidentiary packaging, and policy-driven multi-department emergency dispatch across critical infrastructure and public security zones.

---

## Table of Contents

- [Ecosystem Architecture](#ecosystem-architecture)
- [Strategic Multi-Tier Synergy](#strategic-multi-tier-synergy)
- [Subsystem 1: 5G Suraksha-Net (Aerial UAV & Tactical AI)](#subsystem-1-5g-suraksha-net-aerial-uav--tactical-ai)
  - [Core Capabilities](#core-capabilities)
  - [5G Standalone (SA) Network Slicing & QoS Priority](#5g-standalone-sa-network-slicing--qos-priority)
  - [Drone Telemetry & Pixhawk MAVLink Bridge](#drone-telemetry--pixhawk-mavlink-bridge)
  - [Multi-Modal Vision & Temporal AI Models](#multi-modal-vision--temporal-ai-models)
  - [Automated Evidence Dossier & Operator Safety Gate](#automated-evidence-dossier--operator-safety-gate)
  - [Situational Command Dashboard](#situational-command-dashboard)
- [Subsystem 2: Sparsh CCTV Agentic System (Ground & Traffic CCTV)](#subsystem-2-sparsh-cctv-agentic-system-ground--traffic-cctv)
  - [Core Capabilities](#core-capabilities-1)
  - [Sparsh CCTV Hardware & RTSP Low-Latency Capture](#sparsh-cctv-hardware--rtsp-low-latency-capture)
  - [Two-Tier Hierarchical Accident & Severity Detector](#two-tier-hierarchical-accident--severity-detector)
  - [Fire & Smoke Hazard Detector](#fire--smoke-hazard-detector)
  - [Geolocation & 3 km Emergency Facility Registry](#geolocation--3-km-emergency-facility-registry)
  - [Dynamic Runtime Settings & Embedded HUD Stream](#dynamic-runtime-settings--embedded-hud-stream)
- [Subsystem Comparison Matrix](#subsystem-comparison-matrix)
- [Repository Structure](#repository-structure)
- [Quick Start Guide](#quick-start-guide)
  - [Running Subsystem 1: 5G Suraksha-Net](#running-subsystem-1-5g-suraksha-net)
  - [Running Subsystem 2: Sparsh CCTV System](#running-subsystem-2-sparsh-cctv-system)
- [Responsible AI & Safety Protocols](#responsible-ai--safety-protocols)

---

## Ecosystem Architecture

```mermaid
flowchart TD
    subgraph TIER1["TIER 1: TACTICAL AERIAL UAV (5G Suraksha-Net)"]
        direction TB
        DRONE["Tactical Drone + Pixhawk FC"] -->|MAVLink Telemetry| MAV_BRIDGE["MAVLink Telemetry Bridge"]
        DRONE -->|RTSP / Live Video| CAP1["Low-Latency Capture"]
        
        CAP1 --> W_DET["YOLO11s Weapon Detector<br/>(Pistol / Shotgun / Knife)"]
        CAP1 --> F_DET["ByteTrack + BiGRU RWF-2000<br/>(Armed Altercation & Fight)"]
        CAP1 --> C_DET["Optical Flow Vector Field<br/>(Crowd Panic & Density)"]
        
        W_DET & F_DET & C_DET --> FUSION["Multi-Modal Threat Fusion Engine"]
        MAV_BRIDGE --> FUSION
        
        FUSION --> SLICE["5G SA Network Slicing Controller<br/>NORMAL &rarr; EVENT &rarr; CRITICAL (URLLC)"]
        FUSION --> EVID1["Forensic Packager<br/>(10s MP4 Clip + Keyframe Snapshot)"]
        
        EVID1 --> GATE["Mandatory Human Approval Gate<br/>(Strict Operator Decision Support)"]
        GATE -->|Authorized| POLICE_ALERT["Tactical Police Alert & Dossier"]
        
        FUSION --> UI1["Situational Command Dashboard<br/>(React + TypeScript + Tailwind)"]
    end

    subgraph TIER2["TIER 2: MUNICIPAL GROUND CCTV (Sparsh CCTV Agentic System)"]
        direction TB
        SPARSH["Sparsh CCTV Pole Camera"] -->|RTSP H.264 Stream| CAP2["Stream Capture (15 FPS Pacing)"]
        
        CAP2 --> DET_T1["Tier 1: Spatial Collision Detector<br/>(Vehicle vs Accident ROI)"]
        DET_T1 -->|Accident Crop| DET_T2["Tier 2: YOLO Severity Classifier<br/>(Moderate vs Severe)"]
        CAP2 --> FIRE["YOLO Fire & Smoke Detector<br/>(Flames & Smoke Plumes)"]
        
        DET_T2 & FIRE --> DISPATCHER["Emergency Alert Dispatcher"]
        
        GEO["Geolocation & Haversine Engine"] -->|3.0 km Search Radius| REGISTRY["Emergency Facility Registry<br/>(Hospitals, Police, Fire Stations)"]
        REGISTRY --> DISPATCHER
        
        DISPATCHER --> DISPATCH_EMAIL["Asynchronous SMTP Alert Email<br/>(Multipart HTML + Inline Snapshot)"]
        DISPATCHER --> DISPATCH_SMS["Twilio Emergency SMS"]
        
        CAP2 --> UI2["Embedded Surveillance HUD<br/>(MJPEG Stream + Runtime Settings API)"]
    end

    TIER1 -.->|Unified Operational Theater| SEC_OP["Municipal Emergency Command & Control"]
    TIER2 -.->|Perimeter & Traffic Defense| SEC_OP
```

---

## Strategic Multi-Tier Synergy

Traditional security architectures rely either solely on stationary ground cameras (which suffer from blind spots, occlusions, and inability to track dynamic crowd flows) or isolated drones (which lack continuous fixed-perimeter vantage points). 

This ecosystem unifies both into a synchronized defense matrix:
1. **The Sky View (Tier 1 — Drone)**: Operates above ground clutter to track human threats, weapons, violent brawls, and sudden crowd panics across wide plazas and event zones, leveraging **5G Network Slicing** to guarantee bandwidth and ultra-low latency under network congestion.
2. **The Ground View (Tier 2 — Fixed CCTV)**: Guards vehicular gates, approach roads, and perimeter boundaries, identifying traffic collisions, evaluating vehicle impact severity, detecting combustion/smoke hazards, and routing ambulances, fire trucks, and police units strictly within **3 km**.

---

## Subsystem 1: 5G Suraksha-Net (Aerial UAV & Tactical AI)

### Core Capabilities
- **Tactical Aerial Surveillance**: Processes video streams from UAV camera rigs, RTSP Raspberry Pi transmitters, CCTVs, and local forensic recordings.
- **5G Network Slicing Emulation**: Dynamically transitions network queue priorities (`NORMAL` $\rightarrow$ `EVENT` $\rightarrow$ `CRITICAL`) using 3GPP-aligned URLLC and eMBB slicing policies.
- **Flight Controller Telemetry**: Direct integration with Pixhawk auto-pilots over MAVLink, capturing live GPS coordinates, altitude, attitude angles, heading, battery, and flight mode.
- **Tri-Modal Deep Learning**:
  - Concealed and brandished weapon detection.
  - Multi-person violent altercation and struggle recognition.
  - Dynamic optical-flow crowd movement and panic surge analysis.
- **Operator Safety Safeguard**: Enforces a strict, mandatory human approval requirement before emergency dispatches are released.

### 5G Standalone (SA) Network Slicing & QoS Priority
5G Suraksha-Net integrates a deterministic application-level network policy controller:
- **`NORMAL` (Routine Monitoring)**: Default baseline. Best-effort video encoding, telemetry sampled at nominal rates, background queue priority.
- **`EVENT` (Elevated Incident)**: Triggered upon persistent weapon sighting or crowd panic surge. Uplink priority escalated to high; allocates dedicated event slicing bandwidth.
- **`CRITICAL` (Life-Threatening Incident)**: Triggered upon armed altercation or violent assault. Escalate immediately to URLLC priority queue, reserving low-latency radio bearers for live evidence distribution.

### Drone Telemetry & Pixhawk MAVLink Bridge
- **Asynchronous MAVLink Bridge**: Connects via serial (`COM`, `/dev/tty*`) or UDP (`127.0.0.1:14550`), extracting `HEARTBEAT`, `GLOBAL_POSITION_INT`, `ATTITUDE`, `SYS_STATUS`, and `VFR_HUD` packets.
- **Telemetry Broadcasting**: Broadcasts drone telemetry over WebSocket at 5 Hz, enabling real-time artificial horizon, compass heading, and altitude telemetry displays in the web dashboard.

### Multi-Modal Vision & Temporal AI Models
1. **Weapon Detection Engine (`suraksha.detection.weapon`)**:
   - Model: **YOLO11s** fine-tuned on tactical firearms and edge weapons.
   - Classes: `pistol`, `long_gun`, `knife`.
   - **Temporal Persistence Tracker**: Enforces a multi-frame sliding confirmation window (minimum 3 hits over 5 consecutive frames) and class-specific confidence gating ($\ge 0.70$ for firearms) to eliminate spurious transient false positives.
2. **Violent Fight & Struggle Recognition (`suraksha.fight`)**:
   - Model: **ByteTrack** multi-person spatial tracker paired with a **BiGRU Temporal Action Classifier** trained on RWF-2000.
   - Extracts 8-dimensional interaction vectors (pair approach velocity, bounding box overlap IoU, centroid distance, motion energy, struggle variance).
3. **Crowd Dynamics & Panic Monitor (`suraksha.crowd`)**:
   - Tracks dense optical-flow velocity fields across defined polygon zones.
   - Detects sudden directional divergence, panic velocity spikes ($>2.2$ body-widths/sec), and rapid zone density accumulation.

### Automated Evidence Dossier & Operator Safety Gate
- **Forensic Video Generation**: Upon incident confirmation, captures a trimmed **10-second forensic MP4 clip** ($T-5\text{s}$ pre-incident to $T+5\text{s}$ post-incident) along with a high-resolution annotated keyframe.
- **Decision Support Principle**: The system **never executes autonomous dispatch**. Dispatches remain in `AWAITING_HUMAN_APPROVAL` status until a verified operator authorizes action.

### Situational Command Dashboard
- Built with **React 18, TypeScript, Vite, and Tailwind CSS**.
- Real-time video player with tactical HUD, MAVLink telemetry gauges, 5G Network Slicing status badge, live interactive zones map, and forensic incident log viewer.

---

## Subsystem 2: Sparsh CCTV Agentic System (Ground & Traffic CCTV)

### Core Capabilities
- **Dedicated Sparsh CCTV RTSP Support**: Native communication profile for Sparsh CCTV network cameras.
- **Two-Tier Hierarchical Accident Classification**: Automatically detects traffic collisions and classifies impact severity.
- **Fire & Smoke Hazard Detection**: Real-time localization of flames and smoke plumes.
- **3 km Emergency Geo-Registry**: Geocodes camera nodes, calculates straight-line Haversine distance, and ranks nearby hospitals, police stations, and fire stations within a 3.0 km perimeter.
- **Multi-Channel Alert Dispatch**: Dispatches multipart HTML alert emails with inline evidence images and Twilio emergency SMS.
- **Dynamic Runtime Settings**: Operator configuration API allowing live threshold updates without server reboot.

### Sparsh CCTV Hardware & RTSP Low-Latency Capture
- Connects directly to Sparsh CCTV network endpoints:
  ```
  rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp
  ```
- **15 FPS Frame Pacing Engine**: Throttles processing to 66.6 ms per frame budget to match surveillance standard frame rates while preventing TCP queue lag.
- **Automatic Multi-Source Switching**: Supports live RTSP camera feeds, local MP4 traffic recordings, and USB webcams.

### Two-Tier Hierarchical Accident & Severity Detector
1. **Tier 1 — Spatial Collision Detector (`yolo_detector.pt`)**:
   - Evaluates vehicles across the roadway, outputting bounding boxes for normal traffic flow vs. vehicle collision zones at 50–80 FPS.
2. **Tier 2 — Accident Severity Classifier (`yolo_severity.pt`)**:
   - Automatically crops the collision Region of Interest (ROI) with 10% padding.
   - Classifies structural damage into:
     - **Moderate Accident (Level 2)**: Minor bumper/side impact (Orange HUD indicator).
     - **Severe Accident (Level 3)**: Structural deformation, cabin intrusion, or footprint $>20\%$ of frame area (Red flashing HUD indicator).
3. **Alternative Transformer Backend (`model.safetensors`)**:
   - Hugging Face **DETR ResNet-50 SafeTensors** model with CUDA FP16 support.

### Fire & Smoke Hazard Detector
- **Ultralytics YOLO Fire Detector (`fire_classifier.pt`)**:
- Identifies 3 hazard classes: `fire`, `smoke`, `other`.
- Produces individual bounding boxes with real-time confidence scores, triggering high-priority emergency alerts.

### Geolocation & 3 km Emergency Facility Registry
- **Geolocation Resolution (`geo_service.py`)**: Resolves camera deployment coordinates using Google Maps Geocoding API with offline static coordinate fallbacks.
- **Haversine Distance Engine**: Computes exact straight-line distances in kilometers to surrounding municipal facilities.
- **Strict 3.0 km Radius Filtering (`emergency_registry.py`)**: Filters and ranks nearest facilities:
  - **Fire Stations**: Dispatched for fire and smoke emergencies.
  - **Hospitals & Trauma Centers**: Dispatched for severe vehicular accidents and casualty hazards.
  - **Police & Traffic Inspector Posts**: Dispatched for traffic cordoning, accident clearance, and security.
- **Multi-Department Alert Routing**:
  - `accident` $\rightarrow$ Police, Hospital, Traffic Control
  - `fire` $\rightarrow$ Fire Department, Hospital, Police
  - `smoke` $\rightarrow$ Fire Department, Police

### Dynamic Runtime Settings & Embedded HUD Stream
- **FastAPI Surveillance Gateway**: Serves continuous multipart MJPEG streams (`/api/surveillance/stream`) and structured detection payloads (`/api/surveillance/detections`).
- **Dynamic Reconfiguration API (`/api/settings`)**: Allows operators to update camera IPs, RTSP ports, confidence thresholds, alert cooldowns, and email routing live at runtime without restarting the service.

---

## Subsystem Comparison Matrix

| Capability / Attribute | **5G Suraksha-Net (Aerial UAV Subsystem)** | **Sparsh CCTV (Ground CCTV Subsystem)** |
| :--- | :--- | :--- |
| **Primary Domain** | Aerial UAV Patrol & Wide Plaza Security | Fixed Roadway & Perimeter Surveillance |
| **Video Ingest** | RTSP / Raspberry Pi / Drone Stream / MP4 | Sparsh CCTV RTSP (`channel=1/stream=1.sdp`) |
| **Flight Telemetry** | Pixhawk Autopilot via MAVLink (GPS, Att, Alt) | None (Static Camera Node) |
| **5G Integration** | 5G SA Network Slicing Priority Emulation | Standard TCP/IP Network Delivery |
| **Primary Vision Tasks** | Weapons, Violent Fights, Crowd Panic Flow | Vehicle Collisions, Severity, Fire/Smoke |
| **Model Architectures** | YOLO11s, ByteTrack, BiGRU RWF-2000 | 2-Tier Hierarchical YOLO, DETR ResNet-50 |
| **Incident Verification** | Temporal Persistence Sliding Window | Multi-frame History & BBox Footprint Analysis |
| **Emergency Radius** | Dynamic Geocoded Police/Hospital Routing | Strict 3.0 km Radius Municipal Registry |
| **Dispatch Protocol** | Mandatory Human Authorization Gate | Automated Municipal Alert Dispatch |
| **Delivery Channels** | 10s MP4 Clip, Snapshot, SMTP Email, SMS | Multipart HTML Email, Inline Snap, Twilio SMS |
| **User Interface** | Situational React 18 / TypeScript Web SPA | Embedded MJPEG Web Surveillance Dashboard |

---

## Repository Structure

```text
5G-Suraksha-Net/
│
├── README.md                                  # Master Unified Documentation
├── .gitignore                                 # Master repository ignore file
├── .gitattributes                             # Git LFS configuration (*.safetensors)
│
├── 5G-Suraksha-Net/                           # TIER 1: Aerial UAV & 5G Tactical Subsystem
│   ├── configs/                               # Application, ByteTrack & Zone configs
│   │   ├── app.yaml                           # Pipeline, detector & network thresholds
│   │   ├── bytetrack.yaml                     # Multi-object tracker parameters
│   │   └── zones.yaml                         # Monitored spatial polygon zones
│   ├── frontend/                              # Situational Web Dashboard (React + TS)
│   │   ├── src/                               # Components, hooks, telemetry stores
│   │   └── package.json                       # Frontend dependencies & scripts
│   ├── models/                                # Model weights & checkpoints
│   │   ├── temporal/                          # BiGRU RWF-2000 fight classification models
│   │   └── yolo11s.pt                         # Person & baseline detectors
│   ├── outputs/                               # Forensic artifacts & training weights
│   │   ├── weapon_training/best.pt            # Calibrated YOLO11s weapon detector
│   │   └── demo/                              # Demonstration clips and keyframes
│   ├── scripts/                               # Operation & execution scripts
│   │   ├── run_api.py                         # FastAPI server launcher
│   │   ├── run_pipeline.py                    # Standalone video processing pipeline
│   │   └── start_imc_demo.ps1                 # Full-stack subsystem launcher
│   ├── src/suraksha/                          # Core Python engine
│   │   ├── agents/                            # Multi-agent incident orchestrator
│   │   ├── api/                               # FastAPI endpoints & WebSocket broadcaster
│   │   ├── capture/                           # Low-latency multi-source stream capturer
│   │   ├── crowd/                             # Optical flow crowd velocity & panic analyzer
│   │   ├── detection/                         # YOLO11s weapon detector & ByteTrack
│   │   ├── fight/                             # Spatial candidate & temporal fight recognizer
│   │   ├── incidents/                         # Evidence manager & fusion engine
│   │   ├── location/                          # Geocoding & coordinate providers
│   │   ├── network/                           # 5G SA Network Slicing policy service
│   │   ├── notifications/                     # Email and SMS alert dispatch service
│   │   ├── response/                          # Decision-support emergency response planner
│   │   └── telemetry/                         # MAVLink Pixhawk drone bridge & telemetry service
│   └── tests/                                 # Unit & integration test suites
│
└── Sparsh-CCTV/                               # TIER 2: Municipal Ground CCTV Subsystem
    ├── acc_det/                               # Accident detector training & data pipelines
    │   ├── config.py                          # Training hyper-parameters & dataset paths
    │   ├── pipeline.py                        # Standalone hierarchical pipeline runner
    │   └── weights/                           # Exported YOLO detector & severity models
    ├── api/                                   # FastAPI gateway & Web Surveillance HUD
    │   ├── app.py                             # FastAPI application factory
    │   ├── dashboard.py                       # Embedded surveillance web dashboard
    │   ├── routes_surveillance.py             # Stream control & MJPEG video routes
    │   ├── routes_settings.py                 # Dynamic runtime settings CRUD endpoints
    │   ├── routes_emergency.py                # Geolocation & 3 km emergency facility routes
    │   └── routes_incidents.py                # Incident history & snapshot retrieval
    ├── detectors/                             # Object detectors & classifiers
    │   ├── accident_detector.py               # DETR ResNet-50 transformer detector
    │   ├── fire_detector.py                   # YOLO fire and smoke hazard detector
    │   └── yolo_hierarchical_detector.py      # 2-Tier YOLO accident & severity detector
    ├── models/                                # Model weights directory
    │   ├── accident_classifier/               # Tier 1 detector & Tier 2 severity weights
    │   ├── crash_classifier/                  # SafeTensors DETR ResNet-50 model
    │   └── fire_classifier/                   # YOLO fire & smoke weights
    ├── services/                              # Emergency dispatch microservices
    │   ├── alert_dispatcher.py                # Multi-channel alert dispatcher
    │   ├── email_service.py                   # Asynchronous SMTP HTML email service
    │   ├── emergency_registry.py              # 3 km emergency facility registry
    │   └── geo_service.py                     # Geocoding & Haversine distance calculator
    ├── config.py                              # Central configuration & runtime settings
    ├── main.py                                # Subsystem entry point & CLI
    ├── stream_capture.py                      # Sparsh RTSP low-latency video capturer
    ├── visualizer.py                          # CCTV tactical HUD & overlay renderer
    └── tests/                                 # Subsystem automated test suite
```

---

## Quick Start Guide

### Running Subsystem 1: 5G Suraksha-Net

#### 1. Environment Setup
```powershell
cd 5G-Suraksha-Net
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

#### 2. Configure Environment Variables
Copy `.env.example` to `.env` and provide your API keys and SMTP credentials:
```powershell
cp .env.example .env
```

#### 3. Start Backend Server
```powershell
.venv\Scripts\python.exe scripts/run_api.py
```
*Backend runs at `http://localhost:8100` (API docs at `http://localhost:8100/docs`).*

#### 4. Start Situational Dashboard
In a separate terminal:
```powershell
cd 5G-Suraksha-Net/frontend
npm install
npm run dev
```
*Access the command console at `http://localhost:5173`.*

---

### Running Subsystem 2: Sparsh CCTV System

#### 1. Environment Setup
```powershell
cd Sparsh-CCTV
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # or uv sync
```

#### 2. Start CCTV Surveillance Engine & Web Dashboard
```powershell
python main.py --host 0.0.0.0 --port 8000
```
*Access the embedded MJPEG dashboard at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).*

#### 3. Run Standalone CLI on Video or Sparsh RTSP
```powershell
# Run on sample video
python main.py --source data/videos/accident/v3.mp4

# Connect to Sparsh CCTV RTSP stream
python main.py --rtsp
```

---

## Responsible AI & Safety Protocols

1. **Mandatory Human-in-the-Loop Gate**: All emergency recommendations generated by 5G Suraksha-Net require explicit authorization by a human supervisor prior to dispatch.
2. **Deterministic Evidence Packaging**: Detections are paired with immutable 10-second forensic clips ($T-5\text{s}$ to $T+5\text{s}$) to prevent unverified alarms.
3. **Strict Radius Guardrails**: Emergency facility queries enforce a geographic threshold ($3.0\text{ km}$) to prevent dispatching inappropriate responders.
4. **Privacy & Security Standards**: SMTP credentials, API tokens, and private camera RTSP passwords are strictly quarantined via environment variables and excluded from version control.
