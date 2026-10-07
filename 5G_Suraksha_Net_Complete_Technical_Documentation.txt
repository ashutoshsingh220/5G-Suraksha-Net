# 5G Suraksha-Net — Complete Technical Documentation

> **Combined Repository:** [`github.com/ashutoshsingh220/5G-Suraksha-Net`](https://github.com/ashutoshsingh220/5G-Suraksha-Net)
> **Classification:** Internal Technical Reference — All Figures Extracted from Source Code & Training Artefacts
> **Date:** October 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Subsystem A — Aerial UAV & Tactical AI (5G-Suraksha-Net)](#3-subsystem-a--aerial-uav--tactical-ai)
   - 3.1 [Pipeline Architecture](#31-pipeline-architecture)
   - 3.2 [Person Detection & Multi-Object Tracking](#32-person-detection--multi-object-tracking)
   - 3.3 [Weapon Detection System](#33-weapon-detection-system)
   - 3.4 [Fight Detection System (Temporal BiGRU)](#34-fight-detection-system-temporal-bigru)
   - 3.5 [Crowd Analytics & Panic Detection](#35-crowd-analytics--panic-detection)
   - 3.6 [Multimodal Incident Fusion Engine](#36-multimodal-incident-fusion-engine)
   - 3.7 [5G Network Slicing Policy Engine](#37-5g-network-slicing-policy-engine)
   - 3.8 [Emergency Response Policy & Planner](#38-emergency-response-policy--planner)
   - 3.9 [Drone Telemetry (MAVLink)](#39-drone-telemetry-mavlink)
   - 3.10 [Email Notification System](#310-email-notification-system)
   - 3.11 [AI Agent Assessment (Deterministic + LLM)](#311-ai-agent-assessment)
   - 3.12 [Demo Simulation System](#312-demo-simulation-system)
4. [Subsystem B — Sparsh Municipal Ground CCTV](#4-subsystem-b--sparsh-municipal-ground-cctv)
   - 4.1 [Architecture Overview](#41-architecture-overview)
   - 4.2 [Accident Detection — DETR Backbone](#42-accident-detection--detr-backbone)
   - 4.3 [Accident Detection — YOLO Hierarchical (Primary)](#43-accident-detection--yolo-hierarchical-primary)
   - 4.4 [Fire & Smoke Detection](#44-fire--smoke-detection)
   - 4.5 [Emergency Alert Dispatch Pipeline](#45-emergency-alert-dispatch-pipeline)
   - 4.6 [Emergency Facility Registry](#46-emergency-facility-registry)
   - 4.7 [Surveillance Engine](#47-surveillance-engine)
5. [Datasets Used](#5-datasets-used)
6. [Training Metrics & Evaluation Scores](#6-training-metrics--evaluation-scores)
7. [Threshold Value Reference Table](#7-threshold-value-reference-table)
8. [Hardware & Infrastructure](#8-hardware--infrastructure)
9. [API Surface](#9-api-surface)
10. [Dependencies & Tech Stack](#10-dependencies--tech-stack)

---

## 1. Executive Summary

**5G Suraksha-Net** is a dual-subsystem, AI-powered public safety surveillance platform designed for real-time threat detection, incident classification, and automated emergency response coordination across urban environments. The combined system provides:

| Capability | Subsystem | Status |
|---|---|---|
| Weapon Detection (knife, pistol, long gun) | Aerial UAV (A) | ✅ Active |
| Physical Fight Detection (temporal) | Aerial UAV (A) | ✅ Active |
| Armed Fight Escalation (weapon + fight fusion) | Aerial UAV (A) | ✅ Active |
| Crowd Density & Panic Detection | Aerial UAV (A) | ✅ Active |
| Traffic Accident Detection & Severity | Ground CCTV (B) | ✅ Active |
| Fire & Smoke Detection | Ground CCTV (B) | ✅ Active |
| 5G Network Policy Engine | Aerial UAV (A) | ✅ Application-layer |
| Emergency Email/SMS Dispatch | Both (A + B) | ✅ Active |
| Drone MAVLink Telemetry | Aerial UAV (A) | ✅ Active |
| Raspberry Pi RTSP Edge Streaming | Aerial UAV (A) | ✅ Active |
| Sparsh CCTV RTSP Integration | Ground CCTV (B) | ✅ Active |

> [!IMPORTANT]
> All emergency dispatch actions require **mandatory human approval** before execution. Autonomous emergency dispatch is strictly prohibited by design.

---

## 2. System Architecture Overview

```mermaid
flowchart TB
    subgraph EdgeLayer["Edge Capture Layer"]
        PI["Raspberry Pi 4B<br/>Logitech C270 USB<br/>640x360 @ 15 FPS"]
        SPARSH["Sparsh CCTV Camera<br/>RTSP @ 192.168.128.10:554"]
        WEBCAM["Local Webcam<br/>USB / Built-in"]
        MP4["Pre-recorded MP4<br/>Test & Demo Files"]
    end

    subgraph AerialAI["Subsystem A — Aerial UAV & Tactical AI"]
        PIPE["CrowdFightPipeline<br/>Source-Agnostic Orchestrator"]
        YOLO_P["YOLO11s Person Detector<br/>conf: 0.25 | imgsz: 640"]
        BT["ByteTrack MOT<br/>track_buffer: 60"]
        WPN["YOLO11s Weapon Detector<br/>conf: 0.70 | 3-class"]
        GRU["BiGRU Temporal Classifier<br/>16,321 params | score: 0.65"]
        CROWD["Crowd Analyzer<br/>Density + Optical Flow"]
        FUSION["Multimodal Fusion Engine<br/>2.5s correlation window"]
        NET["5G Network Policy Engine<br/>NORMAL → EVENT → CRITICAL"]
        RESP["Response Planner<br/>human_approval = True"]
        MAV["MAVLink Telemetry Bridge<br/>14550 UDP | READ-ONLY"]
    end

    subgraph GroundCCTV["Subsystem B — Sparsh Ground CCTV"]
        ENGINE["SurveillanceEngine<br/>Background Thread Worker"]
        YOLO_T1["YOLO11s Tier-1 Detector<br/>normal vs accident | 0.70"]
        YOLO_T2["YOLO11n-cls Tier-2<br/>moderate vs severe | 224px"]
        DETR["DETR ResNet-50 Fallback<br/>SafeTensors | 800x800"]
        FIRE["YOLO Fire/Smoke Detector<br/>conf: 0.75 | 3-class"]
        DISP["Emergency Alert Dispatcher<br/>Email + SMS + Webhook"]
        REG["Emergency Facility Registry<br/>3km Haversine radius"]
    end

    subgraph Notification["Alert & Response Layer"]
        EMAIL["SMTP Email Service<br/>Police / Hospital / General"]
        SMS["Twilio SMS Gateway"]
        WH["Webhook Endpoint"]
    end

    PI -->|"RTSP rtsp://:8554/drone"| PIPE
    SPARSH -->|"RTSP rtsp://:554/avstream"| ENGINE
    WEBCAM --> PIPE
    WEBCAM --> ENGINE
    MP4 --> PIPE
    MP4 --> ENGINE

    PIPE --> YOLO_P --> BT --> CROWD
    PIPE --> WPN
    BT --> GRU
    WPN --> FUSION
    GRU --> FUSION
    CROWD --> FUSION
    FUSION --> NET --> RESP --> EMAIL
    MAV -.->|"Telemetry Data"| PIPE

    ENGINE --> YOLO_T1 --> YOLO_T2
    ENGINE --> FIRE
    YOLO_T2 --> DISP
    FIRE --> DISP
    DISP --> REG
    DISP --> EMAIL
    DISP --> SMS
    DISP --> WH
```

---

## 3. Subsystem A — Aerial UAV & Tactical AI

### 3.1 Pipeline Architecture

The core processing pipeline ([`pipeline.py`](file:///C:/Projects/5G%20Suraksha-Net/src/suraksha/pipeline.py)) is a **source-agnostic orchestrator** that processes frames from any video source (webcam, RTSP, MP4) through parallel AI branches:

```
StreamCapture → EvidenceWriter (push_frame buffer)
    ├── Branch A: WeaponDetector → WeaponEvent → FusionEngine
    └── Branch B: PersonDetector → ByteTrack MOT → CrowdAnalyzer
                                                  → FightRecognizer → VerifiedFight → FusionEngine
                                                  
FusionEngine → IncidentManager → EmailNotificationService
             → FrameAnnotator → AnnotatedVideoWriter
```

**Performance Metrics (RTX 4050, 60 frames benchmark):**

| Mode | Total Time | Effective FPS | Detect+Track | Weapon | Crowd | Fight | Peak VRAM |
|---|---|---|---|---|---|---|---|
| Baseline (weapon off) | 2.40s | 25.0 | 27.4ms | 0.01ms | 1.21ms | 0.01ms | 105.5 MB |
| Weapon Enabled | 1.21s | 49.6 | 8.35ms | 9.03ms | 1.17ms | 0.01ms | 165.3 MB |
| Full Fusion | 1.21s | 49.7 | 8.34ms | 9.04ms | 1.16ms | 0.01ms | 238.1 MB |

**Effective FPS Measurement:** Recalculated every `5.0 seconds` using a running average formula: `m[key] += (val - m[key]) / (n + 1)`

**Hot-Swapping:** The `switch_source()` method allows dynamic source changes at runtime, resetting weapon confirmation tracker and active fight candidates.

---

### 3.2 Person Detection & Multi-Object Tracking

| Parameter | Value | Source |
|---|---|---|
| **Model** | YOLO11s (Ultralytics) | `models/yolo11s.pt` |
| **Architecture** | YOLOv11 Small | Pre-trained on COCO |
| **Person Class ID** | `0` (COCO "person") | `detection.person_class_id` |
| **Confidence Threshold** | `0.25` | Calibrated in Phase 4 benchmark (+15.1% yield on 320×240 CCTV) |
| **IoU Threshold (NMS)** | `0.50` | `detection.iou_threshold` |
| **Input Size** | `640 × 640` | `detection.imgsz` |

**Multi-Object Tracker:**

| Parameter | Value |
|---|---|
| **Algorithm** | ByteTrack |
| **Config** | `configs/bytetrack.yaml` |
| **Track Buffer** | `60` frames |
| **Persist Mode** | `true` (tracks survive across frames) |

> [!NOTE]
> The low confidence threshold (0.25) was specifically chosen after Phase 4 benchmarking showed a +15.1% person yield improvement on low-resolution CCTV (320×240) feeds. ByteTrack's high threshold alignment ensures that tracking quality is maintained despite the lower detection threshold.

---

### 3.3 Weapon Detection System

#### 3.3.1 Model Architecture

| Parameter | Value |
|---|---|
| **Model** | YOLO11s (custom fine-tuned) |
| **Weights** | `outputs/weapon_training/best.pt` |
| **Training Epochs** | 51 (best at epoch 50) |
| **Input Size** | `640 × 640` |
| **Device** | Auto (CUDA if available) |
| **Inference Interval** | Every 2nd frame |

#### 3.3.2 Detection Classes

| Class ID | Name | Class-Specific Min Confidence |
|---|---|---|
| `0` | `knife` | `0.65` |
| `1` | `long_gun` | `0.70` |
| `2` | `pistol` | `0.70` |

**Effective threshold:** `max(global_conf_threshold, class_min)` — ensures pistol and long_gun detections require ≥ 70% confidence regardless of global setting.

#### 3.3.3 Temporal Confirmation System

The weapon detector employs a **multi-frame temporal persistence** mechanism to eliminate single-frame false positives:

| Parameter | Value | Purpose |
|---|---|---|
| `conf_threshold` | `0.70` | Minimum raw detection confidence |
| `confirm_conf_threshold` | `0.75` | Confirmation-grade confidence |
| `min_hits` | `3` | Minimum detections in sliding window |
| `window_size` | `5` frames | Sliding window length |
| `iou_match_threshold` | `0.25` | Box matching across frames |
| `expiry_frames` | `15` | Track expiration timeout |

**State Machine:**
```
NORMAL → WEAPON_CANDIDATE (first detection)
       → WEAPON_CONFIRMED (3+ hits in 5-frame window at ≥0.75 conf)
```

**Latching Hysteresis:** Once a weapon track reaches `WEAPON_CONFIRMED` state, it remains confirmed for the lifetime of the track — preventing flicker.

**Matching Score:** `max(IoU, proximity)` where proximity = `max(0.0, 1.0 - (center_dist / (2.0 × avg_diagonal)))`. Class ID must match strictly.

#### 3.3.4 Training Results (Epoch 50 — Best)

| Metric | Overall | Knife | Long Gun | Pistol |
|---|---|---|---|---|
| **Precision** | 0.8629 | 0.8539 | 0.8528 | 0.8819 |
| **Recall** | 0.8289 | 0.8378 | 0.8134 | 0.8355 |
| **mAP@50** | 0.8812 | 0.8697 | 0.8776 | 0.8963 |
| **mAP@50-95** | 0.5832 | 0.6230 | 0.5479 | 0.5786 |

**Final Training Losses (Epoch 50):**
- Box Loss: `0.96817` (train) / `1.03911` (val)
- Cls Loss: `0.62382` (train) / `1.48740` (val)
- DFL Loss: `1.19614` (train) / `1.19446` (val)

**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU, peak VRAM: `3.741 GB`, epoch duration: `442.96s`

#### 3.3.5 Locked Test Set Evaluation

| Metric | Validation | Test | Delta |
|---|---|---|---|
| **mAP@50** | 0.8812 | 0.8740 | -0.82% |
| **mAP@50-95** | 0.5832 | 0.5857 | +0.43% |
| **Precision** | 0.8629 | 0.8377 | -2.92% |
| **Recall** | 0.8289 | 0.8308 | +0.23% |
| **F1** | 0.8456 | 0.8343 | -1.34% |

**Test set size:** 2,703 images, 2,883 ground-truth boxes

**Per-Class Test Results:**

| Class | GT Boxes | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| knife | 596 | 0.796 | 0.8249 | 0.8102 | 0.8516 | 0.615 |
| long_gun | 997 | 0.8529 | 0.8143 | 0.8332 | 0.8682 | 0.5556 |
| pistol | 1,290 | 0.8642 | 0.8533 | 0.8587 | 0.9023 | 0.5866 |

**Confusion Matrix (rows: GT, cols: predicted → knife, long_gun, pistol, background):**

```
knife:     518     0     1   187
long_gun:    2   865    33   211
pistol:      1    33  1157   264
background: 75    99    99     0
```

**Background Hard Negative Analysis:**
- Total background images: **346**
- True negatives (clean): **293** (84.68%)
- False alarm images: **53** (15.32% image-level FPR)
- Total false alarm boxes: **75** (pistol: 23, long_gun: 18, knife: 34)
- `nogun_v5` partition (295 images): **14 false alarms** (4.75% FPR) ← robust
- `weapon_empty` partition (51 images): **39 false alarms** (76.47% FPR) ← adversarial set

#### 3.3.6 Webcam False-Positive Validation

| Metric | Value |
|---|---|
| Source | Webcam (index 0) |
| Resolution | 640 × 480 |
| Frames Processed | 150 |
| Elapsed Time | 6.81s |
| Average FPS | 22.0 |
| Mean Latency | 20.33ms |
| P95 Latency | 13.53ms |
| Peak VRAM | 99.7 MB |
| **Total False Detections** | **0** (0% FPR on clean webcam) |

---

### 3.4 Fight Detection System (Temporal BiGRU)

#### 3.4.1 Fight Candidate Detection

Fight candidate pairs are identified by spatial proximity and motion energy between tracked persons.

**Feature Vector (8 dimensions per pair per frame):**

| Index | Feature | Description |
|---|---|---|
| 0 | `rel_dist` | Center distance / scale |
| 1 | `iou` | Bounding box IoU |
| 2 | `normalized_motion` | `flow_px/frame × fps / scale` (body-widths/sec) |
| 3 | `a.confidence` | Person A detection confidence |
| 4 | `b.confidence` | Person B detection confidence |
| 5 | `a.width / frame_w` | Person A relative width |
| 6 | `b.width / frame_w` | Person B relative width |
| 7 | `1.0` | Bias term |

**Candidate Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `min_pair_speed` | `1.6` body-widths/sec | Minimum relative approach speed |
| `proximity_iou` | `0.08` | IoU threshold for engagement |
| `proximity_distance` | `1.25` body-widths | Max center distance |
| `proximity_min_frames` | `8` (~0.5s at 15 FPS) | Sustained engagement threshold |
| `missed_frames_tolerance` | `2` frames | Jitter tolerance |
| `motion_energy_threshold` | `0.12` body-widths/sec | Minimum motion gate |
| `motion_variance_threshold` | `0.0` | Minimum motion variance |
| `decay_timeout_s` | `1.0` s | Reset if motionless |
| `cooldown_s` | `2.0` s | Post-incident cooldown |

**Optical Flow (Farneback):**
- Downscale: `0.25×` if max dimension > 400px
- Parameters: `pyr_scale=0.5, levels=2, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1`
- Flow energy: averaged over person ROI (min 8×8 patch)

#### 3.4.2 Temporal GRU Classifier Architecture

```
Input: (batch, 32 frames, 8 features)
    ↓
nn.GRU(input=8, hidden=64, layers=1, bidirectional=False)
    ↓
Temporal Pooling: mean(all steps) + last_step → 64-dim
    ↓
nn.Linear(64, 32) → ReLU
    ↓
nn.Linear(32, 1) → sigmoid → fight probability
```

| Parameter | Value |
|---|---|
| **Total Parameters** | **16,321** |
| **Feature Dim** | 8 |
| **Hidden Size** | 64 |
| **Head Hidden** | 32 |
| **Num Layers** | 1 |
| **Bidirectional** | False |
| **Dropout** | 0.0 |
| **Normalization** | Standard (z-score) |
| **Model Weights** | `models/temporal/rwf2000_v2_best.pt` |

**Heuristic Fallback Scorer (when model unavailable):**
```
Score = 0.30 × proximity + 0.20 × contact + 0.30 × energy + 0.20 × oscillation
```
- proximity = mean(rel_dist < 1.5)
- contact = mean(iou > 0.02)
- energy = clip(mean(motion) / 0.7, 0, 1) — 0.7 based on RWF-2000 p90
- oscillation = clip(std(motion) / 0.5, 0, 1) — 0.5 based on RWF-2000 p90

#### 3.4.3 Temporal Verification Pipeline

| Parameter | Value | Description |
|---|---|---|
| `window_frames` | `32` | Frames per classification window |
| `stride_frames` | `8` | New classification every 8 frames |
| `score_threshold` | `0.65` | GRU output threshold |
| `min_duration_s` | `2.5` s | Minimum fight duration |
| `min_consecutive_windows` | `3` | Consecutive positive windows required |
| `min_motion_variance` | `0.0` | Motion variance gate |
| `incident_cooldown_s` | `60` s | Post-incident cooldown |

**Verification States:** `candidate` → `evaluating` → `verified` → (incident) / `expired`

**Rejection Reasons (diagnostic telemetry):**
- `REJECT: insufficient_consecutive_windows`
- `REJECT: insufficient_action_duration`
- `REJECT: score_below_threshold`
- `REJECT: motion_var_below_threshold`
- `REJECT: candidate_expired` (3.0s timeout)

#### 3.4.4 Training Configuration

| Parameter | Value |
|---|---|
| **Primary Dataset** | RWF-2000 |
| **Labels** | `FIGHT`, `NON_FIGHT` |
| **Split Ratios** | Train: 0.70, Val: 0.15, Test: 0.15 |
| **Split Level** | Video-level (no frame leakage) |
| **Seed** | 42 |
| **Window Sizes** | [16, 32] frames |
| **Stride** | 8 frames |
| **Frame Size** | 128 × 128 |
| **Normalization** | Unit (/255 → [0,1]) |
| **Augmentation** | Disabled |
| **Epochs** | 30 |
| **Batch Size** | 64 (eval: 256) |
| **Optimizer** | Adam |
| **Learning Rate** | 1e-3 |
| **Grad Clip** | 5.0 |
| **Balanced Class Weights** | True |
| **Evaluation Threshold** | 0.5 |
| **Leakage Reporting** | Enabled (adjusted metrics) |

---

### 3.5 Crowd Analytics & Panic Detection

**Density Levels:**

| Level | Threshold | Description |
|---|---|---|
| `LOW` | default | Normal crowd density |
| `HIGH` | ≥ 0.35 | 35% zone area covered by person boxes |
| `CRITICAL` | ≥ 0.55 | 55% zone area — dangerous overcrowding |

**Crowd Growth Monitoring:**
- Sliding window: `30 seconds`
- Minimum samples: `5`, minimum time span: `2.0 seconds`
- Growth rate: linear regression slope × 60 (persons/minute)
- Alert threshold: `10.0 persons/minute`

**Panic Movement Detection (Optical Flow):**

| Parameter | Value |
|---|---|
| Flow Grid Size | 32 px |
| Panic Speed Threshold | `2.2` box-widths/sec |
| Panic Min Tracks | `3` tracks |
| Min Panic Persons | `8` total persons in scene |
| Fast Ratio Gate | ≥ 50% of tracks exceed speed threshold |
| Direction Entropy Gate | ≥ 0.6 (high directional disorder) |

**Direction Entropy Calculation:**
- 8 angular bins over [0, 2π], magnitude-weighted
- Normalized by log₂(8) = 3.0 → range [0, 1]
- High entropy = scattered/panicked movement; low entropy = coordinated flow

**Zone Configuration:**
- Camera: `cam_default`
- `market_square`: polygon `[(0.05,0.30), (0.55,0.30), (0.55,0.95), (0.05,0.95)]`
- `street_corridor`: polygon `[(0.58,0.35), (0.98,0.35), (0.98,0.95), (0.58,0.95)]`

---

### 3.6 Multimodal Incident Fusion Engine

The fusion engine correlates detections across weapon, fight, and crowd subsystems to produce escalated compound incidents.

| Parameter | Value |
|---|---|
| `correlation_window_s` | `2.5` seconds |
| `spatial_proximity_threshold` | `1.5` body-widths |
| `escalation_enabled` | `true` |
| `weapon_cooldown_s` | `30.0` s |
| `armed_fight_cooldown_s` | `60.0` s |

**Fusion Outputs:**
- `unarmed_fights` — Fight without weapon
- `weapons_alone` — Weapon without concurrent fight
- `armed_fights` — Weapon + Fight within 2.5s and 1.5 body-widths → **CRITICAL severity**

**Simulation Validation (Phase 2 Benchmark):**
```
Event 1: type=weapon, severity=HIGH, confidence=0.89, escalated=false
Event 2: type=armed_fight, severity=CRITICAL, confidence=0.93, escalated=true
Event 3: type=armed_fight, severity=CRITICAL, status=finalized, escalated=true
```

---

### 3.7 5G Network Slicing Policy Engine

| Policy Level | Trigger | Application Priority |
|---|---|---|
| `NORMAL` | No active incidents | `ROUTINE` |
| `EVENT` | HIGH / MODERATE incident | `HIGH` |
| `CRITICAL` | CRITICAL incident (armed fight) | `CRITICAL` |

**Severity Weights:**
- CRITICAL: `4`, HIGH: `3`, MODERATE/MEDIUM: `2`, LOW: `1`

**Active Window:** `60.0 seconds` — incidents expire from policy consideration after this period.

> [!IMPORTANT]
> **Safety Declaration:** The network policy engine operates at the **application layer only**. It does not have actual 5G control plane access.
> - `actual_network_control`: `false`
> - `control_plane_connected`: `false`
> - `slice_allocated`: `DEFAULT_BE` (Best Effort)
> - `measurement_source`: `APPLICATION_POLICY`

---

### 3.8 Emergency Response Policy & Planner

**Response Policy Matrix:**

| Incident Type | Police Required | Medical Required | Notes |
|---|---|---|---|
| `ARMED_FIGHT` | ✅ (incident severity) | ✅ (incident severity) | Both always dispatched |
| `FIGHT` | ✅ (incident severity) | Only if injury indicator | Medical conditional on ground verification |
| `WEAPON` | ✅ (incident severity) | Only if injury indicator | Medical = CRITICAL if sev=CRITICAL |
| `CROWD_PANIC` | ✅ (incident severity) | Only if injury indicator | — |
| `CROWD_DENSITY_HIGH` | ❌ (low) | ❌ (low) | Routine monitoring only |
| `CROWD_DENSITY_CRITICAL` | ❌ (low) | ❌ (low) | Routine monitoring only |
| `CROWD_RAPID_GROWTH` | ❌ (low) | ❌ (low) | Routine monitoring only |

**Injury Indicator Keys Checked:** `injury`, `injuries`, `person_down`, `casualty`, `medical_need`, `has_injury`, `medical.required`

**Response Planner:**
- Resource directory: `outputs/phase3/nearby_emergency_locations_yashobhoomi.json`
- Facilities sorted by straight-line distance (km), tie-break by name
- All actions: `human_approval_required = True`
- Status: `AWAITING_HUMAN_APPROVAL` (or `RESOURCE_UNAVAILABLE` if missing)

---

### 3.9 Drone Telemetry (MAVLink)

| Parameter | Value |
|---|---|
| Connection | UDP `udpin:127.0.0.1:14550` |
| Serial Baud | 57600 (Pixhawk USB: 115200) |
| Timeout | 3.0 seconds |
| Data Stream Request | `MAV_DATA_STREAM_ALL` at 4 Hz |
| Broadcast Rate | 5.0 Hz |

**Auto-Detection:** Scans COM ports for `ardupilot`, `pixhawk`, `px4`, or USB VID:PID `1209:5741`

**MAVLink Messages Processed:**
- `HEARTBEAT`: armed flag, flight mode, vehicle type
- `GLOBAL_POSITION_INT`: lat/lon, altitude (relative + AMSL), climb rate, heading
- `VFR_HUD`: groundspeed, airspeed, climb rate, heading
- `GPS_RAW_INT`: fix type (No GPS → RTK Fixed), satellites, HDOP
- `SYS_STATUS` / `BATTERY_STATUS`: battery %, voltage, current
- `ATTITUDE`: roll, pitch, yaw (degrees)

**28 ArduCopter flight modes** supported (STABILIZE through TURTLE).

> [!CAUTION]
> The telemetry bridge is **READ-ONLY**. It never transmits flight commands, MAV_CMD, or heartbeat packets. This is a safety-critical design decision.

---

### 3.10 Email Notification System

| Parameter | Value |
|---|---|
| SMTP Port | 587 (TLS) or 465 (SSL) |
| Timeout | 60 seconds |
| Max Attachment | 25 MB |
| Worker Threads | 2 (ThreadPoolExecutor) |
| Deduplication | 1 alert per incident ID |
| Demo Suppression | Suppressed unless `allow_demo=True` |

**Role-Based Email Templates:**

| Role | Subject Format |
|---|---|
| `POLICE` | `[POLICE ALERT & EVIDENCE] {severity} - {type} DETECTED at {location}` |
| `HOSPITAL` | `[EMERGENCY MEDICAL / TRAUMA ALERT] CASUALTY RISK & AMBULANCE DISPATCH at {location}` |
| `GENERAL` | `[5G Suraksha-Net] {severity} - {type} detected at {location}` |

**Evidence Attachments:**
- Video: ~10 second MP4 clip (T-5s to T+5s), max 25 MB
- Image: Keyframe snapshot JPG, max 25 MB

**Pipeline Deferral:** Emails are deferred when incident status = `RECORDING_POST_EVENT`; dispatched once `FINALIZED` or `VERIFIED`.

---

### 3.11 AI Agent Assessment

**Two providers:**
1. **DeterministicAgentProvider** — 100% offline, rule-based reasoning engine (always available)
2. **LLMAgentProvider** — Wraps Gemini/OpenAI API; auto-delegates to Deterministic if no API key

**7-Part Operational Assessment:**
1. `SeverityAssessment` — Deterministic severity classification
2. `LocationAssessment` — Venue coordinates, nearest facilities, GPS fix
3. `EvidenceAssessment` — Snapshot JPG and 10s MP4 clip paths
4. `ResponseAssessment` — Police/medical recommendations, `AWAITING_HUMAN_APPROVAL`
5. `NetworkAssessment` — Policy tier, priority, 0 physical slice claims
6. `Recommended Operational Sequence` (6 steps, all `advisory_only=True`)
7. `Supervisor Briefing` — Complete situation summary with data traceability checklist

---

### 3.12 Demo Simulation System

| Scenario | Type | Severity | Confidence | Network Policy |
|---|---|---|---|---|
| WEAPON | WEAPON | HIGH | 0.93 | EVENT |
| CROWD_PANIC | CROWD_PANIC | HIGH | 0.89 | EVENT |
| ARMED_FIGHT | ARMED_FIGHT | CRITICAL | 0.97 | CRITICAL |
| NORMAL | — | — | — | NORMAL |

**Simulation Timeline:** 7 steps from T+0.0s to T+2.0s → `AWAITING_APPROVAL`

**Evidence Storage:**
- Clips: `outputs/demo/clips/`
- Snapshots: `outputs/demo/snapshots/`
- Incidents: `outputs/demo/incidents/`

---

## 4. Subsystem B — Sparsh Municipal Ground CCTV

### 4.1 Architecture Overview

The Sparsh CCTV subsystem is a **FastAPI-based surveillance application** with a background processing engine, real-time MJPEG streaming, and multi-channel emergency dispatch.

**Processing Loop:**
```
StreamCapture → Frame Read (15 FPS cap)
    ├── AccidentDetector (YOLO Hierarchical or DETR fallback)
    │   ├── Tier 1: Normal vs Accident (YOLO11s, conf: 0.70, iou: 0.65)
    │   └── Tier 2: Moderate vs Severe (YOLO11n-cls, 224×224 crop)
    ├── FireDetector (YOLO, conf: 0.75, 3-class)
    ├── IncidentLogger → EmergencyAlertDispatcher
    │   ├── Email (SMTP, department routing)
    │   ├── SMS (Twilio)
    │   └── Webhook
    └── CCTVVisualizer → Annotated HUD Frame → MJPEG Stream
```

**RTSP Configuration:**
- IP: `192.168.128.10` / Port: `554`
- Channel: `1` / Stream: `1`
- URL Format: `rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp`
- Auto-reconnect interval: `8 seconds`

---

### 4.2 Accident Detection — DETR Backbone

| Parameter | Value |
|---|---|
| **Architecture** | DETR (DEtection TRansformer) |
| **Backbone** | ResNet-50 |
| **Weights Format** | SafeTensors (`model.safetensors`) |
| **Framework** | HuggingFace `transformers` v4.35.2 |
| **Input Size** | 800 × 800 |
| **Num Queries** | 100 (DETR decoder queries) |
| **d_model** | 256 |
| **Encoder/Decoder Layers** | 6 each |
| **Attention Heads** | 8 (encoder and decoder) |
| **FFN Dim** | 2048 |
| **Dropout** | 0.1 |
| **Position Embedding** | Sine |
| **FP16 Inference** | Enabled on CUDA |

**DETR Class Mapping:**

| ID | Label | Is Accident |
|---|---|---|
| 0 | accident | ✅ (Level 3) |
| 1 | accident | ✅ (Level 3) |
| 2 | vehicle | ❌ (Level 0) |

**Loss Coefficients (from config.json):**
- BBox Cost/Loss: `5`, GIoU Cost/Loss: `2`, Class Cost: `1`
- Dice Loss: `1`, Mask Loss: `1`, EOS Coefficient: `0.1`

**ImageNet Normalization:**
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

---

### 4.3 Accident Detection — YOLO Hierarchical (Primary)

This is the **primary** accident detector (`detector_backend = "yolo"`), a two-tier hierarchical system that replaced the slower DETR model:

**Tier 1 — Spatial YOLO Detector:**

| Parameter | Value |
|---|---|
| Model | YOLO11s (fine-tuned) |
| Weights | `models/accident_classifier/yolo_detector.pt` |
| Classes | `{0: "normal", 1: "accident"}` |
| Confidence | `0.70` |
| IoU (NMS) | `0.65` |
| Input Size | 640 × 640 |

**Tier 2 — Severity Classifier:**

| Parameter | Value |
|---|---|
| Model | YOLO11n-cls (classification head) |
| Weights | `models/accident_classifier/yolo_severity.pt` |
| Classes | `{0: "moderate", 1: "severe"}` |
| Input Size | 224 × 224 (cropped ROI with 10% padding) |

**Severity Classification Logic:**
- If classifier predicts "severe" **OR** collision footprint > 20% of frame area → **Severity Level 3 (SEVERE)** — Red
- Otherwise → **Severity Level 2 (MODERATE)** — Orange

**Full Severity Scale:**

| Level | Name | Display | Color (BGR) |
|---|---|---|---|
| 0 | vehicle | Normal Vehicle | (0, 210, 60) — Green |
| 2 | accident_moderate | Moderate Accident | (0, 165, 255) — Orange |
| 3 | accident_severe / accident | Severe Accident | (0, 40, 240) — Red |
| 4 | accident_100 | Totaled Vehicle | (160, 0, 210) — Magenta |

**Speed:** > 50-80 FPS real-time on CUDA (vs ~15-25 FPS for DETR)

#### Training Configuration

| Parameter | Tier 1 (Detector) | Tier 2 (Severity) |
|---|---|---|
| Base Model | `yolo11s.pt` | `yolo11n-cls.pt` |
| Epochs | 35 | 20 |
| Batch Size | 16 | 32 |
| Image Size | 640 | 224 |
| Patience | 10 | — |
| Workers | 4 | 4 |
| Device | `cuda:0` | `cuda:0` |

---

### 4.4 Fire & Smoke Detection

| Parameter | Value |
|---|---|
| **Model** | YOLO (Ultralytics) |
| **Weights** | `models/fire_classifier/fire_classifier.pt` |
| **Confidence Threshold** | `0.75` |
| **Classes** | `["fire", "other", "smoke"]` |

**Detection Colors (BGR):**

| Class | Color | RGB Equivalent |
|---|---|---|
| fire | (0, 69, 255) | Red-Orange |
| smoke | (180, 180, 180) | Light Gray |
| other | (150, 150, 150) | Gray |

**Output Schema (`FireResult`):**
- `has_fire`: boolean
- `has_smoke`: boolean
- `fire_confidence`: max fire detection confidence
- `smoke_confidence`: max smoke detection confidence
- `predicted_label`: "Fire & Smoke" / "Fire" / "Smoke" / "Normal"

---

### 4.5 Emergency Alert Dispatch Pipeline

The Sparsh CCTV uses a **multi-channel emergency alert dispatcher** that routes incidents to appropriate departments:

**Incident-to-Department Routing Matrix:**

| Incident Type | Departments Notified |
|---|---|
| `accident` | Police, Hospital, Traffic Control |
| `fire` | Fire Department, Hospital, Police |
| `smoke` | Fire Department, Police |

**Alert Channels:**
1. **SMTP Email** — Departmental routing with snapshot attachment
2. **Twilio SMS** — Formatted emergency SMS to dispatch number
3. **Webhook** — HTTP POST to configurable URL
4. **Console Banner** — ASCII formatted alert display
5. **JSON Audit Log** — Persistent dispatch record

**Snapshot Cooldown:** `5.0 seconds` — prevents disk flooding during continuous alerts

**Department Email Configuration (from environment):**
- Police: `POLICE_DEPARTMENT_EMAIL`
- Hospital: `HOSPITAL_DEPARTMENT_EMAIL`
- Fire: `FIRE_DEPARTMENT_EMAIL`
- Traffic Control: `TRAFFIC_CONTROL_EMAIL`

---

### 4.6 Emergency Facility Registry

| Parameter | Value |
|---|---|
| Search Radius | `3.0 km` (Haversine distance) |
| Max Facilities Per Type | `2` |
| API | Google Places API (New) |
| Fallback | Curated offline database (9 facilities) |
| Cache | `data/emergency_facilities_cache.json` |

**Curated Offline Facilities (9 total):**
- 1 Fire Station (Dwarka Sector 25)
- 3 Police Stations/Check Posts (Sector 21, Sector 23, Traffic Inspector)
- 5 Hospitals (Maple Care, Ambe, Helpline, Ayushman, Amerix)

**Incident Routing Logic:**
- Fire/Smoke → Fire Stations + Hospitals + Police
- Accident/Crash → Hospitals + Police

**Distance Calculation:** Haversine formula for great-circle distance between camera GPS and facility GPS coordinates.

---

### 4.7 Surveillance Engine

The `SurveillanceEngine` is a **thread-safe background coordinator** managing:

| Feature | Detail |
|---|---|
| Frame Rate Target | 15.0 FPS (`66.66ms` frame budget) |
| Processing Thread | `SparshSurveillanceWorker` (daemon) |
| Stream States | `IDLE`, `STREAMING`, `NO_STREAM`, `PAUSED`, `STOPPED` |
| RTSP Reconnection | Every 8 seconds when camera offline |
| Video File Looping | Automatic rewind at EOF for continuous surveillance |
| Recording | Optional MP4 recording of annotated frames |
| Dynamic Settings | Observer pattern — all thresholds hot-reloadable at runtime |
| Detector Hot-Swap | Backend switchable between `yolo` and `detr` without restart |

**Status Classification:**
- `INCIDENT`: Severity ≥ 2 or fire/smoke detected
- `NORMAL`: No active threats

---

## 5. Datasets Used

### Subsystem A — Aerial UAV

| Dataset | Purpose | Labels | Notes |
|---|---|---|---|
| **RWF-2000** | Fight detection training (primary) | Fight / NonFight | 2,000 clips, video-level split |
| **RLVS** | Fight detection (supplementary) | V / NV | Real-Life Violence Situations |
| **Hockey Fight** | Fight detection (supplementary) | fi / no | Hockey fight dataset |
| **Movies Fight** | Fight detection (supplementary) | fights / noFights | Movie violence clips |
| **UCF-101** (selective) | Hard negatives + ambiguous combat | 13 HARD_NEGATIVE + 8 AMBIGUOUS | Sports/dance classes as negatives |
| **UCF Crime** (subset) | Crime/fight detection | Fighting, Assault → FIGHT | Hard negatives from normal activities |
| **Custom Weapon Dataset** | Weapon detection | knife, long_gun, pistol | Curated dataset with `nogun_v5` negatives |

**Hard Negative Categories (14 classes):**
`normal_walking`, `normal_crowd`, `running`, `gathering`, `dispersing`, `arguing_gesturing`, `hugging`, `handshaking`, `dancing`, `sports_like`, `pushing_shoving`, `waving_gesturing`, `sudden_nonviolent_movement`, `actual_fighting`

### Subsystem B — Sparsh CCTV

| Dataset | Purpose | Source | Notes |
|---|---|---|---|
| **justjuu/traffic-accident-cctv-object-detection** | Tier 1 accident detection | HuggingFace | CCTV accident/vehicle detection |
| **hiennguyen9874/traffic-accident-detection** | Tier 1 supplementary | HuggingFace | Traffic accident dataset |
| **Enos-123/traffic-accident-detection-yolo11x** | Pre-trained weights | HuggingFace | `epoch61.pt` base checkpoint |
| Custom Fire/Smoke Dataset | Fire & smoke detection | — | 3-class: fire, smoke, other |
| DETR Crash Classifier | Accident detection (fallback) | Custom trained | 3-class: accident, accident, vehicle |

---

## 6. Training Metrics & Evaluation Scores

### Weapon Detection (YOLO11s, 51 Epochs)

| Metric | Train (E50) | Val (E50) | Test (Locked) |
|---|---|---|---|
| Precision | — | 0.8629 | 0.8377 |
| Recall | — | 0.8289 | 0.8308 |
| F1 | — | 0.8456 | 0.8343 |
| mAP@50 | — | 0.8812 | 0.8740 |
| mAP@50-95 | — | 0.5832 | 0.5857 |
| Box Loss | 0.968 | 1.039 | — |
| Cls Loss | 0.624 | 1.487 | — |
| DFL Loss | 1.196 | 1.194 | — |

### Fight Detection (BiGRU, RWF-2000)

| Metric | Value | Source |
|---|---|---|
| Score Threshold | 0.65 | `temporal.score_threshold` |
| Precision (RWF-2000 val) | ~85.2% | Phase reporting |
| Precision (CCTV subset) | ~92.3% | Phase reporting |
| Parameters | 16,321 | Model architecture |
| Window Size | 32 frames | Training config |
| Feature Dimension | 8 | Candidate feature vector |

### Accident Detection (YOLO Hierarchical)

| Parameter | Value |
|---|---|
| Base Model | YOLO11s (detection) + YOLO11n-cls (severity) |
| Pre-trained Weights | `epoch61.pt` from HF |
| Training Epochs | 35 (detector) / 20 (severity) |
| Confidence Threshold | 0.70 |
| IoU Threshold | 0.65 |
| Real-time FPS | > 50-80 FPS (CUDA) |

---

## 7. Threshold Value Reference Table

### Subsystem A — Complete Threshold Map

| Subsystem | Parameter | Value | Purpose |
|---|---|---|---|
| Person Detection | `conf_threshold` | `0.25` | COCO person detection |
| Person Detection | `iou_threshold` | `0.50` | NMS overlap |
| Weapon Detection | `conf_threshold` | `0.70` | Raw detection gate |
| Weapon (pistol) | class minimum | `0.70` | Per-class gate |
| Weapon (long_gun) | class minimum | `0.70` | Per-class gate |
| Weapon (knife) | class minimum | `0.65` | Per-class gate |
| Weapon Confirm | `confirm_conf_threshold` | `0.75` | Temporal confirmation |
| Weapon Confirm | `min_hits` | `3 of 5` | Sliding window hits |
| Weapon Confirm | `iou_match_threshold` | `0.25` | Cross-frame matching |
| Weapon Confirm | `expiry_frames` | `15` | Track timeout |
| Fight GRU | `score_threshold` | `0.65` | Classifier output |
| Fight Verify | `min_duration_s` | `2.5` | Minimum fight time |
| Fight Verify | `min_consecutive_windows` | `3` | Positive windows |
| Fight Verify | `incident_cooldown_s` | `60` | Post-fight cooldown |
| Fight Candidate | `proximity_iou` | `0.08` | Engagement gate |
| Fight Candidate | `proximity_distance` | `1.25` | Max body-widths |
| Fight Candidate | `motion_energy_threshold` | `0.12` | Motion gate |
| Crowd | `density_high_threshold` | `0.35` | HIGH density |
| Crowd | `density_critical_threshold` | `0.55` | CRITICAL density |
| Crowd | `growth_alert_per_min` | `10.0` | Growth rate alert |
| Crowd | `panic_speed_threshold` | `2.2` | box-widths/sec |
| Crowd | `min_panic_persons` | `8` | Minimum persons |
| Fusion | `correlation_window_s` | `2.5` | Cross-modal window |
| Fusion | `weapon_cooldown_s` | `30.0` | Weapon incident gap |
| Fusion | `armed_fight_cooldown_s` | `60.0` | Armed fight gap |

### Subsystem B — Complete Threshold Map

| Subsystem | Parameter | Value | Purpose |
|---|---|---|---|
| Accident (YOLO) | `crash_conf` | `0.70` | Tier 1 detection |
| Accident (YOLO) | `crash_iou` | `0.65` | Tier 1 NMS |
| Accident (DETR) | `conf_threshold` | `0.70` | Fallback detector |
| Accident (DETR) | `iou_threshold` | `0.65` | Fallback NMS |
| Severity | Severe gate | ROI > 20% frame OR classifier="severe" | Level 3 escalation |
| Fire/Smoke | `fire_conf` | `0.75` | Fire detection |
| Dispatch | `alert_cooldown_sec` | `5.0` s | Snapshot cooldown |
| Emergency | `search_radius_km` | `3.0` km | Facility search |
| Emergency | `max_dispatch_per_type` | `2` | Max facilities |

---

## 8. Hardware & Infrastructure

### Development & Inference Machine

| Component | Specification |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **CUDA Version** | 12.1 |
| **PyTorch** | CUDA 12.1 build |
| **Peak VRAM (Weapon Training)** | 3.741 GB |
| **Peak VRAM (Full Pipeline)** | 238.1 MB |

### Edge Device — Raspberry Pi

| Component | Specification |
|---|---|
| **Device** | Raspberry Pi 4B |
| **Camera** | Logitech C270 USB HD Webcam |
| **Resolution** | 640 × 360 |
| **FPS** | 15 |
| **Encoder** | `libx264` (`ultrafast` preset, `zerolatency` tune) |
| **Pixel Format** | `yuv420p` |
| **GOP Size** | 15 |
| **Input Format** | MJPEG (from camera) |
| **Streaming Server** | MediaMTX |
| **RTSP Path** | `/drone` on port `8554` |
| **Auto-Start** | `systemd` service (`suraksha-stream.service`) |

### Sparsh CCTV Camera

| Component | Specification |
|---|---|
| **Type** | Sparsh CCTV (RTSP) |
| **IP** | `192.168.128.10` |
| **Port** | 554 |
| **Stream** | `avstream/channel=1/stream=1.sdp` |
| **Auth** | `admin:admin123` (factory default) |

---

## 9. API Surface

### Subsystem A — FastAPI (Port 8100)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/system/status` | GET | Full system status with AI metrics |
| `/video/status` | GET | Current video source and pipeline state |
| `/video/source` | POST | Dynamic video source switching |
| `/reset` | POST | Clear all incidents, snapshots, clips |

### Subsystem B — FastAPI

| Category | Key Endpoints |
|---|---|
| Surveillance | `/api/surveillance/start`, `/stop`, `/status`, `/mjpeg` |
| Emergency | `/api/emergency/dispatches`, `/facilities` |
| Settings | `/api/settings`, `/api/settings/update` |
| Dashboard | HTML dashboard at root |

---

## 10. Dependencies & Tech Stack

### Subsystem A

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11.x | Runtime |
| ultralytics | ≥ 8.3.0 | YOLO11 inference |
| PyTorch | CUDA 12.1 | GPU inference |
| opencv-python | ≥ 4.9.0 | Video processing |
| FastAPI | ≥ 0.110 | REST API |
| uvicorn | ≥ 0.29 | ASGI server |
| pydantic | ≥ 2.6 | Schema validation |
| websockets | ≥ 12.0 | Real-time events |
| scipy | ≥ 1.11 | Scientific computing |
| lap | ≥ 0.4.0 | Linear assignment (tracking) |
| PyYAML | ≥ 6.0 | Configuration |

### Subsystem B

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.14 | Runtime |
| ultralytics | ≥ 8.4.152 | YOLO inference |
| torch | 2.14.0 | GPU inference |
| torchvision | 0.29.0 | Image transforms |
| transformers | ≥ 4.35.0 | DETR model loading |
| safetensors | ≥ 0.4.0 | Weights format |
| timm | ≥ 1.0.0 | Backbone models |
| FastAPI | ≥ 0.142.2 | REST API |
| twilio | ≥ 9.11.2 | SMS alerts |
| groq | ≥ 1.7.0 | LLM integration |
| deepgram-sdk | ≥ 7.11.0 | Speech-to-text |
| livekit-agents | ≥ 0.12.21 | Voice agent |
| googlemaps | ≥ 4.10.0 | Geocoding |

---

> [!NOTE]
> This document is generated from source code inspection and training artefacts. All threshold values, model parameters, and metrics are extracted directly from configuration files, training logs, and source code — not from external documentation.
