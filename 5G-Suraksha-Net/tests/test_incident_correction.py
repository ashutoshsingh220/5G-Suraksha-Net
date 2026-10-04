"""Deterministic test matrix for Phase 1 Incident Detection Correction & Real-World Fix.

Covers all 18 critical operational scenarios:
1. Single person close to camera -> NORMAL
2. Two people standing close -> NORMAL
3. Two people talking -> NORMAL
4. Two people gesturing normally -> NORMAL
5. Two people walking -> NORMAL (GRU scores < 0.65, zero verified fights)
6. Two people crossing paths -> NORMAL (transient overlap < min_frames)
7. Group standing/talking -> NORMAL (density alone creates 0 incidents)
8. Brief contact / pushing -> NORMAL
9. Sustained physical interaction -> CANDIDATE
10. Sustained violent fight -> VERIFIED
11. Verified fight -> MODERATE severity
12. Moderate fight -> NO snapshot (snapshot suppressed, video clip recorded)
13. Candidate decay -> NORMAL (resets after decay_timeout_s without motion)
14. Aggressive but short interaction (<2.5s) -> NORMAL
15. Existing crowd panic logic remains independent (>=8 persons + panic)
16. Existing cooldown remains functional
17. Existing evidence clip lifecycle remains functional (T0-5s to T0+5s)
18. Existing four input sources remain functional
"""
from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from suraksha.config import (
    CandidateConfig,
    CaptureConfig,
    CrowdConfig,
    FightConfig,
    IncidentsConfig,
    TemporalConfig,
    VerifyConfig,
)
from suraksha.crowd.analyzer import (
    CrowdAnalyzer,
    MovementInfo,
    CrowdSnapshot,
    ZoneStat,
)
from suraksha.detection.tracker import TrackedPerson
from suraksha.fight.candidate import FightCandidateDetector
from suraksha.fight.recognizer import FightRecognizer, VerifiedFight
from suraksha.fight.temporal_classifier import TemporalScorer
from suraksha.incidents.evidence import EvidenceWriter
from suraksha.incidents.manager import EventBus, IncidentManager
from suraksha.incidents.schemas import IncidentReport, IncidentStatus, IncidentType, Severity


def _track(tid: int, cx: float, cy: float, w: float = 60.0, h: float = 160.0) -> TrackedPerson:
    return TrackedPerson(
        track_id=tid,
        bbox_xyxy=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        confidence=0.9,
    )


class MockConstantScorer(TemporalScorer):
    """Predictable mock temporal classifier returning a fixed score."""
    def __init__(self, score_val: float):
        self.score_val = score_val

    def score(self, window: np.ndarray) -> float:
        return self.score_val


# ---------------------------------------------------------------------------
# SCENARIO 1: One person standing alone close to camera -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_01_single_person_close_to_camera_no_incident(tmp_path):
    crowd_cfg = CrowdConfig(enable_density_incidents=False)
    inc_cfg = IncidentsConfig(snapshot_dir=str(tmp_path / "snaps"), clip_dir=str(tmp_path / "clips"), report_dir=str(tmp_path / "reps"))
    mgr = IncidentManager("cam_01", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus(), crowd_cfg=crowd_cfg)

    snap = CrowdSnapshot(
        timestamp=time.time(),
        person_count=1,
        zones=[ZoneStat(name="front_gate", person_count=1, density=0.85, level="critical")],
        growth_per_min=0.0,
        growth_alert=False,
        movement=MovementInfo(panic=False, fast_track_ratio=0.0, direction_entropy=0.0),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    reports = mgr.report_crowd(snap, frame)
    assert reports == [], "Single person close to camera must produce 0 incidents"
    assert len(mgr.recent) == 0


# ---------------------------------------------------------------------------
# SCENARIO 2: Two people standing close -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_02_two_people_standing_close_no_incident():
    # rel_dist = 50 / 60 = 0.83 <= 1.25. Motion = 0.02 (< 0.12)
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    det._flow_energy = MagicMock(return_value=0.04)  # motion = 0.04 * 30 / 60 = 0.02
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    cands = []
    for i in range(25):
        t1, t2 = _track(1, 400.0, 400.0), _track(2, 450.0, 400.0)
        cands = det.update(frame, [t1, t2], now=i * 0.033)

    assert cands == [], "Two people standing close without motion must be suppressed (0 candidates)"


# ---------------------------------------------------------------------------
# SCENARIO 3: Two people talking -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_03_two_people_talking_no_incident():
    # Subtle talking motion: head/lips move, motion = 0.06 (< 0.12)
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    det._flow_energy = MagicMock(return_value=0.12)  # motion = 0.12 * 30 / 60 = 0.06
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    cands = []
    for i in range(25):
        t1, t2 = _track(1, 400.0, 400.0), _track(2, 455.0, 400.0)
        cands = det.update(frame, [t1, t2], now=i * 0.033)

    assert cands == [], "Conversational motion (<0.12 bw/s) must produce 0 candidates"


# ---------------------------------------------------------------------------
# SCENARIO 4: Two people gesturing normally -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_04_two_people_gesturing_normally_no_incident():
    # Normal arm gesture while speaking: motion = 0.09 (< 0.12)
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    det._flow_energy = MagicMock(return_value=0.18)  # motion = 0.18 * 30 / 60 = 0.09
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    cands = []
    for i in range(25):
        t1, t2 = _track(1, 400.0, 400.0), _track(2, 460.0, 400.0)
        cands = det.update(frame, [t1, t2], now=i * 0.033)

    assert cands == [], "Normal conversational gesturing must be suppressed (0 candidates)"


# ---------------------------------------------------------------------------
# SCENARIO 5: Two people walking side-by-side -> NORMAL (Classifier rejects)
# ---------------------------------------------------------------------------
def test_scenario_05_two_people_walking_no_verified_fight():
    # Two people walking side by side: steady walking motion
    fcfg = FightConfig()
    rec = FightRecognizer(fcfg, (720, 1280))
    # Real GRU or mock returns 0.15 for walking (< 0.65)
    rec.scorer = MockConstantScorer(0.15)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rec.detector._flow_energy = MagicMock(return_value=0.40)  # motion = 0.20

    verified = []
    for i in range(35):
        x_off = i * 4.0
        t1, t2 = _track(1, 400.0 + x_off, 400.0), _track(2, 460.0 + x_off, 400.0)
        out = rec.update(frame, [t1, t2], now=i * 0.05)
        if out:
            verified.extend(out)

    assert verified == [], "Peaceful walking must never be verified as a fight (score 0.15 < 0.65)"


# ---------------------------------------------------------------------------
# SCENARIO 6: Two people crossing paths -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_06_two_people_crossing_paths_no_candidate():
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    det._flow_energy = MagicMock(return_value=0.6)  # motion = 0.30
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    cands = []
    for i in range(15):
        # Opposite walking paths: cross around frame 7-8
        x1 = 200.0 + i * 30.0
        x2 = 600.0 - i * 30.0
        t1, t2 = _track(1, x1, 400.0), _track(2, x2, 400.0)
        cands = det.update(frame, [t1, t2], now=i * 0.033)

    assert cands == [], "Transient path crossing (<8 frames in proximity) must produce 0 candidates"


# ---------------------------------------------------------------------------
# SCENARIO 7: Group standing/talking -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_07_group_standing_talking_no_incident(tmp_path):
    crowd_cfg = CrowdConfig(enable_density_incidents=False)
    inc_cfg = IncidentsConfig(snapshot_dir=str(tmp_path / "snaps"), clip_dir=str(tmp_path / "clips"), report_dir=str(tmp_path / "reps"))
    mgr = IncidentManager("cam_01", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus(), crowd_cfg=crowd_cfg)

    snap = CrowdSnapshot(
        timestamp=time.time(),
        person_count=10,
        zones=[
            ZoneStat(name="plaza", person_count=10, density=0.72, level="critical"),
            ZoneStat(name="corridor", person_count=5, density=0.58, level="high"),
        ],
        growth_per_min=0.0,
        growth_alert=False,
        movement=MovementInfo(panic=False, fast_track_ratio=0.05, direction_entropy=0.2),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    reports = mgr.report_crowd(snap, frame)
    assert reports == [], "Environmental crowd density alone must produce zero safety incidents"


# ---------------------------------------------------------------------------
# SCENARIO 8: Brief contact / accidental push -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_08_brief_contact_no_candidate():
    # Only 4 frames of contact (< proximity_min_frames=8)
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    det._flow_energy = MagicMock(return_value=1.5)  # high motion during bump
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    cands = []
    for i in range(4):
        t1, t2 = _track(1, 400.0, 400.0), _track(2, 445.0, 400.0)
        cands = det.update(frame, [t1, t2], now=i * 0.033)

    assert cands == [], "Brief contact (<8 frames) must not trigger candidate"


# ---------------------------------------------------------------------------
# SCENARIO 9: Sustained physical interaction / grappling -> CANDIDATE
# ---------------------------------------------------------------------------
def test_scenario_09_sustained_physical_interaction_creates_candidate():
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    # Grappling motion ~0.25 bw/s, leaning posture rel_dist = 1.15
    det._flow_energy = MagicMock(return_value=0.50)  # motion = 0.50 * 30 / 60 = 0.25 (>0.12)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    cands = []
    for i in range(12):
        t1, t2 = _track(1, 400.0, 390.0, w=65.0), _track(2, 470.0, 410.0, w=60.0)
        cands = det.update(frame, [t1, t2], now=i * 0.033)

    assert len(cands) == 1, "Sustained physical grappling must be emitted as a candidate"
    assert cands[0].track_ids == (1, 2)
    assert cands[0].frames_engaged >= 8


# ---------------------------------------------------------------------------
# SCENARIO 10: Sustained fight -> VERIFIED
# ---------------------------------------------------------------------------
def test_scenario_10_sustained_fight_verified():
    fcfg = FightConfig()
    rec = FightRecognizer(fcfg, (720, 1280))
    rec.scorer = MockConstantScorer(0.85)  # High score for fight action
    rec.detector._flow_energy = MagicMock(return_value=0.60)  # motion = 0.30

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    verified = []
    # 35 frames @ 0.1s = 3.5s duration (> 2.5s), stride 8 -> 4 windows
    for i in range(35):
        t1, t2 = _track(1, 400.0, 390.0, w=65.0), _track(2, 460.0, 410.0, w=60.0)
        out = rec.update(frame, [t1, t2], now=10.0 + i * 0.1)
        if out:
            verified.extend(out)

    assert len(verified) >= 1, "Sustained fight must be verified"
    assert verified[0].confidence >= 0.80
    assert verified[0].end_time - verified[0].start_time >= 2.5


# ---------------------------------------------------------------------------
# SCENARIO 11: Verified fight -> MODERATE severity
# ---------------------------------------------------------------------------
def test_scenario_11_verified_fight_emits_moderate_severity(tmp_path):
    inc_cfg = IncidentsConfig(snapshot_dir=str(tmp_path / "snaps"), clip_dir=str(tmp_path / "clips"), report_dir=str(tmp_path / "reps"))
    mgr = IncidentManager("cam_01", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus())

    fight = VerifiedFight(
        track_ids=(1, 2),
        roi_xyxy=np.array([100, 100, 300, 300], dtype=np.float32),
        start_time=100.0,
        end_time=103.0,
        confidence=0.88,
        windows_scored=3,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    report = mgr.report_fight(fight, frame)

    assert report is not None
    assert report.incident_type == IncidentType.FIGHT
    assert report.severity == Severity.MODERATE, "Verified fight must be classified as MODERATE severity"


# ---------------------------------------------------------------------------
# SCENARIO 12: Moderate fight -> NO snapshot (video evidence recorded, snapshot suppressed)
# ---------------------------------------------------------------------------
def test_scenario_12_moderate_fight_no_snapshot(tmp_path):
    inc_cfg = IncidentsConfig(snapshot_dir=str(tmp_path / "snaps"), clip_dir=str(tmp_path / "clips"), report_dir=str(tmp_path / "reps"))
    mgr = IncidentManager("cam_01", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus())

    fight = VerifiedFight(
        track_ids=(1, 2),
        roi_xyxy=np.array([100, 100, 300, 300], dtype=np.float32),
        start_time=100.0,
        end_time=103.0,
        confidence=0.85,
        windows_scored=3,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    report = mgr.report_fight(fight, frame)

    assert report is not None
    assert report.evidence.snapshot_path is None, "Snapshot MUST be suppressed for MODERATE fight"


# ---------------------------------------------------------------------------
# SCENARIO 13: Candidate decay -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_13_candidate_decay_returns_to_normal():
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)

    # 1. Active high-motion engagement for 10 frames
    det._flow_energy = MagicMock(return_value=0.50)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    t = 100.0
    for i in range(10):
        t1, t2 = _track(1, 400.0, 400.0), _track(2, 450.0, 400.0)
        cands = det.update(frame, [t1, t2], now=t)
        t += 0.05
    assert len(cands) == 1, "Candidate must be active during engagement"

    # 2. Motion ceases: pair stops moving, stands still for > 1.0s
    det._flow_energy = MagicMock(return_value=0.0)
    for i in range(25):
        t += 0.05  # 1.25s elapsed without motion
        cands = det.update(frame, [t1, t2], now=t)

    assert cands == [], "Candidate must decay after >1.0s without motion energy"


# ---------------------------------------------------------------------------
# SCENARIO 14: Aggressive but short interaction (<2.5s) -> NORMAL
# ---------------------------------------------------------------------------
def test_scenario_14_short_aggressive_interaction_no_verified_fight():
    fcfg = FightConfig()
    rec = FightRecognizer(fcfg, (720, 1280))
    rec.scorer = MockConstantScorer(0.90)  # High score, but duration is only 1.0s
    rec.detector._flow_energy = MagicMock(return_value=0.60)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    verified = []
    # 10 frames @ 0.1s = 1.0s duration (< 2.5s)
    for i in range(10):
        t1, t2 = _track(1, 400.0, 400.0), _track(2, 450.0, 400.0)
        out = rec.update(frame, [t1, t2], now=10.0 + i * 0.1)
        if out:
            verified.extend(out)

    assert verified == [], "Short interaction (<2.5s) must not verify as a confirmed fight"


# ---------------------------------------------------------------------------
# SCENARIO 15: Existing crowd panic logic remains independent
# ---------------------------------------------------------------------------
def test_scenario_15_crowd_panic_independent(tmp_path):
    crowd_cfg = CrowdConfig(enable_density_incidents=False, min_panic_persons=8)
    inc_cfg = IncidentsConfig(snapshot_dir=str(tmp_path / "snaps"), clip_dir=str(tmp_path / "clips"), report_dir=str(tmp_path / "reps"))
    mgr = IncidentManager("cam_01", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus(), crowd_cfg=crowd_cfg)

    snap = CrowdSnapshot(
        timestamp=time.time(),
        person_count=12,
        zones=[],
        growth_per_min=0.0,
        growth_alert=False,
        movement=MovementInfo(panic=True, fast_track_ratio=0.85, direction_entropy=0.75),
    )
    reports = mgr.report_crowd(snap, np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(reports) == 1
    r = reports[0]
    assert r.incident_type == IncidentType.CROWD_PANIC
    assert r.severity == Severity.HIGH
    assert r.person_count == 12


# ---------------------------------------------------------------------------
# SCENARIO 16: Existing evidence clip lifecycle: T0 - 5s to T0 + 5s
# ---------------------------------------------------------------------------
def test_scenario_16_evidence_clip_recording_lifecycle(tmp_path):
    inc_cfg = IncidentsConfig(
        clip_fps=15,
        clip_seconds_before=5,
        clip_seconds_after=5,
        snapshot_dir=str(tmp_path / "snaps"),
        clip_dir=str(tmp_path / "clips"),
        report_dir=str(tmp_path / "reps"),
    )
    writer = EvidenceWriter(inc_cfg)
    mgr = IncidentManager("cam_01", VerifyConfig(), writer, EventBus())

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # 1. Push 5 seconds of pre-event frames (t = 100.0 to 105.0, 75 frames @ 15fps)
    t = 100.0
    dt = 1.0 / 15.0
    for _ in range(75):
        writer.push_frame(frame, t)
        t += dt

    # Trigger incident at T0 = 105.0
    fight = VerifiedFight(
        track_ids=(1, 2),
        roi_xyxy=np.array([50, 50, 200, 200], dtype=np.float32),
        start_time=102.5,
        end_time=105.0,
        confidence=0.89,
        windows_scored=4,
    )
    report = mgr.report_fight(fight, frame)
    assert report is not None
    assert report.severity == Severity.MODERATE
    assert report.status == IncidentStatus.RECORDING_POST_EVENT
    assert report.evidence.snapshot_path is None  # Suppressed for MODERATE
    assert report.evidence.clip_path is None

    # 2. Push 5+ seconds of post-event frames (80 frames @ 15fps)
    completed_any = False
    for _ in range(80):
        t += dt
        done_ids = writer.push_frame(frame, t)
        if report.incident_id in done_ids:
            completed_any = True

    assert completed_any is True, "Recording session must complete at target_end_time"
    assert report.status == IncidentStatus.FINALIZED
    assert report.evidence.clip_path is not None
    assert Path(report.evidence.clip_path).exists()


# ---------------------------------------------------------------------------
# SCENARIO 17: Cooldown / deduplication remains functional
# ---------------------------------------------------------------------------
def test_scenario_17_incident_deduplication_cooldown(tmp_path):
    vcfg = VerifyConfig(incident_cooldown_s=10.0)
    inc_cfg = IncidentsConfig(snapshot_dir=str(tmp_path / "snaps"), clip_dir=str(tmp_path / "clips"), report_dir=str(tmp_path / "reps"))
    mgr = IncidentManager("cam_01", vcfg, EvidenceWriter(inc_cfg), EventBus())

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    fight = VerifiedFight(
        track_ids=(1, 2),
        roi_xyxy=np.array([50, 50, 200, 200], dtype=np.float32),
        start_time=100.0,
        end_time=102.5,
        confidence=0.85,
        windows_scored=3,
    )

    r1 = mgr.report_fight(fight, frame)
    assert r1 is not None

    r2 = mgr.report_fight(fight, frame)
    assert r2 is None, "Duplicate fight report within cooldown window must be suppressed"


# ---------------------------------------------------------------------------
# SCENARIO 18: Existing four input sources remain functional
# ---------------------------------------------------------------------------
def test_scenario_18_four_sources_configuration():
    from suraksha.capture.stream import StreamCapture

    # 1. Webcam source
    cfg_webcam = CaptureConfig(rtsp_url="0")
    cap_webcam = StreamCapture(cfg_webcam)
    assert cap_webcam.source_kind == "webcam"

    # 2. Raspberry Pi RTSP
    cfg_pi = CaptureConfig(rtsp_url="rtsp://10.254.18.48:8554/drone")
    cap_pi = StreamCapture(cfg_pi)
    assert cap_pi.source_kind == "stream"

    # 3. Sparsh / generic RTSP
    cfg_sparsh = CaptureConfig(rtsp_url="rtsp://192.168.1.100:554/live")
    cap_sparsh = StreamCapture(cfg_sparsh)
    assert cap_sparsh.source_kind == "stream"

    # 4. File / MP4
    cfg_file = CaptureConfig(rtsp_url="datasets/test.mp4")
    cap_file = StreamCapture(cfg_file)
    assert cap_file.source_kind == "file"

