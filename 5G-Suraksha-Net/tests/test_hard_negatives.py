"""Regression and hard-negative validation tests.

Proves:
1. Candidate with GRU score 0.13 does NOT verify.
2. Candidate below 0.65 does NOT verify.
3. One positive temporal window does NOT verify.
4. Two positive windows do NOT verify when three are required.
5. Insufficient action duration does NOT verify.
6. Standing close remains NORMAL.
7. Talking remains NORMAL.
8. Walking remains NORMAL.
9. Crossing remains NORMAL.
10. Dense peaceful crowd remains NORMAL.
11. Sustained fight with valid temporal evidence becomes MODERATE.
12. MODERATE fight produces no snapshot.
13. Existing video evidence lifecycle remains functional.
14. Existing cooldown/deduplication remains functional.
15. All four input sources remain intact.
16. Diagnostic telemetry exposes all required fields and rejection reasons.
17. Hard-case manifest updates idempotently.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from suraksha.capture.stream import StreamCapture
from suraksha.config import (
    CandidateConfig,
    CaptureConfig,
    CrowdConfig,
    FightConfig,
    IncidentsConfig,
    VerifyConfig,
)
from suraksha.crowd.analyzer import CrowdAnalyzer, CrowdSnapshot, MovementInfo, ZoneStat
from suraksha.detection.tracker import TrackedPerson
from suraksha.fight.candidate import FightCandidateDetector
from suraksha.fight.recognizer import FightRecognizer, VerifiedFight
from suraksha.fight.temporal_classifier import TemporalScorer
from suraksha.incidents.evidence import EvidenceWriter
from suraksha.incidents.manager import EventBus, IncidentManager
from suraksha.incidents.schemas import IncidentType, Severity


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


class MockSequenceScorer(TemporalScorer):
    """Returns a predefined sequence of scores."""
    def __init__(self, scores: list[float]):
        self.scores = list(scores)
        self.idx = 0

    def score(self, window: np.ndarray) -> float:
        if self.idx < len(self.scores):
            s = self.scores[self.idx]
            self.idx += 1
            return s
        return self.scores[-1] if self.scores else 0.0


# ---------------------------------------------------------------------------
# 1. Candidate with GRU score 0.13 does NOT verify
# ---------------------------------------------------------------------------
def test_gru_score_013_does_not_verify():
    cfg = FightConfig()
    cfg.temporal.score_threshold = 0.65
    cfg.verify.min_consecutive_windows = 3
    cfg.verify.min_duration_s = 2.5
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8

    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockConstantScorer(0.13)  # Real webcam test score from hard negative
    rec.detector._flow_energy = MagicMock(return_value=1.5)  # simulate movement

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = 1000.0

    verified = []
    for i in range(50):
        t = t0 + i * 0.1
        tracks = [_track(46, 500, 360, w=65.0), _track(76, 545, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=t, fps=10.0)
        verified.extend(vf)

    assert len(verified) == 0, "GRU score 0.13 must never produce a verified fight"
    telemetry = rec.get_telemetry()
    assert len(telemetry) >= 1
    t_item = telemetry[0]
    assert t_item["current_gru_score"] == 0.13
    assert t_item["verified_state"] is False
    assert t_item["consecutive_positive_windows"] == 0
    assert "score_below_threshold" in str(t_item["last_rejection_reason"])


# ---------------------------------------------------------------------------
# 2. Candidate below 0.65 does NOT verify
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("score", [0.0, 0.13, 0.45, 0.64])
def test_candidate_below_065_does_not_verify(score):
    cfg = FightConfig()
    cfg.temporal.score_threshold = 0.65
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockConstantScorer(score)
    rec.detector._flow_energy = MagicMock(return_value=1.5)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = 1000.0

    for i in range(40):
        t = t0 + i * 0.1
        tracks = [_track(1, 500, 360, w=65.0), _track(2, 545, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=t, fps=10.0)
        assert len(vf) == 0


# ---------------------------------------------------------------------------
# 3. One positive temporal window does NOT verify
# ---------------------------------------------------------------------------
def test_one_positive_window_does_not_verify():
    cfg = FightConfig()
    cfg.verify.min_consecutive_windows = 3
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockSequenceScorer([0.85, 0.20, 0.20, 0.20])
    rec.detector._flow_energy = MagicMock(return_value=1.5)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = 1000.0

    verified = []
    for i in range(40):
        t = t0 + i * 0.1
        tracks = [_track(1, 500, 360, w=65.0), _track(2, 545, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=t, fps=10.0)
        verified.extend(vf)

    assert len(verified) == 0, "Single isolated positive window must not verify"


# ---------------------------------------------------------------------------
# 4. Two positive windows do NOT verify when three are required
# ---------------------------------------------------------------------------
def test_two_positive_windows_do_not_verify_when_three_required():
    cfg = FightConfig()
    cfg.verify.min_consecutive_windows = 3
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockSequenceScorer([0.80, 0.85, 0.20, 0.20])
    rec.detector._flow_energy = MagicMock(return_value=1.5)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = 1000.0

    verified = []
    for i in range(40):
        t = t0 + i * 0.1
        tracks = [_track(1, 500, 360, w=65.0), _track(2, 545, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=t, fps=10.0)
        verified.extend(vf)

    assert len(verified) == 0, "Two positive windows when three required must not verify"


# ---------------------------------------------------------------------------
# 5. Insufficient action duration does NOT verify
# ---------------------------------------------------------------------------
def test_insufficient_action_duration_does_not_verify():
    cfg = FightConfig()
    cfg.verify.min_consecutive_windows = 2
    cfg.verify.min_duration_s = 3.0  # requires 3.0 seconds
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockConstantScorer(0.90)  # high score
    rec.detector._flow_energy = MagicMock(return_value=1.5)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = 1000.0

    # Only 1.5 seconds elapsed (15 frames at 10 fps)
    verified = []
    for i in range(15):
        t = t0 + i * 0.1
        tracks = [_track(1, 500, 360, w=65.0), _track(2, 545, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=t, fps=10.0)
        verified.extend(vf)

    assert len(verified) == 0, "High scores with duration < min_duration_s must not verify"


# ---------------------------------------------------------------------------
# 6. Standing close remains NORMAL
# ---------------------------------------------------------------------------
def test_standing_close_remains_normal():
    cfg = CandidateConfig(motion_energy_threshold=0.12)
    det = FightCandidateDetector(cfg, (720, 1280))
    det._flow_energy = MagicMock(return_value=0.0)  # zero optical flow
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 2 people standing completely still next to each other
    for i in range(30):
        tracks = [_track(10, 500, 360), _track(11, 550, 360)]
        cands = det.update(frame, tracks, now=1000.0 + i * 0.05, fps=20.0)

    assert len(cands) == 0, "Still standing close must not become candidate"


# ---------------------------------------------------------------------------
# 7. Talking remains NORMAL
# ---------------------------------------------------------------------------
def test_talking_remains_normal():
    cfg = FightConfig()
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockConstantScorer(0.12)  # Normal conversational gesturing score
    rec.detector._flow_energy = MagicMock(return_value=0.3)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    verified = []
    for i in range(30):
        tracks = [_track(1, 400, 360, w=65.0), _track(2, 450, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=1000.0 + i * 0.1, fps=10.0)
        verified.extend(vf)

    assert len(verified) == 0


# ---------------------------------------------------------------------------
# 8. Walking remains NORMAL
# ---------------------------------------------------------------------------
def test_walking_parallel_remains_normal():
    cfg = FightConfig()
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockConstantScorer(0.14)  # Typical walking score
    rec.detector._flow_energy = MagicMock(return_value=0.3)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    verified = []
    for i in range(30):
        # Moving together in parallel
        tracks = [_track(1, 400 + i * 5, 360, w=65.0), _track(2, 450 + i * 5, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=1000.0 + i * 0.1, fps=10.0)
        verified.extend(vf)

    assert len(verified) == 0


# ---------------------------------------------------------------------------
# 9. Crossing remains NORMAL
# ---------------------------------------------------------------------------
def test_crossing_remains_normal():
    cfg = CandidateConfig(proximity_min_frames=8)
    det = FightCandidateDetector(cfg, (720, 1280))
    det._flow_energy = MagicMock(return_value=1.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Cross paths: only near each other for 3 frames
    for i in range(20):
        x1 = 200 + i * 30  # moving right
        x2 = 800 - i * 30  # moving left
        tracks = [_track(1, x1, 360), _track(2, x2, 360)]
        cands = det.update(frame, tracks, now=1000.0 + i * 0.1, fps=10.0)
        assert len(cands) == 0, "Transient crossing must not trigger candidate"


# ---------------------------------------------------------------------------
# 10. Dense peaceful crowd remains NORMAL
# ---------------------------------------------------------------------------
def test_dense_peaceful_crowd_remains_normal(tmp_path):
    crowd_cfg = CrowdConfig(enable_density_incidents=False)
    inc_cfg = IncidentsConfig(
        snapshot_dir=str(tmp_path / "snaps"),
        clip_dir=str(tmp_path / "clips"),
        report_dir=str(tmp_path / "reps"),
    )
    mgr = IncidentManager("cam_test", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus(), crowd_cfg=crowd_cfg)

    # 15 people in scene, zero panic, zero rapid growth
    snap = CrowdSnapshot(
        timestamp=time.time(),
        person_count=15,
        zones=[
            ZoneStat(name="plaza", person_count=15, density=0.75, level="high"),
        ],
        growth_per_min=0.0,
        growth_alert=False,
        movement=MovementInfo(panic=False, fast_track_ratio=0.05, direction_entropy=0.2),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    reports = mgr.report_crowd(snap, frame)

    fight_reports = [r for r in reports if r.incident_type == IncidentType.FIGHT]
    assert len(fight_reports) == 0, "Peaceful crowd must never generate fight incidents"


# ---------------------------------------------------------------------------
# 11. Sustained fight with valid temporal evidence becomes MODERATE
# ---------------------------------------------------------------------------
def test_sustained_fight_becomes_moderate(tmp_path):
    cfg = FightConfig()
    cfg.verify.min_consecutive_windows = 3
    cfg.verify.min_duration_s = 2.0
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockConstantScorer(0.88)  # Real fight model score
    rec.detector._flow_energy = MagicMock(return_value=1.5)

    inc_cfg = IncidentsConfig(
        snapshot_dir=str(tmp_path / "snaps"),
        clip_dir=str(tmp_path / "clips"),
        report_dir=str(tmp_path / "reps"),
    )
    mgr = IncidentManager("cam_test", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = 1000.0

    verified = []
    # 40 frames (~4.0 seconds) of sustained interaction
    for i in range(40):
        t = t0 + i * 0.1
        tracks = [_track(1, 500, 360, w=65.0), _track(2, 545, 360, w=60.0)]
        vf = rec.update(frame, tracks, now=t, fps=10.0)
        verified.extend(vf)

    assert len(verified) >= 1
    report = mgr.report_fight(verified[0], frame)
    assert report is not None
    assert report.severity == Severity.MODERATE, "Fight severity must be MODERATE"
    assert report.incident_type == IncidentType.FIGHT


# ---------------------------------------------------------------------------
# 12. MODERATE fight produces no snapshot
# ---------------------------------------------------------------------------
def test_moderate_fight_produces_no_snapshot(tmp_path):
    inc_cfg = IncidentsConfig(
        snapshot_dir=str(tmp_path / "snaps"),
        clip_dir=str(tmp_path / "clips"),
        report_dir=str(tmp_path / "reps"),
    )
    mgr = IncidentManager("cam_test", VerifyConfig(), EvidenceWriter(inc_cfg), EventBus())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    vf = VerifiedFight(
        track_ids=(1, 2),
        roi_xyxy=np.array([400, 200, 600, 500]),
        start_time=1000.0,
        end_time=1003.0,
        confidence=0.88,
        windows_scored=5,
    )
    report = mgr.report_fight(vf, frame)
    assert report is not None
    assert report.evidence is not None
    assert report.evidence.snapshot_path is None, "MODERATE fight must suppress snapshot"


# ---------------------------------------------------------------------------
# 13. Existing video evidence lifecycle remains functional
# ---------------------------------------------------------------------------
def test_video_evidence_lifecycle_functional():
    mock_writer = MagicMock()
    mgr = IncidentManager("cam_test", VerifyConfig(), mock_writer, EventBus())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    vf = VerifiedFight(
        track_ids=(10, 20),
        roi_xyxy=np.array([100, 100, 300, 300]),
        start_time=1000.0,
        end_time=1003.0,
        confidence=0.85,
        windows_scored=4,
    )
    report = mgr.report_fight(vf, frame)
    assert report is not None
    mock_writer.start_clip_session.assert_called_once()


# ---------------------------------------------------------------------------
# 14. Existing cooldown/deduplication remains functional
# ---------------------------------------------------------------------------
def test_cooldown_deduplication(tmp_path):
    inc_cfg = IncidentsConfig(
        snapshot_dir=str(tmp_path / "snaps"),
        clip_dir=str(tmp_path / "clips"),
        report_dir=str(tmp_path / "reps"),
    )
    mgr = IncidentManager("cam_test", VerifyConfig(cooldown_s=10.0), EvidenceWriter(inc_cfg), EventBus())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    vf = VerifiedFight(
        track_ids=(1, 2),
        roi_xyxy=np.array([100, 100, 200, 200]),
        start_time=1000.0,
        end_time=1003.0,
        confidence=0.85,
        windows_scored=3,
    )
    r1 = mgr.report_fight(vf, frame)
    assert r1 is not None

    r2 = mgr.report_fight(vf, frame)
    assert r2 is None, "Second report within cooldown must be suppressed"


# ---------------------------------------------------------------------------
# 15. All four input sources remain intact
# ---------------------------------------------------------------------------
def test_four_sources_intact():
    # Webcam
    c1 = StreamCapture(CaptureConfig(rtsp_url="0"))
    assert c1.source_kind == "webcam"

    # Pi RTSP
    c2 = StreamCapture(CaptureConfig(rtsp_url="rtsp://10.254.18.48:8554/drone"))
    assert c2.source_kind == "stream"

    # Sparsh RTSP
    c3 = StreamCapture(CaptureConfig(rtsp_url="rtsp://192.168.1.50:554/live"))
    assert c3.source_kind == "stream"

    # File / MP4
    c4 = StreamCapture(CaptureConfig(rtsp_url="test.mp4"))
    assert c4.source_kind == "file"


# ---------------------------------------------------------------------------
# 16. Diagnostic telemetry completeness (all fields & rejection reasons)
# ---------------------------------------------------------------------------
def test_diagnostic_telemetry_completeness():
    cfg = FightConfig()
    cfg.temporal.stride_frames = 2
    cfg.temporal.window_frames = 8
    rec = FightRecognizer(cfg, frame_shape=(720, 1280), device="cpu")
    rec.scorer = MockConstantScorer(0.13)
    rec.detector._flow_energy = MagicMock(return_value=1.5)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(25):
        tracks = [_track(46, 500, 360, w=65.0), _track(76, 545, 360, w=60.0)]
        rec.update(frame, tracks, now=1000.0 + i * 0.1, fps=10.0)

    telemetry = rec.get_telemetry()
    assert len(telemetry) == 1
    t = telemetry[0]

    required_keys = [
        "pair_id", "track_a", "track_b", "distance", "normalized_distance",
        "iou", "motion_energy", "motion_variance", "candidate_duration",
        "candidate_state", "temporal_window_count", "current_gru_score",
        "verification_threshold", "consecutive_positive_windows",
        "required_positive_windows", "action_duration",
        "minimum_action_duration", "verified_state", "incident_created",
        "last_rejection_reason",
    ]
    for k in required_keys:
        assert k in t, f"Missing telemetry field: {k}"

    assert t["track_a"] == 46
    assert t["track_b"] == 76
    assert t["current_gru_score"] == 0.13
    assert t["verified_state"] is False
    assert "score_below_threshold" in str(t["last_rejection_reason"])


# ---------------------------------------------------------------------------
# 17. Hard cases manifest updates idempotently
# ---------------------------------------------------------------------------
def test_manifest_idempotency(tmp_path):
    from scripts.update_hard_cases_manifest import update_manifest

    manifest_file = tmp_path / "hard_cases.csv"
    scan_dir = tmp_path / "hard_negative"
    scan_dir.mkdir()

    # Create dummy images
    (scan_dir / "Screenshot 2026-10-01 220440.png").write_bytes(b"dummy1")
    (scan_dir / "Screenshot 2026-10-01 214107.png").write_bytes(b"dummy2")

    count1 = update_manifest(manifest_file, scan_dir)
    assert count1 == 2

    # Second pass: must not duplicate
    count2 = update_manifest(manifest_file, scan_dir)
    assert count2 == 2

    with open(manifest_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        categories = {r["category"] for r in reader}
        assert "standing_close" in categories
