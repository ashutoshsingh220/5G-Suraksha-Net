"""Regression and integration tests for low-resolution detection improvements."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from suraksha.config import AppConfig, DetectionConfig, PROJECT_ROOT, load_config
from suraksha.detection.tracker import TrackedPerson
from suraksha.fight.candidate import FightCandidateDetector
from suraksha.fight.temporal_classifier import TorchTemporalClassifier, build_scorer
from suraksha.incidents.schemas import IncidentReport


def test_detection_config_loads_calibrated_threshold():
    """Verify configs/app.yaml loads conf_threshold = 0.25 calibrated for low-res CCTV."""
    load_config.cache_clear()
    cfg = load_config()
    assert cfg.detection.conf_threshold == 0.25
    assert cfg.detection.imgsz == 640
    assert cfg.detection.iou_threshold == 0.50
    assert Path(cfg.detection.weights).name in ("yolo11s.pt", "yolo11m.pt")


def test_detection_config_matches_bytetrack_threshold():
    """Verify detection conf_threshold (0.25) aligns with ByteTrack track_high_thresh (0.25)."""
    import yaml
    with open(PROJECT_ROOT / "configs" / "bytetrack.yaml", "r", encoding="utf-8") as f:
        bytetrack_cfg = yaml.safe_load(f)

    cfg = load_config()
    # Detection threshold must not exceed track_high_thresh, otherwise ByteTrack's primary tier is starved
    assert cfg.detection.conf_threshold <= bytetrack_cfg["track_high_thresh"]
    assert bytetrack_cfg["track_high_thresh"] == 0.25
    assert bytetrack_cfg["track_buffer"] == 60


def test_tracked_person_contract():
    """Verify TrackedPerson dataclass preserves bounding box, center, and width contracts."""
    bbox = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float32)
    p = TrackedPerson(track_id=1, bbox_xyxy=bbox, confidence=0.28)
    assert p.track_id == 1
    assert p.confidence == 0.28
    assert p.width == 40.0
    np.testing.assert_allclose(p.center, np.array([30.0, 60.0], dtype=np.float32))


def test_candidate_pairing_with_calibrated_detections():
    """Verify FightCandidateDetector correctly pairs tracks produced under the calibrated detection threshold."""
    load_config.cache_clear()
    cfg = load_config()
    candidate_cfg = cfg.fight.candidate.model_copy()
    candidate_cfg.proximity_min_frames = 5
    candidate_cfg.motion_energy_threshold = 0.0
    detector = FightCandidateDetector(candidate_cfg, (240, 320))

    # Two close persons at 320x240
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    p1 = TrackedPerson(track_id=1, bbox_xyxy=np.array([100.0, 80.0, 140.0, 160.0], dtype=np.float32), confidence=0.28)
    p2 = TrackedPerson(track_id=2, bbox_xyxy=np.array([120.0, 80.0, 160.0, 160.0], dtype=np.float32), confidence=0.26)

    # Simulate 6 frames of close engagement
    cands = []
    for f in range(6):
        cands = detector.update(frame, [p1, p2], now=f * 0.033, fps=30.0)

    assert len(cands) == 1
    cand = cands[0]
    assert cand.track_ids == (1, 2)
    assert len(cand.features) == 6
    # 8-dimensional feature representation preserved
    assert cand.features[-1].shape == (8,)
    assert FightCandidateDetector.FEATURE_DIM == 8


def test_active_checkpoint_loads_and_accepts_feature_window():
    """Verify active champion checkpoint rwf2000_v2_best.pt loads and accepts [32, 8] input."""
    load_config.cache_clear()
    cfg = load_config()
    ckpt_path = PROJECT_ROOT / cfg.fight.temporal.model_weights
    assert ckpt_path.exists(), f"Active checkpoint {ckpt_path} missing"

    scorer = build_scorer(str(ckpt_path), device="cpu")
    assert isinstance(scorer, TorchTemporalClassifier)

    # Window of shape (32, 8)
    window = np.random.randn(32, 8).astype(np.float32)
    score = scorer.score(window)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_incident_schema_unchanged():
    """Verify IncidentReport schema remains strictly intact with version 1.0."""
    report = IncidentReport(
        incident_id="test_incident_123",
        camera_id="cam_01",
        incident_type="fight",
        severity="high",
        confidence=0.78,
        start_time="2026-09-30T18:30:00Z",
        details={"windows_scored": 3, "detector": "temporal"},
    )
    d = report.model_dump(mode="json")
    assert d["schema_version"] == "1.0"
    assert d["source_module"] == "crowd_fight"
    assert d["incident_type"] == "fight"
    assert d["confidence"] == 0.78
