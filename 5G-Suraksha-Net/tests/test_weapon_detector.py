"""Unit and integration tests for WeaponDetector and temporal persistence tracking."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from suraksha.capture.stream import FrameEvent
from suraksha.config import AppConfig, WeaponConfig, WeaponConfirmationConfig
from suraksha.detection.weapon import (
    RawWeaponDetection,
    WeaponDetector,
    WeaponEvent,
    WeaponPersistenceTracker,
    WeaponState,
    compute_iou,
    compute_proximity,
)


# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------

def make_frame_event(frame_idx: int, frame: np.ndarray | None = None, timestamp: float | None = None) -> FrameEvent:
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    if timestamp is None:
        timestamp = 1000.0 + frame_idx * 0.066
    return FrameEvent(
        frame=frame,
        frame_idx=frame_idx,
        timestamp=timestamp,
        source_fps=15.0,
    )


def make_box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class MockYOLOModel:
    """Mock YOLO model that returns scripted detections."""

    def __init__(self, script: dict[int, list[tuple[list[float], float, int]]] | None = None) -> None:
        self.script = script or {}
        self.call_count = 0

    def predict(self, frame: np.ndarray, **kwargs) -> list[Any]:
        self.call_count += 1
        # kwargs doesn't have frame_idx, so we can use call_count
        dets_for_call = self.script.get(self.call_count, [])
        if not dets_for_call:
            mock_res = MagicMock()
            mock_res.boxes = None
            return [mock_res]

        boxes_mock = MagicMock()
        boxes_list = [d[0] for d in dets_for_call]
        confs_list = [d[1] for d in dets_for_call]
        clss_list = [d[2] for d in dets_for_call]

        boxes_mock.__len__.return_value = len(boxes_list)
        boxes_mock.xyxy = MagicMock()
        boxes_mock.xyxy.cpu().numpy.return_value = np.array(boxes_list, dtype=np.float32)
        boxes_mock.conf = MagicMock()
        boxes_mock.conf.cpu().numpy.return_value = np.array(confs_list, dtype=np.float32)
        boxes_mock.cls = MagicMock()
        boxes_mock.cls.cpu().numpy.return_value = np.array(clss_list, dtype=np.float32)

        mock_res = MagicMock()
        mock_res.boxes = boxes_mock
        return [mock_res]


# ---------------------------------------------------------------------------
# Unit Tests: Geometry & Matching
# ---------------------------------------------------------------------------

def test_compute_iou_identical_and_disjoint():
    b1 = make_box(10.0, 10.0, 50.0, 50.0)
    b2 = make_box(10.0, 10.0, 50.0, 50.0)
    assert pytest.approx(compute_iou(b1, b2), 0.01) == 1.0

    b3 = make_box(100.0, 100.0, 150.0, 150.0)
    assert compute_iou(b1, b3) == 0.0


def test_compute_proximity():
    b1 = make_box(10.0, 10.0, 50.0, 50.0)
    b2 = make_box(12.0, 12.0, 52.0, 52.0)
    prox = compute_proximity(b1, b2)
    assert prox > 0.90


# ---------------------------------------------------------------------------
# Unit Tests: Persistence State Machine & Confirmation Rules
# ---------------------------------------------------------------------------

def test_no_weapon_returns_normal():
    """No detections in frame keeps detector state in NORMAL."""
    tracker = WeaponPersistenceTracker()
    state, telemetry, cand_cnt, conf_cnt = tracker.update([], timestamp=100.0)
    assert state == WeaponState.NORMAL
    assert len(telemetry) == 0
    assert cand_cnt == 0
    assert conf_cnt == 0


def test_one_frame_weapon_detection_is_candidate_only():
    """A single high-confidence detection creates a candidate, NOT a confirmed alert."""
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=3, window_size=5)
    det = RawWeaponDetection(
        bbox_xyxy=make_box(50.0, 50.0, 100.0, 100.0),
        confidence=0.88,
        cls_id=2,
        class_name="pistol",
    )
    state, telemetry, cand_cnt, conf_cnt = tracker.update([det], timestamp=100.0)
    assert state == WeaponState.WEAPON_CANDIDATE
    assert cand_cnt == 1
    assert conf_cnt == 0
    assert len(telemetry) == 1
    assert telemetry[0].confirmation_state == WeaponState.WEAPON_CANDIDATE.value
    assert telemetry[0].persistence_count == 1
    assert telemetry[0].weapon_class == "pistol"


def test_two_of_five_detections_is_candidate_only():
    """Two detections out of five frames (2/5) does not meet min_hits=3, remains candidate."""
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=3, window_size=5)
    box = make_box(50.0, 50.0, 100.0, 100.0)
    det = RawWeaponDetection(bbox_xyxy=box, confidence=0.85, cls_id=0, class_name="knife")

    # Frame 1: Hit
    state1, _, cand1, conf1 = tracker.update([det], timestamp=100.0)
    assert state1 == WeaponState.WEAPON_CANDIDATE
    assert cand1 == 1 and conf1 == 0

    # Frame 2: Miss
    state2, _, cand2, conf2 = tracker.update([], timestamp=100.1)
    assert state2 == WeaponState.WEAPON_CANDIDATE

    # Frame 3: Hit (2 hits so far)
    state3, _, cand3, conf3 = tracker.update([det], timestamp=100.2)
    assert state3 == WeaponState.WEAPON_CANDIDATE
    assert cand3 == 1 and conf3 == 0

    # Frame 4: Miss
    state4, _, cand4, conf4 = tracker.update([], timestamp=100.3)
    assert state4 == WeaponState.WEAPON_CANDIDATE

    # Frame 5: Miss
    state5, telemetry5, cand5, conf5 = tracker.update([], timestamp=100.4)
    assert state5 == WeaponState.WEAPON_CANDIDATE
    assert cand5 == 1 and conf5 == 0
    assert telemetry5[0].persistence_count == 2


def test_three_of_five_detections_confirms_weapon():
    """Three detections out of five frames (3/5) meets min_hits=3, transitions to CONFIRMED."""
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=3, window_size=5)
    box = make_box(50.0, 50.0, 120.0, 120.0)
    det = RawWeaponDetection(bbox_xyxy=box, confidence=0.78, cls_id=1, class_name="long_gun")

    # Frame 1: Hit (1/1)
    tracker.update([det], timestamp=100.0)
    # Frame 2: Miss (1/2)
    tracker.update([], timestamp=100.1)
    # Frame 3: Hit (2/3)
    tracker.update([det], timestamp=100.2)
    # Frame 4: Miss (2/4)
    tracker.update([], timestamp=100.3)
    # Frame 5: Hit (3/5 -> CONFIRMED!)
    state5, telemetry5, cand5, conf5 = tracker.update([det], timestamp=100.4)

    assert state5 == WeaponState.WEAPON_CONFIRMED
    assert conf5 == 1
    assert cand5 == 0
    assert len(telemetry5) == 1
    assert telemetry5[0].confirmation_state == WeaponState.WEAPON_CONFIRMED.value
    assert telemetry5[0].weapon_class == "long_gun"
    assert telemetry5[0].persistence_count == 3


def test_confidence_below_confirm_threshold_does_not_confirm():
    """Low-confidence detections (e.g. 0.35, below confirm_conf_threshold 0.65) stay CANDIDATE."""
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=3, window_size=5)
    box = make_box(50.0, 50.0, 100.0, 100.0)
    low_conf_det = RawWeaponDetection(bbox_xyxy=box, confidence=0.35, cls_id=2, class_name="pistol")

    for i in range(5):
        state, _, cand, conf = tracker.update([low_conf_det], timestamp=100.0 + i * 0.1)

    # 5 detections occurred, but none met confirm_conf_threshold (0.65), so remains CANDIDATE
    assert state == WeaponState.WEAPON_CANDIDATE
    assert cand == 1
    assert conf == 0


def test_class_change_resets_and_does_not_incorrectly_confirm():
    """If detection changes class at the same location, it does NOT aggregate hits to confirm."""
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=3, window_size=5, expiry_frames=5)
    box = make_box(50.0, 50.0, 100.0, 100.0)

    knife_det = RawWeaponDetection(bbox_xyxy=box, confidence=0.80, cls_id=0, class_name="knife")
    pistol_det = RawWeaponDetection(bbox_xyxy=box, confidence=0.80, cls_id=2, class_name="pistol")

    # Frame 1: Knife (knife hits=1)
    tracker.update([knife_det], timestamp=100.0)
    # Frame 2: Knife (knife hits=2)
    tracker.update([knife_det], timestamp=100.1)

    # Frame 3: Class changes to Pistol! (Knife missed, Pistol new track hits=1)
    state3, telemetry3, cand3, conf3 = tracker.update([pistol_det], timestamp=100.2)
    assert conf3 == 0
    # Both knife and pistol are separate candidate tracks, neither confirmed
    assert cand3 == 2
    classes = {t.weapon_class for t in telemetry3}
    assert classes == {"knife", "pistol"}

    # Frame 4: Pistol (Pistol hits=2, Knife missed again)
    state4, _, cand4, conf4 = tracker.update([pistol_det], timestamp=100.3)
    assert conf4 == 0

    # Neither track reached 3 hits for its own class
    assert state4 == WeaponState.WEAPON_CANDIDATE


def test_disappearing_weapon_expires_correctly():
    """Disappearing weapon tracks expire after expiry_frames without detections, returning to NORMAL."""
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=2, window_size=5, expiry_frames=3)
    box = make_box(40.0, 40.0, 80.0, 80.0)
    det = RawWeaponDetection(bbox_xyxy=box, confidence=0.90, cls_id=2, class_name="pistol")

    # Hit 1 & 2 -> Confirmed
    tracker.update([det], timestamp=100.0)
    state2, _, _, conf2 = tracker.update([det], timestamp=100.1)
    assert state2 == WeaponState.WEAPON_CONFIRMED
    assert conf2 == 1

    # Frame 3: Miss 1
    tracker.update([], timestamp=100.2)
    # Frame 4: Miss 2
    tracker.update([], timestamp=100.3)
    # Frame 5: Miss 3 (consecutive_misses == expiry_frames -> track dropped!)
    state5, telemetry5, cand5, conf5 = tracker.update([], timestamp=100.4)

    assert state5 == WeaponState.NORMAL
    assert len(telemetry5) == 0
    assert cand5 == 0 and conf5 == 0


def test_multiple_simultaneous_weapons():
    """Multiple distinct weapons (e.g. knife and pistol) tracked simultaneously."""
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=2, window_size=5)

    knife_box = make_box(20.0, 20.0, 60.0, 60.0)
    pistol_box = make_box(200.0, 200.0, 260.0, 260.0)

    det_knife = RawWeaponDetection(bbox_xyxy=knife_box, confidence=0.85, cls_id=0, class_name="knife")
    det_pistol = RawWeaponDetection(bbox_xyxy=pistol_box, confidence=0.90, cls_id=2, class_name="pistol")

    # Frame 1: Both detected (Candidates)
    state1, tele1, cand1, conf1 = tracker.update([det_knife, det_pistol], timestamp=100.0)
    assert state1 == WeaponState.WEAPON_CANDIDATE
    assert cand1 == 2 and conf1 == 0

    # Frame 2: Both detected again -> Both CONFIRMED
    state2, tele2, cand2, conf2 = tracker.update([det_knife, det_pistol], timestamp=100.1)
    assert state2 == WeaponState.WEAPON_CONFIRMED
    assert conf2 == 2 and cand2 == 0
    assert len(tele2) == 2
    classes = {t.weapon_class for t in tele2}
    assert classes == {"knife", "pistol"}


def test_all_three_classes_supported():
    """Verify knife, long_gun, and pistol are all recognized and mapped correctly."""
    tracker = WeaponPersistenceTracker(min_hits=1, confirm_conf_threshold=0.5)

    classes_to_test = [(0, "knife"), (1, "long_gun"), (2, "pistol")]
    for cid, cname in classes_to_test:
        tracker.reset()
        det = RawWeaponDetection(
            bbox_xyxy=make_box(10.0, 10.0, 50.0, 50.0),
            confidence=0.85,
            cls_id=cid,
            class_name=cname,
        )
        state, tele, _, conf = tracker.update([det], timestamp=100.0)
        assert state == WeaponState.WEAPON_CONFIRMED
        assert tele[0].weapon_class == cname


# ---------------------------------------------------------------------------
# Unit Tests: WeaponDetector Wrapper & Gating
# ---------------------------------------------------------------------------

def test_detector_disabled_returns_normal_zero_latency():
    """When detector is disabled in config, update returns NORMAL immediately with 0 latency."""
    cfg = WeaponConfig(enabled=False)
    detector = WeaponDetector(cfg=cfg, model=None)

    ev = make_frame_event(1)
    weap_event = detector.update(ev)

    assert weap_event.state == WeaponState.NORMAL
    assert weap_event.inference_latency_ms == 0.0
    assert len(weap_event.telemetry) == 0
    assert detector.inference_count == 0


def test_inference_interval_skips_model_forward():
    """With inference_interval=2, model is only evaluated on even frame indices."""
    mock_model = MockYOLOModel()
    cfg = WeaponConfig(enabled=True, inference_interval=2)
    detector = WeaponDetector(cfg=cfg, model=mock_model)

    # Frame 0: should infer (0 % 2 == 0)
    ev0 = make_frame_event(0)
    detector.update(ev0)
    assert mock_model.call_count == 1

    # Frame 1: should skip forward pass (1 % 2 != 0)
    ev1 = make_frame_event(1)
    detector.update(ev1)
    assert mock_model.call_count == 1  # Not incremented!

    # Frame 2: should infer (2 % 2 == 0)
    ev2 = make_frame_event(2)
    detector.update(ev2)
    assert mock_model.call_count == 2


def test_subscribers_receive_weapon_events():
    """Subscribers receive published WeaponEvents synchronously."""
    mock_model = MockYOLOModel(script={
        1: [([10.0, 10.0, 60.0, 60.0], 0.88, 2)],
    })
    cfg = WeaponConfig(enabled=True)
    detector = WeaponDetector(cfg=cfg, model=mock_model)

    received_events = []
    detector.subscribe(lambda ev: received_events.append(ev))

    ev = make_frame_event(0)
    weap_event = detector.update(ev)

    assert len(received_events) == 1
    assert received_events[0] is weap_event
    assert received_events[0].state == WeaponState.WEAPON_CANDIDATE


def test_weapon_observability_metrics():
    """Verify get_metrics() returns all required telemetry keys."""
    mock_model = MockYOLOModel(script={
        1: [([10.0, 10.0, 60.0, 60.0], 0.88, 0)],
    })
    cfg = WeaponConfig(enabled=True)
    detector = WeaponDetector(cfg=cfg, model=mock_model)

    detector.update(make_frame_event(0))
    metrics = detector.get_metrics()

    assert "enabled" in metrics and metrics["enabled"] is True
    assert "state" in metrics
    assert "inference_latency_ms" in metrics
    assert "mean_latency_ms" in metrics
    assert "p95_latency_ms" in metrics
    assert "inference_count" in metrics and metrics["inference_count"] == 1
    assert "total_frames_processed" in metrics and metrics["total_frames_processed"] == 1
    assert "candidate_count" in metrics
    assert "confirmed_count" in metrics
    assert "per_class_detections" in metrics
    assert metrics["per_class_detections"]["knife"] == 1


# ---------------------------------------------------------------------------
# Integration Tests: FrameEvent -> WeaponDetector -> WeaponEvent
# ---------------------------------------------------------------------------

def test_frame_event_to_weapon_event_full_chain():
    """End-to-end integration: FrameEvent -> WeaponDetector -> WeaponEvent with telemetry."""
    # Scripted mock to simulate 3 consecutive pistol detections followed by a miss
    mock_model = MockYOLOModel(script={
        1: [([100.0, 100.0, 200.0, 200.0], 0.75, 2)],
        2: [([102.0, 101.0, 201.0, 202.0], 0.82, 2)],
        3: [([101.0, 100.0, 203.0, 201.0], 0.88, 2)],
        4: [],
    })
    confirm_cfg = WeaponConfirmationConfig(confirm_conf_threshold=0.65, min_hits=3, window_size=5)
    cfg = WeaponConfig(enabled=True, confirmation=confirm_cfg)
    detector = WeaponDetector(cfg=cfg, model=mock_model)

    # Frame 1: Candidate
    ev1 = detector.update(make_frame_event(1))
    assert ev1.state == WeaponState.WEAPON_CANDIDATE
    assert ev1.candidate_count == 1
    assert ev1.confirmed_count == 0

    # Frame 2: Candidate (2 hits)
    ev2 = detector.update(make_frame_event(2))
    assert ev2.state == WeaponState.WEAPON_CANDIDATE

    # Frame 3: Confirmed (3 hits)
    ev3 = detector.update(make_frame_event(3))
    assert ev3.state == WeaponState.WEAPON_CONFIRMED
    assert ev3.confirmed_count == 1
    assert len(ev3.telemetry) == 1
    t = ev3.telemetry[0]
    assert t.weapon_class == "pistol"
    assert t.confidence == pytest.approx(0.88, 0.01)
    assert t.persistence_count == 3
    assert t.confirmation_state == "WEAPON_CONFIRMED"

    # Frame 4: Miss (Window holds 3 hits -> remains confirmed)
    ev4 = detector.update(make_frame_event(4))
    assert ev4.state == WeaponState.WEAPON_CONFIRMED
    assert ev4.telemetry[0].persistence_count == 3


def test_pipeline_dual_branch_isolation(tmp_path):
    """Verify FrameEvent concurrently dispatches to both WeaponDetector and Person/Crowd/Fight branches without interference."""
    from suraksha.config import load_config
    from suraksha.pipeline import CrowdFightPipeline
    from suraksha.incidents.manager import EventBus

    load_config.cache_clear()
    cfg = load_config()
    cfg.incidents.snapshot_dir = str(tmp_path / "snapshots")
    cfg.incidents.clip_dir = str(tmp_path / "clips")
    cfg.incidents.report_dir = str(tmp_path / "incidents")

    bus = EventBus()
    received_weapons = []
    bus.subscribe_weapon(lambda w: received_weapons.append(w))

    pipeline = CrowdFightPipeline(cfg=cfg, bus=bus)

    # Mock the weapon model to return a knife
    mock_weapon_model = MockYOLOModel(script={
        1: [([50.0, 50.0, 100.0, 100.0], 0.85, 0)],
    })
    pipeline.weapon_detector.model = mock_weapon_model

    # Run the init components manually
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipeline._init_components(frame.shape[:2], source_fps=15.0)

    # Create FrameEvent
    ev = FrameEvent(frame=frame, frame_idx=0, timestamp=1000.0, source_fps=15.0)

    # Process frame through Branch A and Branch B
    # Branch A: Weapon
    weapon_ev = pipeline.weapon_detector.update(ev)
    # Branch B: Tracker + Crowd + Fight
    tracks = pipeline.tracker.update(ev.frame)
    crowd = pipeline.crowd.update(ev.frame, tracks, ev.timestamp)
    verified = pipeline.fight.update(ev.frame, tracks, ev.timestamp, fps=ev.source_fps)

    # Assertions
    assert weapon_ev.state == WeaponState.WEAPON_CANDIDATE
    assert len(received_weapons) == 1
    assert received_weapons[0].telemetry[0].weapon_class == "knife"

    # Branch B results intact
    assert isinstance(tracks, list)
    assert crowd.person_count >= 0
    assert isinstance(verified, list)

