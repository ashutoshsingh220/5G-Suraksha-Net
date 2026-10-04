"""Comprehensive deterministic test suite for Phase 2 Multimodal Incident Fusion.

Covers all 25 mandatory scenarios:
 1. No weapon + no fight -> no weapon incident
 2. Weapon candidate only -> no incident
 3. Confirmed weapon alone -> HIGH WEAPON incident
 4. Confirmed knife alone -> HIGH WEAPON incident
 5. Confirmed pistol alone -> HIGH WEAPON incident
 6. Confirmed long_gun alone -> HIGH WEAPON incident
 7. Fight without weapon -> existing fight behavior unchanged (MODERATE)
 8. Confirmed weapon + verified fight -> CRITICAL ARMED_FIGHT
 9. Weapon near Person A while Person B is fighting elsewhere -> NOT ARMED_FIGHT (bystander isolation)
10. Weapon disappears before fight verification -> must NOT produce false ARMED_FIGHT
11. Fight begins shortly after weapon confirmation -> deterministic escalation
12. Fight ends while weapon remains confirmed -> remains weapon incident, no duplicate recreation
13. Same weapon remains visible for 30+ frames -> exactly one active incident
14. Multiple weapons -> deterministic handling (two distinct incidents)
15. Class switch knife -> pistol -> existing persistence rules respected
16. Weapon confidence below threshold -> no escalation
17. Weapon + crowd density only -> do not automatically create ARMED_FIGHT
18. Weapon + crowd panic without fight -> HIGH weapon incident, no false ARMED_FIGHT
19. Evidence generation -> snapshot at incident time
20. Evidence generation -> approximately T-5 to T+5 clip
21. Evidence generation -> no blocking of inference loop (async_write)
22. Cooldown -> no duplicate incidents
23. New event after cooldown -> new incident allowed
24. Existing fight evidence -> unchanged (snapshot suppressed, clip recorded)
25. Existing crowd behavior -> unchanged
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from suraksha.config import (
    CrowdConfig,
    FusionConfig,
    IncidentsConfig,
    VerifyConfig,
)
from suraksha.crowd.analyzer import CrowdSnapshot, MovementInfo, ZoneStat
from suraksha.detection.tracker import TrackedPerson
from suraksha.detection.weapon import (
    RawWeaponDetection,
    WeaponEvent,
    WeaponPersistenceTracker,
    WeaponState,
    WeaponTelemetry,
)
from suraksha.fight.recognizer import VerifiedFight
from suraksha.incidents.evidence import EvidenceWriter
from suraksha.incidents.fusion import CorrelatedArmedFight, IncidentFusionEngine
from suraksha.incidents.manager import EventBus, IncidentManager
from suraksha.incidents.schemas import IncidentReport, IncidentStatus, IncidentType, Severity


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _track(tid: int, cx: float, cy: float, w: float = 60.0, h: float = 160.0) -> TrackedPerson:
    """Helper to create TrackedPerson centered at (cx, cy)."""
    return TrackedPerson(
        track_id=tid,
        bbox_xyxy=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        confidence=0.90,
    )


def _fight(
    track_ids: tuple[int, ...] = (1, 2),
    roi_xyxy: tuple[float, float, float, float] = (50.0, 50.0, 200.0, 200.0),
    start_time: float = 100.0,
    end_time: float = 102.5,
    confidence: float = 0.88,
    windows_scored: int = 4,
) -> VerifiedFight:
    return VerifiedFight(
        track_ids=track_ids,
        roi_xyxy=np.array(roi_xyxy, dtype=np.float32),
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
        windows_scored=windows_scored,
    )


def _weapon_telemetry(
    weapon_class: str = "pistol",
    confidence: float = 0.85,
    bbox: list[float] | None = None,
    timestamp: float = 102.5,
    track_id: int = 1,
    confirmation_state: str = "WEAPON_CONFIRMED",
    persistence_count: int = 4,
) -> WeaponTelemetry:
    if bbox is None:
        bbox = [70.0, 70.0, 110.0, 110.0]
    return WeaponTelemetry(
        weapon_class=weapon_class,
        confidence=confidence,
        bbox=bbox,
        frame_timestamp=timestamp,
        persistence_count=persistence_count,
        confirmation_state=confirmation_state,
        track_id=track_id,
    )


def _weapon_event(
    frame_idx: int = 10,
    timestamp: float = 102.5,
    state: WeaponState = WeaponState.WEAPON_CONFIRMED,
    telemetry: list[WeaponTelemetry] | None = None,
) -> WeaponEvent:
    tels = telemetry if telemetry is not None else [_weapon_telemetry()]
    cand_cnt = sum(1 for t in tels if t.confirmation_state == "WEAPON_CANDIDATE")
    conf_cnt = sum(1 for t in tels if t.confirmation_state == "WEAPON_CONFIRMED")
    return WeaponEvent(
        frame_idx=frame_idx,
        timestamp=timestamp,
        state=state,
        telemetry=tels,
        candidate_count=cand_cnt,
        confirmed_count=conf_cnt,
        detections=[],
        inference_latency_ms=12.5,
    )


def _make_manager(tmp_path, verify_cfg: VerifyConfig | None = None, async_write: bool = False):
    vcfg = verify_cfg or VerifyConfig(incident_cooldown_s=30.0)
    inc_cfg = IncidentsConfig(
        clip_fps=15,
        clip_seconds_before=5,
        clip_seconds_after=5,
        snapshot_dir=str(tmp_path / "snaps"),
        clip_dir=str(tmp_path / "clips"),
        report_dir=str(tmp_path / "reps"),
        async_write=async_write,
    )
    writer = EvidenceWriter(inc_cfg)
    bus = EventBus()
    mgr = IncidentManager("cam_01", vcfg, writer, bus)
    return mgr, writer, bus


# ---------------------------------------------------------------------------
# SCENARIO 1: No weapon + no fight -> no weapon incident
# ---------------------------------------------------------------------------
def test_scenario_01_no_weapon_no_fight(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()

    ev = WeaponEvent(
        frame_idx=1,
        timestamp=100.0,
        state=WeaponState.NORMAL,
        telemetry=[],
        candidate_count=0,
        confirmed_count=0,
        detections=[],
    )
    tracks = [_track(1, 100.0, 100.0)]
    decision = engine.evaluate(ev, [], tracks, timestamp=100.0)

    assert not decision.has_armed_fight
    assert not decision.has_weapon
    assert len(decision.weapons_alone) == 0
    assert len(decision.unarmed_fights) == 0
    assert len(mgr.recent) == 0


# ---------------------------------------------------------------------------
# SCENARIO 2: Weapon candidate only -> no incident
# ---------------------------------------------------------------------------
def test_scenario_02_weapon_candidate_only_no_incident(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()

    cand_telemetry = _weapon_telemetry(confirmation_state="WEAPON_CANDIDATE", persistence_count=1)
    ev = _weapon_event(state=WeaponState.WEAPON_CANDIDATE, telemetry=[cand_telemetry])
    tracks = [_track(1, 90.0, 90.0)]

    decision = engine.evaluate(ev, [], tracks, timestamp=100.0)
    assert not decision.has_armed_fight
    assert not decision.has_weapon  # Confirmed weapon required for incident
    assert len(decision.weapons_alone) == 0


# ---------------------------------------------------------------------------
# SCENARIO 3: Confirmed weapon alone -> HIGH WEAPON incident
# ---------------------------------------------------------------------------
def test_scenario_03_confirmed_weapon_alone_high_incident(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.88, track_id=1)
    ev = _weapon_event(state=WeaponState.WEAPON_CONFIRMED, telemetry=[wt])
    tracks = [_track(1, 90.0, 90.0)]

    decision = engine.evaluate(ev, [], tracks, timestamp=100.0)
    assert len(decision.weapons_alone) == 1
    assert len(decision.armed_fights) == 0

    report = mgr.report_weapon(decision.weapons_alone[0], frame, timestamp=100.0)
    assert report is not None
    assert report.incident_type == IncidentType.WEAPON
    assert report.severity == Severity.HIGH
    assert report.confidence == 0.88
    assert report.details["weapon_class"] == "pistol"
    assert report.evidence.snapshot_path is not None
    assert Path(report.evidence.snapshot_path).exists()


# ---------------------------------------------------------------------------
# SCENARIO 4: Confirmed knife alone -> HIGH WEAPON incident
# ---------------------------------------------------------------------------
def test_scenario_04_confirmed_knife_alone(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="knife", confidence=0.82, track_id=10)
    report = mgr.report_weapon(wt, frame, timestamp=100.0)

    assert report is not None
    assert report.incident_type == IncidentType.WEAPON
    assert report.severity == Severity.HIGH
    assert report.details["weapon_class"] == "knife"


# ---------------------------------------------------------------------------
# SCENARIO 5: Confirmed pistol alone -> HIGH WEAPON incident
# ---------------------------------------------------------------------------
def test_scenario_05_confirmed_pistol_alone(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.91, track_id=11)
    report = mgr.report_weapon(wt, frame, timestamp=100.0)

    assert report is not None
    assert report.incident_type == IncidentType.WEAPON
    assert report.severity == Severity.HIGH
    assert report.details["weapon_class"] == "pistol"


# ---------------------------------------------------------------------------
# SCENARIO 6: Confirmed long_gun alone -> HIGH WEAPON incident
# ---------------------------------------------------------------------------
def test_scenario_06_confirmed_long_gun_alone(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="long_gun", confidence=0.79, track_id=12)
    report = mgr.report_weapon(wt, frame, timestamp=100.0)

    assert report is not None
    assert report.incident_type == IncidentType.WEAPON
    assert report.severity == Severity.HIGH
    assert report.details["weapon_class"] == "long_gun"


# ---------------------------------------------------------------------------
# SCENARIO 7: Fight without weapon -> existing fight behavior unchanged (MODERATE)
# ---------------------------------------------------------------------------
def test_scenario_07_fight_without_weapon_moderate(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fight = _fight(track_ids=(1, 2), confidence=0.86)
    tracks = [_track(1, 80.0, 100.0), _track(2, 120.0, 100.0)]

    decision = engine.evaluate(None, [fight], tracks, timestamp=102.5)
    assert len(decision.unarmed_fights) == 1
    assert len(decision.armed_fights) == 0

    report = mgr.report_fight(decision.unarmed_fights[0], frame, timestamp=102.5)
    assert report is not None
    assert report.incident_type == IncidentType.FIGHT
    assert report.severity == Severity.MODERATE
    assert report.evidence.snapshot_path is None  # Suppressed for MODERATE


# ---------------------------------------------------------------------------
# SCENARIO 8: Confirmed weapon + verified fight -> CRITICAL ARMED_FIGHT
# ---------------------------------------------------------------------------
def test_scenario_08_confirmed_weapon_plus_fight_armed_fight(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Combatants at (80, 100) and (120, 100)
    tracks = [_track(1, 80.0, 100.0), _track(2, 120.0, 100.0)]
    fight = _fight(track_ids=(1, 2), roi_xyxy=(50.0, 50.0, 160.0, 180.0), end_time=102.5)

    # Confirmed weapon held by track 1 (within upper body of track 1)
    wt = _weapon_telemetry(weapon_class="knife", confidence=0.92, bbox=[70.0, 70.0, 95.0, 95.0], timestamp=102.5, track_id=1)
    ev = _weapon_event(timestamp=102.5, telemetry=[wt])

    decision = engine.evaluate(ev, [fight], tracks, timestamp=102.5)
    assert len(decision.armed_fights) == 1
    assert len(decision.weapons_alone) == 0
    assert len(decision.unarmed_fights) == 0

    af = decision.armed_fights[0]
    assert af.carrier_track_id == 1

    report = mgr.report_armed_fight(af, frame, timestamp=102.5)
    assert report is not None
    assert report.incident_type == IncidentType.ARMED_FIGHT
    assert report.severity == Severity.CRITICAL
    assert report.evidence.snapshot_path is not None
    assert Path(report.evidence.snapshot_path).exists()


# ---------------------------------------------------------------------------
# SCENARIO 9: Bystander with weapon -> must NOT incorrectly create ARMED_FIGHT
# ---------------------------------------------------------------------------
def test_scenario_09_bystander_isolation_no_armed_fight(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Combatants at (100, 100) and (150, 100)
    p1 = _track(1, 100.0, 100.0)
    p2 = _track(2, 150.0, 100.0)
    fight = _fight(track_ids=(1, 2), roi_xyxy=(70.0, 50.0, 180.0, 180.0), end_time=102.5)

    # Bystander Person 9 standing far away at (800, 500) holding confirmed pistol
    p9 = _track(9, 800.0, 500.0)
    wt_bystander = _weapon_telemetry(
        weapon_class="pistol",
        confidence=0.89,
        bbox=[780.0, 470.0, 820.0, 510.0],
        timestamp=102.5,
        track_id=9,
    )
    ev = _weapon_event(timestamp=102.5, telemetry=[wt_bystander])

    tracks = [p1, p2, p9]
    decision = engine.evaluate(ev, [fight], tracks, timestamp=102.5)

    # Strict assertion: NO armed fight!
    assert len(decision.armed_fights) == 0
    assert len(decision.weapons_alone) == 1
    assert decision.weapons_alone[0].track_id == 9
    assert len(decision.unarmed_fights) == 1
    assert decision.unarmed_fights[0].track_ids == (1, 2)


# ---------------------------------------------------------------------------
# SCENARIO 10: Weapon disappears before fight verification -> no false ARMED_FIGHT
# ---------------------------------------------------------------------------
def test_scenario_10_disappeared_weapon_no_armed_fight(tmp_path):
    engine = IncidentFusionEngine()
    tracks = [_track(1, 80.0, 100.0), _track(2, 120.0, 100.0)]
    fight = _fight(track_ids=(1, 2), end_time=105.0)

    # Weapon was candidate at t=100, but is gone at t=105 (state NORMAL, empty telemetry)
    ev = WeaponEvent(
        frame_idx=75,
        timestamp=105.0,
        state=WeaponState.NORMAL,
        telemetry=[],
        candidate_count=0,
        confirmed_count=0,
        detections=[],
    )

    decision = engine.evaluate(ev, [fight], tracks, timestamp=105.0)
    assert len(decision.armed_fights) == 0
    assert len(decision.unarmed_fights) == 1


# ---------------------------------------------------------------------------
# SCENARIO 11: Fight begins after weapon confirmed -> deterministic escalation
# ---------------------------------------------------------------------------
def test_scenario_11_escalation_from_weapon_to_armed_fight(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tracks = [_track(1, 100.0, 100.0), _track(2, 140.0, 100.0)]

    # Step 1: Confirmed weapon alone at t=100.0
    wt = _weapon_telemetry(weapon_class="knife", confidence=0.85, bbox=[90.0, 80.0, 115.0, 105.0], timestamp=100.0, track_id=1)
    ev1 = _weapon_event(frame_idx=1, timestamp=100.0, telemetry=[wt])
    dec1 = engine.evaluate(ev1, [], tracks, timestamp=100.0)

    r_weapon = mgr.report_weapon(dec1.weapons_alone[0], frame, timestamp=100.0)
    assert r_weapon is not None
    assert r_weapon.incident_type == IncidentType.WEAPON
    assert r_weapon.severity == Severity.HIGH
    weapon_inc_id = r_weapon.incident_id

    # Step 2: Fight erupts at t=101.5 involving track 1
    fight = _fight(track_ids=(1, 2), roi_xyxy=(70.0, 50.0, 170.0, 180.0), start_time=100.5, end_time=101.5)
    wt2 = _weapon_telemetry(weapon_class="knife", confidence=0.88, bbox=[90.0, 80.0, 115.0, 105.0], timestamp=101.5, track_id=1)
    ev2 = _weapon_event(frame_idx=23, timestamp=101.5, telemetry=[wt2])

    dec2 = engine.evaluate(ev2, [fight], tracks, timestamp=101.5)
    assert len(dec2.armed_fights) == 1

    r_armed = mgr.report_armed_fight(dec2.armed_fights[0], frame, timestamp=101.5)
    assert r_armed is not None
    # Escalated in place!
    assert r_armed.incident_id == weapon_inc_id
    assert r_armed.incident_type == IncidentType.ARMED_FIGHT
    assert r_armed.severity == Severity.CRITICAL
    assert r_armed.details.get("escalated") is True


# ---------------------------------------------------------------------------
# SCENARIO 12: Fight ends while weapon remains -> incident remains, not recreated
# ---------------------------------------------------------------------------
def test_scenario_12_fight_ends_weapon_remains_no_duplicate(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Initial armed fight report
    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.90, track_id=1, timestamp=100.0)
    fight = _fight(track_ids=(1, 2), end_time=100.0)
    af = CorrelatedArmedFight(
        fight=fight,
        weapon_telemetry=wt,
        carrier_track_id=1,
        correlation_confidence=0.90,
        roi_xyxy=fight.roi_xyxy,
    )
    r1 = mgr.report_armed_fight(af, frame, timestamp=100.0)
    assert r1 is not None

    # Fight ends, but weapon track 1 is still detected alone at t=102.0
    r_weapon = mgr.report_weapon(wt, frame, timestamp=102.0)
    # Weapon track 1 is suppressed because it was part of an active armed fight / entity cooldown
    assert r_weapon is None, "Should not emit duplicate conflicting weapon incident when weapon is already active"


# ---------------------------------------------------------------------------
# SCENARIO 13: Same weapon remains visible for 30+ frames -> exactly one active incident
# ---------------------------------------------------------------------------
def test_scenario_13_weapon_visible_30_frames_one_incident(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wt = _weapon_telemetry(weapon_class="knife", confidence=0.85, track_id=5, timestamp=100.0)

    # Frame 1
    r_first = mgr.report_weapon(wt, frame, timestamp=100.0)
    assert r_first is not None

    # Next 35 frames (every frame for 2+ seconds)
    t = 100.0
    for i in range(35):
        t += 0.066
        wt_subsequent = _weapon_telemetry(weapon_class="knife", confidence=0.85, track_id=5, timestamp=t)
        r_sub = mgr.report_weapon(wt_subsequent, frame, timestamp=t)
        assert r_sub is None, f"Frame {i+2} must not create duplicate active incident"

    # Total weapon incidents in recent
    weapon_incidents = [r for r in mgr.recent if r.incident_type == IncidentType.WEAPON]
    assert len(weapon_incidents) == 1


# ---------------------------------------------------------------------------
# SCENARIO 14: Multiple weapons -> deterministic handling
# ---------------------------------------------------------------------------
def test_scenario_14_multiple_distinct_weapons(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt1 = _weapon_telemetry(weapon_class="knife", confidence=0.85, track_id=1, timestamp=100.0)
    wt2 = _weapon_telemetry(weapon_class="pistol", confidence=0.92, track_id=2, timestamp=100.0)

    r1 = mgr.report_weapon(wt1, frame, timestamp=100.0)
    r2 = mgr.report_weapon(wt2, frame, timestamp=100.0)

    assert r1 is not None and r2 is not None
    assert r1.incident_id != r2.incident_id
    assert r1.details["weapon_class"] == "knife"
    assert r2.details["weapon_class"] == "pistol"


# ---------------------------------------------------------------------------
# SCENARIO 15: Class switch knife -> pistol -> existing persistence rules respected
# ---------------------------------------------------------------------------
def test_scenario_15_class_switch_does_not_cross_confirm():
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=3, window_size=5)
    box = np.array([50, 50, 100, 100], dtype=np.float32)

    det_knife = RawWeaponDetection(bbox_xyxy=box, confidence=0.85, cls_id=0, class_name="knife")
    det_pistol = RawWeaponDetection(bbox_xyxy=box, confidence=0.85, cls_id=2, class_name="pistol")

    # 2 hits knife
    tracker.update([det_knife], timestamp=100.0)
    tracker.update([det_knife], timestamp=100.1)

    # 1 hit pistol (class change!)
    state, telemetry, cand_cnt, conf_cnt = tracker.update([det_pistol], timestamp=100.2)

    # Neither track reached 3 hits for its own class
    assert conf_cnt == 0
    assert state == WeaponState.WEAPON_CANDIDATE


# ---------------------------------------------------------------------------
# SCENARIO 16: Weapon confidence below threshold -> no escalation
# ---------------------------------------------------------------------------
def test_scenario_16_weapon_conf_below_threshold_stays_candidate():
    tracker = WeaponPersistenceTracker(confirm_conf_threshold=0.65, min_hits=3, window_size=5)
    box = np.array([50, 50, 100, 100], dtype=np.float32)
    low_conf_det = RawWeaponDetection(bbox_xyxy=box, confidence=0.45, cls_id=2, class_name="pistol")

    for i in range(5):
        state, _, cand_cnt, conf_cnt = tracker.update([low_conf_det], timestamp=100.0 + i * 0.1)

    assert state == WeaponState.WEAPON_CANDIDATE
    assert conf_cnt == 0


# ---------------------------------------------------------------------------
# SCENARIO 17: Weapon + crowd density only -> do not automatically create ARMED_FIGHT
# ---------------------------------------------------------------------------
def test_scenario_17_weapon_plus_crowd_density_no_armed_fight(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.88, track_id=1)
    ev = _weapon_event(telemetry=[wt])
    tracks = [_track(1, 100.0, 100.0)]

    # Critical crowd density, but ZERO fights
    snap = CrowdSnapshot(
        timestamp=100.0,
        person_count=15,
        zones=[ZoneStat(name="plaza", person_count=15, density=0.90, level="critical")],
        growth_per_min=0.0,
        growth_alert=False,
        movement=MovementInfo(panic=False, fast_track_ratio=0.0, direction_entropy=0.0),
    )

    dec = engine.evaluate(ev, [], tracks, timestamp=100.0)
    assert not dec.has_armed_fight
    assert len(dec.weapons_alone) == 1

    r_weapon = mgr.report_weapon(dec.weapons_alone[0], frame, timestamp=100.0)
    assert r_weapon is not None
    assert r_weapon.incident_type == IncidentType.WEAPON


# ---------------------------------------------------------------------------
# SCENARIO 18: Weapon + crowd panic without fight -> HIGH weapon incident, no false ARMED_FIGHT
# ---------------------------------------------------------------------------
def test_scenario_18_weapon_plus_crowd_panic_no_armed_fight(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    engine = IncidentFusionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="long_gun", confidence=0.90, track_id=2)
    ev = _weapon_event(telemetry=[wt])
    tracks = [_track(2, 200.0, 200.0)]

    # Panic in scene
    snap = CrowdSnapshot(
        timestamp=100.0,
        person_count=10,
        zones=[],
        growth_per_min=0.0,
        growth_alert=False,
        movement=MovementInfo(panic=True, fast_track_ratio=0.75, direction_entropy=0.8),
    )

    dec = engine.evaluate(ev, [], tracks, timestamp=100.0)
    assert len(dec.armed_fights) == 0
    assert len(dec.weapons_alone) == 1

    # Report crowd and weapon independently
    r_crowd = mgr.report_crowd(snap, frame)
    r_weapon = mgr.report_weapon(dec.weapons_alone[0], frame, timestamp=100.0)

    assert r_weapon is not None
    assert r_weapon.incident_type == IncidentType.WEAPON
    assert r_weapon.severity == Severity.HIGH

    # Crowd panic should be reported if criteria met
    crowd_types = [r.incident_type for r in r_crowd]
    assert IncidentType.CROWD_PANIC in crowd_types


# ---------------------------------------------------------------------------
# SCENARIO 19: Evidence generation -> snapshot at incident time
# ---------------------------------------------------------------------------
def test_scenario_19_evidence_snapshot_at_incident_time(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.92, track_id=3)
    report = mgr.report_weapon(wt, frame, timestamp=100.0)

    assert report.evidence.snapshot_path is not None
    snap_p = Path(report.evidence.snapshot_path)
    assert snap_p.exists()
    assert snap_p.suffix == ".jpg"
    assert snap_p.stem == report.incident_id


# ---------------------------------------------------------------------------
# SCENARIO 20: Evidence generation -> approximately T-5 to T+5 clip
# ---------------------------------------------------------------------------
def test_scenario_20_evidence_t5_to_t5_clip(tmp_path):
    mgr, writer, _ = _make_manager(tmp_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # 1. Push 5 seconds of pre-event frames (75 frames @ 15fps)
    t = 100.0
    dt = 1.0 / 15.0
    for _ in range(75):
        writer.push_frame(frame, t)
        t += dt

    # Trigger weapon incident at t=105.0
    wt = _weapon_telemetry(weapon_class="knife", confidence=0.88, track_id=4, timestamp=105.0)
    report = mgr.report_weapon(wt, frame, timestamp=105.0)
    assert report.status == IncidentStatus.RECORDING_POST_EVENT

    # 2. Push 5+ seconds of post-event frames (80 frames @ 15fps)
    completed_any = False
    for _ in range(80):
        t += dt
        done_ids = writer.push_frame(frame, t)
        if report.incident_id in done_ids:
            completed_any = True

    assert completed_any is True
    assert report.status == IncidentStatus.FINALIZED
    assert report.evidence.clip_path is not None
    assert Path(report.evidence.clip_path).exists()


# ---------------------------------------------------------------------------
# SCENARIO 21: Evidence generation -> no blocking of inference loop (async_write)
# ---------------------------------------------------------------------------
def test_scenario_21_evidence_async_write_non_blocking(tmp_path):
    mgr, writer, _ = _make_manager(tmp_path, async_write=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    t = 100.0
    dt = 1.0 / 15.0
    for _ in range(75):
        writer.push_frame(frame, t)
        t += dt

    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.90, track_id=8, timestamp=105.0)
    report = mgr.report_weapon(wt, frame, timestamp=105.0)

    # Measuring push_frame latency during completion frames
    completed_any = False
    max_latency_ms = 0.0
    for _ in range(80):
        t += dt
        t_start = time.perf_counter()
        done_ids = writer.push_frame(frame, t)
        lat = (time.perf_counter() - t_start) * 1000.0
        max_latency_ms = max(max_latency_ms, lat)
        if report.incident_id in done_ids:
            completed_any = True

    # With async_write=True, push_frame delegates clip writing and returns in < 50ms
    assert max_latency_ms < 50.0, f"push_frame blocked for {max_latency_ms:.2f}ms"
    assert completed_any is True

    # Wait for background thread to complete
    writer.wait_pending(timeout=5.0)
    assert report.status == IncidentStatus.FINALIZED
    assert Path(report.evidence.clip_path).exists()


# ---------------------------------------------------------------------------
# SCENARIO 22: Cooldown -> no duplicate incidents
# ---------------------------------------------------------------------------
def test_scenario_22_weapon_cooldown_prevents_duplicate(tmp_path):
    mgr, _, _ = _make_manager(tmp_path, verify_cfg=VerifyConfig(incident_cooldown_s=30.0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.90, track_id=1, timestamp=100.0)
    r1 = mgr.report_weapon(wt, frame, timestamp=100.0)
    assert r1 is not None

    # Mark r1 as finalized so active status check doesn't overshadow cooldown check
    r1.status = IncidentStatus.FINALIZED

    # Attempt second report within cooldown at t=110.0
    r2 = mgr.report_weapon(wt, frame, timestamp=110.0)
    assert r2 is None, "Weapon on entity cooldown must be suppressed"


# ---------------------------------------------------------------------------
# SCENARIO 23: New event after cooldown -> new incident allowed
# ---------------------------------------------------------------------------
def test_scenario_23_new_event_after_cooldown_allowed(tmp_path):
    mgr, _, _ = _make_manager(tmp_path, verify_cfg=VerifyConfig(incident_cooldown_s=30.0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    wt = _weapon_telemetry(weapon_class="pistol", confidence=0.90, track_id=1, timestamp=100.0)
    r1 = mgr.report_weapon(wt, frame, timestamp=100.0)
    assert r1 is not None
    r1.status = IncidentStatus.FINALIZED

    # Attempt report after cooldown at t=135.0 (135 > 100 + 30)
    r2 = mgr.report_weapon(wt, frame, timestamp=135.0)
    assert r2 is not None, "After cooldown expires, a new incident must be created"
    assert r2.incident_id != r1.incident_id


# ---------------------------------------------------------------------------
# SCENARIO 24: Existing fight evidence -> unchanged (snapshot suppressed, clip retained)
# ---------------------------------------------------------------------------
def test_scenario_24_existing_fight_evidence_unchanged(tmp_path):
    mgr, _, _ = _make_manager(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fight = _fight(track_ids=(1, 2), confidence=0.87)
    report = mgr.report_fight(fight, frame, timestamp=102.5)

    assert report is not None
    assert report.severity == Severity.MODERATE
    assert report.evidence.snapshot_path is None  # Suppressed for MODERATE
    assert report.status == IncidentStatus.RECORDING_POST_EVENT


# ---------------------------------------------------------------------------
# SCENARIO 25: Existing crowd behavior -> unchanged
# ---------------------------------------------------------------------------
def test_scenario_25_existing_crowd_behavior_unchanged(tmp_path):
    crowd_cfg = CrowdConfig(enable_density_incidents=False)
    vcfg = VerifyConfig()
    inc_cfg = IncidentsConfig(snapshot_dir=str(tmp_path / "snaps"), clip_dir=str(tmp_path / "clips"), report_dir=str(tmp_path / "reps"))
    mgr = IncidentManager("cam_01", vcfg, EvidenceWriter(inc_cfg), EventBus(), crowd_cfg=crowd_cfg)

    snap = CrowdSnapshot(
        timestamp=100.0,
        person_count=12,
        zones=[ZoneStat(name="plaza", person_count=12, density=0.85, level="critical")],
        growth_per_min=0.0,
        growth_alert=False,
        movement=MovementInfo(panic=False, fast_track_ratio=0.0, direction_entropy=0.0),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    reports = mgr.report_crowd(snap, frame)
    assert reports == [], "With enable_density_incidents=False, density context must not trigger incidents"
