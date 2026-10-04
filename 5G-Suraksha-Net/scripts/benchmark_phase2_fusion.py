"""Benchmark and End-to-End Simulation Script for Phase 2 Multimodal Fusion.

Evaluates:
1. Throughput & latency comparison across 3 modes:
   - Mode 1: Baseline (Fight + Crowd only, Weapon disabled)
   - Mode 2: Phase 1 (Dual branch: Fight + Crowd + Weapon, Fusion disabled)
   - Mode 3: Phase 2 (Multimodal Incident Fusion enabled)
2. Synthetic End-to-End Escalation Simulation:
   - Phase A: Normal walking (T=0 to T=2s)
   - Phase B: Weapon confirmed (T=2s to T=3s) -> HIGH WEAPON
   - Phase C: Combat begins (T=3s to T=5s) -> Escalates to CRITICAL ARMED_FIGHT
   - Phase D: Evidence generation (T=5s to T=7s) -> Snapshot & clip verified
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch

from suraksha.capture.stream import FrameEvent, StreamCapture
from suraksha.config import (
    AppConfig,
    CaptureConfig,
    FusionConfig,
    IncidentsConfig,
    VerifyConfig,
    WeaponConfig,
    load_config,
)
from suraksha.detection.tracker import TrackedPerson
from suraksha.detection.weapon import (
    RawWeaponDetection,
    WeaponDetector,
    WeaponEvent,
    WeaponState,
    WeaponTelemetry,
)
from suraksha.fight.recognizer import VerifiedFight
from suraksha.incidents.evidence import EvidenceWriter
from suraksha.incidents.fusion import CorrelatedArmedFight, IncidentFusionEngine
from suraksha.incidents.manager import EventBus, IncidentManager
from suraksha.incidents.schemas import IncidentReport, IncidentStatus, IncidentType, Severity
from suraksha.pipeline import CrowdFightPipeline


def run_cctv_benchmark(video_path: str, max_frames: int = 60) -> dict:
    """Run benchmark comparison on CCTV test video."""
    print("=" * 70)
    print("RUNNING CCTV BENCHMARK COMPARISON")
    print(f"Source video: {video_path} ({max_frames} frames)")
    print("=" * 70)

    results = {}
    modes = [
        ("Mode 1: Baseline (Weapon Disabled)", False, False),
        ("Mode 2: Phase 1 (Weapon Enabled, Fusion Off)", True, False),
        ("Mode 3: Phase 2 (Weapon + Multimodal Fusion)", True, True),
    ]

    for label, weapon_en, fusion_en in modes:
        cfg = load_config()
        cfg.capture.rtsp_url = video_path
        cfg.capture.max_frames = max_frames
        cfg.capture.pace = "fast"  # unthrottled benchmark mode
        cfg.weapon.enabled = weapon_en
        cfg.fusion.enabled = fusion_en
        cfg.incidents.async_write = True

        pipeline = CrowdFightPipeline(cfg)
        pipeline.start()

        # Wait for max_frames or timeout
        t0 = time.perf_counter()
        while pipeline.state.running and pipeline.state.frames_processed < max_frames:
            time.sleep(0.05)
            if time.perf_counter() - t0 > 30.0:
                break
        total_time = time.perf_counter() - t0
        frames_done = pipeline.state.frames_processed
        pipeline.stop()

        fps = frames_done / max(total_time, 1e-4)
        m = pipeline.state.metrics

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        results[label] = {
            "frames": frames_done,
            "total_time_s": round(total_time, 2),
            "effective_fps": round(fps, 1),
            "detect_track_ms": round(m.get("detect_track_ms", 0.0), 2),
            "weapon_ms": round(m.get("weapon_ms", 0.0), 2),
            "crowd_ms": round(m.get("crowd_ms", 0.0), 2),
            "fight_ms": round(m.get("fight_ms", 0.0), 2),
            "total_latency_ms": round(m.get("total_ms", 0.0), 2),
            "peak_vram_mb": round(vram_mb, 1),
        }
        print(f"[{label}] -> {frames_done} frames in {total_time:.2f}s ({fps:.1f} FPS) | Total Latency: {m.get('total_ms', 0.0):.1f}ms")

    return results


def run_e2e_escalation_simulation(output_dir: Path) -> dict:
    """Run deterministic synthetic integration simulating real escalation lifecycle."""
    print("\n" + "=" * 70)
    print("RUNNING END-TO-END ESCALATION SIMULATION")
    print("=" * 70)

    sim_snaps = output_dir / "sim_snapshots"
    sim_clips = output_dir / "sim_clips"
    sim_reps = output_dir / "sim_incidents"
    sim_snaps.mkdir(parents=True, exist_ok=True)
    sim_clips.mkdir(parents=True, exist_ok=True)
    sim_reps.mkdir(parents=True, exist_ok=True)

    inc_cfg = IncidentsConfig(
        clip_fps=15,
        clip_seconds_before=3,
        clip_seconds_after=3,
        snapshot_dir=str(sim_snaps),
        clip_dir=str(sim_clips),
        report_dir=str(sim_reps),
        async_write=False,
    )
    writer = EvidenceWriter(inc_cfg)
    bus = EventBus()
    mgr = IncidentManager("sim_cam_01", VerifyConfig(incident_cooldown_s=30.0), writer, bus)
    fusion = IncidentFusionEngine(FusionConfig(enabled=True, correlation_window_s=2.5))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "SIMULATION CCTV", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    events_log = []

    def on_bus_event(rep: IncidentReport):
        events_log.append({
            "incident_id": rep.incident_id,
            "type": rep.incident_type.value,
            "severity": rep.severity.value,
            "status": rep.status.value,
            "confidence": rep.confidence,
            "escalated": rep.details.get("escalated", False),
        })

    bus.subscribe(on_bus_event)

    t = 100.0
    dt = 1.0 / 15.0

    # 1. Push 45 frames of normal baseline (3.0s)
    print("Phase 1: Normal pre-event baseline (T=100.0s to 103.0s)...")
    for _ in range(45):
        writer.push_frame(frame, t)
        t += dt

    # 2. Weapon confirmed at T=103.0s (Person 1 pulls pistol)
    print("Phase 2: Weapon Confirmed (T=103.0s)...")
    p1 = TrackedPerson(track_id=1, bbox_xyxy=np.array([100, 100, 160, 260], dtype=np.float32), confidence=0.92)
    p2 = TrackedPerson(track_id=2, bbox_xyxy=np.array([200, 100, 260, 260], dtype=np.float32), confidence=0.90)
    tracks = [p1, p2]

    wt = WeaponTelemetry(
        weapon_class="pistol",
        confidence=0.89,
        bbox=[110.0, 120.0, 140.0, 160.0],
        frame_timestamp=t,
        persistence_count=4,
        confirmation_state="WEAPON_CONFIRMED",
        track_id=1,
    )
    weap_ev = WeaponEvent(
        frame_idx=45,
        timestamp=t,
        state=WeaponState.WEAPON_CONFIRMED,
        telemetry=[wt],
        candidate_count=0,
        confirmed_count=1,
    )

    dec_phase2 = fusion.evaluate(weap_ev, [], tracks, timestamp=t)
    assert len(dec_phase2.weapons_alone) == 1
    r_weapon = mgr.report_weapon(dec_phase2.weapons_alone[0], frame, timestamp=t)
    print(f" -> Emitted: ID={r_weapon.incident_id} type={r_weapon.incident_type.value} severity={r_weapon.severity.value}")
    weapon_inc_id = r_weapon.incident_id

    # 3. Combat struggle ensues at T=104.5s
    print("Phase 3: Violent Struggle with Weapon (T=104.5s)...")
    for _ in range(22):
        writer.push_frame(frame, t)
        t += dt

    fight = VerifiedFight(
        track_ids=(1, 2),
        roi_xyxy=np.array([90, 80, 270, 280], dtype=np.float32),
        start_time=103.5,
        end_time=t,
        confidence=0.91,
        windows_scored=5,
    )
    wt2 = WeaponTelemetry(
        weapon_class="pistol",
        confidence=0.93,
        bbox=[120.0, 130.0, 150.0, 170.0],
        frame_timestamp=t,
        persistence_count=15,
        confirmation_state="WEAPON_CONFIRMED",
        track_id=1,
    )
    weap_ev2 = WeaponEvent(
        frame_idx=67,
        timestamp=t,
        state=WeaponState.WEAPON_CONFIRMED,
        telemetry=[wt2],
        confirmed_count=1,
    )

    dec_phase3 = fusion.evaluate(weap_ev2, [fight], tracks, timestamp=t)
    assert len(dec_phase3.armed_fights) == 1
    r_armed = mgr.report_armed_fight(dec_phase3.armed_fights[0], frame, timestamp=t)
    print(f" -> Escalated: ID={r_armed.incident_id} type={r_armed.incident_type.value} severity={r_armed.severity.value}")
    assert r_armed.incident_id == weapon_inc_id
    assert r_armed.incident_type == IncidentType.ARMED_FIGHT
    assert r_armed.severity == Severity.CRITICAL

    # 4. Post-event frames to finalize video clip
    print("Phase 4: Post-event recording & evidence finalization (T=104.5s to 108.5s)...")
    completed = False
    for _ in range(60):
        t += dt
        done_ids = writer.push_frame(frame, t)
        if weapon_inc_id in done_ids:
            completed = True

    print(f" -> Clip recording completed: {completed}")
    print(f" -> Final incident status: {r_armed.status.value}")
    print(f" -> Evidence snapshot: {r_armed.evidence.snapshot_path}")
    print(f" -> Evidence clip: {r_armed.evidence.clip_path}")

    return {
        "incident_id": weapon_inc_id,
        "final_type": r_armed.incident_type.value,
        "final_severity": r_armed.severity.value,
        "final_status": r_armed.status.value,
        "snapshot_exists": Path(r_armed.evidence.snapshot_path).exists() if r_armed.evidence.snapshot_path else False,
        "clip_exists": Path(r_armed.evidence.clip_path).exists() if r_armed.evidence.clip_path else False,
        "bus_events_count": len(events_log),
        "events_log": events_log,
    }


def main():
    video_path = "datasets/videos/external/cctv_test_01.mp4"
    out_dir = Path("outputs/weapon_training/integration_phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_res = run_cctv_benchmark(video_path, max_frames=60)
    sim_res = run_e2e_escalation_simulation(out_dir)

    summary = {
        "benchmark": benchmark_res,
        "simulation": sim_res,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_file = out_dir / "phase2_benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK & SIMULATION COMPLETE! Summary written to: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
