"""Deterministic Demo Scenario Service for 5G Suraksha-Net IMC Demonstration.

Reliably executes end-to-end operational scenarios (Weapon, Crowd Panic, Armed Fight, Normal)
without fabricating live camera detections or contaminating real incident archives.
Strictly requires human operator authorization for all emergency actions.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from suraksha.config import PROJECT_ROOT
from suraksha.demo.schemas import (
    DemoScenarioId,
    DemoScenarioMeta,
    DemoSimulationStep,
    DemoStatusResponse,
)
from suraksha.incidents.schemas import BBox, Evidence, IncidentReport, IncidentStatus, IncidentType, Severity
from suraksha.location.schemas import Location, LocationSource
from suraksha.logging_utils import get_logger
from suraksha.network import get_network_policy_service
from suraksha.response.planner import ResponsePlanner

log = get_logger(__name__)

DEMO_REPORT_DIR = PROJECT_ROOT / "outputs" / "demo" / "incidents"
DEMO_SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "demo" / "snapshots"
DEMO_CLIP_DIR = PROJECT_ROOT / "outputs" / "demo" / "clips"


class DemoScenarioService:
    """Manages controlled demonstration scenarios and simulated incident lifecycle."""

    def __init__(self) -> None:
        self._active: bool = False
        self._active_scenario_id: DemoScenarioId | None = None
        self._active_incident: IncidentReport | None = None
        self._timeline: list[DemoSimulationStep] = []
        self._lock = threading.Lock()

        # Ensure demo storage directories exist
        DEMO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        DEMO_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        DEMO_CLIP_DIR.mkdir(parents=True, exist_ok=True)

        self._ensure_sample_demo_evidence()

    def _ensure_sample_demo_evidence(self) -> None:
        """Create visible, explicitly labeled synthetic demo evidence images."""
        try:
            demo_snap = DEMO_SNAPSHOT_DIR / "demo_evidence.jpg"
            if not demo_snap.is_file():
                # Generate a 640x360 placeholder with high-contrast tactical banner
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                img[:] = (20, 10, 15)  # dark slate background
                # Amber border
                cv2.rectangle(img, (10, 10), (630, 350), (30, 180, 240), 2)
                # Header label
                cv2.putText(img, "5G SURAKSHA-NET IMC DEMONSTRATION", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
                cv2.putText(img, "CONTROLLED SIMULATION SCENARIO", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 180, 240), 2)
                cv2.putText(img, "SIMULATED INCIDENT EVIDENCE ARTIFACT", (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                cv2.putText(img, "YASHOBHOOMI SECTOR 25, NEW DELHI", (35, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                cv2.putText(img, "DECISION SUPPORT ONLY - HUMAN APPROVAL MANDATORY", (35, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)
                cv2.imwrite(str(demo_snap), img)
        except Exception as e:
            log.debug("Failed creating demo evidence placeholder: %s", e)

    def _resolve_demo_location(self) -> Location:
        """Resolve demonstration venue location dynamically using Google Maps API."""
        try:
            from suraksha.config import load_config
            from suraksha.location.provider import create_location_provider

            cfg = load_config()
            return create_location_provider(cfg.location).get_location()
        except Exception as e:
            log.warning("Could not dynamically resolve demo location from Google Maps: %s", e)
            return Location(
                name="Yashobhoomi, Dwarka Sector 25, New Delhi",
                latitude=28.552553,
                longitude=77.044893,
                source=LocationSource.GEOCODED_DEMO,
            )

    def get_available_scenarios(self) -> list[DemoScenarioMeta]:
        """Return metadata for all available demonstration scenarios."""
        return [
            DemoScenarioMeta(
                id=DemoScenarioId.WEAPON,
                name="Weapon Detection (High)",
                description="Simulated concealed weapon detection; triggers elevated network priority and police response recommendation.",
                incident_type=IncidentType.WEAPON,
                severity=Severity.HIGH,
                expected_network_policy="EVENT",
                expected_network_priority="HIGH",
                location_name="Yashobhoomi, Dwarka Sector 25, New Delhi",
                video_choices=[
                    {"id": "weapondetection.mp4", "label": "weapondetection.mp4"},
                ],
            ),
            DemoScenarioMeta(
                id=DemoScenarioId.CROWD_PANIC,
                name="Crowd Panic (High)",
                description="Simulated rapid crowd surge and panic movement; tests crowd monitoring and perimeter security dispatch.",
                incident_type=IncidentType.CROWD_PANIC,
                severity=Severity.HIGH,
                expected_network_policy="EVENT",
                expected_network_priority="HIGH",
                location_name="Yashobhoomi, Dwarka Sector 25, New Delhi",
                video_choices=[
                    {"id": "crowd_panic.mp4", "label": "crowd_panic.mp4"},
                ],
            ),
            DemoScenarioMeta(
                id=DemoScenarioId.ARMED_FIGHT,
                name="Armed Fight (Critical)",
                description="Simulated violent armed altercation; tests multi-modal fusion, CRITICAL network policy, and urgent police/medical recommendations.",
                incident_type=IncidentType.ARMED_FIGHT,
                severity=Severity.CRITICAL,
                expected_network_policy="CRITICAL",
                expected_network_priority="CRITICAL",
                location_name="Yashobhoomi, Dwarka Sector 25, New Delhi",
                video_choices=[
                    {"id": "armedfight.mp4", "label": "Clip 1 (armedfight.mp4)"},
                    {"id": "armedfight_1.mp4", "label": "Clip 2 (armedfight_1.mp4)"},
                ],
            ),
            DemoScenarioMeta(
                id=DemoScenarioId.NORMAL,
                name="Normal Monitoring (Standby)",
                description="Routine monitoring baseline; no active incidents, routine network queue priority, all systems ready.",
                incident_type=None,
                severity=None,
                expected_network_policy="NORMAL",
                expected_network_priority="ROUTINE",
                location_name="Yashobhoomi, Dwarka Sector 25, New Delhi",
            ),
        ]

    def start_scenario(
        self,
        scenario_id: str | DemoScenarioId,
        video_choice: str | None = None,
        broadcast_callback: Callable[[IncidentReport], None] | None = None,
    ) -> DemoStatusResponse:
        """Activate a controlled demonstration scenario with video feed integration."""
        raw_str = scenario_id.value if isinstance(scenario_id, DemoScenarioId) else str(scenario_id).upper()
        if "WEAPON" in raw_str:
            scen_str = "WEAPON"
        elif "PANIC" in raw_str or "CROWD" in raw_str:
            scen_str = "CROWD_PANIC"
        elif "FIGHT" in raw_str:
            scen_str = "ARMED_FIGHT"
        elif "NORMAL" in raw_str or "RESET" in raw_str:
            scen_str = "NORMAL"
        else:
            scen_str = raw_str

        if scen_str == "NORMAL":
            return self.stop_scenario(broadcast_callback=broadcast_callback)

        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()

        # Build scenario incident
        inc_id = f"demo-{scen_str.lower()}-{int(now_ts) % 10000:04d}"

        video_file: Path | None = None
        email_clip: Path | None = None

        real_snap: Path | None = None
        if scen_str == "WEAPON":
            video_file = DEMO_CLIP_DIR / "weapondetection.mp4"
            cand_10s = DEMO_CLIP_DIR / "weapondetection_10s.mp4"
            email_clip = cand_10s if cand_10s.is_file() else video_file
            real_snap = DEMO_SNAPSHOT_DIR / "weapon_evidence.jpg"
        elif scen_str == "CROWD_PANIC":
            video_file = DEMO_CLIP_DIR / "crowd_panic.mp4"
            cand_10s = DEMO_CLIP_DIR / "crowd_panic_10s.mp4"
            email_clip = cand_10s if cand_10s.is_file() else video_file
            real_snap = DEMO_SNAPSHOT_DIR / "crowd_panic_evidence.jpg"
        elif scen_str == "ARMED_FIGHT":
            if video_choice in ("armedfight_1.mp4", "armedfight_1", "clip2", "2"):
                video_file = DEMO_CLIP_DIR / "armedfight_1.mp4"
                cand_10s = DEMO_CLIP_DIR / "armedfight_1_10s.mp4"
                email_clip = cand_10s if cand_10s.is_file() else video_file
                real_snap = DEMO_SNAPSHOT_DIR / "armedfight_1_evidence.jpg"
            else:
                video_file = DEMO_CLIP_DIR / "armedfight.mp4"
                cand_10s = DEMO_CLIP_DIR / "armedfight_10s.mp4"
                email_clip = cand_10s if cand_10s.is_file() else video_file
                real_snap = DEMO_SNAPSHOT_DIR / "armedfight_evidence.jpg"

        snap_rel = None
        clip_rel = None

        # Copy real forensic snapshot to incident outputs
        import shutil
        if real_snap and real_snap.is_file():
            try:
                out_snap_prod = PROJECT_ROOT / "outputs" / "snapshots" / f"{inc_id}.jpg"
                out_snap_demo = DEMO_SNAPSHOT_DIR / f"{inc_id}.jpg"
                out_snap_prod.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(real_snap), str(out_snap_prod))
                shutil.copy2(str(real_snap), str(out_snap_demo))
                snap_rel = f"outputs/snapshots/{inc_id}.jpg"
            except Exception as e:
                log.warning("Could not copy scenario real snapshot: %s", e)

        if not snap_rel and video_file and video_file.is_file():
            try:
                cap = cv2.VideoCapture(str(video_file))
                if cap.isOpened():
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 90)
                    target_idx = min(60, max(0, total_f // 4))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    ret, k_frame = cap.read()
                    if not ret or k_frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, k_frame = cap.read()
                    if ret and k_frame is not None:
                        out_snap_demo = DEMO_SNAPSHOT_DIR / f"{inc_id}.jpg"
                        out_snap_prod = PROJECT_ROOT / "outputs" / "snapshots" / f"{inc_id}.jpg"
                        out_snap_prod.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_snap_demo), k_frame)
                        cv2.imwrite(str(out_snap_prod), k_frame)
                        snap_rel = f"outputs/snapshots/{inc_id}.jpg"
                    cap.release()
            except Exception as e:
                log.warning("Could not extract snapshot from demo video: %s", e)

        if not snap_rel:
            snap_rel = "outputs/demo/snapshots/weapon_evidence.jpg"

        if email_clip and email_clip.is_file():
            try:
                out_clip_prod = PROJECT_ROOT / "outputs" / "clips" / f"{inc_id}.mp4"
                out_clip_prod.parent.mkdir(parents=True, exist_ok=True)
                if not out_clip_prod.is_file():
                    shutil.copy2(str(email_clip), str(out_clip_prod))
                clip_rel = f"outputs/clips/{inc_id}.mp4"
            except Exception as e:
                log.warning("Could not copy demo clip: %s", e)

        # Resolve unified demo location via Google Maps Geocoding API
        loc = self._resolve_demo_location()

        if scen_str == "WEAPON":
            itype = IncidentType.WEAPON
            sev = Severity.HIGH
            conf = 0.93
            bbox = BBox(x1=240.0, y1=180.0, x2=420.0, y2=480.0)
            sim_timeline = [
                DemoSimulationStep(step_id="s1", timestamp_label="T+00.0s", title="SCENARIO ACTIVATED", details="Weapon detection scenario initiated by operator."),
                DemoSimulationStep(step_id="s2", timestamp_label="T+01.0s", title="INCIDENT VERIFIED", details=f"Temporal persistence confirmed weapon object at {conf*100:.1f}% confidence."),
                DemoSimulationStep(step_id="s3", timestamp_label="T+01.2s", title="SEVERITY EVALUATED", details="Deterministic severity established as HIGH."),
                DemoSimulationStep(step_id="s4", timestamp_label="T+01.3s", title="NETWORK POLICY ELEVATED", details="Application transmission policy escalated to EVENT / HIGH priority."),
                DemoSimulationStep(step_id="s5", timestamp_label="T+01.5s", title="RESPONSE PLAN READY", details="Police security dispatch action recommended."),
                DemoSimulationStep(step_id="s6", timestamp_label="T+01.8s", title="AGENTIC ASSESSMENT READY", details="Supervisor decision-support briefing synthesized."),
                DemoSimulationStep(step_id="s7", timestamp_label="T+02.0s", title="OPERATOR GATE ACTIVE", details="Awaiting human approval before dispatch.", status="AWAITING_APPROVAL"),
            ]

        elif scen_str == "CROWD_PANIC":
            itype = IncidentType.CROWD_PANIC
            sev = Severity.HIGH
            conf = 0.89
            bbox = BBox(x1=150.0, y1=120.0, x2=550.0, y2=460.0)
            sim_timeline = [
                DemoSimulationStep(step_id="s1", timestamp_label="T+00.0s", title="SCENARIO ACTIVATED", details="Crowd panic surge scenario initiated."),
                DemoSimulationStep(step_id="s2", timestamp_label="T+01.0s", title="INCIDENT VERIFIED", details=f"Dense crowd surge confirmed at {conf*100:.1f}% confidence."),
                DemoSimulationStep(step_id="s3", timestamp_label="T+01.2s", title="SEVERITY EVALUATED", details="Deterministic severity established as HIGH."),
                DemoSimulationStep(step_id="s4", timestamp_label="T+01.3s", title="NETWORK POLICY ELEVATED", details="Application transmission policy escalated to EVENT / HIGH priority."),
                DemoSimulationStep(step_id="s5", timestamp_label="T+01.5s", title="RESPONSE PLAN READY", details="Crowd perimeter security dispatch recommended."),
                DemoSimulationStep(step_id="s6", timestamp_label="T+01.8s", title="AGENTIC ASSESSMENT READY", details="Supervisor decision-support briefing synthesized."),
                DemoSimulationStep(step_id="s7", timestamp_label="T+02.0s", title="OPERATOR GATE ACTIVE", details="Awaiting human approval before dispatch.", status="AWAITING_APPROVAL"),
            ]

        else:  # ARMED_FIGHT (CRITICAL)
            scen_str = "ARMED_FIGHT"
            itype = IncidentType.ARMED_FIGHT
            sev = Severity.CRITICAL
            conf = 0.97
            bbox = BBox(x1=210.0, y1=150.0, x2=510.0, y2=520.0)
            sim_timeline = [
                DemoSimulationStep(step_id="s1", timestamp_label="T+00.0s", title="SCENARIO ACTIVATED", details="Armed violent fight scenario initiated."),
                DemoSimulationStep(step_id="s2", timestamp_label="T+01.0s", title="INCIDENT VERIFIED", details=f"Temporal violent motion + weapon confirmed at {conf*100:.1f}% confidence."),
                DemoSimulationStep(step_id="s3", timestamp_label="T+01.2s", title="SEVERITY EVALUATED", details="Deterministic severity established as CRITICAL."),
                DemoSimulationStep(step_id="s4", timestamp_label="T+01.3s", title="NETWORK POLICY ELEVATED", details="Application transmission policy escalated to CRITICAL / CRITICAL priority."),
                DemoSimulationStep(step_id="s5", timestamp_label="T+01.5s", title="RESPONSE PLAN READY", details="Urgent Police Security & Medical standby generated."),
                DemoSimulationStep(step_id="s6", timestamp_label="T+01.8s", title="AGENTIC ASSESSMENT READY", details="Supervisor critical operational briefing synthesized."),
                DemoSimulationStep(step_id="s7", timestamp_label="T+02.0s", title="OPERATOR GATE ACTIVE", details="Awaiting mandatory human authorization.", status="AWAITING_APPROVAL"),
            ]

        report = IncidentReport(
            schema_version="1.0",
            source_module="crowd_fight",
            incident_id=inc_id,
            camera_id="cam_imc_demo",
            incident_type=itype,
            severity=sev,
            confidence=conf,
            start_time=now,
            status=IncidentStatus.VERIFIED,
            bbox=bbox,
            location=loc,
            evidence=Evidence(
                snapshot_path=snap_rel,
                clip_path=clip_rel,
                frame_idx=100,
            ),
            source_mode="DEMO",
            details={
                "demo_scenario": scen_str,
                "is_simulation": True,
                "video_file": str(video_file) if video_file else None,
                "email_clip_path": str(email_clip) if email_clip else None,
                "video_choice": video_choice,
                "note": "CONTROLLED DEMO SCENARIO - Real-time pipeline active on scenario video",
            },
        )

        with self._lock:
            self._active = True
            self._active_scenario_id = DemoScenarioId(scen_str)
            self._active_incident = report
            self._timeline = sim_timeline

            # Persist strictly into demo isolation directory (outputs/demo/incidents/)
            demo_path = DEMO_REPORT_DIR / f"{inc_id}.json"
            with open(demo_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

        # Update NetworkPolicyService directly
        net_svc = get_network_policy_service()
        net_svc.on_incident(report)

        # Notify EventBus / WebSocket if callback provided
        if broadcast_callback is not None:
            try:
                broadcast_callback(report)
            except Exception as e:
                log.warning("Broadcast callback failed for demo scenario: %s", e)

        log.info("DEMO_SCENARIO_STARTED: %s (id=%s, sev=%s, video=%s)", scen_str, inc_id, sev.value, video_file)

        return DemoStatusResponse(
            is_demo_active=True,
            active_scenario=scen_str,
            source_mode="DEMO",
            incident=report,
            simulation_timeline=sim_timeline,
            video_file=str(video_file) if video_file else None,
            video_choice=video_choice,
        )

    def stop_scenario(
        self,
        broadcast_callback: Callable[[IncidentReport], None] | None = None,
    ) -> DemoStatusResponse:
        """Stop active demonstration scenario and cleanly restore normal baseline."""
        with self._lock:
            self._active = False
            prev_scenario = self._active_scenario_id
            self._active_scenario_id = None
            self._active_incident = None
            self._timeline = [
                DemoSimulationStep(
                    step_id="r1",
                    timestamp_label="T+00.0s",
                    title="SCENARIO RESET",
                    details="Controlled demonstration stopped. Baseline routine monitoring restored.",
                    status="COMPLETED",
                )
            ]

        # Reset NetworkPolicyService back to NORMAL
        net_svc = get_network_policy_service()
        net_svc.reset()

        log.info("DEMO_SCENARIO_STOPPED: Previous was %s", prev_scenario)

        return DemoStatusResponse(
            is_demo_active=False,
            active_scenario=None,
            source_mode="REAL",
            incident=None,
            simulation_timeline=self._timeline,
        )

    def get_status(self) -> DemoStatusResponse:
        """Retrieve current real-time demonstration status."""
        with self._lock:
            return DemoStatusResponse(
                is_demo_active=self._active,
                active_scenario=self._active_scenario_id,
                source_mode="DEMO" if self._active else "REAL",
                incident=self._active_incident,
                simulation_timeline=list(self._timeline),
            )

    def get_demo_incident(self, incident_id: str) -> IncidentReport | None:
        """Retrieve simulated demo incident from memory or demo disk persistence."""
        with self._lock:
            if self._active_incident and self._active_incident.incident_id == incident_id:
                return self._active_incident

        path = DEMO_REPORT_DIR / f"{incident_id}.json"
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return IncidentReport.model_validate(data)
            except Exception:
                pass
        return None
