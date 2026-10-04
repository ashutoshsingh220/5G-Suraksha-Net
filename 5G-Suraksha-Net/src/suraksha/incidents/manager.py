"""Incident manager: verification, dedupe/cooldown, persistence, pub/sub."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from suraksha.config import CrowdConfig, PROJECT_ROOT, VerifyConfig
from suraksha.crowd.analyzer import CrowdSnapshot
from suraksha.detection.weapon import WeaponTelemetry
from suraksha.fight.recognizer import VerifiedFight
from suraksha.incidents.evidence import EvidenceWriter
from suraksha.incidents.fusion import CorrelatedArmedFight
from suraksha.incidents.schemas import (
    BBox,
    Evidence,
    IncidentReport,
    IncidentStatus,
    IncidentType,
    Severity,
)
from suraksha.location.provider import LocationProvider
from suraksha.location.schemas import Location
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


class EventBus:
    """Thread-safe in-process pub/sub. The FastAPI layer subscribes for WebSocket push."""

    def __init__(self) -> None:
        self._subs: list[Callable[[IncidentReport], None]] = []
        self._weapon_subs: list[Callable[[Any], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[IncidentReport], None]) -> None:
        with self._lock:
            self._subs.append(fn)

    def unsubscribe(self, fn: Callable[[IncidentReport], None]) -> None:
        with self._lock:
            if fn in self._subs:
                self._subs.remove(fn)

    def publish(self, report: IncidentReport) -> None:
        with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(report)
            except Exception:
                log.exception("EventBus subscriber failed")

    def subscribe_weapon(self, fn: Callable[[Any], None]) -> None:
        with self._lock:
            self._weapon_subs.append(fn)

    def unsubscribe_weapon(self, fn: Callable[[Any], None]) -> None:
        with self._lock:
            if fn in self._weapon_subs:
                self._weapon_subs.remove(fn)

    def publish_weapon(self, event: Any) -> None:
        with self._lock:
            subs = list(self._weapon_subs)
        for fn in subs:
            try:
                fn(event)
            except Exception:
                log.exception("EventBus weapon subscriber failed")


class IncidentManager:
    """Turns raw detections into deduplicated, evidence-backed IncidentReports."""

    MAX_RECENT = 200

    def __init__(
        self,
        camera_id: str,
        verify_cfg: VerifyConfig,
        evidence: EvidenceWriter,
        bus: EventBus,
        crowd_cfg: CrowdConfig | None = None,
        location_provider: LocationProvider | None = None,
        response_planner: Any | None = None,
    ):
        self.camera_id = camera_id
        self.verify_cfg = verify_cfg
        self.evidence = evidence
        self.bus = bus
        self.crowd_cfg = crowd_cfg or CrowdConfig()
        self.location_provider = location_provider
        self.response_planner = response_planner
        self.response_plans: dict[str, Any] = {}
        self.recent: deque[IncidentReport] = deque(maxlen=self.MAX_RECENT)
        self._cooldowns: dict[IncidentType, float] = {}
        self._entity_cooldowns: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def get_response_plan(self, incident_id: str) -> Any | None:
        """Retrieve or generate response plan for an incident."""
        with self._lock:
            for report in self.recent:
                if report.incident_id == incident_id:
                    if self.response_planner is not None:
                        plan = self.response_planner.plan(report)
                        self.response_plans[incident_id] = plan
                        return plan
            return self.response_plans.get(incident_id)

    def _resolve_location(self, now: float | None = None) -> Location | None:
        """Resolve current location using the configured LocationProvider, if available."""
        if self.location_provider is not None:
            try:
                return self.location_provider.get_location(camera_id=self.camera_id, timestamp=now)
            except Exception:
                log.exception("Failed to retrieve location from provider")
                return None
        return None

    def _on_cooldown(self, itype: IncidentType, now: float) -> bool:
        until = self._cooldowns.get(itype, 0.0)
        return now < until

    def _on_entity_cooldown(self, key: tuple, now: float) -> bool:
        until = self._entity_cooldowns.get(key, 0.0)
        return now < until

    def _persist(self, report: IncidentReport) -> None:
        report_dir = PROJECT_ROOT / self.evidence.cfg.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{report.incident_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

    def _emit(
        self,
        report: IncidentReport,
        frame: np.ndarray | None = None,
        now: float | None = None,
        entity_key: tuple | None = None,
    ) -> None:
        now_val = now if (now is not None and now > 0) else time.time()
        with self._lock:
            self.recent.append(report)
            self._cooldowns[report.incident_type] = now_val + self.verify_cfg.incident_cooldown_s
            if entity_key is not None:
                self._entity_cooldowns[entity_key] = now_val + self.verify_cfg.incident_cooldown_s
        self._persist(report)
        if self.response_planner is not None:
            try:
                self.response_plans[report.incident_id] = self.response_planner.plan(report)
            except Exception:
                log.exception("Failed to generate response plan for incident %s", report.incident_id)
        log.info(
            "INCIDENT %s type=%s severity=%s conf=%.2f zone=%s status=%s",
            report.incident_id, report.incident_type.value,
            report.severity.value, report.confidence, report.zone,
            report.status.value,
        )
        self.bus.publish(report)

    # ---------- fight ----------
    def report_fight(
        self,
        fight: VerifiedFight,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> IncidentReport | None:
        now = timestamp if (timestamp is not None and timestamp > 0) else (fight.end_time if fight.end_time > 0 else time.time())
        if self._on_cooldown(IncidentType.FIGHT, now):
            return None
        report = IncidentReport(
            camera_id=self.camera_id,
            incident_type=IncidentType.FIGHT,
            severity=Severity.MODERATE,
            confidence=round(min(fight.confidence, 1.0), 3),
            status=IncidentStatus.VERIFIED,
            start_time=datetime.fromtimestamp(fight.start_time, tz=timezone.utc),
            end_time=datetime.fromtimestamp(fight.end_time, tz=timezone.utc),
            track_ids=list(fight.track_ids),
            bbox=BBox.from_xyxy(fight.roi_xyxy),
            details={"windows_scored": fight.windows_scored, "detector": "temporal"},
            location=self._resolve_location(now),
        )
        if frame is not None:
            snap = None
            if report.severity == Severity.CRITICAL:
                snap = self.evidence.save_snapshot(
                    frame, report.incident_id,
                    annotations=[(fight.roi_xyxy, f"FIGHT {report.confidence:.2f}", (0, 0, 255))],
                )
            report.evidence = Evidence(snapshot_path=snap, clip_path=None)
            report.status = IncidentStatus.RECORDING_POST_EVENT

            def on_clip_complete(inc_id: str, clip_path: str) -> None:
                with self._lock:
                    for r in self.recent:
                        if r.incident_id == inc_id:
                            r.status = IncidentStatus.FINALIZED
                            r.finalized_at = datetime.now(timezone.utc)
                            if r.evidence:
                                r.evidence.clip_path = clip_path
                            self._persist(r)
                            self.bus.publish(r)
                            log.info("INCIDENT_FINALIZED %s clip=%s", inc_id, clip_path)
                            break

            self.evidence.start_clip_session(
                report.incident_id, fight.start_time, fight.end_time, callback=on_clip_complete
            )
        self._emit(report, frame, now=now)
        return report

    # ---------- weapon ----------
    def report_weapon(
        self,
        telemetry: WeaponTelemetry,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> IncidentReport | None:
        now = timestamp if (timestamp is not None and timestamp > 0) else (telemetry.frame_timestamp if telemetry.frame_timestamp > 0 else time.time())
        entity_key = (IncidentType.WEAPON, telemetry.track_id)

        # Entity cooldown check
        if self._on_entity_cooldown(entity_key, now):
            return None

        # Check if already active in WEAPON or ARMED_FIGHT
        with self._lock:
            for r in reversed(self.recent):
                if (
                    r.incident_type in (IncidentType.WEAPON, IncidentType.ARMED_FIGHT)
                    and (
                        telemetry.track_id in r.track_ids
                        or r.details.get("track_id") == telemetry.track_id
                        or r.details.get("weapon_track_id") == telemetry.track_id
                    )
                    and r.status in (IncidentStatus.VERIFIED, IncidentStatus.RECORDING_POST_EVENT)
                ):
                    return None

        report = IncidentReport(
            camera_id=self.camera_id,
            incident_type=IncidentType.WEAPON,
            severity=Severity.HIGH,
            confidence=round(min(telemetry.confidence, 1.0), 3),
            status=IncidentStatus.VERIFIED,
            start_time=datetime.fromtimestamp(now, tz=timezone.utc),
            end_time=datetime.fromtimestamp(now, tz=timezone.utc),
            track_ids=[telemetry.track_id],
            bbox=BBox.from_xyxy(telemetry.bbox),
            details={
                "weapon_class": telemetry.weapon_class,
                "track_id": telemetry.track_id,
                "persistence_count": telemetry.persistence_count,
                "confidence": round(telemetry.confidence, 4),
            },
            location=self._resolve_location(now),
        )

        if frame is not None:
            snap = self.evidence.save_snapshot(
                frame,
                report.incident_id,
                annotations=[
                    (
                        telemetry.bbox,
                        f"WEAPON: {telemetry.weapon_class.upper()} {report.confidence:.2f}",
                        (0, 0, 255),
                    )
                ],
            )
            report.evidence = Evidence(snapshot_path=snap, clip_path=None)
            report.status = IncidentStatus.RECORDING_POST_EVENT

            def on_clip_complete(inc_id: str, clip_path: str) -> None:
                with self._lock:
                    for r in self.recent:
                        if r.incident_id == inc_id:
                            r.status = IncidentStatus.FINALIZED
                            r.finalized_at = datetime.now(timezone.utc)
                            if r.evidence:
                                r.evidence.clip_path = clip_path
                            self._persist(r)
                            self.bus.publish(r)
                            log.info("INCIDENT_FINALIZED %s clip=%s", inc_id, clip_path)
                            break

            self.evidence.start_clip_session(
                report.incident_id, now, now, callback=on_clip_complete
            )

        self._emit(report, frame, now=now, entity_key=entity_key)
        return report

    # ---------- armed fight (fusion) ----------
    def report_armed_fight(
        self,
        armed_fight: CorrelatedArmedFight,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> IncidentReport | None:
        now = timestamp if (timestamp is not None and timestamp > 0) else time.time()
        fight_tids = tuple(sorted(armed_fight.fight.track_ids))
        entity_key = (IncidentType.ARMED_FIGHT, fight_tids)

        if self._on_entity_cooldown(entity_key, now):
            return None

        # Check for active weapon or fight incident to escalate
        target_weapon_tid = armed_fight.weapon_telemetry.track_id
        with self._lock:
            # 1. Escalate active weapon incident if present
            for r in reversed(self.recent):
                if (
                    r.incident_type == IncidentType.WEAPON
                    and r.details.get("track_id") == target_weapon_tid
                    and r.status in (IncidentStatus.VERIFIED, IncidentStatus.RECORDING_POST_EVENT)
                ):
                    r.incident_type = IncidentType.ARMED_FIGHT
                    r.severity = Severity.CRITICAL
                    r.confidence = round(min(max(r.confidence, armed_fight.correlation_confidence), 1.0), 3)
                    r.end_time = datetime.fromtimestamp(now, tz=timezone.utc)
                    r.bbox = BBox.from_xyxy(armed_fight.roi_xyxy)
                    all_ids = set(r.track_ids) | set(armed_fight.fight.track_ids)
                    if armed_fight.carrier_track_id is not None:
                        all_ids.add(armed_fight.carrier_track_id)
                    r.track_ids = list(all_ids)
                    r.details["escalated"] = True
                    r.details["escalated_from"] = IncidentType.WEAPON.value
                    r.details.update(armed_fight.to_dict())

                    if frame is not None:
                        snap = self.evidence.save_snapshot(
                            frame,
                            r.incident_id,
                            annotations=[
                                (armed_fight.fight.roi_xyxy, f"FIGHT {armed_fight.fight.confidence:.2f}", (0, 0, 255)),
                                (armed_fight.weapon_telemetry.bbox, f"ARMED: {armed_fight.weapon_telemetry.weapon_class.upper()} {armed_fight.weapon_telemetry.confidence:.2f}", (0, 0, 255)),
                            ],
                        )
                        if r.evidence:
                            r.evidence.snapshot_path = snap
                        else:
                            r.evidence = Evidence(snapshot_path=snap)

                    self._entity_cooldowns[entity_key] = now + self.verify_cfg.incident_cooldown_s
                    self._persist(r)
                    self.bus.publish(r)
                    log.info("ESCALATED_WEAPON_TO_ARMED_FIGHT %s severity=%s", r.incident_id, r.severity.value)
                    return r

            # 2. Escalate active unarmed fight incident if present
            for r in reversed(self.recent):
                if (
                    r.incident_type == IncidentType.FIGHT
                    and tuple(sorted(r.track_ids)) == fight_tids
                    and r.status in (IncidentStatus.VERIFIED, IncidentStatus.RECORDING_POST_EVENT)
                ):
                    r.incident_type = IncidentType.ARMED_FIGHT
                    r.severity = Severity.CRITICAL
                    r.confidence = round(min(max(r.confidence, armed_fight.correlation_confidence), 1.0), 3)
                    r.end_time = datetime.fromtimestamp(now, tz=timezone.utc)
                    r.bbox = BBox.from_xyxy(armed_fight.roi_xyxy)
                    if armed_fight.carrier_track_id is not None and armed_fight.carrier_track_id not in r.track_ids:
                        r.track_ids.append(armed_fight.carrier_track_id)
                    r.details["escalated"] = True
                    r.details["escalated_from"] = IncidentType.FIGHT.value
                    r.details.update(armed_fight.to_dict())

                    if frame is not None:
                        snap = self.evidence.save_snapshot(
                            frame,
                            r.incident_id,
                            annotations=[
                                (armed_fight.fight.roi_xyxy, f"FIGHT {armed_fight.fight.confidence:.2f}", (0, 0, 255)),
                                (armed_fight.weapon_telemetry.bbox, f"ARMED: {armed_fight.weapon_telemetry.weapon_class.upper()} {armed_fight.weapon_telemetry.confidence:.2f}", (0, 0, 255)),
                            ],
                        )
                        if r.evidence:
                            r.evidence.snapshot_path = snap
                        else:
                            r.evidence = Evidence(snapshot_path=snap)

                    self._entity_cooldowns[entity_key] = now + self.verify_cfg.incident_cooldown_s
                    self._persist(r)
                    self.bus.publish(r)
                    log.info("ESCALATED_FIGHT_TO_ARMED_FIGHT %s severity=%s", r.incident_id, r.severity.value)
                    return r

        # 3. Create fresh ARMED_FIGHT incident
        all_tids = list(set(list(armed_fight.fight.track_ids) + ([armed_fight.carrier_track_id] if armed_fight.carrier_track_id is not None else [])))
        report = IncidentReport(
            camera_id=self.camera_id,
            incident_type=IncidentType.ARMED_FIGHT,
            severity=Severity.CRITICAL,
            confidence=round(min(armed_fight.correlation_confidence, 1.0), 3),
            status=IncidentStatus.VERIFIED,
            start_time=datetime.fromtimestamp(armed_fight.fight.start_time, tz=timezone.utc),
            end_time=datetime.fromtimestamp(now, tz=timezone.utc),
            track_ids=all_tids,
            bbox=BBox.from_xyxy(armed_fight.roi_xyxy),
            details=armed_fight.to_dict(),
            location=self._resolve_location(now),
        )

        if frame is not None:
            snap = self.evidence.save_snapshot(
                frame,
                report.incident_id,
                annotations=[
                    (armed_fight.fight.roi_xyxy, f"FIGHT {armed_fight.fight.confidence:.2f}", (0, 0, 255)),
                    (armed_fight.weapon_telemetry.bbox, f"ARMED: {armed_fight.weapon_telemetry.weapon_class.upper()} {armed_fight.weapon_telemetry.confidence:.2f}", (0, 0, 255)),
                ],
            )
            report.evidence = Evidence(snapshot_path=snap, clip_path=None)
            report.status = IncidentStatus.RECORDING_POST_EVENT

            def on_clip_complete(inc_id: str, clip_path: str) -> None:
                with self._lock:
                    for r in self.recent:
                        if r.incident_id == inc_id:
                            r.status = IncidentStatus.FINALIZED
                            r.finalized_at = datetime.now(timezone.utc)
                            if r.evidence:
                                r.evidence.clip_path = clip_path
                            self._persist(r)
                            self.bus.publish(r)
                            log.info("INCIDENT_FINALIZED %s clip=%s", inc_id, clip_path)
                            break

            self.evidence.start_clip_session(
                report.incident_id, armed_fight.fight.start_time, armed_fight.fight.end_time, callback=on_clip_complete
            )

        self._emit(report, frame, now=now, entity_key=entity_key)
        with self._lock:
            self._entity_cooldowns[(IncidentType.WEAPON, armed_fight.weapon_telemetry.track_id)] = now + self.verify_cfg.incident_cooldown_s
        return report

    # ---------- crowd ----------
    def report_crowd(self, snapshot: CrowdSnapshot, frame: np.ndarray | None = None) -> list[IncidentReport]:
        reports: list[IncidentReport] = []
        now = snapshot.timestamp

        # Decouple environmental density context from incident generation:
        # Zone density and rapid growth only generate incidents if explicitly enabled in configuration.
        if self.crowd_cfg.enable_density_incidents:
            for z in snapshot.zones:
                if z.level == "critical":
                    itype, sev = IncidentType.CROWD_DENSITY_CRITICAL, Severity.CRITICAL
                elif z.level == "high":
                    itype, sev = IncidentType.CROWD_DENSITY_HIGH, Severity.MEDIUM
                else:
                    continue
                if self._on_cooldown(itype, now):
                    continue
                r = IncidentReport(
                    camera_id=self.camera_id,
                    incident_type=itype,
                    severity=sev,
                    confidence=round(min(z.density / max(0.55, 1e-6), 1.0), 3),
                    status=IncidentStatus.VERIFIED,
                    start_time=datetime.fromtimestamp(now, tz=timezone.utc),
                    zone=z.name,
                    person_count=snapshot.person_count,
                    details={"density": round(z.density, 4), "zone_person_count": z.person_count},
                    location=self._resolve_location(now),
                )
                if frame is not None:
                    r.evidence = Evidence(
                        snapshot_path=self.evidence.save_snapshot(frame, r.incident_id)
                    )
                reports.append(r)
                self._emit(r, frame)

            if snapshot.growth_alert and not self._on_cooldown(IncidentType.CROWD_RAPID_GROWTH, now):
                r = IncidentReport(
                    camera_id=self.camera_id,
                    incident_type=IncidentType.CROWD_RAPID_GROWTH,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    status=IncidentStatus.VERIFIED,
                    start_time=datetime.fromtimestamp(now, tz=timezone.utc),
                    person_count=snapshot.person_count,
                    details={"growth_per_min": round(snapshot.growth_per_min, 2)},
                    location=self._resolve_location(now),
                )
                reports.append(r)
                self._emit(r, frame)

        # Panic incident: strictly guarded by multi-signal verification: panic AND min_panic_persons
        if (
            snapshot.movement.panic
            and snapshot.person_count >= self.crowd_cfg.min_panic_persons
            and not self._on_cooldown(IncidentType.CROWD_PANIC, now)
        ):
            r = IncidentReport(
                camera_id=self.camera_id,
                incident_type=IncidentType.CROWD_PANIC,
                severity=Severity.HIGH,
                confidence=0.75,
                status=IncidentStatus.VERIFIED,
                start_time=datetime.fromtimestamp(now, tz=timezone.utc),
                person_count=snapshot.person_count,
                details={
                    "fast_track_ratio": round(snapshot.movement.fast_track_ratio, 3),
                    "direction_entropy": round(snapshot.movement.direction_entropy, 3),
                },
                location=self._resolve_location(now),
            )
            if frame is not None:
                r.evidence = Evidence(
                    snapshot_path=self.evidence.save_snapshot(frame, r.incident_id)
                )
            reports.append(r)
            self._emit(r, frame)

        return reports
