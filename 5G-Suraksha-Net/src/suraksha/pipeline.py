"""End-to-end pipeline orchestrator (runs in a background thread)."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from suraksha.config import AppConfig, load_config
from suraksha.crowd.analyzer import CrowdAnalyzer, CrowdSnapshot
from suraksha.detection.tracker import MultiObjectTracker, TrackedPerson
from suraksha.detection.weapon import WeaponDetector, WeaponEvent, WeaponState
from suraksha.capture.stream import StreamCapture
from suraksha.device import detect_device
from suraksha.fight.recognizer import FightRecognizer, VerifiedFight
from suraksha.incidents.evidence import EvidenceWriter
from suraksha.incidents.fusion import FusionDecision, IncidentFusionEngine
from suraksha.incidents.manager import EventBus, IncidentManager
from suraksha.location import LocationProvider, create_location_provider
from suraksha.logging_utils import get_logger
from suraksha.notifications import EmailNotificationService
from suraksha.response import ResponsePlanner
from suraksha.visualization.annotator import FrameAnnotator
from suraksha.visualization.video_writer import AnnotatedVideoWriter

log = get_logger(__name__)


@dataclass
class PipelineState:
    """Shared read-mostly state, consumed by the API layer."""
    running: bool = False
    started_at: float | None = None
    frames_processed: int = 0
    last_frame_time: float | None = None
    effective_fps: float = 0.0
    device: str = "cpu"
    person_count: int = 0
    active_fights_count: int = 0
    latest_crowd: dict = field(default_factory=dict)
    latest_weapon: dict = field(default_factory=dict)
    latest_incidents: list[dict] = field(default_factory=list)
    # running averages in ms; detect_track is a single ultralytics call
    # (YOLO inference + ByteTrack association are not separately measurable)
    metrics: dict = field(default_factory=lambda: {
        "detect_track_ms": 0.0, "crowd_ms": 0.0, "fight_ms": 0.0, "weapon_ms": 0.0, "total_ms": 0.0,
        "resolution": None, "source_fps": 0.0,
    })


class CrowdFightPipeline:
    """Sparsh RTSP -> YOLO11s+ByteTrack -> crowd analytics -> fight recognition -> incidents."""

    def __init__(
        self,
        cfg: AppConfig | None = None,
        bus: EventBus | None = None,
        preview_queue: queue.Queue | None = None,
        save_video_path: str | Path | None = None,
        auditor: Any | None = None,
    ):
        self.cfg = cfg or load_config()
        self.auditor = auditor
        self.device_info = detect_device(self.cfg.device)
        self.device = self.device_info.device
        self.bus = bus or EventBus()
        self.state = PipelineState(device=self.device)
        self.evidence = EvidenceWriter(self.cfg.incidents)
        self.location_provider: LocationProvider = create_location_provider(self.cfg.location)
        self.planner = ResponsePlanner()
        self.manager = IncidentManager(
            self.cfg.camera_id,
            self.cfg.fight.verify,
            self.evidence,
            self.bus,
            crowd_cfg=self.cfg.crowd,
            location_provider=self.location_provider,
            response_planner=self.planner,
        )
        self.weapon_detector = WeaponDetector(self.cfg.weapon, device=self.device)
        self.weapon_detector.subscribe(self.bus.publish_weapon)
        self.fusion = IncidentFusionEngine(self.cfg.fusion)
        self.email_notifier = EmailNotificationService(config=self.cfg.email, planner=self.planner)
        self.bus.subscribe(self.email_notifier.handle_incident_event)

        self.capture = StreamCapture(self.cfg.capture)
        self.preview_queue = preview_queue
        self.save_video_path = Path(save_video_path) if save_video_path else None
        self.annotator: FrameAnnotator | None = None
        self.video_writer: AnnotatedVideoWriter | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()

    def _init_components(self, frame_shape: tuple[int, int], source_fps: float | None = None):
        """Shape-dependent components, built lazily on the first frame."""
        self.tracker = MultiObjectTracker(self.cfg.detection, self.cfg.tracking, self.device)
        self.crowd = CrowdAnalyzer(self.cfg.crowd, self.cfg.camera_id, frame_shape)
        self.fight = FightRecognizer(self.cfg.fight, frame_shape, self.device)
        if self.save_video_path or self.preview_queue is not None:
            self.annotator = FrameAnnotator()
        if self.save_video_path:
            h, w = frame_shape
            fps = source_fps if (source_fps and source_fps > 0) else float(self.cfg.capture.target_fps)
            self.video_writer = AnnotatedVideoWriter(
                output_path=self.save_video_path,
                fps=fps,
                frame_size=(w, h),
            )

    def _loop(self) -> None:
        first = True
        fps_t0 = time.time()
        fps_n = 0
        try:
            for ev in self.capture.frames():
                if self._stop.is_set():
                    break
                while self._pause.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                if self._stop.is_set():
                    break
                if first:
                    self._init_components(ev.frame.shape[:2], ev.source_fps)
                    first = False

                self.evidence.push_frame(ev.frame, ev.timestamp)
                h, w = ev.frame.shape[:2]

                # Branch A: Weapon Detection (independent stream)
                t_w0 = time.perf_counter()
                weapon_ev: WeaponEvent = self.weapon_detector.update(ev)
                t_w1 = time.perf_counter()
                weapon_ms = (t_w1 - t_w0) * 1000.0

                if self.auditor is not None and weapon_ev.detections:
                    self.auditor.record_detections(
                        ev.frame,
                        weapon_ev.detections,
                        ev.frame_idx,
                        ev.timestamp,
                    )

                # Branch B: Person Detection + ByteTrack -> Crowd -> Fight
                t0 = time.perf_counter()
                tracks: list[TrackedPerson] = self.tracker.update(ev.frame)
                t1 = time.perf_counter()
                crowd: CrowdSnapshot = self.crowd.update(ev.frame, tracks, ev.timestamp)
                t2 = time.perf_counter()
                verified = self.fight.update(ev.frame, tracks, ev.timestamp, fps=ev.source_fps)
                t3 = time.perf_counter()

                m = self.state.metrics
                n = self.state.frames_processed
                # running average over processed frames
                m["weapon_ms"] += (weapon_ms - m.get("weapon_ms", 0.0)) / (n + 1)
                m["detect_track_ms"] += ((t1 - t0) * 1000 - m["detect_track_ms"]) / (n + 1)
                m["crowd_ms"] += ((t2 - t1) * 1000 - m["crowd_ms"]) / (n + 1)
                m["fight_ms"] += ((t3 - t2) * 1000 - m["fight_ms"]) / (n + 1)
                m["total_ms"] += (((t3 - t0) + (t_w1 - t_w0)) * 1000 - m["total_ms"]) / (n + 1)
                m["resolution"] = f"{w}x{h}"
                m["source_fps"] = ev.source_fps

                self.state.person_count = crowd.person_count
                self.state.active_fights_count = len(verified)
                self.state.latest_crowd = crowd.to_dict()
                self.state.latest_weapon = weapon_ev.to_dict()
                self.state.frames_processed += 1
                self.state.last_frame_time = ev.timestamp

                # Multimodal Incident Fusion Layer
                fusion_decision: FusionDecision = self.fusion.evaluate(
                    weapon_event=weapon_ev,
                    verified_fights=verified,
                    tracks=tracks,
                    timestamp=ev.timestamp,
                )

                self.manager.report_crowd(crowd, ev.frame)
                for uf in fusion_decision.unarmed_fights:
                    self.manager.report_fight(uf, ev.frame, timestamp=ev.timestamp)
                for wt in fusion_decision.weapons_alone:
                    self.manager.report_weapon(wt, ev.frame, timestamp=ev.timestamp)
                for af in fusion_decision.armed_fights:
                    self.manager.report_armed_fight(af, ev.frame, timestamp=ev.timestamp)

                self.state.latest_incidents = [
                    r.model_dump(mode="json") for r in list(self.manager.recent)[-20:]
                ]

                fps_n += 1
                if time.time() - fps_t0 >= 5.0:
                    self.state.effective_fps = fps_n / (time.time() - fps_t0)
                    fps_t0, fps_n = time.time(), 0

                # Render and dispatch visual preview / video export if enabled
                if self.annotator is not None:
                    active_cands = list(self.fight.latest_candidates)
                    active_verified = list(verified)
                    for key, af in self.fight._active.items():
                        if af.verified and key not in [v.track_ids for v in active_verified]:
                            active_verified.append(VerifiedFight(
                                track_ids=af.candidate.track_ids,
                                roi_xyxy=af.candidate.roi_xyxy.copy(),
                                start_time=af.candidate.first_seen,
                                end_time=ev.timestamp,
                                confidence=af.window_scores[-1] if af.window_scores else 0.0,
                                windows_scored=len(af.window_scores),
                            ))

                    all_scores = [af.window_scores[-1] for af in self.fight._active.values() if af.window_scores]
                    max_fight_score = max(all_scores) if all_scores else None

                    if fusion_decision.armed_fights:
                        af_first = fusion_decision.armed_fights[0]
                        inc_state = f"! ARMED FIGHT ({af_first.weapon_telemetry.weapon_class.upper()}) [CRITICAL] !"
                    elif fusion_decision.weapons_alone:
                        w_cls = fusion_decision.weapons_alone[0].weapon_class.upper()
                        inc_state = f"! WEAPON DETECTED ({w_cls}) [HIGH] !"
                    elif active_verified:
                        inc_state = "! FIGHT DETECTED (MODERATE) !"
                    elif active_cands:
                        inc_state = f"CANDIDATE ({len(active_cands)} pairs)"
                    elif crowd.growth_alert:
                        inc_state = "CROWD RAPID GROWTH"
                    else:
                        inc_state = "NORMAL"

                    annotated_frame = self.annotator.annotate(
                        frame=ev.frame,
                        tracks=tracks,
                        candidates=active_cands,
                        verified_fights=active_verified,
                        crowd=crowd,
                        weapon_event=weapon_ev,
                        metrics=self.state.metrics,
                        effective_fps=self.state.effective_fps,
                        max_fight_score=max_fight_score,
                        score_threshold=self.cfg.fight.temporal.score_threshold,
                        incident_state=inc_state,
                        frame_idx=ev.frame_idx,
                    )

                    if self.video_writer is not None:
                        self.video_writer.write(annotated_frame)

                    if self.preview_queue is not None:
                        try:
                            self.preview_queue.put_nowait(annotated_frame)
                        except queue.Full:
                            try:
                                _ = self.preview_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.preview_queue.put_nowait(annotated_frame)
                            except queue.Full:
                                pass
        except Exception:
            log.exception("Pipeline loop crashed")
        finally:
            # set the flag FIRST so waiters always unblock, even if cleanup fails
            self.state.running = False
            try:
                self.evidence.finalize_all()
            except Exception:
                log.exception("Evidence finalization failed")
            if self.auditor is not None:
                try:
                    self.auditor.finalize()
                except Exception:
                    log.exception("Auditor finalization failed")
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    log.exception("Video writer release failed")
            if self.preview_queue is not None:
                try:
                    self.preview_queue.put(None)
                except Exception:
                    pass
            try:
                self.capture.release()
            except Exception:
                log.exception("Capture release failed")
            log.info("Pipeline stopped after %d frames", self.state.frames_processed)

    def start(self) -> None:
        if self.state.running:
            return
        self._stop.clear()
        # Peek one frame to learn the shape, then start the loop thread.
        self.state.running = True
        self.state.started_at = time.time()
        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self._thread.start()
        log.info("Pipeline started (device=%s)", self.device)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._pause.clear()  # unblock pause if thread is waiting
        if self._thread:
            self._thread.join(timeout=timeout)
        self.state.running = False
        if hasattr(self, "email_notifier") and self.email_notifier is not None:
            self.email_notifier.shutdown(wait=False)

    def pause(self) -> None:
        """Pause frame processing."""
        self._pause.set()

    def resume(self) -> None:
        """Resume frame processing."""
        self._pause.clear()

    @property
    def is_paused(self) -> bool:
        """Whether the pipeline processing loop is currently paused."""
        return self._pause.is_set()

    def switch_source(self, url: str, source_type: str = "file", pace: str = "realtime") -> bool:
        """Dynamically hot-swap video capture source on the fly without reloading YOLO models."""
        if hasattr(self, "capture") and hasattr(self.capture, "switch_source"):
            ok = self.capture.switch_source(url=url, source_kind=source_type, pace=pace)
            # Reset detector transient candidate state so scenarios don't bleed into baseline
            if hasattr(self, "weapon_detector") and hasattr(self.weapon_detector, "confirmation"):
                self.weapon_detector.confirmation.reset()
            if hasattr(self, "fight") and hasattr(self.fight, "_active"):
                self.fight._active.clear()
                if hasattr(self.fight, "latest_candidates"):
                    self.fight.latest_candidates.clear()
            return ok
        return False


def main(cfg: AppConfig | None = None) -> None:
    from suraksha.logging_utils import setup_logging

    cfg = cfg or load_config()
    setup_logging(cfg.log_level)
    pipeline = CrowdFightPipeline(cfg)
    pipeline.start()
    try:
        while pipeline.state.running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
