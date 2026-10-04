"""Weapon detection and temporal persistence tracking module for 5G Suraksha-Net.

Consumes FrameEvent stream and produces structured WeaponEvent telemetry.
Independent capability parallel to PersonDetector / ByteTrack / VideoMAE.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from suraksha.capture.stream import FrameEvent
from suraksha.config import WeaponConfig
from suraksha.logging_utils import get_logger

log = get_logger(__name__)

WEAPON_CLASSES = {
    0: "knife",
    1: "long_gun",
    2: "pistol",
}
WEAPON_CLASS_NAMES = set(WEAPON_CLASSES.values())


class WeaponState(str, Enum):
    """Temporal persistence state for weapon monitoring."""
    NORMAL = "NORMAL"
    WEAPON_CANDIDATE = "WEAPON_CANDIDATE"
    WEAPON_CONFIRMED = "WEAPON_CONFIRMED"


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter_area
    if union <= 1e-6:
        return 0.0
    return float(inter_area / union)


def compute_proximity(box1: np.ndarray, box2: np.ndarray) -> float:
    """Center distance proximity normalized by average bounding box diagonal."""
    c1_x = (float(box1[0]) + float(box1[2])) / 2.0
    c1_y = (float(box1[1]) + float(box1[3])) / 2.0
    c2_x = (float(box2[0]) + float(box2[2])) / 2.0
    c2_y = (float(box2[1]) + float(box2[3])) / 2.0

    diag1 = np.hypot(float(box1[2]) - float(box1[0]), float(box1[3]) - float(box1[1]))
    diag2 = np.hypot(float(box2[2]) - float(box2[0]), float(box2[3]) - float(box2[1]))
    avg_diag = (diag1 + diag2) / 2.0
    if avg_diag <= 1e-6:
        return 0.0

    dist = np.hypot(c1_x - c2_x, c1_y - c2_y)
    return float(max(0.0, 1.0 - (dist / (2.0 * avg_diag))))


@dataclass
class RawWeaponDetection:
    """Raw single-frame YOLO prediction."""
    bbox_xyxy: np.ndarray  # float32 [x1, y1, x2, y2]
    confidence: float
    cls_id: int
    class_name: str


@dataclass
class WeaponTelemetry:
    """Structured telemetry payload for one detected/tracked weapon instance."""
    weapon_class: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    frame_timestamp: float
    persistence_count: int
    confirmation_state: str  # NORMAL | WEAPON_CANDIDATE | WEAPON_CONFIRMED
    track_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "weapon_class": self.weapon_class,
            "confidence": round(self.confidence, 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "frame_timestamp": self.frame_timestamp,
            "persistence_count": self.persistence_count,
            "confirmation_state": self.confirmation_state,
            "track_id": self.track_id,
        }


@dataclass
class WeaponEvent:
    """Published internal event containing current frame weapon status and telemetry."""
    frame_idx: int
    timestamp: float
    state: WeaponState
    telemetry: list[WeaponTelemetry] = field(default_factory=list)
    candidate_count: int = 0
    confirmed_count: int = 0
    detections: list[RawWeaponDetection] = field(default_factory=list)
    inference_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "candidate_count": self.candidate_count,
            "confirmed_count": self.confirmed_count,
            "detections_count": len(self.detections),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "telemetry": [t.to_dict() for t in self.telemetry],
        }


class WeaponTrack:
    """Maintains multi-frame temporal consistency and confirmation state for a weapon instance."""

    def __init__(
        self,
        track_id: int,
        detection: RawWeaponDetection,
        timestamp: float,
        window_size: int = 5,
        confirm_conf_threshold: float = 0.70,
        min_hits: int = 3,
    ) -> None:
        self.track_id = track_id
        self.cls_id = detection.cls_id
        self.class_name = detection.class_name
        self.last_bbox = detection.bbox_xyxy.copy()
        self.last_conf = detection.confidence
        self.window_size = window_size
        self.confirm_conf_threshold = confirm_conf_threshold
        self.min_hits = min_hits

        # Sliding window of hits (True/False) for eligible confirmation frames
        self.history: deque[bool] = deque(maxlen=window_size)
        self.score_history: deque[float] = deque(maxlen=window_size)

        # Initial hit
        is_hit = detection.confidence >= confirm_conf_threshold
        self.history.append(is_hit)
        self.score_history.append(detection.confidence)

        self.total_hits = 1
        self.consecutive_misses = 0
        self.first_seen_timestamp = timestamp
        self.last_seen_timestamp = timestamp
        self.confirmed_timestamp: float | None = None

        # Check confirmation status
        self._update_state()

    def update_hit(self, detection: RawWeaponDetection, timestamp: float) -> None:
        """Update track with a matched detection in the current frame."""
        self.last_bbox = detection.bbox_xyxy.copy()
        self.last_conf = detection.confidence
        self.last_seen_timestamp = timestamp
        self.total_hits += 1
        self.consecutive_misses = 0

        is_hit = detection.confidence >= self.confirm_conf_threshold
        self.history.append(is_hit)
        self.score_history.append(detection.confidence)

        self._update_state()

    def update_miss(self) -> None:
        """Register a frame where this track was not detected."""
        self.history.append(False)
        self.consecutive_misses += 1
        self._update_state()

    def _update_state(self) -> None:
        hits = sum(1 for h in self.history if h)
        if hits >= self.min_hits and self.last_conf >= self.confirm_conf_threshold:
            self.state = WeaponState.WEAPON_CONFIRMED
            if self.confirmed_timestamp is None:
                self.confirmed_timestamp = self.last_seen_timestamp
        elif getattr(self, "state", None) == WeaponState.WEAPON_CONFIRMED:
            # Latching hysteresis: once confirmed, maintain CONFIRMED status while track remains active
            pass
        else:
            self.state = WeaponState.WEAPON_CANDIDATE

    @property
    def hits_in_window(self) -> int:
        return sum(1 for h in self.history if h)


class WeaponPersistenceTracker:
    """Manages weapon tracks and evaluates spatial/temporal persistence across frames."""

    def __init__(
        self,
        confirm_conf_threshold: float = 0.70,
        min_hits: int = 3,
        window_size: int = 5,
        iou_match_threshold: float = 0.2,
        expiry_frames: int = 15,
    ) -> None:
        self.confirm_conf_threshold = confirm_conf_threshold
        self.min_hits = min_hits
        self.window_size = window_size
        self.iou_match_threshold = iou_match_threshold
        self.expiry_frames = expiry_frames

        self._tracks: dict[int, WeaponTrack] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[RawWeaponDetection],
        timestamp: float,
    ) -> tuple[WeaponState, list[WeaponTelemetry], int, int]:
        """Update persistence state with new detections."""
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Match detections to existing tracks
        # Strict rule: class MUST match; class change resets / does not match existing track
        if self._tracks and detections:
            track_ids = list(self._tracks.keys())
            # Build cost / score matrix for matching
            candidates: list[tuple[float, int, int]] = []
            for d_idx, det in enumerate(detections):
                for t_id in track_ids:
                    track = self._tracks[t_id]
                    if det.cls_id != track.cls_id:
                        # Class mismatch: cannot match
                        continue
                    iou = compute_iou(det.bbox_xyxy, track.last_bbox)
                    prox = compute_proximity(det.bbox_xyxy, track.last_bbox)
                    if iou >= self.iou_match_threshold or prox >= 0.70:
                        score = max(iou, prox)
                        candidates.append((score, d_idx, t_id))

            # Greedy match by descending score
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, d_idx, t_id in candidates:
                if d_idx in matched_dets or t_id in matched_tracks:
                    continue
                matched_dets.add(d_idx)
                matched_tracks.add(t_id)
                self._tracks[t_id].update_hit(detections[d_idx], timestamp)

        # Unmatched existing tracks record a miss
        for t_id, track in list(self._tracks.items()):
            if t_id not in matched_tracks:
                track.update_miss()
                if track.consecutive_misses >= self.expiry_frames:
                    del self._tracks[t_id]

        # Unmatched detections spawn new candidate tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                t_id = self._next_track_id
                self._next_track_id += 1
                new_track = WeaponTrack(
                    track_id=t_id,
                    detection=det,
                    timestamp=timestamp,
                    window_size=self.window_size,
                    confirm_conf_threshold=self.confirm_conf_threshold,
                    min_hits=self.min_hits,
                )
                self._tracks[t_id] = new_track

        # Build telemetry and evaluate overall state
        telemetry: list[WeaponTelemetry] = []
        candidate_count = 0
        confirmed_count = 0

        for track in self._tracks.values():
            if track.state == WeaponState.WEAPON_CONFIRMED:
                confirmed_count += 1
            elif track.state == WeaponState.WEAPON_CANDIDATE:
                candidate_count += 1

            telemetry.append(
                WeaponTelemetry(
                    weapon_class=track.class_name,
                    confidence=float(track.last_conf),
                    bbox=[float(v) for v in track.last_bbox],
                    frame_timestamp=track.last_seen_timestamp,
                    persistence_count=track.hits_in_window,
                    confirmation_state=track.state.value,
                    track_id=track.track_id,
                )
            )

        if confirmed_count > 0:
            overall_state = WeaponState.WEAPON_CONFIRMED
        elif candidate_count > 0:
            overall_state = WeaponState.WEAPON_CANDIDATE
        else:
            overall_state = WeaponState.NORMAL

        return overall_state, telemetry, candidate_count, confirmed_count

    def get_current_state(self) -> tuple[WeaponState, list[WeaponTelemetry], int, int]:
        """Compute current state and telemetry without modifying track histories or registering misses."""
        telemetry: list[WeaponTelemetry] = []
        candidate_count = 0
        confirmed_count = 0

        for track in self._tracks.values():
            if track.state == WeaponState.WEAPON_CONFIRMED:
                confirmed_count += 1
            elif track.state == WeaponState.WEAPON_CANDIDATE:
                candidate_count += 1

            telemetry.append(
                WeaponTelemetry(
                    weapon_class=track.class_name,
                    confidence=float(track.last_conf),
                    bbox=[float(v) for v in track.last_bbox],
                    frame_timestamp=track.last_seen_timestamp,
                    persistence_count=track.hits_in_window,
                    confirmation_state=track.state.value,
                    track_id=track.track_id,
                )
            )

        if confirmed_count > 0:
            overall_state = WeaponState.WEAPON_CONFIRMED
        elif candidate_count > 0:
            overall_state = WeaponState.WEAPON_CANDIDATE
        else:
            overall_state = WeaponState.NORMAL

        return overall_state, telemetry, candidate_count, confirmed_count

    def reset(self) -> None:
        """Reset all active tracks."""
        self._tracks.clear()
        self._next_track_id = 1


class WeaponDetector:
    """Standalone weapon detector and temporal persistence orchestrator."""

    def __init__(
        self,
        cfg: WeaponConfig | None = None,
        device: str = "auto",
        model: Any = None,
    ) -> None:
        self.cfg = cfg or WeaponConfig()
        self.device = device
        self.model = model
        self._subscribers: list[Callable[[WeaponEvent], None]] = []

        # Persistence tracker
        confirm_cfg = self.cfg.confirmation
        self.tracker = WeaponPersistenceTracker(
            confirm_conf_threshold=confirm_cfg.confirm_conf_threshold,
            min_hits=confirm_cfg.min_hits,
            window_size=confirm_cfg.window_size,
            iou_match_threshold=confirm_cfg.iou_match_threshold,
            expiry_frames=confirm_cfg.expiry_frames,
        )

        # Observability metrics
        self.last_latency_ms: float = 0.0
        self._latencies: deque[float] = deque(maxlen=200)
        self.inference_count: int = 0
        self.total_frames_processed: int = 0
        self.per_class_detections: dict[str, int] = {"knife": 0, "long_gun": 0, "pistol": 0}
        self.current_state: WeaponState = WeaponState.NORMAL
        self.latest_event: WeaponEvent | None = None
        self._fps_t0: float = time.time()
        self._fps_n: int = 0
        self.effective_fps: float = 0.0

        if self.cfg.enabled and self.model is None:
            self._init_model()

    def _init_model(self) -> None:
        """Load YOLO11s weapon weights."""
        from ultralytics import YOLO

        target_device = "cuda:0" if self.device in ("cuda", "cuda:0") or (self.device == "auto" and Path(self.cfg.weights).exists()) else "cpu"
        # Ultralytics device resolution
        device_str = "0" if target_device.startswith("cuda") else "cpu"

        weights_path = Path(self.cfg.weights)
        if not weights_path.exists():
            log.warning("Weapon weights path not found: %s. Weapon inference will be disabled.", weights_path)
            self.model = None
            return

        log.info("Loading YOLO11s weapon detector from: %s (device=%s)", weights_path, device_str)
        self.model = YOLO(str(weights_path))
        self.device_str = device_str

    def subscribe(self, fn: Callable[[WeaponEvent], None]) -> None:
        """Subscribe to published WeaponEvent stream."""
        if fn not in self._subscribers:
            self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[WeaponEvent], None]) -> None:
        """Unsubscribe from WeaponEvent stream."""
        if fn in self._subscribers:
            self._subscribers.remove(fn)

    def _publish(self, event: WeaponEvent) -> None:
        """Publish WeaponEvent to subscribers."""
        for fn in list(self._subscribers):
            try:
                fn(event)
            except Exception:
                log.exception("Error in WeaponEvent subscriber callback")

    def detect_frame(self, frame: np.ndarray) -> tuple[list[RawWeaponDetection], float]:
        """Execute raw YOLO11s forward pass on frame."""
        if self.model is None:
            return [], 0.0

        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "conf": self.cfg.conf_threshold,
            "iou": self.cfg.iou_threshold,
            "imgsz": self.cfg.imgsz,
            "classes": [0, 1, 2],  # knife, long_gun, pistol
            "verbose": False,
        }
        if hasattr(self, "device_str"):
            kwargs["device"] = self.device_str

        results = self.model.predict(frame, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Strict firearm & weapon confidence gating:
        # Firearms (pistol, long_gun/shotgun) must achieve at least 0.70 confidence
        # to prevent spurious false positives from ambient handheld items (phones, remotes).
        class_conf_minimums = {
            "pistol": 0.70,
            "long_gun": 0.70,
            "knife": 0.65,
        }

        detections: list[RawWeaponDetection] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                has_boxes = False
                try:
                    has_boxes = len(boxes) > 0
                except TypeError:
                    if hasattr(boxes, "xyxy"):
                        has_boxes = len(boxes.xyxy) > 0

                if has_boxes:
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
                    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)

                    for b, c, cl in zip(xyxy, confs, clss):
                        cid = int(cl)
                        cname = WEAPON_CLASSES.get(cid)
                        class_min = class_conf_minimums.get(cname, 0.70)
                        effective_thresh = max(self.cfg.conf_threshold, class_min)
                        if cname is not None and float(c) >= effective_thresh:
                            detections.append(
                                RawWeaponDetection(
                                    bbox_xyxy=b.astype(np.float32),
                                    confidence=float(c),
                                    cls_id=cid,
                                    class_name=cname,
                                )
                            )
                            self.per_class_detections[cname] = self.per_class_detections.get(cname, 0) + 1

        return detections, latency_ms

    def update(self, event: FrameEvent) -> WeaponEvent:
        """Process a FrameEvent, perform detection + persistence tracking, and emit WeaponEvent."""
        self.total_frames_processed += 1
        frame_idx = event.frame_idx
        timestamp = event.timestamp

        # 1. Disabled mode
        if not self.cfg.enabled:
            weap_event = WeaponEvent(
                frame_idx=frame_idx,
                timestamp=timestamp,
                state=WeaponState.NORMAL,
                telemetry=[],
                candidate_count=0,
                confirmed_count=0,
                detections=[],
                inference_latency_ms=0.0,
            )
            self.current_state = WeaponState.NORMAL
            self.latest_event = weap_event
            self._publish(weap_event)
            return weap_event

        # 2. Inference interval gating (rate limiting forward passes)
        interval = max(1, self.cfg.inference_interval)
        should_infer = (frame_idx % interval == 0)

        raw_dets: list[RawWeaponDetection] = []
        latency_ms = 0.0

        if should_infer and self.model is not None:
            raw_dets, latency_ms = self.detect_frame(event.frame)
            self.inference_count += 1
            self.last_latency_ms = latency_ms
            self._latencies.append(latency_ms)

            # Update persistence tracker with new detections
            state, telemetry, cand_cnt, conf_cnt = self.tracker.update(raw_dets, timestamp)
        else:
            # Re-use current persistence state without registering false misses on skipped frames
            state, telemetry, cand_cnt, conf_cnt = self.tracker.get_current_state()

        self.current_state = state

        # Observability FPS
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t0 >= 2.0:
            self.effective_fps = round(self._fps_n / max(now - self._fps_t0, 1e-4), 1)
            self._fps_t0, self._fps_n = now, 0

        weap_event = WeaponEvent(
            frame_idx=frame_idx,
            timestamp=timestamp,
            state=state,
            telemetry=telemetry,
            candidate_count=cand_cnt,
            confirmed_count=conf_cnt,
            detections=raw_dets,
            inference_latency_ms=latency_ms,
        )

        self.latest_event = weap_event
        self._publish(weap_event)
        return weap_event

    def get_metrics(self) -> dict[str, Any]:
        """Return operational telemetry dictionary."""
        mean_lat = float(np.mean(self._latencies)) if self._latencies else 0.0
        p95_lat = float(np.percentile(self._latencies, 95)) if self._latencies else 0.0
        return {
            "enabled": self.cfg.enabled,
            "state": self.current_state.value,
            "inference_latency_ms": round(self.last_latency_ms, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "inference_count": self.inference_count,
            "total_frames_processed": self.total_frames_processed,
            "effective_fps": self.effective_fps,
            "candidate_count": self.latest_event.candidate_count if self.latest_event else 0,
            "confirmed_count": self.latest_event.confirmed_count if self.latest_event else 0,
            "per_class_detections": dict(self.per_class_detections),
        }

    def reset(self) -> None:
        """Reset state, tracker, and telemetry."""
        self.tracker.reset()
        self.last_latency_ms = 0.0
        self._latencies.clear()
        self.inference_count = 0
        self.total_frames_processed = 0
        self.per_class_detections = {"knife": 0, "long_gun": 0, "pistol": 0}
        self.current_state = WeaponState.NORMAL
        self.latest_event = None
