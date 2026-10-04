"""Fight candidate detection from tracked-person kinematics.

A candidate is a *pair of tracks* showing sustained aggressive engagement:
rapid mutual approach + close proximity + high local motion energy.
Candidates are cheap heuristics — the temporal classifier makes the decision.
"""
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from suraksha.config import CandidateConfig
from suraksha.detection.tracker import TrackedPerson
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class FightCandidate:
    track_ids: tuple[int, int]
    roi_xyxy: np.ndarray          # union bbox of the pair (pixels)
    first_seen: float
    last_seen: float
    score: float                  # 0..1 heuristic strength
    frames_engaged: int = 0
    features: list[np.ndarray] = field(default_factory=list)  # per-frame feature vectors


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
    ], dtype=np.float32)


@dataclass
class _PairState:
    engaged_frames: int = 0
    missed_frames: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    features: list = field(default_factory=list)
    roi: np.ndarray | None = None
    last_cooldown_until: float = 0.0
    motion_history: list[float] = field(default_factory=list)
    last_high_motion_time: float = 0.0


class FightCandidateDetector:
    """Emits/updates FightCandidates per frame from track kinematics + flow."""

    FEATURE_DIM = 8  # see _pair_features
    DEFAULT_FPS = 30.0

    def __init__(self, cfg: CandidateConfig, frame_shape: tuple[int, int],
                 fps: float | None = None):
        self.cfg = cfg
        self._pairs: dict[tuple[int, int], _PairState] = {}
        self._prev_gray: np.ndarray | None = None
        self._frame_shape = frame_shape  # (h, w)
        self._frame_seq = 0              # incremented once per update() call
        self._frame_mag: np.ndarray | None = None
        self._frame_mag_seq = -1
        # Source frame rate, needed to convert per-frame optical-flow
        # displacement into a per-second speed. Callers that know the real fps
        # (live pipeline: ev.source_fps; offline extractor: per-clip fps) pass it
        # to update()/set it directly; otherwise we assume DEFAULT_FPS.
        self.fps = float(fps) if fps and fps > 0 else self.DEFAULT_FPS

    def reset(self) -> None:
        """Clear all cross-frame state.

        The live pipeline never calls this mid-stream. Offline feature
        extraction MUST call it between clips: otherwise the previous clip's
        last gray frame is differenced against the next clip's first frame, and
        per-pair engagement counters carry over, making the output depend on the
        order clips were processed in.
        """
        self._pairs.clear()
        self._prev_gray = None
        self._frame_mag = None
        self._frame_mag_seq = -1
        self._frame_seq = 0

    # ---- optical motion energy inside an ROI ----
    def _frame_flow_magnitude(self, frame: np.ndarray) -> np.ndarray | None:
        """Optical-flow magnitude for the WHOLE frame, computed once per frame.

        Previously the flow was computed on the pair's ROI crop and compared
        against the previous crop of the same pair. The ROI is the union of two
        *moving* bounding boxes, so its integer shape changed on ~96% of frames
        (measured on RWF-2000); the shape guard then returned 0.0 and reset, and
        the motion-energy feature was dead almost everywhere. Computing one
        full-frame flow and masking it to the ROI removes the shape dependency
        entirely, adds no resize distortion, and costs one Farneback pass per
        frame instead of one per pair.
        """
        if self._frame_mag is not None and self._frame_mag_seq == self._frame_seq:
            return self._frame_mag
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_mag_seq = self._frame_seq
        # Rescale high-res frames for real-time flow calculation (matches CrowdAnalyzer 0.25 downscaling)
        scale = 0.25 if max(h, w) > 400 else 1.0
        if scale < 1.0:
            small_w, small_h = max(int(w * scale), 64), max(int(h * scale), 64)
            gray_proc = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            gray_proc = gray

        if self._prev_gray is None or self._prev_gray.shape != gray_proc.shape:
            self._prev_gray = gray_proc
            self._frame_mag = None
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray_proc, None, 0.5, 2, 15, 2, 5, 1.1, 0
        )
        self._prev_gray = gray_proc
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if scale < 1.0:
            mag = cv2.resize(mag * (1.0 / scale), (w, h), interpolation=cv2.INTER_LINEAR)
        self._frame_mag = mag
        return mag

    def _flow_energy(self, frame: np.ndarray, roi: np.ndarray) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0
        mag = self._frame_flow_magnitude(frame)
        if mag is None:
            return 0.0
        return float(mag[y1:y2, x1:x2].mean())

    # ---- per-pair kinematic features (used by the temporal classifier too) ----
    def _normalize_motion(self, flow: float, scale: float) -> float:
        """Convert raw optical-flow magnitude into a camera-independent speed.

        `flow` is the mean Farneback displacement magnitude inside the pair ROI,
        in PIXELS PER FRAME-INTERVAL. That raw unit depends on three things that
        have nothing to do with how violent the interaction is:

          * resolution — displacement in px grows ~linearly with frame width, so
            the same punch reads ~3.7x larger at 1280px than at 320px;
          * frame rate — per-frame displacement is inversely proportional to fps,
            so the same motion reads ~2.9x larger at 10fps than at 30fps;
          * ROI/person size — a distant person moves fewer px than a near one.

        Dividing by `scale` (the mean body width in px, the SAME spatial scale
        already used for `rel_dist`) cancels resolution and person size, turning
        px into body-widths. Multiplying by `fps` cancels the frame-interval,
        turning per-frame into per-second. The result —

            normalized_motion = flow_px_per_frame * fps / body_width_px

        — is a speed in BODY-WIDTHS PER SECOND, invariant to resolution, fps and
        ROI size (measured: <10% drift across 320..1280px and 10..30fps, versus
        3.7x / 2.9x for the raw unit). It is also directly comparable to
        `rel_dist`, which is already expressed in body-widths, so the feature
        vector stays internally consistent.
        """
        return float(flow) * self.fps / max(scale, 1e-3)

    def _pair_features(
        self, a: TrackedPerson, b: TrackedPerson, iou: float, flow: float
    ) -> np.ndarray:
        ca, cb = a.center, b.center
        dist = float(np.linalg.norm(ca - cb))
        scale = max((a.width + b.width) / 2.0, 1e-3)
        rel_dist = dist / scale
        return np.array([
            rel_dist,
            iou,
            self._normalize_motion(flow, scale),  # motion energy, body-widths/sec
            a.confidence,
            b.confidence,
            a.width / max(self._frame_shape[1], 1),
            b.width / max(self._frame_shape[1], 1),
            1.0,                  # bias
        ], dtype=np.float32)

    def update(
        self, frame: np.ndarray, tracks: list[TrackedPerson], now: float | None = None,
        fps: float | None = None,
    ) -> list[FightCandidate]:
        """Return currently-active candidates (may be empty)."""
        now = now if now is not None else time.time()
        if fps is not None and fps > 0:
            self.fps = float(fps)
        self._frame_seq += 1     # one frame in -> one cached flow field
        by_id = {t.track_id: t for t in tracks}
        active_pairs: set[tuple[int, int]] = set()
        candidates: list[FightCandidate] = []

        for ta, tb in itertools.combinations(tracks, 2):
            key = (min(ta.track_id, tb.track_id), max(ta.track_id, tb.track_id))
            iou = _iou(ta.bbox_xyxy, tb.bbox_xyxy)
            st = self._pairs.get(key)
            if st and now < st.last_cooldown_until:
                continue

            ca, cb = ta.center, tb.center
            dist = float(np.linalg.norm(ca - cb))
            scale = max((ta.width + tb.width) / 2.0, 1e-3)
            rel_dist = dist / scale

            max_dist = getattr(self.cfg, "proximity_distance", 0.0)
            is_engaged = (iou >= self.cfg.proximity_iou) or (max_dist > 0.0 and rel_dist <= max_dist)

            if is_engaged:
                roi = _union(ta.bbox_xyxy, tb.bbox_xyxy)
                flow = self._flow_energy(frame, roi)
                motion = self._normalize_motion(flow, scale)
                st = st or _PairState(first_seen=now, last_high_motion_time=now)
                st.engaged_frames += 1
                st.missed_frames = 0
                st.last_seen = now
                st.roi = roi
                st.features.append(self._pair_features(ta, tb, iou, flow))
                if len(st.features) > 256:
                    st.features = st.features[-256:]
                st.motion_history.append(motion)
                if len(st.motion_history) > 64:
                    st.motion_history = st.motion_history[-64:]

                motion_energy_thr = self.cfg.motion_energy_threshold
                if motion >= motion_energy_thr:
                    st.last_high_motion_time = now

                # Motion variance tracking (oscillation check)
                var_thr = getattr(self.cfg, "motion_variance_threshold", 0.0)
                if len(st.motion_history) >= 5 and motion_energy_thr > 0.0:
                    motion_var = float(np.var(st.motion_history[-15:]))
                else:
                    motion_var = 0.0

                # Candidate decay: if motion dropped below threshold for > decay_timeout_s
                decay_s = getattr(self.cfg, "decay_timeout_s", 1.0)
                if (
                    motion_energy_thr > 0.0
                    and (now - st.last_high_motion_time) > decay_s
                    and st.engaged_frames >= self.cfg.proximity_min_frames
                ):
                    log.debug("CANDIDATE_DECAY pair=%s motionless_s=%.2f", key, now - st.last_high_motion_time)
                    st.engaged_frames = 0
                    st.features.clear()
                    st.motion_history.clear()

                self._pairs[key] = st
                active_pairs.add(key)

                motion_ok = (motion_energy_thr <= 0.0) or (motion >= motion_energy_thr)
                var_ok = (motion_energy_thr <= 0.0) or (var_thr <= 0.0) or (motion_var >= var_thr)

                if (
                    st.engaged_frames >= self.cfg.proximity_min_frames
                    and motion_ok
                    and var_ok
                ):
                    score = 1.0 if motion_energy_thr <= 0 else min(1.0, motion / (2.0 * motion_energy_thr))
                    if st.engaged_frames == self.cfg.proximity_min_frames:
                        log.info(
                            "CANDIDATE_STARTED pair=%s rel_dist=%.2f iou=%.2f motion=%.2f var=%.4f",
                            key, rel_dist, iou, motion, motion_var,
                        )
                    candidates.append(FightCandidate(
                        track_ids=key,
                        roi_xyxy=roi,
                        first_seen=st.first_seen,
                        last_seen=now,
                        score=score,
                        frames_engaged=st.engaged_frames,
                        features=list(st.features),
                    ))
                elif st.engaged_frames >= self.cfg.proximity_min_frames:
                    log.debug(
                        "CANDIDATE_SUPPRESSED pair=%s rel_dist=%.2f iou=%.2f motion=%.2f (thr=%.2f) var=%.4f (thr=%.2f)",
                        key, rel_dist, iou, motion, motion_energy_thr, motion_var, var_thr,
                    )
            else:
                # pair momentarily not engaged: allow grace period for tracking jitter during grappling
                if st:
                    max_missed = getattr(self.cfg, "missed_frames_tolerance", 2)
                    st.missed_frames += 1
                    if st.missed_frames <= max_missed:
                        active_pairs.add(key)
                    else:
                        if st.engaged_frames >= self.cfg.proximity_min_frames:
                            st.last_cooldown_until = now + self.cfg.cooldown_s
                        del self._pairs[key]

        # prune stale pairs whose tracks vanished
        for key in list(self._pairs):
            if key not in active_pairs and not all(k in by_id for k in key):
                del self._pairs[key]

        return candidates
