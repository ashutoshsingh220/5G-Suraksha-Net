"""Fight recognition orchestrator: candidates -> temporal windows -> verified incidents."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from suraksha.config import FightConfig
from suraksha.detection.tracker import TrackedPerson
from suraksha.fight.candidate import FightCandidate, FightCandidateDetector
from suraksha.fight.temporal_classifier import TemporalScorer, build_scorer
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class VerifiedFight:
    """A confirmed fight, ready for incident packaging."""
    track_ids: tuple[int, int]
    roi_xyxy: np.ndarray
    start_time: float
    end_time: float
    confidence: float
    windows_scored: int
    feature_window: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]


@dataclass
class CandidateDiagnostic:
    """Diagnostic telemetry snapshot for an active candidate pair."""
    pair_id: tuple[int, int]
    track_a: int
    track_b: int
    distance: float
    normalized_distance: float
    iou: float
    motion_energy: float
    motion_variance: float
    candidate_duration: float
    candidate_state: str  # "candidate" | "evaluating" | "verified" | "expired"
    temporal_window_count: int
    current_gru_score: float | None
    verification_threshold: float
    consecutive_positive_windows: int
    required_positive_windows: int
    action_duration: float
    minimum_action_duration: float
    verified_state: bool
    incident_created: bool
    last_rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "pair_id": list(self.pair_id),
            "track_a": self.track_a,
            "track_b": self.track_b,
            "distance": round(self.distance, 1),
            "normalized_distance": round(self.normalized_distance, 3),
            "iou": round(self.iou, 3),
            "motion_energy": round(self.motion_energy, 3),
            "motion_variance": round(self.motion_variance, 4),
            "candidate_duration": round(self.candidate_duration, 2),
            "candidate_state": self.candidate_state,
            "temporal_window_count": self.temporal_window_count,
            "current_gru_score": round(self.current_gru_score, 3) if self.current_gru_score is not None else None,
            "verification_threshold": round(self.verification_threshold, 3),
            "consecutive_positive_windows": self.consecutive_positive_windows,
            "required_positive_windows": self.required_positive_windows,
            "action_duration": round(self.action_duration, 2),
            "minimum_action_duration": round(self.minimum_action_duration, 2),
            "verified_state": self.verified_state,
            "incident_created": self.incident_created,
            "last_rejection_reason": self.last_rejection_reason,
        }


@dataclass
class _ActiveFight:
    candidate: FightCandidate
    window_scores: deque = field(default_factory=lambda: deque(maxlen=32))
    consecutive_hits: int = 0
    frames_since_window: int = 0
    last_update: float = 0.0
    verified: bool = False
    last_rejection_reason: str | None = None


class FightRecognizer:
    """Accumulates candidates over time and applies the temporal scorer.

    Fighting is temporal behavior: a single frame never triggers an incident.
    Verification requires:
      1. candidate engagement persisting >= verify.min_duration_s
      2. at least verify.min_consecutive_windows temporal windows scoring
         above fight.temporal.score_threshold
    """

    def __init__(self, cfg: FightConfig, frame_shape: tuple[int, int], device: str = "auto"):
        self.cfg = cfg
        self.detector = FightCandidateDetector(cfg.candidate, frame_shape)
        self.scorer: TemporalScorer = build_scorer(cfg.temporal.model_weights, device=device)
        self._active: dict[tuple[int, int], _ActiveFight] = {}
        self.latest_candidates: list[FightCandidate] = []
        self.diagnostics: dict[tuple[int, int], CandidateDiagnostic] = {}

    def get_telemetry(self) -> list[dict]:
        """Return structured diagnostic telemetry for all currently tracked candidate pairs."""
        return [diag.to_dict() for diag in self.diagnostics.values()]

    def update(
        self, frame: np.ndarray, tracks: list[TrackedPerson], now: float | None = None,
        fps: float | None = None,
    ) -> list[VerifiedFight]:
        now = now if now is not None else time.time()
        candidates = self.detector.update(frame, tracks, now, fps=fps)
        self.latest_candidates = list(candidates)
        seen_keys: set[tuple[int, int]] = set()
        verified: list[VerifiedFight] = []
        by_id = {t.track_id: t for t in tracks}

        for cand in candidates:
            key = cand.track_ids
            seen_keys.add(key)
            af = self._active.get(key)
            if af is None:
                af = _ActiveFight(candidate=cand, last_update=now)
                self._active[key] = af
            af.candidate = cand
            af.last_update = now
            af.frames_since_window += 1

            # Score a temporal window every stride frames
            if af.frames_since_window >= self.cfg.temporal.stride_frames:
                af.frames_since_window = 0
                window = self._make_window(cand)
                if window is not None:
                    s = self.scorer.score(window)
                    af.window_scores.append(s)

                    # Window-level motion variance check (action must oscillate / show struggle dynamics)
                    min_m_var = getattr(self.cfg.verify, "min_motion_variance", 0.0)
                    window_motion_var = float(np.var(window[:, 2]))
                    var_ok = (
                        min_m_var <= 0.0
                        or self.cfg.candidate.motion_energy_threshold <= 0.0
                        or window_motion_var >= min_m_var
                    )

                    duration = now - cand.first_seen

                    if s >= self.cfg.temporal.score_threshold and var_ok:
                        af.consecutive_hits += 1
                        if af.consecutive_hits == 1:
                            log.info(
                                "FIGHT_VERIFICATION_STARTED tracks=%s score=%.2f motion_var=%.3f (thr=%.2f)",
                                key, s, window_motion_var, self.cfg.temporal.score_threshold,
                            )

                        if (
                            af.consecutive_hits >= self.cfg.verify.min_consecutive_windows
                            and duration >= self.cfg.verify.min_duration_s
                            and not af.verified
                        ):
                            af.verified = True
                            af.last_rejection_reason = None
                            conf = float(np.mean(list(af.window_scores)[-af.consecutive_hits:]))
                            log.warning(
                                "FIGHT_VERIFIED tracks=%s conf=%.2f duration=%.1fs consecutive_windows=%d",
                                key, conf, duration, af.consecutive_hits,
                            )
                            verified.append(VerifiedFight(
                                track_ids=key,
                                roi_xyxy=cand.roi_xyxy.copy(),
                                start_time=cand.first_seen,
                                end_time=now,
                                confidence=conf,
                                windows_scored=len(af.window_scores),
                                feature_window=window,
                            ))
                        elif not af.verified:
                            if af.consecutive_hits < self.cfg.verify.min_consecutive_windows:
                                af.last_rejection_reason = "REJECT: insufficient_consecutive_windows"
                                log.debug(
                                    "REJECT: insufficient_consecutive_windows pair=%s hits=%d req=%d (score=%.2f)",
                                    key, af.consecutive_hits, self.cfg.verify.min_consecutive_windows, s,
                                )
                            elif duration < self.cfg.verify.min_duration_s:
                                af.last_rejection_reason = "REJECT: insufficient_action_duration"
                                log.debug(
                                    "REJECT: insufficient_action_duration pair=%s duration=%.2fs req=%.2fs (score=%.2f)",
                                    key, duration, self.cfg.verify.min_duration_s, s,
                                )
                    else:
                        if s < self.cfg.temporal.score_threshold:
                            af.last_rejection_reason = "REJECT: score_below_threshold"
                            log.debug(
                                "REJECT: score_below_threshold pair=%s score=%.2f thr=%.2f (windows=%d)",
                                key, s, self.cfg.temporal.score_threshold, len(af.window_scores),
                            )
                        elif not var_ok:
                            af.last_rejection_reason = "REJECT: motion_var_below_threshold"
                            log.debug(
                                "REJECT: motion_var_below_threshold pair=%s var=%.4f thr=%.4f",
                                key, window_motion_var, min_m_var,
                            )

                        if af.consecutive_hits > 0:
                            log.debug(
                                "FIGHT_VERIFICATION_RESET tracks=%s score=%.2f (thr=%.2f) motion_var=%.3f (thr=%.2f)",
                                key, s, self.cfg.temporal.score_threshold, window_motion_var, min_m_var,
                            )
                        af.consecutive_hits = 0

            # Record / update telemetry for active pair
            t1, t2 = key
            dist = 0.0
            if t1 in by_id and t2 in by_id:
                dist = float(np.linalg.norm(by_id[t1].center - by_id[t2].center))
            last_feat = cand.features[-1] if cand.features else None
            norm_dist = float(last_feat[0]) if last_feat is not None else 0.0
            iou = float(last_feat[1]) if last_feat is not None else 0.0
            motion_e = float(last_feat[2]) if last_feat is not None else 0.0
            motion_v = float(np.var([f[2] for f in cand.features])) if len(cand.features) > 1 else 0.0
            cand_dur = now - cand.first_seen

            state_str = "verified" if af.verified else ("evaluating" if af.window_scores else "candidate")
            self.diagnostics[key] = CandidateDiagnostic(
                pair_id=key,
                track_a=t1,
                track_b=t2,
                distance=dist,
                normalized_distance=norm_dist,
                iou=iou,
                motion_energy=motion_e,
                motion_variance=motion_v,
                candidate_duration=cand_dur,
                candidate_state=state_str,
                temporal_window_count=len(af.window_scores),
                current_gru_score=float(af.window_scores[-1]) if af.window_scores else None,
                verification_threshold=float(self.cfg.temporal.score_threshold),
                consecutive_positive_windows=af.consecutive_hits,
                required_positive_windows=int(self.cfg.verify.min_consecutive_windows),
                action_duration=cand_dur,
                minimum_action_duration=float(self.cfg.verify.min_duration_s),
                verified_state=af.verified,
                incident_created=af.verified,
                last_rejection_reason=af.last_rejection_reason,
            )

        # expire stale actives (pair no longer a candidate)
        stale_timeout = 3.0
        for key in list(self._active):
            if key not in seen_keys and now - self._active[key].last_update > stale_timeout:
                log.debug("REJECT: candidate_expired pair=%s elapsed_s=%.2f", key, now - self._active[key].last_update)
                del self._active[key]
                if key in self.diagnostics:
                    self.diagnostics[key].candidate_state = "expired"
                    self.diagnostics[key].last_rejection_reason = "REJECT: candidate_expired"

        return verified

    def _make_window(self, cand: FightCandidate) -> np.ndarray | None:
        """Stack the last window_frames pair-feature vectors -> (T, D)."""
        feats = cand.features
        if len(feats) < max(4, self.cfg.temporal.stride_frames):
            return None
        window = feats[-self.cfg.temporal.window_frames:]
        # pad short windows by repeating the first frame
        while len(window) < self.cfg.temporal.window_frames:
            window.insert(0, window[0])
        return np.stack(window).astype(np.float32)

