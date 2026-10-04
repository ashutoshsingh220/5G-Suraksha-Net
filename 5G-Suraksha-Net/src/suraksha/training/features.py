"""Offline extraction of the GRU's 8-dim per-frame pair features from video files.

WHY THIS MODULE EXISTS — the genuine incompatibility found in Step 1
-------------------------------------------------------------------
Two different "temporal representations" exist in this repository and they are
NOT interchangeable:

  * `suraksha.data.sequences.SequenceSampler` -> `(T, 128, 128, 3)` normalised
    PIXELS, lazily decoded from a file. Built for a future clip-CNN.
  * `suraksha.fight.temporal_classifier.TorchTemporalClassifier` -> `(B, T, 8)`
    PER-FRAME PAIR KINEMATICS, produced live by
    `FightCandidateDetector._pair_features()` from YOLO+ByteTrack person pairs.

The shipped GRU has `input_size=8`. It cannot consume pixels: 128*128*3 = 49152
values per frame. So the training pipeline must generate the *same* 8-dim
features the inference pipeline generates, from the same extractor. Feeding it
anything else would produce a checkpoint that loads but means nothing.

Features are therefore extracted by running the EXISTING `MultiObjectTracker`
and the EXISTING `FightCandidateDetector._pair_features` over each video. One
extractor, used by both training and inference, is what makes Step 14
(checkpoint compatibility) a property of the design rather than a hope.

Storage: 8 float32 per frame -> a 32-frame window is 1 KiB. 30 000 windows
~= 30 MiB, so windows are cached as a single compressed .npz per video and NO
frame images are ever written to disk.

Note on the flow gate
---------------------
At inference the recognizer only scores windows from *emitted* candidates, which
requires `flow >= candidate.motion_energy_threshold`. That threshold is itself a
fight heuristic, so selecting training windows with it would bias the training
set toward obvious fights and starve NON_FIGHT. This extractor therefore records
every window from an ENGAGED pair and stores `gate_passed` per window, so both
distributions can be measured and the choice can be made from data.

Flow is computed by the detector itself (`FightCandidateDetector._flow_energy`),
which caches one full-frame Farneback pass per frame and masks it to the pair
ROI — the same code path inference uses, so a training window and an inference
window over the same clip are the same tensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from suraksha.fight.candidate import FightCandidateDetector, _iou, _union
from suraksha.logging_utils import get_logger

log = get_logger(__name__)

FEATURE_DIM = FightCandidateDetector.FEATURE_DIM   # 8 — never redefined here


@dataclass
class VideoFeatures:
    """All temporal windows extracted from one video."""
    clip_id: str
    path: str
    label: str
    split: str
    fps: float
    frame_count: int
    windows: np.ndarray = field(default_factory=lambda: np.empty((0, 0, FEATURE_DIM), np.float32))
    gate_passed: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    # Per-window mean of feature[2] = normalized motion in body-widths/sec
    # (NOT raw optical-flow px/frame). Kept under the historical `mean_flow`
    # name for cache compatibility; the unit changed in the PROMPT 5 fix.
    mean_flow: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    tracks_per_frame: np.ndarray = field(default_factory=lambda: np.empty(0, np.int16))
    engaged_pairs_per_frame: np.ndarray = field(default_factory=lambda: np.empty(0, np.int16))
    frames_decoded: int = 0
    note: str = ""

    @property
    def n_windows(self) -> int:
        return int(self.windows.shape[0])


class PairFeatureExtractor:
    """Video file -> (N, window_frames, FEATURE_DIM) float32 windows.

    Deterministic: frames are decoded in order, pairs are enumerated in
    sorted-track-id order, and windows are emitted every `stride` engaged frames
    with the same front-padding rule as `FightRecognizer._make_window`.
    """

    def __init__(self, tracker, detector: FightCandidateDetector,
                 window_frames: int = 32, stride_frames: int = 8,
                 motion_energy_threshold: float = 0.05):
        self.tracker = tracker
        self.detector = detector
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.motion_energy_threshold = motion_energy_threshold

    def extract(self, clip_id: str, path: str | Path, label: str, split: str,
                fps: float, frame_count: int, max_frames: int = 0) -> VideoFeatures:
        import cv2

        path = Path(path)
        out = VideoFeatures(clip_id=clip_id, path=str(path), label=label, split=split,
                            fps=fps, frame_count=frame_count)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            out.note = "UNOPENABLE"
            log.warning("cannot open %s", path)
            return out

        det = self.detector
        proximity_iou = det.cfg.proximity_iou
        frame_shape = det._frame_shape
        # Feature[2] is normalized motion in body-widths/sec, which needs the
        # clip's frame rate to convert per-frame flow displacement into a
        # per-second speed. Set it on the detector so _pair_features and the
        # gate below both use the real fps, not the detector's default.
        det.fps = float(fps) if fps and fps > 0 else det.DEFAULT_FPS

        # Offline extraction reuses one tracker+detector across many clips, so
        # both must be wiped between clips. Without this, clip B inherits clip
        # A's ByteTrack IDs / Kalman state and A's last gray frame leaks into B's
        # first flow computation — making the features depend on processing
        # order, which would quietly break reproducibility.
        det.reset()
        tracker_reset = getattr(self.tracker, "reset", None)
        if callable(tracker_reset):
            tracker_reset()

        # per-pair rolling state: feature list + engaged-frame counter
        feats: dict[tuple[int, int], list[np.ndarray]] = {}
        engaged: dict[tuple[int, int], int] = {}
        since_window: dict[tuple[int, int], int] = {}

        wins: list[np.ndarray] = []
        gates: list[bool] = []
        flows: list[float] = []
        tracks_per_frame: list[int] = []
        pairs_per_frame: list[int] = []

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if max_frames and idx >= max_frames:
                    break
                h, w = frame.shape[:2]
                if (w, h) != (frame_shape[1], frame_shape[0]):
                    frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]),
                                       interpolation=cv2.INTER_AREA)

                # Same per-frame flow cache the inference path uses: bump the
                # frame counter once, then every pair in this frame reads the
                # identical ROI-masked magnitude.
                det._frame_seq += 1

                tracks = self.tracker.update(frame)
                tracks_per_frame.append(len(tracks))
                active: set[tuple[int, int]] = set()
                if len(tracks) >= 2:
                    proximity_dist = getattr(det.cfg, "proximity_distance", 0.0)
                    for ta, tb in _sorted_pairs(tracks):
                        key = (min(ta.track_id, tb.track_id), max(ta.track_id, tb.track_id))
                        iou = _iou(ta.bbox_xyxy, tb.bbox_xyxy)
                        ca, cb = ta.center, tb.center
                        dist = float(np.linalg.norm(ca - cb))
                        scale = max((ta.width + tb.width) / 2.0, 1e-3)
                        rel_dist = dist / scale

                        is_engaged = (iou >= proximity_iou) or (
                            proximity_dist > 0.0 and rel_dist <= proximity_dist
                        )
                        if not is_engaged:
                            feats.pop(key, None); engaged.pop(key, None)
                            since_window.pop(key, None)
                            continue
                        roi = _union(ta.bbox_xyxy, tb.bbox_xyxy)
                        flow = det._flow_energy(frame, roi)
                        feat = det._pair_features(ta, tb, iou, flow)
                        feats.setdefault(key, []).append(feat)
                        if len(feats[key]) > 256:
                            feats[key] = feats[key][-256:]
                        engaged[key] = engaged.get(key, 0) + 1
                        since_window[key] = since_window.get(key, 0) + 1
                        active.add(key)

                        if since_window[key] >= self.stride_frames:
                            since_window[key] = 0
                            win = _pad_window(feats[key], self.window_frames)
                            if win is not None:
                                wins.append(win)
                                # feat[2] is normalized motion (body-widths/sec),
                                # the same quantity the live gate compares against
                                # motion_energy_threshold.
                                gates.append(float(feat[2]) >= self.motion_energy_threshold)
                                flows.append(float(np.mean(win[:, 2])))
                pairs_per_frame.append(len(active))
                idx += 1
        finally:
            cap.release()

        out.frames_decoded = idx
        if idx == 0:
            out.note = "NO_FRAMES_DECODED"
            return out
        out.tracks_per_frame = np.asarray(tracks_per_frame, dtype=np.int16)
        out.engaged_pairs_per_frame = np.asarray(pairs_per_frame, dtype=np.int16)
        if not wins:
            out.note = "NO_ENGAGED_PAIR"
            out.windows = np.empty((0, self.window_frames, FEATURE_DIM), np.float32)
            return out
        out.windows = np.stack(wins).astype(np.float32)
        out.gate_passed = np.asarray(gates, dtype=bool)
        out.mean_flow = np.asarray(flows, dtype=np.float32)
        return out


def _sorted_pairs(tracks):
    """Deterministic pair enumeration: sorted by (id_a, id_b)."""
    import itertools

    ordered = sorted(tracks, key=lambda t: t.track_id)
    return list(itertools.combinations(ordered, 2))


def _pad_window(feats: list[np.ndarray], window_frames: int) -> np.ndarray | None:
    """Last `window_frames` features, front-padded by repeating the first frame.

    Identical rule to `FightRecognizer._make_window`, so a training window and an
    inference window of the same clip are the same tensor.
    """
    if len(feats) < 4:
        return None
    win = list(feats[-window_frames:])
    while len(win) < window_frames:
        win.insert(0, win[0])
    return np.stack(win).astype(np.float32)


def save_features(vf: VideoFeatures, dest: str | Path) -> Path:
    """Compact per-video cache — windows only, never frame images."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        windows=vf.windows,
        gate_passed=vf.gate_passed,
        mean_flow=vf.mean_flow,
        tracks_per_frame=vf.tracks_per_frame,
        engaged_pairs_per_frame=vf.engaged_pairs_per_frame,
        clip_id=vf.clip_id, label=vf.label, split=vf.split,
        fps=np.float32(vf.fps), frame_count=np.int32(vf.frame_count),
        frames_decoded=np.int32(vf.frames_decoded), note=vf.note,
    )
    return dest


def load_features(src: str | Path) -> VideoFeatures:
    with np.load(src, allow_pickle=False) as z:
        return VideoFeatures(
            clip_id=str(z["clip_id"]), path="", label=str(z["label"]),
            split=str(z["split"]), fps=float(z["fps"]),
            frame_count=int(z["frame_count"]),
            windows=z["windows"], gate_passed=z["gate_passed"],
            mean_flow=z["mean_flow"],
            tracks_per_frame=z["tracks_per_frame"],
            engaged_pairs_per_frame=z["engaged_pairs_per_frame"],
            frames_decoded=int(z["frames_decoded"]), note=str(z["note"]),
        )
