"""Temporal fight recognition — PyTorch.

WHY PyTorch (not TensorFlow/Keras):
  - Ultralytics YOLO11 and ByteTrack already run on PyTorch; one framework
    means one CUDA context, consistent device management and simpler deps.
  - The classifier consumes kinematic features extracted from YOLO+ByteTrack
    outputs, so sharing tensors with the detection stack is natural.
  - Deployment stays a single process on GPU or CPU with no dual-framework
    memory overhead.

The model is feature-temporal (per-frame kinematic vectors -> GRU -> sigmoid),
NOT a single-frame object detector: fighting is decided over a window of
frames. A raw-clip CNN variant can be added later behind the same interface.

Until a trained checkpoint exists, HeuristicTemporalScorer provides a
conservative fallback so the pipeline is runnable end-to-end.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from suraksha.logging_utils import get_logger

log = get_logger(__name__)

FEATURE_DIM = 8  # must match FightCandidateDetector.FEATURE_DIM


class TemporalScorer(ABC):
    """Scores a temporal window of pair features. 1.0 = certainly a fight."""

    @abstractmethod
    def score(self, feature_window: np.ndarray) -> float:
        """feature_window: (T, FEATURE_DIM) float32."""


class HeuristicTemporalScorer(TemporalScorer):
    """Training-free fallback using window statistics.

    Fight-like windows show: sustained close proximity, high and *oscillating*
    motion energy (strikes), moderate relative distance (not hugging = hug,
    not far = bystanders).

    Feature[2] is normalized motion in BODY-WIDTHS/SEC (see
    FightCandidateDetector._normalize_motion), so the divisors below are
    calibrated on that unit, measured on the RWF-2000 train split:
      * FIGHT window-mean motion p50~0.36, p90~0.70  -> energy divisor 0.7
      * FIGHT window-std  motion p50~0.38, p90~1.02  -> oscillation divisor 0.5
    These map a strongly fight-like window to ~1.0 while a typical NON_FIGHT
    window (mean p50~0.29, std p50~0.10) stays low. They are only a fallback:
    once a GRU checkpoint exists it replaces this scorer entirely.
    """

    def score(self, feature_window: np.ndarray) -> float:
        if feature_window.shape[0] < 4:
            return 0.0
        rel_dist = feature_window[:, 0]
        iou = feature_window[:, 1]
        motion = feature_window[:, 2]   # body-widths/sec

        proximity = float(np.mean(rel_dist < 1.5))
        contact = float(np.mean(iou > 0.02))
        energy = float(np.clip(np.mean(motion) / 0.7, 0.0, 1.0))
        # oscillation: motion variance — strikes alternate push/pull
        oscillation = float(np.clip(np.std(motion) / 0.5, 0.0, 1.0))

        s = 0.30 * proximity + 0.20 * contact + 0.30 * energy + 0.20 * oscillation
        return float(np.clip(s, 0.0, 1.0))


class TorchTemporalClassifier(TemporalScorer):
    """GRU over per-frame kinematic feature vectors.

    Input:  (B, T, FEATURE_DIM)
    Output: (B,) fight probability.

    The network comes from `suraksha.training.model.build_gru` so a checkpoint
    produced by the training pipeline is guaranteed to match this architecture.
    If the checkpoint carries `normalization` statistics (fit on training data
    only), they are applied here — inference must preprocess exactly as training
    did, or the logits are meaningless.
    """

    def __init__(self, weights_path: str | Path, device: str = "auto", feature_dim: int = FEATURE_DIM):
        import numpy as np
        import torch

        from suraksha.training.model import HIDDEN_SIZE, build_gru

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        blob = state.get("model_state", state) if isinstance(state, dict) else state

        ckpt_dim = state.get("feature_dim") if isinstance(state, dict) else None
        if ckpt_dim is not None and ckpt_dim != feature_dim:
            log.warning("checkpoint feature_dim=%s overrides requested %s", ckpt_dim, feature_dim)
            feature_dim = int(ckpt_dim)
        hidden = int(state.get("hidden_size", HIDDEN_SIZE)) if isinstance(state, dict) else HIDDEN_SIZE

        self.feature_dim = feature_dim
        self.model = build_gru(feature_dim, hidden).to(self.device)
        self.model.load_state_dict(blob)
        self.model.eval()

        norm = state.get("normalization") if isinstance(state, dict) else None
        self.norm_method = (norm or {}).get("method", "none")
        if self.norm_method == "standard":
            self.norm_mean = np.asarray(norm["mean"], dtype=np.float32)
            self.norm_std = np.asarray(norm["std"], dtype=np.float32)
        else:
            self.norm_mean = self.norm_std = None
        log.info("Loaded temporal fight classifier: %s on %s (feature_dim=%d, normalization=%s)",
                 weights_path, self.device, feature_dim, self.norm_method)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.norm_mean is None:
            return x
        return (x - self.norm_mean) / self.norm_std

    def score(self, feature_window: np.ndarray) -> float:
        torch = self._torch
        with torch.no_grad():
            x = np.ascontiguousarray(feature_window, dtype=np.float32)
            x = self._normalize(x)
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)
            return float(torch.sigmoid(self.model(x)).item())


def build_scorer(weights_path: str, device: str = "auto") -> TemporalScorer:
    """Factory: trained model if weights exist, else heuristic fallback."""
    if weights_path and Path(weights_path).exists():
        try:
            return TorchTemporalClassifier(weights_path, device=device)
        except Exception as e:
            log.error("Failed to load fight classifier %s: %s — using heuristic", weights_path, e)
    else:
        log.info("No trained fight classifier — using HeuristicTemporalScorer")
    return HeuristicTemporalScorer()
