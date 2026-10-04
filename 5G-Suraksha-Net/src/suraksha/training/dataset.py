"""Split-isolated window corpus + TRAIN-ONLY normalization.

Two invariants drive this module, both required by PROMPT 4:

1. **A video belongs entirely to one split.** Windows are never shuffled or
   re-split independently of their source clip, so no temporal window straddles
   the train/val boundary. The split is inherited from the manifest entry, which
   in turn preserves RWF-2000's official folder split verbatim.

2. **Normalization statistics are fitted on training windows only.** Val
   statistics are never touched. The fitted parameters travel inside the
   checkpoint (`normalization` key) and are re-applied by
   `fight.temporal_classifier.TorchTemporalClassifier`, so train and inference
   cannot drift apart.

Storage: no frames or images are ever written. Each clip's windows live in one
compressed `.npz` under `datasets/processed/features/<manifest>/<clip_id>.npz`
(8 float32 per frame per window — ~1 KiB per 32-frame window). The whole corpus
is loaded into RAM: 18 750 windows x 32 x 8 x 4 B is ~19 MiB, so lazy per-item
file reads would cost more than they save.

Augmentation: **there is none.** `data/sequences.py` implements horizontal flip
and brightness jitter, but those are *pixel-space* transforms for the unused
(T,128,128,3) clip path. The GRU consumes 8 kinematic features per frame;
flipping a bbox pair changes nothing measurable and brightness does not exist in
this representation. Rather than invent a feature-space augmentation that the
inference path cannot reproduce, the pipeline trains unaugmented. This is
stated explicitly in the report instead of being left implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from suraksha.logging_utils import get_logger
from suraksha.training.features import FEATURE_DIM, load_features

log = get_logger(__name__)

LABEL_FIGHT = "FIGHT"
LABEL_NON_FIGHT = "NON_FIGHT"
LABEL_UNKNOWN = "UNKNOWN"
LABEL_TO_INT = {LABEL_NON_FIGHT: 0, LABEL_FIGHT: 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

TRAINABLE_LABELS = frozenset(LABEL_TO_INT)


def label_to_int(label: str) -> int:
    """Canonical label -> {0,1}. Anything else raises: no UNKNOWN may train."""
    key = str(label).strip().upper()
    if key not in LABEL_TO_INT:
        raise ValueError(
            f"label {label!r} is not trainable (only {sorted(TRAINABLE_LABELS)}); "
            "UNKNOWN/ambiguous clips must be filtered out before the corpus is built"
        )
    return LABEL_TO_INT[key]


@dataclass
class WindowRef:
    clip_id: str
    window_index: int
    label: int
    split: str
    gate_passed: bool


@dataclass
class CorpusStats:
    clips_requested: int = 0
    clips_loaded: int = 0
    clips_missing: list[tuple[str, str]] = field(default_factory=list)
    clips_zero_yield: list[str] = field(default_factory=list)
    labels_rejected: dict[str, int] = field(default_factory=dict)


@dataclass
class Corpus:
    """In-memory window matrix with the provenance of every row.

    `clip_ids` is a plain list[str], NOT a numpy string array: `np.full(n, id,
    dtype=np.str_)` silently produces dtype '<U1' and truncates every id to its
    first character, which would make per-clip provenance (and therefore the
    leakage-adjusted validation split) meaningless.
    """

    windows: np.ndarray                 # (N, T, FEATURE_DIM) float32
    labels: np.ndarray                  # (N,) int64
    gate_passed: np.ndarray             # (N,) bool
    clip_ids: list[str]                 # len N — source clip per row
    window_index: np.ndarray            # (N,) int32 position within the clip
    split: str = ""
    stats: CorpusStats = field(default_factory=CorpusStats)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.windows.shape)

    def counts(self) -> dict[str, int]:
        return {INT_TO_LABEL[k]: int(v) for k, v in
                zip(*np.unique(self.labels, return_counts=True))}

    def class_ratio(self) -> float:
        """FIGHT : NON_FIGHT at WINDOW level (not video level — they differ)."""
        c = self.counts()
        pos, neg = c.get(LABEL_FIGHT, 0), c.get(LABEL_NON_FIGHT, 0)
        return float("inf") if neg == 0 else pos / max(neg, 1)

    def mask_excluding_clips(self, clip_ids: Iterable[str]) -> np.ndarray:
        drop = set(clip_ids)
        return np.array([c not in drop for c in self.clip_ids], dtype=bool)

    def subset(self, mask: np.ndarray, split: str | None = None) -> "Corpus":
        return Corpus(
            windows=self.windows[mask],
            labels=self.labels[mask],
            gate_passed=self.gate_passed[mask],
            clip_ids=[c for c, keep in zip(self.clip_ids, mask) if keep],
            window_index=self.window_index[mask],
            split=self.split if split is None else split,
            stats=self.stats,
        )

    def clip_ids_present(self) -> list[str]:
        return sorted(set(self.clip_ids))


def cache_path(cache_dir: Path | str, manifest: str, clip_id: str) -> Path:
    return Path(cache_dir) / manifest / f"{clip_id}.npz"


def build_corpus(
    entries: Sequence,
    cache_dir: Path | str,
    manifest: str = "rwf2000_v1",
    split: str = "",
    require_split_isolation: bool = True,
) -> Corpus:
    """Load every cached window for `entries` into one (N, T, 8) matrix.

    `require_split_isolation` asserts that all supplied entries share one split,
    which is how the "a video belongs entirely to one split" rule is enforced
    mechanically rather than by convention.
    """
    splits = {e.split for e in entries}
    if require_split_isolation and len(splits) > 1:
        raise ValueError(
            f"refusing to build a corpus over mixed splits {sorted(splits)}: "
            "windows must inherit their video's split"
        )

    stats = CorpusStats(clips_requested=len(entries))
    win_chunks: list[np.ndarray] = []
    lab_chunks: list[np.ndarray] = []
    gate_chunks: list[np.ndarray] = []
    clip_chunks: list[list[str]] = []
    widx_chunks: list[np.ndarray] = []

    for e in entries:
        label = str(e.label).strip().upper()
        if label not in TRAINABLE_LABELS:
            stats.labels_rejected[label] = stats.labels_rejected.get(label, 0) + 1
            continue
        src = cache_path(cache_dir, manifest, e.clip_id)
        if not src.exists():
            stats.clips_missing.append((e.clip_id, "no cached features"))
            continue
        vf = load_features(src)
        if vf is None:
            stats.clips_missing.append((e.clip_id, "unreadable cache file"))
            continue
        stats.clips_loaded += 1
        if vf.n_windows == 0:
            stats.clips_zero_yield.append(e.clip_id)
            continue
        y = label_to_int(vf.label or label)
        n = vf.n_windows
        win_chunks.append(vf.windows)
        lab_chunks.append(np.full(n, y, dtype=np.int64))
        gate_chunks.append(vf.gate_passed.astype(bool))
        clip_chunks.append([e.clip_id] * n)
        widx_chunks.append(np.arange(n, dtype=np.int32))

    if not win_chunks:
        empty = np.zeros((0, 0, FEATURE_DIM), dtype=np.float32)
        return Corpus(
            windows=empty,
            labels=np.zeros(0, dtype=np.int64),
            gate_passed=np.zeros(0, dtype=bool),
            clip_ids=[],
            window_index=np.zeros(0, dtype=np.int32),
            split=split or (splits.pop() if len(splits) == 1 else ""),
            stats=stats,
        )

    corpus = Corpus(
        windows=np.concatenate(win_chunks, axis=0).astype(np.float32, copy=False),
        labels=np.concatenate(lab_chunks, axis=0),
        gate_passed=np.concatenate(gate_chunks, axis=0),
        clip_ids=[c for chunk in clip_chunks for c in chunk],
        window_index=np.concatenate(widx_chunks, axis=0),
        split=split or (splits.pop() if len(splits) == 1 else ""),
        stats=stats,
    )
    _assert_shape(corpus)
    log.info(
        "corpus[%s]: %d windows from %d/%d clips (zero-yield=%d, missing=%d) "
        "labels=%s ratio=%.2f:1",
        corpus.split or "?", len(corpus), corpus.stats.clips_loaded,
        corpus.stats.clips_requested, len(corpus.stats.clips_zero_yield),
        len(corpus.stats.clips_missing), corpus.counts(), corpus.class_ratio(),
    )
    return corpus


def _assert_shape(corpus: Corpus) -> None:
    n = corpus.windows.shape[0]
    if corpus.windows.ndim != 3 or corpus.windows.shape[2] != FEATURE_DIM:
        raise ValueError(
            f"expected (N, T, {FEATURE_DIM}) windows, got {corpus.windows.shape}"
        )
    for name, arr in (("labels", corpus.labels), ("gate_passed", corpus.gate_passed),
                      ("window_index", corpus.window_index)):
        if len(arr) != n:
            raise ValueError(f"{name} length {len(arr)} != window count {n}")
    if len(corpus.clip_ids) != n:
        raise ValueError(
            f"clip_ids length {len(corpus.clip_ids)} != window count {n}: "
            "per-window provenance is broken"
        )
    if len(set(corpus.clip_ids)) < 1:
        raise ValueError("clip provenance is empty")
    if not np.isfinite(corpus.windows).all():
        raise ValueError("non-finite feature values in corpus")


# --------------------------------------------------------------------------
# Normalization — fitted on TRAIN windows only
# --------------------------------------------------------------------------

NORM_NONE = "none"
NORM_STANDARD = "standard"
NORM_MINMAX = "minmax"
NORM_METHODS = (NORM_NONE, NORM_STANDARD, NORM_MINMAX)
_EPS = 1e-6


def fit_normalization(windows: np.ndarray, method: str = NORM_STANDARD) -> dict:
    """Fit per-feature statistics over the supplied (N, T, 8) TRAIN windows.

    Passing validation windows here is a bug: the caller is responsible for
    only ever handing in training data, and `train_temporal.py` does so by
    construction (it fits before the val corpus is loaded).
    """
    if method not in NORM_METHODS:
        raise ValueError(f"unknown normalization {method!r}, expected one of {NORM_METHODS}")
    if method == NORM_NONE:
        return {"method": NORM_NONE}
    if windows.ndim != 3 or windows.shape[2] != FEATURE_DIM:
        raise ValueError(f"expected (N, T, {FEATURE_DIM}), got {windows.shape}")
    if windows.shape[0] == 0:
        raise ValueError("cannot fit normalization on an empty corpus")

    flat = windows.reshape(-1, FEATURE_DIM).astype(np.float64)
    if method == NORM_STANDARD:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < _EPS, 1.0, std)   # constant feature -> leave as is
        return {
            "method": NORM_STANDARD,
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "fitted_on": "train",
            "n_windows": int(windows.shape[0]),
        }
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    span = np.where((hi - lo) < _EPS, 1.0, hi - lo)
    return {
        "method": NORM_MINMAX,
        "min": lo.astype(np.float32).tolist(),
        "scale": span.astype(np.float32).tolist(),
        "fitted_on": "train",
        "n_windows": int(windows.shape[0]),
    }


def apply_normalization(windows: np.ndarray, params: dict | None) -> np.ndarray:
    method = (params or {}).get("method", NORM_NONE)
    x = np.asarray(windows, dtype=np.float32)
    if method == NORM_NONE:
        return x
    if method == NORM_STANDARD:
        mean = np.asarray(params["mean"], dtype=np.float32)
        std = np.asarray(params["std"], dtype=np.float32)
        return (x - mean) / std
    if method == NORM_MINMAX:
        lo = np.asarray(params["min"], dtype=np.float32)
        scale = np.asarray(params["scale"], dtype=np.float32)
        return (x - lo) / scale
    raise ValueError(f"unknown normalization method {method!r}")


def class_weights(labels: np.ndarray) -> np.ndarray | None:
    """Inverse-frequency weights, or None when the split is already balanced.

    PROMPT 4 Step 9: only introduce weighting if the *actual* training
    distribution is imbalanced. Measured on RWF-2000 it is — heavily, and at the
    window level only. Videos are 800:800 train / 200:200 val, but requiring
    `proximity_iou >= 0.05` engagement is itself correlated with fighting, so
    NON_FIGHT clips yield far fewer windows. The ratio is returned to the caller
    for the report rather than being hidden inside the loss.
    """
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    if counts.min() == 0:
        raise ValueError(f"a class is absent from training labels: counts={counts}")
    ratio = counts.max() / counts.min()
    if ratio < 1.25:                      # balanced enough — leave the loss alone
        return None
    total = counts.sum()
    w = total / (2.0 * counts)
    return (w / w.sum() * 2.0).astype(np.float32)


# --------------------------------------------------------------------------
# torch Dataset
# --------------------------------------------------------------------------

class TemporalFightDataset:
    """torch.utils.data.Dataset over a normalized window matrix.

    Implemented without importing torch at module level so the corpus logic
    stays usable from plain-python tests and scripts.
    """

    def __init__(self, corpus: Corpus, normalization: dict | None = None):
        import torch

        self._torch = torch
        self.corpus = corpus
        self.normalization = normalization or {"method": NORM_NONE}
        x = apply_normalization(corpus.windows, self.normalization)
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(corpus.labels.astype(np.int64))
        self.gate = torch.from_numpy(corpus.gate_passed.astype(bool))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return tuple(self.x.shape)

    def make_loader(self, batch_size: int, shuffle: bool, num_workers: int = 0,
                    drop_last: bool = False, generator=None):
        return self._torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, generator=generator,
        )
