"""Loss and validation metrics for the temporal fight classifier.

The GRU emits a RAW LOGIT (`training.model.build_gru` ends in `Linear(32,1)`
with no activation), so the loss is `BCEWithLogitsLoss` — never `BCELoss` on a
pre-sigmoid value, which would double-apply the sigmoid and destabilize
gradients.

scikit-learn is not installed and PROMPT 4 Step 16 forbids changing the
environment, so ROC-AUC and PR-AUC are computed here with numpy:
  * ROC-AUC via the rank form of the Mann-Whitney U statistic (tie-corrected
    with average ranks), which is exact and O(n log n).
  * PR-AUC as sklearn's `average_precision`: a step sum over descending scores,
    not a trapezoid under the interpolated curve.

No threshold search is performed. The 0.60 value in `configs/app.yaml` is the
*incident verification* threshold and PROMPT 4 forbids tuning it here; binary
metrics are therefore reported at the neutral 0.5 decision boundary only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Tie-corrected Mann-Whitney AUC. None if either class is absent."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0    # average rank, 1-based
        i = j + 1
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Average precision (sklearn-compatible). None if no positives."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # collapse ties: only the last occurrence of each distinct score counts
    distinct = np.ones(len(s_sorted), dtype=bool)
    distinct[:-1] = s_sorted[1:] != s_sorted[:-1]
    return float((precision[distinct] * np.diff(np.concatenate([[0.0], recall]))[distinct]).sum())


@dataclass
class Metrics:
    n: int = 0
    loss: float = float("nan")
    accuracy: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    roc_auc: float | None = None
    pr_auc: float | None = None
    threshold: float = 0.5
    confusion: tuple[int, int, int, int] = (0, 0, 0, 0)   # tn, fp, fn, tp
    extra: dict = field(default_factory=dict)

    @property
    def tn(self) -> int: return self.confusion[0]
    @property
    def fp(self) -> int: return self.confusion[1]
    @property
    def fn(self) -> int: return self.confusion[2]
    @property
    def tp(self) -> int: return self.confusion[3]

    @property
    def specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else float("nan")

    @property
    def balanced_accuracy(self) -> float:
        """Mean of per-class recall: (TPR + TNR) / 2. NaN if either class absent."""
        r, s = self.recall, self.specificity
        if r != r or s != s:               # NaN check without math.isnan import
            return float("nan")
        return (r + s) / 2.0

    def as_dict(self) -> dict:
        return {
            "n": self.n, "loss": self.loss, "accuracy": self.accuracy,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "roc_auc": self.roc_auc, "pr_auc": self.pr_auc,
            "threshold": self.threshold, "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "confusion": {"tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp},
            **self.extra,
        }

    def table(self, title: str = "metrics") -> str:
        d = self.as_dict()
        fmt = lambda v: "  n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        rows = [
            ("n", str(self.n)),
            ("loss", fmt(d["loss"])),
            ("accuracy", fmt(d["accuracy"])),
            ("precision", fmt(d["precision"])),
            ("recall", fmt(d["recall"])),
            ("f1", fmt(d["f1"])),
            ("specificity", fmt(d["specificity"])),
            ("roc_auc", fmt(d["roc_auc"])),
            ("pr_auc", fmt(d["pr_auc"])),
            ("threshold", fmt(d["threshold"])),
            ("confusion tn/fp/fn/tp", f"{self.tn}/{self.fp}/{self.fn}/{self.tp}"),
        ]
        width = max(len(k) for k, _ in rows) + 2
        head = f"{title} (n={self.n})"
        return "\n".join([head, "-" * len(head)] +
                         [f"  {k:<{width}}{v}" for k, v in rows])


def confusion_matrix(pred: np.ndarray, labels: np.ndarray) -> tuple[int, int, int, int]:
    p = np.asarray(pred, dtype=np.int64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    return tn, fp, fn, tp


def compute_metrics(logits: np.ndarray, labels: np.ndarray, loss: float = float("nan"),
                    threshold: float = 0.5) -> Metrics:
    """Binary metrics from raw logits. Tolerates a single-class input."""
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if logits.shape != labels.shape:
        raise ValueError(f"logits {logits.shape} vs labels {labels.shape}")
    m = Metrics(n=len(labels), loss=float(loss), threshold=float(threshold),
                confusion=confusion_matrix(logits > threshold, labels))
    if len(labels) == 0:
        return m
    pred = (logits > threshold).astype(np.int64)
    tp, fp, fn, tn = m.tp, m.fp, m.fn, m.tn
    m.accuracy = (tp + tn) / len(labels)
    m.precision = tp / (tp + fp) if (tp + fp) else 0.0
    m.recall = tp / (tp + fn) if (tp + fn) else 0.0
    m.f1 = (2 * m.precision * m.recall / (m.precision + m.recall)
            if (m.precision + m.recall) else 0.0)
    probs = sigmoid(logits)
    m.roc_auc = roc_auc(probs, labels)
    m.pr_auc = pr_auc(probs, labels)
    return m


def binary_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Reference numpy implementation used to cross-check torch's loss."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    # log(1+exp(-|z|)) + max(z,0) - z*y  is the numerically stable form
    return float(np.mean(np.logaddexp(0.0, -np.abs(z)) + np.maximum(z, 0.0) - z * y))


def compute_metrics_from_probs(probs: np.ndarray, labels: np.ndarray,
                               threshold: float = 0.5,
                               loss: float = float("nan")) -> Metrics:
    """Binary metrics from probabilities in [0,1] at an explicit threshold.

    `compute_metrics` thresholds raw LOGITS, so its `threshold` is in logit space
    and cannot express the production `score_threshold = 0.60`, which is defined
    on the sigmoid score. This variant takes probabilities directly so the same
    0.60 boundary the live recognizer uses can be evaluated honestly. AUC/PR-AUC
    are threshold-independent and identical to the logit-space computation.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} vs labels {y.shape}")
    m = Metrics(n=len(y), loss=float(loss), threshold=float(threshold),
                confusion=confusion_matrix(p > threshold, y))
    if len(y) == 0:
        return m
    tp, fp, fn, tn = m.tp, m.fp, m.fn, m.tn
    m.accuracy = (tp + tn) / len(y)
    m.precision = tp / (tp + fp) if (tp + fp) else 0.0
    m.recall = tp / (tp + fn) if (tp + fn) else 0.0
    m.f1 = (2 * m.precision * m.recall / (m.precision + m.recall)
            if (m.precision + m.recall) else 0.0)
    m.roc_auc = roc_auc(p, y)
    m.pr_auc = pr_auc(p, y)
    return m


VIDEO_AGG_METHODS = ("mean", "max")


def video_level_aggregate(probs: np.ndarray, clip_ids: Sequence[str],
                          labels: np.ndarray, method: str = "mean"):
    """Collapse per-window probabilities into one score per source clip.

    A clip's windows all share its label, so the clip label is taken from the
    first window seen. `mean` is the documented primary aggregation (a clip is a
    fight if its average window score is high); `max` is reported alongside it as
    a sensitive alternative. Insertion order of clips is preserved.
    """
    if method not in VIDEO_AGG_METHODS:
        raise ValueError(f"unknown video aggregation {method!r}, expected {VIDEO_AGG_METHODS}")
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if len(p) != len(y) or len(p) != len(clip_ids):
        raise ValueError("probs, labels and clip_ids must all have the same length")
    groups: dict[str, list[float]] = {}
    first_label: dict[str, int] = {}
    for cid, pi, yi in zip(clip_ids, p, y):
        groups.setdefault(cid, []).append(float(pi))
        first_label.setdefault(cid, int(yi))
    ids = list(groups.keys())
    if method == "mean":
        scores = np.array([float(np.mean(groups[c])) for c in ids], dtype=np.float64)
    else:
        scores = np.array([float(np.max(groups[c])) for c in ids], dtype=np.float64)
    clip_labels = np.array([first_label[c] for c in ids], dtype=np.int64)
    return ids, scores, clip_labels


def video_level_metrics(probs: np.ndarray, clip_ids: Sequence[str], labels: np.ndarray,
                        threshold: float = 0.5, method: str = "mean") -> Metrics:
    """VIDEO-LEVEL metrics: one prediction per source clip, not per window.

    Distinct from window-level metrics — a single high-scoring window does not
    make a video a fight here; the aggregated clip score must clear the
    threshold. `extra` records the aggregation method and per-clip scores.
    """
    ids, scores, clip_labels = video_level_aggregate(probs, clip_ids, labels, method)
    m = compute_metrics_from_probs(scores, clip_labels, threshold=threshold)
    m.extra = {
        "level": "video",
        "aggregation": method,
        "n_videos": len(ids),
        "video_ids": ids,
        "video_scores": [round(float(s), 6) for s in scores],
        "video_labels": clip_labels.tolist(),
    }
    return m
