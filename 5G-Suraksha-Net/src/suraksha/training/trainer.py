"""Training loop, checkpoint format, smoke/overfit/benchmark harnesses.

Nothing here starts a full training run on its own — `scripts/train_temporal.py`
is the only entry point and PROMPT 4 forbids executing it for real until the
user says so. The functions below exist so the pipeline can be *verified*
cheaply first:

    smoke_test      one forward + backward + optimizer step, shapes and grads
    tiny_overfit    8-32 videos driven to near-zero loss (a plumbing test, NOT
                    a performance claim — see the docstring)
    benchmark       throughput, peak VRAM, projected epoch time

Checkpoint contract (must stay loadable by
`fight.temporal_classifier.TorchTemporalClassifier`):

    {
      "model_state":  <state_dict>,
      "feature_dim":  8,
      "hidden_size":  64,
      "normalization": {"method": ..., "mean": [...], "std": [...]},
      "architecture": {...}, "training": {...}, "metrics": {...}, ...
    }

The classifier reads `model_state` / `feature_dim` / `hidden_size` /
`normalization` and tolerates a bare state_dict, so adding metadata keys can
never break inference.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from suraksha.logging_utils import get_logger
from suraksha.training.dataset import (
    FEATURE_DIM,
    NORM_NONE,
    TemporalFightDataset,
    class_weights,
    fit_normalization,
)
from suraksha.training.metrics import Metrics, binary_cross_entropy_with_logits, compute_metrics
from suraksha.training.model import HIDDEN_SIZE, build_gru

log = get_logger(__name__)

CHECKPOINT_VERSION = 1


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def set_seed(seed: int, deterministic_cuda: bool = False) -> dict[str, Any]:
    """Seed python/numpy/torch. Returns what was actually enforced.

    PROMPT 4 Step 15: do not claim bit-for-bit reproducibility while CUDA
    nondeterminism remains. `deterministic_cuda=True` enables the cudnn
    deterministic flag and `use_deterministic_algorithms`, which costs throughput
    and can raise on ops lacking a deterministic kernel, so it stays opt-in and
    the returned dict records the truth either way.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic_cuda)
    torch.backends.cudnn.benchmark = not deterministic_cuda

    strict = False
    if deterministic_cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            strict = True
        except Exception as exc:                       # pragma: no cover
            log.warning("use_deterministic_algorithms failed: %s", exc)

    return {
        "seed": seed,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": bool(deterministic_cuda),
        "cudnn_benchmark": not deterministic_cuda,
        "torch_deterministic_algorithms": strict,
        "bitwise_reproducible_on_cuda": bool(strict and torch.cuda.is_available()),
    }


def resolve_device(device: str = "auto"):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

def save_checkpoint(
    path: Path | str,
    model,
    normalization: dict | None,
    *,
    architecture: dict | None = None,
    training: dict | None = None,
    metrics: dict | None = None,
    dataset_info: dict | None = None,
    extra: dict | None = None,
) -> Path:
    import torch

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "feature_dim": int(getattr(model, "feature_dim", FEATURE_DIM)),
        "hidden_size": int(getattr(model, "hidden_size", HIDDEN_SIZE)),
        "normalization": normalization or {"method": NORM_NONE},
        "architecture": architecture or (model.architecture() if hasattr(model, "architecture") else {}),
        "training": training or {},
        "metrics": metrics or {},
        "dataset": dataset_info or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, dest)
    log.info("checkpoint written -> %s (%.1f KiB)", dest, dest.stat().st_size / 1024)
    return dest


def inspect_checkpoint(path: Path | str) -> dict[str, Any]:
    """Summarize a checkpoint without loading tensors onto the GPU."""
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return {"format": "bare state_dict", "keys": []}
    blob = state.get("model_state", state)
    shapes = {k: tuple(v.shape) for k, v in blob.items() if hasattr(v, "shape")}
    n_params = int(sum(int(np.prod(s)) for s in shapes.values()))
    return {
        "format": "wrapped" if "model_state" in state else "bare state_dict",
        "top_level_keys": sorted(state.keys()),
        "feature_dim": state.get("feature_dim"),
        "hidden_size": state.get("hidden_size"),
        "normalization": state.get("normalization"),
        "param_tensors": shapes,
        "n_parameters": n_params,
        "architecture": state.get("architecture"),
        "metrics": state.get("metrics"),
        "dataset": state.get("dataset"),
    }


# --------------------------------------------------------------------------
# loss / train / eval
# --------------------------------------------------------------------------

def make_loss(pos_weight: np.ndarray | float | None = None):
    import torch
    import torch.nn as nn

    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    if isinstance(pos_weight, np.ndarray):
        # weights are [w_neg, w_pos]; BCEWithLogitsLoss only scales positives
        w_neg, w_pos = float(pos_weight[0]), float(pos_weight[1])
        pw = torch.tensor(w_pos / max(w_neg, 1e-9), dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    balanced: bool = True
    seed: int = 42
    device: str = "auto"


@dataclass
class EpochResult:
    epoch: int
    train: Metrics
    val: Metrics | None = None
    seconds: float = 0.0
    lr: float = 0.0


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip: float = 5.0) -> tuple[Metrics, float]:
    import torch

    model.train()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    total = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        if out.shape != yb.shape:
            raise ValueError(f"model output {tuple(out.shape)} != labels {tuple(yb.shape)}")
        loss = loss_fn(out, yb.float())
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
        logits_all.append(out.detach().cpu().numpy())
        labels_all.append(yb.detach().cpu().numpy())

    logits = np.concatenate(logits_all) if logits_all else np.zeros(0)
    labels = np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=np.int64)
    mean_loss = total / n_batches if n_batches else float("nan")
    return compute_metrics(logits, labels, loss=mean_loss), mean_loss


def evaluate(model, loader, device, loss_fn=None) -> Metrics:
    import torch

    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            if loss_fn is not None:
                total += float(loss_fn(out, yb.float()).item())
                n_batches += 1
            logits_all.append(out.cpu().numpy())
            labels_all.append(yb.cpu().numpy())
    logits = np.concatenate(logits_all) if logits_all else np.zeros(0)
    labels = np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=np.int64)
    mean_loss = total / n_batches if n_batches else float("nan")
    return compute_metrics(logits, labels, loss=mean_loss)


# --------------------------------------------------------------------------
# Step 12 — forward / backward / optimizer-step smoke test
# --------------------------------------------------------------------------

@dataclass
class SmokeResult:
    passed: bool
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    loss: float
    loss_reference: float
    grad_norm: float
    grads_finite: bool
    params_before: float
    params_after: float
    params_changed: bool
    n_parameters: int
    device: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["input_shape"] = list(self.input_shape)
        d["output_shape"] = list(self.output_shape)
        return d

    def table(self) -> str:
        lines = ["FORWARD / BACKWARD / OPTIMIZER SMOKE TEST", "-" * 40]
        lines += [
            f"  device              {self.device}",
            f"  input shape         {self.input_shape}",
            f"  output shape        {self.output_shape}  (expected (B,))",
            f"  parameters          {self.n_parameters}",
            f"  loss (torch BCE)    {self.loss:.6f}",
            f"  loss (numpy ref)    {self.loss_reference:.6f}",
            f"  grad norm           {self.grad_norm:.6f}",
            f"  grads finite        {self.grads_finite}",
            f"  params changed      {self.params_changed}",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        lines.append(f"  RESULT              {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def smoke_test(batch_size: int = 8, window_frames: int = 32, seed: int = 0,
               device: str = "auto", feature_dim: int = FEATURE_DIM) -> SmokeResult:
    """Prove the graph is wired: shapes, finite grads, one real weight update."""
    import torch

    dev = resolve_device(device)
    notes: list[str] = []
    torch.manual_seed(seed)

    model = build_gru(feature_dim, HIDDEN_SIZE).to(dev)
    x = torch.randn(batch_size, window_frames, feature_dim, device=dev)
    y = torch.randint(0, 2, (batch_size,), device=dev).float()

    out = model(x)
    if out.shape != (batch_size,):
        raise AssertionError(f"expected output {(batch_size,)}, got {tuple(out.shape)}")

    before = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(out, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if len(grads) != sum(1 for _ in model.parameters()):
        notes.append("some parameters received no gradient")
    grad_norm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)).item()) if grads else 0.0
    grads_finite = bool(all(torch.isfinite(g).all().item() for g in grads))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy()

    loss_val = float(loss.item())
    ref = binary_cross_entropy_with_logits(out.detach().cpu().numpy(), y.cpu().numpy())
    if abs(loss_val - ref) > 1e-4:
        notes.append(f"torch loss {loss_val:.6f} disagrees with numpy reference {ref:.6f}")

    changed = bool(not np.allclose(before, after))
    if not changed:
        notes.append("optimizer step did not modify weights")
    n_params = int(sum(p.numel() for p in model.parameters()))

    return SmokeResult(
        passed=bool(grads_finite and grad_norm > 0 and changed and out.shape == (batch_size,)),
        input_shape=tuple(x.shape), output_shape=tuple(out.shape),
        loss=loss_val, loss_reference=ref, grad_norm=grad_norm,
        grads_finite=grads_finite,
        params_before=float(before.sum()), params_after=float(after.sum()),
        params_changed=changed, n_parameters=n_params, device=str(dev), notes=notes,
    )


# --------------------------------------------------------------------------
# Step 13 — tiny overfit (a plumbing test, not a performance claim)
# --------------------------------------------------------------------------

@dataclass
class OverfitResult:
    passed: bool
    n_videos: int
    n_windows: int
    epochs: int
    final_loss: float
    initial_loss: float
    loss_reduction: float
    final_accuracy: float
    history: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    model: Any = None            # the trained net, so callers can checkpoint it
    class_weights: list | None = None

    def table(self) -> str:
        lines = ["TINY OVERFIT TEST (plumbing check — NOT model performance)", "-" * 56]
        lines += [
            f"  videos / windows    {self.n_videos} / {self.n_windows}",
            f"  epochs              {self.epochs}",
            f"  loss  first->last   {self.initial_loss:.4f} -> {self.final_loss:.4f}",
            f"  loss reduction      {self.loss_reduction:.1%}",
            f"  final accuracy      {self.final_accuracy:.4f}",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        lines.append(f"  RESULT              {'PASS (learns)' if self.passed else 'FAIL — STOP AND DIAGNOSE'}")
        return "\n".join(lines)


def tiny_overfit(train_ds: TemporalFightDataset, epochs: int = 200, batch_size: int = 8,
                 lr: float = 3e-3, device: str = "auto", seed: int = 0,
                 loss_target: float = 0.05, n_videos: int = 0,
                 val_ds: TemporalFightDataset | None = None) -> OverfitResult:
    """Drive a handful of videos to near-zero training loss.

    This proves gradients reach every layer, labels are not shuffled relative to
    features, and normalization is finite. Memorizing 8-32 clips says nothing
    about generalization and must never be reported as accuracy.
    """
    import torch

    dev = resolve_device(device)
    set_seed(seed)
    notes: list[str] = []

    model = build_gru(FEATURE_DIM, HIDDEN_SIZE).to(dev)
    weights = class_weights(train_ds.corpus.labels) if len(train_ds) else None
    loss_fn = make_loss(weights)
    if weights is not None:
        notes.append(f"class weights applied {np.round(weights, 3).tolist()}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = train_ds.make_loader(batch_size=batch_size, shuffle=True)

    history: list[dict] = []
    initial = float("nan")
    final = float("nan")
    final_acc = float("nan")
    for ep in range(1, epochs + 1):
        m, _ = train_one_epoch(model, loader, opt, loss_fn, dev, grad_clip=0.0)
        if ep == 1:
            initial = m.loss
        final, final_acc = m.loss, m.accuracy
        history.append({"epoch": ep, "loss": m.loss, "accuracy": m.accuracy})
        if ep % max(1, epochs // 5) == 0 or ep == 1:
            log.info("overfit epoch %3d/%d  loss=%.4f  acc=%.4f", ep, epochs, m.loss, m.accuracy)
        if m.loss <= loss_target and m.accuracy >= 0.99:
            notes.append(f"converged early at epoch {ep}")
            break

    if val_ds is not None and len(val_ds):
        vm = evaluate(model, val_ds.make_loader(batch_size=batch_size, shuffle=False), dev, loss_fn)
        notes.append(f"held-out videos: loss={vm.loss:.4f} acc={vm.accuracy:.4f} "
                     "(expected to be poor — this test memorizes on purpose)")

    reduction = (initial - final) / initial if initial and np.isfinite(initial) else 0.0
    passed = bool(final <= max(loss_target * 4, 0.2) and reduction > 0.5 and final_acc >= 0.9)
    if not passed:
        notes.append("did not learn: check label alignment, normalization finiteness, "
                     "feature all-zero rows, and that gradients are not vanishing")

    return OverfitResult(
        passed=passed, n_videos=n_videos, n_windows=len(train_ds), epochs=len(history),
        final_loss=final, initial_loss=initial, loss_reduction=reduction,
        final_accuracy=final_acc, history=history, notes=notes,
        model=model,
        class_weights=(np.round(weights, 4).tolist() if weights is not None else None),
    )


# --------------------------------------------------------------------------
# Step 17 — GPU / throughput benchmark
# --------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    device: str
    gpu_name: str
    vram_total_gb: float
    batch_size: int
    window_frames: int
    feature_dim: int
    n_parameters: int
    forward_ms: float
    backward_ms: float
    step_ms: float
    windows_per_second: float
    peak_vram_mb: float
    projected_epoch_minutes: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["projected_epoch_minutes"] = dict(self.projected_epoch_minutes)
        return d

    def table(self) -> str:
        lines = ["THROUGHPUT / VRAM BENCHMARK", "-" * 40]
        lines += [
            f"  device              {self.device}",
            f"  gpu                 {self.gpu_name}",
            f"  vram total          {self.vram_total_gb:.2f} GB",
            f"  batch x T x F       {self.batch_size} x {self.window_frames} x {self.feature_dim}",
            f"  parameters          {self.n_parameters:,}",
            f"  forward             {self.forward_ms:.2f} ms/batch",
            f"  backward            {self.backward_ms:.2f} ms/batch",
            f"  fwd+bwd+step        {self.step_ms:.2f} ms/batch",
            f"  throughput          {self.windows_per_second:,.0f} windows/s",
            f"  peak vram           {self.peak_vram_mb:.1f} MiB",
        ]
        for k, v in self.projected_epoch_minutes.items():
            lines.append(f"  epoch @ {k:>8} win   {v:.2f} min")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def benchmark(batch_sizes: Sequence[int] = (8, 16, 32, 64), window_frames: int = 32,
              iters: int = 30, warmup: int = 5, device: str = "auto",
              corpus_sizes: Sequence[int] = (5000, 18750)) -> list[BenchmarkResult]:
    """Measure the cost of one training step on synthetic-but-correctly-shaped data.

    Uses random tensors of the real shape rather than real windows: the GRU cost
    depends only on (B, T, 8), so this isolates compute from disk/CPU feature
    extraction without touching the dataset.
    """
    import torch

    dev = resolve_device(device)
    is_cuda = dev.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    vram_total = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3) if is_cuda else 0.0
    results: list[BenchmarkResult] = []

    for bs in batch_sizes:
        model = build_gru(FEATURE_DIM, HIDDEN_SIZE).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        x = torch.randn(bs, window_frames, FEATURE_DIM, device=dev)
        y = torch.randint(0, 2, (bs,), device=dev).float()

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)

        def one_step() -> tuple[float, float]:
            t0 = time.perf_counter()
            out = model(x)
            if is_cuda:
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if is_cuda:
                torch.cuda.synchronize(dev)
            return (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000

        for _ in range(warmup):
            one_step()
        fwd: list[float] = []
        bwd: list[float] = []
        for _ in range(iters):
            a, b = one_step()
            fwd.append(a)
            bwd.append(b)

        forward_ms = float(np.median(fwd))
        backward_ms = float(np.median(bwd))
        step_ms = forward_ms + backward_ms
        peak_mb = (torch.cuda.max_memory_allocated(dev) / 1024 ** 2) if is_cuda else 0.0
        n_params = int(sum(p.numel() for p in model.parameters()))
        notes: list[str] = []
        if is_cuda and peak_mb > 0.85 * vram_total * 1024:
            notes.append("peak VRAM near the card's limit")

        results.append(BenchmarkResult(
            device=str(dev), gpu_name=gpu_name, vram_total_gb=vram_total,
            batch_size=bs, window_frames=window_frames, feature_dim=FEATURE_DIM,
            n_parameters=n_params, forward_ms=forward_ms, backward_ms=backward_ms,
            step_ms=step_ms,
            windows_per_second=(bs / (step_ms / 1000)) if step_ms else 0.0,
            peak_vram_mb=peak_mb,
            projected_epoch_minutes={
                str(n): (n / max(bs, 1)) * (step_ms / 1000) / 60 for n in corpus_sizes
            },
            notes=notes,
        ))
        del model, opt, x, y
        if is_cuda:
            torch.cuda.empty_cache()

    return results


# --------------------------------------------------------------------------
# misc helpers used by the CLI
# --------------------------------------------------------------------------

def write_json(path: Path | str, payload: dict) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return dest


def fit_and_report_normalization(train_windows: np.ndarray, method: str) -> dict:
    params = fit_normalization(train_windows, method)
    if params.get("method") == "standard":
        log.info("normalization fitted on TRAIN only: n_windows=%d", params["n_windows"])
        for i, (mu, sd) in enumerate(zip(params["mean"], params["std"])):
            log.info("  feature %d  mean=%+.4f  std=%.4f", i, mu, sd)
    return params
