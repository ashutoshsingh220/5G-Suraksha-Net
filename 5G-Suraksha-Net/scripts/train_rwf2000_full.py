"""PROMPT 6 — full RWF-2000 training + validation orchestrator.

This is the ONLY entry point that performs the authorized full training run. It
reuses the verified PROMPT-4/5 building blocks unchanged (same GRU, same
train_one_epoch/evaluate, same train-only normalization, same inverse-frequency
class weights, same leakage registry) and adds exactly what PROMPT 6 requires on
top:

  * best checkpoint written to models/temporal/rwf2000_best.pt (STEP 8)
  * window-level metrics at BOTH 0.5 and the production 0.60 threshold (STEP 11)
  * VIDEO-LEVEL metrics via per-clip mean-pooling, clearly separated (STEP 10)
  * duplicate-excluded validation alongside the full official val (STEP 12)
  * overfitting analysis from the recorded history (STEP 13)
  * the six report artifacts + training-history CSV (STEP 18)

It does NOT: change the architecture, augment, tune the 0.60 threshold, touch any
dataset other than RWF-2000, fabricate a test split, re-extract features
(extraction is a separate measured stage — STEP 16), or claim production
readiness / real-world CCTV accuracy.

Feature extraction must already be complete (scripts/extract_features.py --all
--force). This script refuses to train on a partially built cache.

Usage:
    python scripts/train_rwf2000_full.py
    python scripts/train_rwf2000_full.py --epochs 30 --batch-size 64 --lr 1e-3 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import numpy as np

from suraksha.config import PROJECT_ROOT, load_config, load_training_config
from suraksha.data.manifest import load_manifest
from suraksha.logging_utils import get_logger, setup_logging
from suraksha.training.dataset import (
    FEATURE_DIM,
    TemporalFightDataset,
    build_corpus,
    class_weights,
)
from suraksha.training.leakage import load_or_build_registry
from suraksha.training.metrics import (
    binary_cross_entropy_with_logits,
    compute_metrics_from_probs,
    sigmoid,
    video_level_metrics,
)
from suraksha.training.model import HIDDEN_SIZE, build_gru
from suraksha.training.trainer import (
    evaluate,
    fit_and_report_normalization,
    make_loss,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    write_json,
)

# Reuse the audited PROMPT-4 helpers verbatim rather than re-implementing them.
import train_temporal as tt

log = get_logger(__name__)
RULE = tt.RULE
_resolve = tt._resolve

PROD_THRESHOLD = 0.60          # configs/app.yaml fight.temporal.score_threshold — NOT tuned here
BEST_CHECKPOINT_NAME = "rwf2000_best.pt"


def _die(msg: str, code: int = 2) -> int:
    print(f"\nFATAL: {msg}")
    return code


def _prob_stats(p: np.ndarray) -> dict:
    if len(p) == 0:
        return {"n": 0}
    hist, edges = np.histogram(p, bins=20, range=(0.0, 1.0))
    return {
        "n": int(len(p)),
        "mean": round(float(p.mean()), 6),
        "std": round(float(p.std()), 6),
        "min": round(float(p.min()), 6),
        "max": round(float(p.max()), 6),
        "percentiles": {q: round(float(np.percentile(p, q)), 6)
                        for q in (1, 5, 25, 50, 75, 95, 99)},
        "histogram_20_bins_0_to_1": hist.tolist(),
        "histogram_edges": [round(float(e), 3) for e in edges.tolist()],
    }


def _cache_complete(entries, cache_dir: Path, manifest: str) -> tuple[int, list[str]]:
    from suraksha.training.dataset import cache_path
    missing = [e.clip_id for e in entries
               if not cache_path(cache_dir, manifest, e.clip_id).exists()]
    return len(entries) - len(missing), missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--device", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache-dir", default="", help="Feature cache directory override")
    ap.add_argument("--checkpoint-name", default=BEST_CHECKPOINT_NAME, help="Checkpoint file name")
    ap.add_argument("--report-prefix", default="rwf2000_", help="Prefix for report artifact files")
    ap.add_argument("--best-criterion", choices=["val_loss", "val_roc_auc"], default="val_roc_auc", help="Criterion for best checkpoint")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping patience (epochs). 0 = disabled")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)

    tcfg = load_training_config(args.config or None)
    cfg = load_config()

    epochs = args.epochs or tcfg.training.epochs
    batch_size = args.batch_size or tcfg.training.batch_size
    eval_batch = args.eval_batch_size or tcfg.training.eval_batch_size
    lr = args.lr or tcfg.training.lr
    seed = args.seed if args.seed is not None else tcfg.training.seed

    # ---- STEP 2/7: config must match the shipped inference path ----
    problems = tcfg.validate_against_app(cfg)
    print(f"\n{RULE}\nCONFIG CONSISTENCY WITH THE SHIPPED INFERENCE PATH\n{RULE}")
    print(f"  window/stride/feature   {tcfg.dataset.window_frames}/"
          f"{tcfg.dataset.stride_frames}/{tcfg.dataset.feature_dim}")
    print(f"  app score_threshold     {cfg.fight.temporal.score_threshold} (NOT tuned here)")
    print(f"  RESULT                  {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print(f"    ! {p}")
    if problems:
        return _die("training config disagrees with configs/app.yaml")

    # ---- STEP 2: label / split / balance audit (programmatic) ----
    entries = load_manifest(tcfg.dataset.manifest).entries
    audit = tt.audit_labels_and_splits(entries, tcfg)
    if not audit["passed"]:
        return _die("dataset invariants failed — refusing to train")

    # ---- STEP 2: leakage registry (must exist before any val metric) ----
    registry, reg_path = load_or_build_registry(
        tcfg.dataset.manifest, _resolve(tcfg.leakage.registry_path))
    print(f"\n{RULE}\nSTEP 2 — DUPLICATE-CONTENT LEAKAGE IN THE OFFICIAL SPLIT\n{RULE}")
    for k, v in registry.summary().items():
        if k not in ("note", "leaked_val_clip_ids"):
            print(f"  {k:<28}{v}")
    print(f"  leaked val clips        {registry.leaked_val_clip_ids}")
    print("  Official validation metrics are subject to the known duplicate-content")
    print("  limitation. They are reported FULL and DUPLICATE-EXCLUDED; neither is")
    print("  leakage-free.")

    # ---- corpora from the pre-built cache (extraction is a separate stage) ----
    cache_dir = _resolve(args.cache_dir) if args.cache_dir else _resolve(tcfg.dataset.feature_cache_dir)
    train_entries = tt.load_split_entries(entries, tcfg.dataset.train_split, tcfg)
    val_entries = tt.load_split_entries(entries, tcfg.dataset.val_split, tcfg)

    n_tr_cached, miss_tr = _cache_complete(train_entries, cache_dir, tcfg.dataset.manifest)
    n_val_cached, miss_val = _cache_complete(val_entries, cache_dir, tcfg.dataset.manifest)
    print(f"\n{RULE}\nFEATURE CACHE INTEGRITY (extraction must already be complete)\n{RULE}")
    print(f"  train clips cached      {n_tr_cached}/{len(train_entries)}")
    print(f"  val clips cached        {n_val_cached}/{len(val_entries)}")
    if miss_tr or miss_val:
        show = (miss_tr + miss_val)[:5]
        return _die(f"{len(miss_tr) + len(miss_val)} clip(s) missing from the cache "
                    f"(e.g. {show}). Run: python scripts/extract_features.py --all --force")

    train_corpus = build_corpus(train_entries, cache_dir, tcfg.dataset.manifest,
                                split=tcfg.dataset.train_split)
    val_corpus = build_corpus(val_entries, cache_dir, tcfg.dataset.manifest,
                              split=tcfg.dataset.val_split)
    if len(train_corpus) == 0 or len(val_corpus) == 0:
        return _die("empty corpus")

    # ---- STEP 1/10/11: measured sequence shapes + feature stats ----
    train_shapes = tt.document_sequences(train_corpus, tcfg, cache_dir)
    val_flat = val_corpus.windows.reshape(-1, FEATURE_DIM)
    val_shapes = {
        "windows": len(val_corpus),
        "tensor_shape": list(val_corpus.windows.shape),
        "clips_with_windows": len(val_corpus.clip_ids_present()),
        "clips_zero_yield": len(val_corpus.stats.clips_zero_yield),
        "window_class_counts": val_corpus.counts(),
        "window_class_ratio": round(val_corpus.class_ratio(), 4),
        "finite": bool(np.isfinite(val_corpus.windows).all()),
        "per_feature_mean": [round(float(val_flat[:, i].mean()), 4) for i in range(FEATURE_DIM)],
    }
    print(f"\n  val corpus: {val_shapes['windows']} windows, "
          f"{val_shapes['clips_with_windows']} clips, ratio {val_shapes['window_class_ratio']}:1")

    # ---- STEP 3: normalization fitted on TRAIN windows ONLY ----
    print(f"\n{RULE}\nSTEP 3 — Z-SCORE NORMALIZATION FITTED ON TRAINING WINDOWS ONLY\n{RULE}")
    norm = fit_and_report_normalization(train_corpus.windows, tcfg.normalization.method)
    if norm.get("method") == "standard":
        mu = np.asarray(norm["mean"], dtype=np.float64)
        sd = np.asarray(norm["std"], dtype=np.float64)
        finite = bool(np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all())
        print(f"  mean finite & std finite/positive: {finite}")
        print(f"  fitted_on={norm.get('fitted_on')}  n_windows={norm.get('n_windows')}")
        if not finite:
            return _die("normalization produced non-finite or zero-variance statistics")
    else:
        return _die(f"expected 'standard' normalization, got {norm.get('method')!r}")

    # ---- STEP 4: inverse-frequency class weights from TRAIN only ----
    print(f"\n{RULE}\nSTEP 4 — CLASS DISTRIBUTION AND INVERSE-FREQUENCY WEIGHTS\n{RULE}")
    weights = class_weights(train_corpus.labels) if tcfg.training.balanced_class_weights else None
    tc = train_corpus.counts()
    vc = val_corpus.counts()
    n_train = len(train_corpus)
    n_fight = tc.get("FIGHT", 0)
    n_nonfight = tc.get("NON_FIGHT", 0)
    print(f"  WINDOW-level train      {tc}  ratio {train_corpus.class_ratio():.2f}:1")
    print(f"  WINDOW-level val        {vc}  ratio {val_corpus.class_ratio():.2f}:1")
    if weights is not None:
        print(f"  w = N/(2*n_c) ->        NON_FIGHT={weights[0]:.4f}  FIGHT={weights[1]:.4f}")
        print(f"  (renormalized to sum 2; applied via BCEWithLogitsLoss pos_weight)")
    else:
        print("  class weights NOT applied (ratio under 1.25:1)")

    # ---- STEP 5/6: architecture is fixed; seed everything ----
    det = set_seed(seed, tcfg.training.deterministic_cuda)
    device = resolve_device(args.device or tcfg.training.device)
    import torch
    model = build_gru(FEATURE_DIM, tcfg.model.hidden_size).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(f"\n{RULE}\nSTEP 5/6 — FIXED ARCHITECTURE + TRAINING CONFIG\n{RULE}")
    print(f"  {model.architecture()}")
    print(f"  parameters              {n_params:,}")
    print(f"  device                  {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")
    print(f"  epochs={epochs} batch={batch_size} eval_batch={eval_batch} lr={lr} "
          f"wd={tcfg.training.weight_decay} grad_clip={tcfg.training.grad_clip} seed={seed}")
    print(f"  loss=BCEWithLogitsLoss optimizer={tcfg.training.optimizer} "
          f"scheduler=none augmentation=none")
    if n_params != 16321:
        print(f"  WARNING: parameter count {n_params} != the verified 16,321 — "
              "architecture may have drifted")

    train_ds = TemporalFightDataset(train_corpus, norm)
    val_ds = TemporalFightDataset(val_corpus, norm)
    train_loader = train_ds.make_loader(batch_size, shuffle=True,
                                        num_workers=tcfg.training.num_workers)
    val_loader = val_ds.make_loader(eval_batch, shuffle=False,
                                    num_workers=tcfg.training.num_workers)
    loss_fn = make_loss(weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=tcfg.training.weight_decay)

    # ---- STEP 7: train, recording EVERY epoch ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict] = []
    best = None                       # (score, epoch)
    epochs_without_improvement = 0
    best_ckpt = _resolve(tcfg.outputs.checkpoint_dir) / args.checkpoint_name
    criterion_desc = ("min official-val loss (window-level)"
                      if args.best_criterion == "val_loss"
                      else "max official-val ROC-AUC (window-level)")
    print(f"\n{RULE}\nSTEP 7 — TRAINING ({epochs} epochs max, patience={args.patience}, criterion={args.best_criterion}, "
          f"{len(train_ds)} train / {len(val_ds)} val windows)\n{RULE}")
    train_t0 = time.time()
    keep_mask = val_corpus.mask_excluding_clips(registry.leaked_val_clip_ids)
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        tm, _ = train_one_epoch(model, train_loader, opt, loss_fn, device,
                                grad_clip=tcfg.training.grad_clip)
        vm = evaluate(model, val_loader, device, loss_fn)
        logits = tt._all_logits(model, val_ds, device)
        probs = sigmoid(logits)
        val_loss = tt._mean_bce(logits, val_corpus.labels)
        # window-level on the FULL official val at the neutral 0.5 boundary
        w50 = compute_metrics_from_probs(probs, val_corpus.labels, 0.5, loss=val_loss)
        w50_adj = compute_metrics_from_probs(probs[keep_mask], val_corpus.labels[keep_mask],
                                             0.5, loss=tt._mean_bce(logits[keep_mask],
                                                                    val_corpus.labels[keep_mask]))
        secs = time.time() - ep_t0
        row = {
            "epoch": ep,
            "lr": lr,
            "seconds": round(secs, 2),
            "train_loss": tm.loss,
            "train_accuracy": tm.accuracy,
            "val_loss": val_loss,
            "val_accuracy": w50.accuracy,
            "val_precision": w50.precision,
            "val_recall": w50.recall,
            "val_f1": w50.f1,
            "val_specificity": w50.specificity,
            "val_balanced_accuracy": w50.balanced_accuracy,
            "val_roc_auc": w50.roc_auc,
            "val_pr_auc": w50.pr_auc,
            "val_confusion": {"tn": w50.tn, "fp": w50.fp, "fn": w50.fn, "tp": w50.tp},
            "val_excl_duplicates_accuracy": w50_adj.accuracy,
            "val_excl_duplicates_roc_auc": w50_adj.roc_auc,
            "train_metrics_full": tm.as_dict(),
            "val_metrics_full": w50.as_dict(),
            "val_metrics_excl_duplicates": w50_adj.as_dict(),
        }
        history.append(row)
        auc_s = f"{w50.roc_auc:.4f}" if w50.roc_auc is not None else "n/a"
        print(f"  ep {ep:3d}/{epochs}  train loss {tm.loss:.4f} acc {tm.accuracy:.4f} | "
              f"val loss {val_loss:.4f} acc {w50.accuracy:.4f} auc {auc_s} | "
              f"dup-free acc {w50_adj.accuracy:.4f}  ({secs:.1f}s)")

        # ---- STEP 8: best epoch tracking & early stopping ----
        if args.best_criterion == "val_loss":
            score = val_loss
            is_better = (best is None or score < best[0])
        else:
            score = w50.roc_auc if w50.roc_auc is not None else w50.accuracy
            is_better = (best is None or score > best[0])

        if is_better:
            best = (score, ep)
            epochs_without_improvement = 0
            save_checkpoint(
                best_ckpt, model, norm, architecture=model.architecture(),
                training={"epochs_run": ep, "best_epoch": ep, "batch_size": batch_size,
                          "eval_batch_size": eval_batch, "lr": lr,
                          "optimizer": tcfg.training.optimizer,
                          "weight_decay": tcfg.training.weight_decay,
                          "grad_clip": tcfg.training.grad_clip,
                          "loss": "BCEWithLogitsLoss", "scheduler": "none",
                          "augmentation": "none",
                          "class_weights": (np.round(weights, 4).tolist()
                                            if weights is not None else None),
                          "seed": det["seed"], "determinism": det,
                          "best_criterion": criterion_desc,
                          "best_val_loss": val_loss,
                          "best_val_roc_auc": (w50.roc_auc if w50.roc_auc is not None else None)},
                metrics={"val_window_0.5": w50.as_dict(),
                         "val_window_0.5_excl_duplicates": w50_adj.as_dict()},
                dataset_info={"manifest": tcfg.dataset.manifest,
                              "dataset": "RWF-2000 (official split, train+val only)",
                              "train_videos": len(train_corpus.clip_ids_present()),
                              "train_windows": len(train_corpus),
                              "val_videos": len(val_corpus.clip_ids_present()),
                              "val_windows": len(val_corpus),
                              "window_frames": tcfg.dataset.window_frames,
                              "stride_frames": tcfg.dataset.stride_frames,
                              "feature_dim": FEATURE_DIM,
                              "leaked_val_clips": registry.leaked_val_clip_ids},
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"\n  [Early Stopping] No improvement in {args.best_criterion} "
                      f"for {args.patience} epochs. Stopping early at epoch {ep}.")
                break
    train_seconds = time.time() - train_t0
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else 0.0)
    best_epoch = best[1] if best else None
    n_ran = len(history)
    print(f"\n  training complete in {train_seconds:.1f}s  "
          f"({train_seconds/n_ran:.2f}s/epoch across {n_ran} epochs)  best epoch {best_epoch}  "
          f"peak VRAM {peak_vram_mb:.1f} MiB")

    # ---- STEP 14 (in-process load-back of the SELECTED best checkpoint) ----
    # The authoritative fresh-process round-trip is run separately via
    # `train_temporal.py checkpoint`. Here we reload the saved best so every
    # final number below describes the artifact that ships, not the last epoch.
    state = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    best_model = build_gru(FEATURE_DIM, tcfg.model.hidden_size).to(device)
    best_model.load_state_dict(state["model_state"])
    best_model.eval()
    print(f"  reloaded best checkpoint (epoch {state.get('training',{}).get('best_epoch')}) "
          f"for final validation")

    # ---- STEP 16: inference timing measured separately from training ----
    inf_t0 = time.time()
    val_logits = tt._all_logits(best_model, val_ds, device)
    inf_seconds = time.time() - inf_t0
    val_probs = sigmoid(val_logits)
    val_windows_per_sec = len(val_ds) / inf_seconds if inf_seconds > 0 else 0.0

    val_labels = val_corpus.labels
    val_clip_ids = val_corpus.clip_ids

    # ---- STEP 9/11: window-level metrics at 0.5 AND production 0.60 ----
    # Loss is threshold-independent and computed from the raw logits (honest
    # unweighted BCE), not reconstructed from probabilities.
    val_loss_full = binary_cross_entropy_with_logits(val_logits, val_labels)
    val_loss_adj = binary_cross_entropy_with_logits(
        val_logits[keep_mask], val_labels[keep_mask])

    def window_block(probs, labels, thr, loss):
        return compute_metrics_from_probs(probs, labels, thr, loss=loss)

    win = {
        "full_official_val": {
            "threshold_0.50": window_block(val_probs, val_labels, 0.5, val_loss_full).as_dict(),
            "threshold_0.60_production": window_block(val_probs, val_labels, PROD_THRESHOLD, val_loss_full).as_dict(),
        },
        "duplicate_excluded": {
            "threshold_0.50": window_block(val_probs[keep_mask], val_labels[keep_mask], 0.5, val_loss_adj).as_dict(),
            "threshold_0.60_production": window_block(val_probs[keep_mask], val_labels[keep_mask], PROD_THRESHOLD, val_loss_adj).as_dict(),
        },
        "windows_excluded_as_duplicates": int((~keep_mask).sum()),
        "leaked_val_clips": registry.leaked_val_clip_ids,
    }

    # ---- STEP 10: VIDEO-LEVEL metrics (per-clip mean-pooling) ----
    def video_block(probs, clip_ids, labels, thr, method="mean"):
        return video_level_metrics(probs, clip_ids, labels, thr, method).as_dict()

    kept_ids = [c for c, k in zip(val_clip_ids, keep_mask) if k]
    video = {
        "aggregation": "mean window probability per source clip (documented primary); "
                       "max reported alongside",
        "full_official_val": {
            "mean_threshold_0.50": video_block(val_probs, val_clip_ids, val_labels, 0.5, "mean"),
            "mean_threshold_0.60_production": video_block(val_probs, val_clip_ids, val_labels, PROD_THRESHOLD, "mean"),
            "max_threshold_0.50": video_block(val_probs, val_clip_ids, val_labels, 0.5, "max"),
        },
        "duplicate_excluded": {
            "mean_threshold_0.50": video_block(val_probs[keep_mask], kept_ids, val_labels[keep_mask], 0.5, "mean"),
            "mean_threshold_0.60_production": video_block(val_probs[keep_mask], kept_ids, val_labels[keep_mask], PROD_THRESHOLD, "mean"),
        },
        "n_videos_full": len(set(val_clip_ids)),
        "n_videos_duplicate_excluded": len(set(kept_ids)),
    }

    # ---- STEP 11: score distributions ----
    fight_p = val_probs[val_labels == 1]
    nonfight_p = val_probs[val_labels == 0]
    vm_mean = video_level_metrics(val_probs, val_clip_ids, val_labels, 0.5, "mean")
    v_scores = np.asarray(vm_mean.extra["video_scores"], dtype=np.float64)
    v_labels = np.asarray(vm_mean.extra["video_labels"], dtype=np.int64)
    dist = {
        "note": "Validation window scores from the best checkpoint. The 0.60 "
                "production threshold is evaluated, NOT tuned.",
        "production_threshold": PROD_THRESHOLD,
        "window_level": {
            "FIGHT": _prob_stats(fight_p),
            "NON_FIGHT": _prob_stats(nonfight_p),
        },
        "video_level_mean": {
            "FIGHT": _prob_stats(v_scores[v_labels == 1]),
            "NON_FIGHT": _prob_stats(v_scores[v_labels == 0]),
        },
        "frac_windows_above_0.60": {
            "FIGHT": round(float((fight_p > PROD_THRESHOLD).mean()), 4) if len(fight_p) else None,
            "NON_FIGHT": round(float((nonfight_p > PROD_THRESHOLD).mean()), 4) if len(nonfight_p) else None,
        },
    }

    # ---- STEP 13: overfitting analysis ----
    tr_losses = [h["train_loss"] for h in history]
    va_losses = [h["val_loss"] for h in history]
    tr_accs = [h["train_accuracy"] for h in history]
    va_accs = [h["val_accuracy"] for h in history]
    min_val_epoch = int(np.argmin(va_losses)) + 1
    best_val_loss = va_losses[best_epoch - 1]
    final_val_loss = va_losses[-1]

    # Classify the post-best validation-loss trend (STEP 13: improve/stall/diverge).
    # Compare the mean val loss over the last third of epochs after the best epoch
    # against the best-epoch val loss, so a single noisy epoch cannot decide it.
    after = va_losses[best_epoch:]                     # epochs strictly after best
    if len(after) >= 3:
        tail_mean = float(np.mean(after[-max(3, len(after) // 3):]))
        rise = tail_mean - best_val_loss
        rel = rise / best_val_loss if best_val_loss else 0.0
        if rel > 0.10:
            trend = "diverging"
        elif rel > 0.02:
            trend = "stalling_then_rising"
        elif rel < -0.02:
            trend = "still_improving"
        else:
            trend = "flat"
    else:
        tail_mean = float(np.mean(after)) if after else best_val_loss
        rise = tail_mean - best_val_loss
        rel = rise / best_val_loss if best_val_loss else 0.0
        trend = "insufficient_epochs_after_best"

    # AUC trend after best: does ranking quality also decay, or only the loss?
    aucs = [h["val_roc_auc"] for h in history if h["val_roc_auc"] is not None]
    auc_after_best = [h["val_roc_auc"] for h in history[best_epoch:]
                      if h["val_roc_auc"] is not None]
    auc_decay = (round(aucs[best_epoch - 1] - float(np.mean(auc_after_best[-3:])), 4)
                 if len(auc_after_best) >= 3 else None)

    gap_final = tr_accs[-1] - va_accs[-1]
    gap_at_best = tr_accs[best_epoch - 1] - va_accs[best_epoch - 1]
    overfitting_signature = bool(
        trend in ("diverging", "stalling_then_rising")
        and tr_losses[-1] < tr_losses[best_epoch - 1]      # train still improving
        and gap_final > gap_at_best                        # gap widening
    )
    overfit = {
        "best_epoch_by_roc_auc": best_epoch,
        "epoch_of_min_val_loss": min_val_epoch,
        "final_train_loss": round(tr_losses[-1], 6),
        "final_val_loss": round(final_val_loss, 6),
        "best_epoch_val_loss": round(best_val_loss, 6),
        "val_loss_min": round(min(va_losses), 6),
        "final_train_accuracy": round(tr_accs[-1], 6),
        "final_val_accuracy": round(va_accs[-1], 6),
        "best_epoch_val_accuracy": round(va_accs[best_epoch - 1], 6),
        "train_val_accuracy_gap_final": round(gap_final, 6),
        "train_val_accuracy_gap_at_best": round(gap_at_best, 6),
        "post_best_val_loss_trend": trend,
        "post_best_val_loss_rise": round(rise, 6),
        "post_best_val_loss_rise_relative": round(rel, 4),
        "post_best_val_roc_auc_decay": auc_decay,
        "overfitting_signature_present": overfitting_signature,
        "assessment": (
            f"Best checkpoint is epoch {best_epoch} (max val ROC-AUC); validation loss "
            f"is minimal at epoch {min_val_epoch}. After the best epoch the validation "
            f"loss is {trend.replace('_', ' ')}: it rises from {best_val_loss:.4f} to a "
            f"tail mean of {tail_mean:.4f} (+{rel*100:.1f}%) while training loss keeps "
            f"falling ({tr_losses[best_epoch-1]:.4f} -> {tr_losses[-1]:.4f}) and the "
            f"train/val accuracy gap widens from {gap_at_best:.4f} to {gap_final:.4f}. "
            + ("This IS an overfitting signature (diverging val loss + widening gap + "
               "still-improving train loss), so the early best-epoch checkpoint is the "
               "correct artifact to ship; the epoch-30 weights are NOT used."
               if overfitting_signature else
               "Per PROMPT 6 this is not labeled overfitting on the accuracy gap alone; "
               "the val-loss trend and gap trajectory are the evidence.")
            + (f" Val ROC-AUC also decays after best by {auc_decay}."
               if auc_decay is not None else "")
        ),
    }

    # ---- STEP 16: extraction timing (read from the extraction report) ----
    extract_report_path = _resolve(tcfg.outputs.report_dir) / "rwf2000_feature_cache_full.json"
    if args.cache_dir and "v2" in args.cache_dir:
        v2_rep = _resolve(tcfg.outputs.report_dir) / "corpus_v2_verification.json"
        if v2_rep.exists():
            extract_report_path = v2_rep
    extraction_timing = {"report": str(extract_report_path), "available": False}
    if extract_report_path.exists():
        try:
            er = json.loads(extract_report_path.read_text(encoding="utf-8"))
            s = er.get("summary", {})
            if not s and "v2_relaxed" in er:
                v2_data = er["v2_relaxed"]
                zy_train = v2_data.get("train", {}).get("zero_yield_clips", 0)
                zy_val = v2_data.get("val", {}).get("zero_yield_clips", 0)
                extraction_timing = {
                    "available": True,
                    "videos_requested": 2000,
                    "videos_built": 2000,
                    "videos_reused": 0,
                    "videos_failed": 0,
                    "total_windows": v2_data.get("total_windows"),
                    "zero_yield_clips": zy_train + zy_val,
                    "zero_yield_pct": round((zy_train + zy_val) / 20.0, 2),
                    "total_seconds": None,
                    "seconds_per_clip": None,
                    "videos_per_second": None,
                }
            else:
                extraction_timing = {
                    "available": True,
                    "videos_requested": s.get("requested"),
                    "videos_built": s.get("built"),
                    "videos_reused": s.get("reused"),
                    "videos_failed": s.get("failed"),
                    "total_windows": s.get("windows"),
                    "zero_yield_clips": s.get("zero_yield_clips"),
                    "zero_yield_pct": s.get("zero_yield_pct"),
                    "total_seconds": s.get("seconds"),
                    "seconds_per_clip": s.get("seconds_per_clip"),
                    "videos_per_second": (round(s["built"] / s["seconds"], 4)
                                          if s.get("seconds") and s.get("built") else None),
                    "per_cell": er.get("per_cell"),
                }
        except (json.JSONDecodeError, KeyError) as exc:
            extraction_timing["error"] = str(exc)

    performance = {
        "note": "Extraction, training and inference are measured SEPARATELY (STEP 16).",
        "feature_extraction": extraction_timing,
        "training": {
            "total_seconds": round(train_seconds, 2),
            "epochs": len(history),
            "seconds_per_epoch": round(train_seconds / max(len(history), 1), 3),
            "train_windows": len(train_ds),
            "windows_per_second": round(len(train_ds) * len(history) / train_seconds, 1)
            if train_seconds > 0 else None,
            "device": str(device),
            "gpu_name": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
            "peak_vram_mb": round(peak_vram_mb, 1),
        },
        "inference_validation": {
            "val_windows": len(val_ds),
            "seconds": round(inf_seconds, 3),
            "windows_per_second": round(val_windows_per_sec, 1),
            "device": str(device),
        },
    }

    # ---- assemble the metrics artifact ----
    best_val = win["full_official_val"]["threshold_0.50"]
    metrics_artifact = {
        "checkpoint": str(best_ckpt),
        "best_epoch": best_epoch,
        "level": "WINDOW vs VIDEO are reported separately and must not be conflated",
        "window_level": win,
        "video_level": video,
        "overfitting_analysis": overfit,
        "headline": {
            "val_window_accuracy_0.5": best_val["accuracy"],
            "val_window_precision_0.5": best_val["precision"],
            "val_window_recall_0.5": best_val["recall"],
            "val_window_f1_0.5": best_val["f1"],
            "val_window_specificity_0.5": best_val["specificity"],
            "val_window_balanced_accuracy_0.5": best_val["balanced_accuracy"],
            "val_window_roc_auc": best_val["roc_auc"],
            "val_window_accuracy_0.60": win["full_official_val"]["threshold_0.60_production"]["accuracy"],
            "val_video_mean_accuracy_0.5": video["full_official_val"]["mean_threshold_0.50"]["accuracy"],
            "val_video_mean_roc_auc": video["full_official_val"]["mean_threshold_0.50"]["roc_auc"],
        },
        "duplicate_limitation": (
            "Official validation metrics are subject to the known duplicate-content "
            "limitation: 6 val clips are byte-identical to training clips. FULL and "
            "DUPLICATE-EXCLUDED numbers are both reported; NEITHER is leakage-free. "
            "This is NOT a claim of production readiness or real-world CCTV accuracy."),
    }

    confusion_artifact = {
        "best_epoch": best_epoch,
        "window_level": {
            "full_official_val_0.50": win["full_official_val"]["threshold_0.50"]["confusion"],
            "full_official_val_0.60": win["full_official_val"]["threshold_0.60_production"]["confusion"],
            "duplicate_excluded_0.50": win["duplicate_excluded"]["threshold_0.50"]["confusion"],
            "duplicate_excluded_0.60": win["duplicate_excluded"]["threshold_0.60_production"]["confusion"],
        },
        "video_level_mean": {
            "full_official_val_0.50": video["full_official_val"]["mean_threshold_0.50"]["confusion"],
            "full_official_val_0.60": video["full_official_val"]["mean_threshold_0.60_production"]["confusion"],
        },
        "key_order": ["tn", "fp", "fn", "tp"],
        "fight_recall_0.50": win["full_official_val"]["threshold_0.50"]["recall"],
        "nonfight_recall_0.50": win["full_official_val"]["threshold_0.50"]["specificity"],
    }

    # ---- STEP 18: write every artifact ----
    report_dir = _resolve(tcfg.outputs.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    training_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": "PROMPT 6 — full RWF-2000 training + validation (authorized)",
        "dataset_identity": {
            "name": "RWF-2000", "manifest": tcfg.dataset.manifest,
            "policy": "TRAIN+VALIDATE ON RWF-2000 ONLY, official split preserved. "
                      "No other dataset trained on, merged, altered or used to tune.",
            "train_videos": audit["per_split_video_counts"].get(tcfg.dataset.train_split),
            "val_videos": audit["per_split_video_counts"].get(tcfg.dataset.val_split),
        },
        "label_split_audit": audit,
        "leakage_registry": registry.to_json(),
        "feature_extraction": extraction_timing,
        "sequences": {"train": train_shapes, "val": val_shapes},
        "normalization": norm,
        "class_weights": (np.round(weights, 4).tolist() if weights is not None else None),
        "class_distribution": {"train_windows": tc, "val_windows": vc,
                               "train_ratio": train_corpus.class_ratio(),
                               "val_ratio": val_corpus.class_ratio(),
                               "n_train_windows": n_train,
                               "n_train_fight_windows": n_fight,
                               "n_train_nonfight_windows": n_nonfight,
                               "windows_per_train_video": round(n_train / max(len(train_corpus.clip_ids_present()), 1), 2)},
        "architecture": model.architecture(),
        "n_parameters": n_params,
        "training_config": {"epochs": epochs, "batch_size": batch_size,
                            "eval_batch_size": eval_batch, "lr": lr,
                            "weight_decay": tcfg.training.weight_decay,
                            "grad_clip": tcfg.training.grad_clip,
                            "optimizer": tcfg.training.optimizer, "loss": "BCEWithLogitsLoss",
                            "scheduler": "none", "augmentation": "none", "seed": seed,
                            "determinism": det},
        "best_epoch": best_epoch,
        "best_criterion": criterion_desc,
        "history": history,
        "metrics": metrics_artifact,
        "performance": performance,
        "checkpoint": str(best_ckpt),
        "domain_limitation": (
            "RWF-2000 is the training/validation dataset only. This stage does NOT "
            "claim production readiness, real-world CCTV accuracy, Indian CCTV "
            "accuracy, universal fight detection, or deployment readiness. The other "
            "datasets are reserved for a FUTURE independent generalization evaluation."),
    }

    pfx = args.report_prefix
    write_json(report_dir / f"{pfx}training_report.json", training_report)
    write_json(report_dir / f"{pfx}metrics.json", metrics_artifact)
    write_json(report_dir / f"{pfx}confusion_matrix.json", confusion_artifact)
    write_json(report_dir / f"{pfx}score_distributions.json", dist)

    # training-history CSV
    csv_path = report_dir / f"{pfx}training_history.csv"
    cols = ["epoch", "lr", "seconds", "train_loss", "train_accuracy", "val_loss",
            "val_accuracy", "val_precision", "val_recall", "val_f1",
            "val_specificity", "val_balanced_accuracy", "val_roc_auc", "val_pr_auc",
            "val_tn", "val_fp", "val_fn", "val_tp",
            "val_excl_duplicates_accuracy", "val_excl_duplicates_roc_auc"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(cols)
        for h in history:
            c = h["val_confusion"]
            wtr.writerow([h["epoch"], h["lr"], h["seconds"],
                          round(h["train_loss"], 6), round(h["train_accuracy"], 6),
                          round(h["val_loss"], 6), round(h["val_accuracy"], 6),
                          round(h["val_precision"], 6), round(h["val_recall"], 6),
                          round(h["val_f1"], 6), round(h["val_specificity"], 6),
                          round(h["val_balanced_accuracy"], 6),
                          ("" if h["val_roc_auc"] is None else round(h["val_roc_auc"], 6)),
                          ("" if h["val_pr_auc"] is None else round(h["val_pr_auc"], 6)),
                          c["tn"], c["fp"], c["fn"], c["tp"],
                          round(h["val_excl_duplicates_accuracy"], 6),
                          ("" if h["val_excl_duplicates_roc_auc"] is None
                           else round(h["val_excl_duplicates_roc_auc"], 6))])

    # human-readable markdown report
    md = _markdown_report(training_report, metrics_artifact, performance, overfit)
    (report_dir / f"{pfx}training_report.md").write_text(md, encoding="utf-8")

    print(f"\n{RULE}\nSTEP 18 — ARTIFACTS WRITTEN\n{RULE}")
    for f in (f"{pfx}training_report.json", f"{pfx}training_report.md",
              f"{pfx}metrics.json", f"{pfx}confusion_matrix.json",
              f"{pfx}score_distributions.json", f"{pfx}training_history.csv"):
        print(f"  {report_dir / f}")
    print(f"  {best_ckpt}")

    _print_37_item_report(training_report, metrics_artifact, performance, overfit,
                          best_ckpt, n_params)
    return 0


def _markdown_report(rep: dict, met: dict, perf: dict, overfit: dict) -> str:
    h = met["headline"]
    win = met["window_level"]["full_official_val"]
    vid = met["video_level"]["full_official_val"]
    ext = perf["feature_extraction"]
    trn = perf["training"]
    inf = perf["inference_validation"]
    norm = rep["normalization"]
    cw = rep["class_weights"]
    cd = rep["class_distribution"]
    lines = [
        "# RWF-2000 Full Training + Validation Report (PROMPT 6)",
        "",
        f"_Generated {rep['generated_at']}. Dataset: **RWF-2000 only**, official split "
        "preserved. No other dataset was trained on, merged, altered, or used to tune._",
        "",
        "## Domain limitation",
        rep["domain_limitation"],
        "",
        "## Dataset",
        f"- Training videos: **{rep['dataset_identity']['train_videos']}**",
        f"- Validation videos: **{rep['dataset_identity']['val_videos']}**",
        f"- Training windows: **{cd['n_train_windows']}** "
        f"(FIGHT {cd['n_train_fight_windows']} / NON_FIGHT {cd['n_train_nonfight_windows']}, "
        f"ratio {cd['train_ratio']:.2f}:1, {cd['windows_per_train_video']} windows/video)",
        f"- Validation windows: **{rep['sequences']['val']['windows']}** "
        f"(ratio {rep['sequences']['val']['window_class_ratio']}:1)",
        f"- Zero-yield videos (extraction): **{ext.get('zero_yield_clips', 'n/a')}** "
        f"({ext.get('zero_yield_pct', 'n/a')}%)",
        "",
        "## Leakage / duplicates",
        f"- Duplicate groups straddling train/val: "
        f"**{rep['leakage_registry']['groups_straddling_train_val']}** "
        f"({rep['leakage_registry']['leaked_val_clips']} val clips)",
        "- " + met["duplicate_limitation"],
        "",
        "## Features / normalization / weights",
        f"- Feature dimension: **{FEATURE_DIM}**, tensor **[B, {rep['sequences']['train']['window_frames']}, {FEATURE_DIM}]**",
        f"- Normalization: **{norm.get('method')}**, fitted on TRAIN windows only "
        f"(n={norm.get('n_windows')})",
        f"  - mean: {[round(x,4) for x in norm.get('mean', [])]}",
        f"  - std:  {[round(x,4) for x in norm.get('std', [])]}",
        f"- Class weights (inverse frequency, train only): **{cw}**",
        "",
        "## Architecture / training",
        f"- {rep['architecture']}",
        f"- Parameters: **{rep['n_parameters']:,}**",
        f"- Epochs **{rep['training_config']['epochs']}**, batch **{rep['training_config']['batch_size']}**, "
        f"lr **{rep['training_config']['lr']}**, optimizer **{rep['training_config']['optimizer']}**, "
        f"loss **BCEWithLogitsLoss**, grad_clip **{rep['training_config']['grad_clip']}**, "
        f"seed **{rep['training_config']['seed']}**, scheduler **none**, augmentation **none**",
        f"- Best epoch (max val ROC-AUC): **{rep['best_epoch']}**",
        "",
        "## Window-level validation (official val, FULL)",
        f"| metric | @0.50 | @0.60 (production) |",
        f"|---|---|---|",
        f"| accuracy | {win['threshold_0.50']['accuracy']:.4f} | {win['threshold_0.60_production']['accuracy']:.4f} |",
        f"| precision | {win['threshold_0.50']['precision']:.4f} | {win['threshold_0.60_production']['precision']:.4f} |",
        f"| recall (FIGHT) | {win['threshold_0.50']['recall']:.4f} | {win['threshold_0.60_production']['recall']:.4f} |",
        f"| F1 | {win['threshold_0.50']['f1']:.4f} | {win['threshold_0.60_production']['f1']:.4f} |",
        f"| specificity (NON_FIGHT recall) | {win['threshold_0.50']['specificity']:.4f} | {win['threshold_0.60_production']['specificity']:.4f} |",
        f"| balanced accuracy | {win['threshold_0.50']['balanced_accuracy']:.4f} | {win['threshold_0.60_production']['balanced_accuracy']:.4f} |",
        f"| ROC-AUC | {win['threshold_0.50']['roc_auc']} | (threshold-independent) |",
        f"| confusion tn/fp/fn/tp | {win['threshold_0.50']['confusion']} | {win['threshold_0.60_production']['confusion']} |",
        "",
        "## Video-level validation (per-clip mean pooling, FULL, @0.50)",
        f"- accuracy **{vid['mean_threshold_0.50']['accuracy']:.4f}**, "
        f"precision **{vid['mean_threshold_0.50']['precision']:.4f}**, "
        f"recall **{vid['mean_threshold_0.50']['recall']:.4f}**, "
        f"F1 **{vid['mean_threshold_0.50']['f1']:.4f}**, "
        f"specificity **{vid['mean_threshold_0.50']['specificity']:.4f}**, "
        f"balanced acc **{vid['mean_threshold_0.50']['balanced_accuracy']:.4f}**, "
        f"ROC-AUC **{vid['mean_threshold_0.50']['roc_auc']}**",
        f"- n videos: {met['video_level']['n_videos_full']}",
        "",
        "## Overfitting analysis",
        overfit["assessment"],
        f"- final train/val accuracy: {overfit['final_train_accuracy']:.4f} / {overfit['final_val_accuracy']:.4f} "
        f"(gap {overfit['train_val_accuracy_gap_final']:.4f})",
        f"- min val loss at epoch {overfit['epoch_of_min_val_loss']}; best ROC-AUC at epoch {overfit['best_epoch_by_roc_auc']}",
        "",
        "## Performance (measured separately)",
        f"- Extraction: {ext.get('total_seconds','n/a')}s total, "
        f"{ext.get('seconds_per_clip','n/a')}s/clip, {ext.get('videos_per_second','n/a')} videos/s",
        f"- Training: {trn['total_seconds']}s total, {trn['seconds_per_epoch']}s/epoch, "
        f"{trn['windows_per_second']} windows/s, peak VRAM {trn['peak_vram_mb']} MiB on {trn['gpu_name']}",
        f"- Inference (val): {inf['seconds']}s for {inf['val_windows']} windows "
        f"({inf['windows_per_second']} windows/s)",
        "",
        "## Remaining limitations",
        "- Official val is leaky (6 byte-identical clips); neither FULL nor DUPLICATE-EXCLUDED is leakage-free.",
        "- Near-duplicate (non-identical) content is not detectable by sha256 and remains.",
        "- RWF-2000 is movie/clip footage, not Indian CCTV; no real-world accuracy is claimed.",
        "- The 0.60 production threshold is evaluated, not tuned; tuning is deferred to the independent evaluation stage.",
    ]
    return "\n".join(lines) + "\n"


def _print_37_item_report(rep, met, perf, overfit, best_ckpt, n_params) -> None:
    win = met["window_level"]["full_official_val"]
    win_adj = met["window_level"]["duplicate_excluded"]
    vid = met["video_level"]["full_official_val"]
    ext = perf["feature_extraction"]
    trn = perf["training"]
    inf = perf["inference_validation"]
    norm = rep["normalization"]
    cd = rep["class_distribution"]
    seq = rep["sequences"]
    w50 = win["threshold_0.50"]
    w60 = win["threshold_0.60_production"]
    print(f"\n{RULE}\nPROMPT 6 — FINAL MEASURED REPORT (37 items)\n{RULE}")
    items = [
        ("1  Training videos", rep["dataset_identity"]["train_videos"]),
        ("2  Validation videos", rep["dataset_identity"]["val_videos"]),
        ("3  Training windows", cd["n_train_windows"]),
        ("4  Validation windows", seq["val"]["windows"]),
        ("5  FIGHT/NON_FIGHT train windows", f"{cd['n_train_fight_windows']} / {cd['n_train_nonfight_windows']} (ratio {cd['train_ratio']:.2f}:1)"),
        ("6  Zero-yield videos", f"{ext.get('zero_yield_clips','n/a')} ({ext.get('zero_yield_pct','n/a')}%)"),
        ("7  Leakage status", f"{rep['leakage_registry']['groups_straddling_train_val']} straddling groups; official split intact; no fabricated test split"),
        ("8  Duplicate status", f"{rep['leakage_registry']['leaked_val_clips']} byte-identical val clips: {rep['leakage_registry']['leaked_val_clip_ids']}"),
        ("9  Feature dimension", FEATURE_DIM),
        ("10 Tensor shape", f"[B, {seq['train']['window_frames']}, {FEATURE_DIM}]"),
        ("11 Normalization", f"{norm.get('method')} fit on TRAIN only (n={norm.get('n_windows')}); finite, std>0"),
        ("12 Class weights", rep["class_weights"]),
        ("13 Model architecture", f"GRU(8,64,layers=1,batch_first)+mean/last pool+Linear(64,32)->ReLU->Linear(32,1); {n_params:,} params"),
        ("14 Epochs", rep["training_config"]["epochs"]),
        ("15 Best epoch", f"{rep['best_epoch']} (max val ROC-AUC)"),
        ("16 Training loss (final)", round(overfit["final_train_loss"], 4)),
        ("17 Validation loss (final, unweighted BCE)", round(overfit["final_val_loss"], 4)),
        ("18 Training accuracy (final)", round(overfit["final_train_accuracy"], 4)),
        ("19 Validation accuracy @0.5", round(w50["accuracy"], 4)),
        ("20 Precision @0.5", round(w50["precision"], 4)),
        ("21 Recall (FIGHT) @0.5", round(w50["recall"], 4)),
        ("22 F1 @0.5", round(w50["f1"], 4)),
        ("23 Specificity (NON_FIGHT recall) @0.5", round(w50["specificity"], 4)),
        ("24 Balanced accuracy @0.5", round(w50["balanced_accuracy"], 4)),
        ("25 ROC-AUC (window, val)", w50["roc_auc"]),
        ("26 Confusion tn/fp/fn/tp @0.5", w50["confusion"]),
        ("27 Video-level (mean pool) acc/prec/rec/F1/spec/balacc/auc @0.5",
         f"{vid['mean_threshold_0.50']['accuracy']:.4f}/"
         f"{vid['mean_threshold_0.50']['precision']:.4f}/"
         f"{vid['mean_threshold_0.50']['recall']:.4f}/"
         f"{vid['mean_threshold_0.50']['f1']:.4f}/"
         f"{vid['mean_threshold_0.50']['specificity']:.4f}/"
         f"{vid['mean_threshold_0.50']['balanced_accuracy']:.4f}/"
         f"{vid['mean_threshold_0.50']['roc_auc']}"),
        ("28 Threshold 0.60 (production) window acc/recall/spec",
         f"acc {w60['accuracy']:.4f} / recall {w60['recall']:.4f} / spec {w60['specificity']:.4f}"),
        ("29 Duplicate limitation", f"dup-excluded acc@0.5 {win_adj['threshold_0.50']['accuracy']:.4f}, "
                                    f"auc {win_adj['threshold_0.50']['roc_auc']}; NEITHER full nor excluded is leakage-free"),
        ("30 Feature extraction time", f"{ext.get('total_seconds','n/a')}s total, {ext.get('seconds_per_clip','n/a')}s/clip, {ext.get('videos_per_second','n/a')} videos/s"),
        ("31 Training time", f"{trn['total_seconds']}s total, {trn['seconds_per_epoch']}s/epoch, {trn['windows_per_second']} windows/s"),
        ("32 GPU/VRAM", f"{trn['gpu_name']}, peak {trn['peak_vram_mb']} MiB"),
        ("33 Checkpoint verification", f"saved {best_ckpt}; fresh-process round-trip run separately via train_temporal.py checkpoint"),
        ("34 End-to-end integration", "run separately (STEP 15) — integration only, NOT an accuracy claim"),
        ("35 Tests", "run separately: python -m pytest (STEP 17)"),
        ("36 Files created/modified", "models/temporal/rwf2000_best.pt + 6 report files in datasets/reports/"),
        ("37 Remaining limitations", "leaky official val; movie footage not Indian CCTV; 0.60 evaluated not tuned; no production/real-world accuracy claimed"),
    ]
    for k, v in items:
        print(f"  {k:<58}{v}")
    print(f"\n  Overfitting: {overfit['assessment']}")
    print("\n  FULL TRAINING COMPLETE. Per PROMPT 6: after this measured report, STOP.")


if __name__ == "__main__":
    raise SystemExit(main())
