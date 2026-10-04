"""Temporal fight-classifier pipeline: verification stages and (guarded) training.

Subcommands
-----------
  verify        run every PROMPT 4 verification stage on a small deterministic
                sample and write datasets/reports/rwf2000_pipeline_verification.json
                (label/split audit, leakage registry, sequence shapes, train-only
                normalization, forward/backward smoke test, tiny overfit,
                checkpoint round-trip, GPU benchmark). This is the default.
  smoke         Step 12 only: one forward + backward + optimizer step.
  overfit       Step 13 only: memorize 8-32 clips (a plumbing test, not accuracy).
  benchmark     Step 17 only: throughput + peak VRAM at several batch sizes.
  checkpoint    Step 14 only: load a checkpoint through the real inference path.
  train         the full run. REFUSES to start unless --confirm-full-training is
                passed, because PROMPT 4 forbids executing it at this stage.

Nothing here downloads data, modifies raw videos, merges datasets, fabricates a
test split, or tunes the 0.60 incident threshold.

Usage:
    python scripts/train_temporal.py verify
    python scripts/train_temporal.py verify --per-cell 20 --overfit-epochs 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles/pipes default to the locale codepage, which mangles the
# non-ASCII rules and dashes this report prints. Force UTF-8 on both streams.
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
    LABEL_FIGHT,
    LABEL_NON_FIGHT,
    TRAINABLE_LABELS,
    Corpus,
    TemporalFightDataset,
    apply_normalization,
    build_corpus,
    class_weights,
    fit_normalization,
)
from suraksha.training.extract import build_cache, select_cells
from suraksha.training.leakage import load_or_build_registry
from suraksha.training.metrics import (
    binary_cross_entropy_with_logits,
    compute_metrics,
    sigmoid,
)
from suraksha.training.model import HIDDEN_SIZE, build_gru
from suraksha.training.trainer import (
    benchmark as run_benchmark,
    evaluate,
    fit_and_report_normalization,
    inspect_checkpoint,
    make_loss,
    resolve_device,
    save_checkpoint,
    set_seed,
    smoke_test,
    tiny_overfit,
    train_one_epoch,
    write_json,
)

log = get_logger(__name__)

RULE = "=" * 92


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------
# Steps 3 / 4 / 9 — programmatic label, split and balance verification
# --------------------------------------------------------------------------

def audit_labels_and_splits(entries, tcfg) -> dict:
    """Assert the dataset invariants in code rather than trusting the audit doc."""
    problems: list[str] = []
    labels = Counter(e.label for e in entries)
    splits = Counter(e.split for e in entries)

    bad_labels = {k: v for k, v in labels.items() if k not in TRAINABLE_LABELS}
    if bad_labels:
        problems.append(f"non-trainable labels present: {bad_labels}")
    if "test" in splits:
        problems.append("a 'test' split exists — PROMPT 4 forbids fabricating one")

    # folder name must agree with the canonical label (RWF-2000 uses Fight/NonFight)
    folder_mismatch = []
    for e in entries:
        parts = {p.lower() for p in Path(e.path).parts}
        expected = "fight" if e.label == LABEL_FIGHT else "nonfight"
        if expected not in parts:
            folder_mismatch.append(e.clip_id)
    if folder_mismatch:
        problems.append(f"{len(folder_mismatch)} clip(s) whose folder disagrees with "
                        f"the label, e.g. {folder_mismatch[:3]}")

    # Step 4: a video belongs entirely to one split
    vid_splits = defaultdict(set)
    for e in entries:
        vid_splits[e.group_id or e.video_id or e.clip_id].add(e.split)
    crossing = {k: sorted(v) for k, v in vid_splits.items() if len(v) > 1}
    if crossing:
        problems.append(f"{len(crossing)} source video(s) span multiple splits "
                        f"(window-level leakage), e.g. {list(crossing.items())[:3]}")

    missing_meta = [e.clip_id for e in entries
                    if not e.effective_frames or not e.fps or not e.sha256]
    if missing_meta:
        problems.append(f"{len(missing_meta)} clip(s) missing frame_count/fps/sha256")

    per_split = {s: dict(Counter(e.label for e in entries if e.split == s))
                 for s in sorted(splits)}
    video_ratio = {}
    for s, c in per_split.items():
        pos, neg = c.get(LABEL_FIGHT, 0), c.get(LABEL_NON_FIGHT, 0)
        video_ratio[s] = {"FIGHT": pos, "NON_FIGHT": neg,
                          "ratio": round(pos / neg, 4) if neg else None}

    result = {
        "clips": len(entries),
        "labels": dict(labels),
        "splits": dict(splits),
        "per_split_video_counts": video_ratio,
        "unique_source_videos": len(vid_splits),
        "videos_crossing_splits": len(crossing),
        "clips_missing_metadata": len(missing_meta),
        "folder_label_mismatches": len(folder_mismatch),
        "unknown_or_ambiguous_labels": sum(v for k, v in labels.items()
                                           if k not in TRAINABLE_LABELS),
        "problems": problems,
        "passed": not problems,
    }
    print(f"\n{RULE}\nSTEP 3/4/9 — LABEL, SPLIT AND BALANCE AUDIT (programmatic)\n{RULE}")
    print(f"  clips                     {result['clips']}")
    print(f"  labels                    {result['labels']}")
    print(f"  splits                    {result['splits']}")
    for s, v in video_ratio.items():
        print(f"  {s:<24}{v['FIGHT']} FIGHT / {v['NON_FIGHT']} NON_FIGHT  "
              f"(ratio {v['ratio']})")
    print(f"  UNKNOWN/ambiguous labels  {result['unknown_or_ambiguous_labels']}")
    print(f"  videos crossing splits    {result['videos_crossing_splits']}")
    print(f"  clips missing metadata    {result['clips_missing_metadata']}")
    print(f"  folder/label mismatches   {result['folder_label_mismatches']}")
    print(f"  RESULT                    {'PASS' if result['passed'] else 'FAIL'}")
    for p in problems:
        print(f"    ! {p}")
    return result


# --------------------------------------------------------------------------
# Steps 10 / 11 — sequence sample and shape documentation
# --------------------------------------------------------------------------

def document_sequences(corpus: Corpus, tcfg, cache_dir: Path) -> dict:
    """Report the ACTUAL tensor shapes instead of assuming an ordering."""
    w = corpus.windows
    info = {
        "windows": len(corpus),
        "ndim": int(w.ndim),
        "tensor_shape": list(w.shape),
        "shape_meaning": ["N_windows", "T_frames_per_window", "feature_dim"],
        "dtype": str(w.dtype),
        "bytes_per_window": int(w[0].nbytes) if len(w) else 0,
        "corpus_mib": round(float(w.nbytes) / 1024 ** 2, 2),
        "window_frames": tcfg.dataset.window_frames,
        "stride_frames": tcfg.dataset.stride_frames,
        "feature_dim": FEATURE_DIM,
        "clips_with_windows": len(corpus.clip_ids_present()),
        "clips_zero_yield": len(corpus.stats.clips_zero_yield),
        "clips_missing_cache": len(corpus.stats.clips_missing),
        "labels_rejected": corpus.stats.labels_rejected,
        "window_class_counts": corpus.counts(),
        "window_class_ratio": round(corpus.class_ratio(), 4),
        "gate_passed_windows": int(corpus.gate_passed.sum()),
        "gate_passed_pct": round(100 * float(corpus.gate_passed.mean()), 2) if len(corpus) else 0.0,
        "finite": bool(np.isfinite(w).all()) if len(w) else True,
        "all_zero_rows_pct": round(100 * float((np.abs(w).sum(axis=(1, 2)) == 0).mean()), 2)
        if len(w) else 0.0,
    }
    per_feature = []
    if len(w):
        flat = w.reshape(-1, FEATURE_DIM)
        names = ["rel_dist", "iou", "norm_motion(bw/s)", "conf_a", "conf_b",
                 "width_a/frame_w", "width_b/frame_w", "bias"]
        for i in range(FEATURE_DIM):
            col = flat[:, i]
            per_feature.append({
                "index": i, "name": names[i],
                "mean": round(float(col.mean()), 4), "std": round(float(col.std()), 4),
                "min": round(float(col.min()), 4), "max": round(float(col.max()), 4),
                "nonzero_pct": round(100 * float((col != 0).mean()), 2),
            })
    info["per_feature_stats"] = per_feature

    print(f"\n{RULE}\nSTEP 10/11 — SEQUENCE SHAPES (measured, not assumed)\n{RULE}")
    print(f"  window tensor           {tuple(w.shape)}  {w.dtype}")
    print(f"  axis meaning            {info['shape_meaning']}")
    print(f"  bytes/window            {info['bytes_per_window']}  "
          f"corpus={info['corpus_mib']} MiB")
    print(f"  window_frames/stride    {info['window_frames']}/{info['stride_frames']}")
    print(f"  clips with windows      {info['clips_with_windows']}  "
          f"zero-yield={info['clips_zero_yield']}  missing-cache={info['clips_missing_cache']}")
    print(f"  class counts            {info['window_class_counts']}  "
          f"ratio={info['window_class_ratio']}:1")
    print(f"  gate-passed windows     {info['gate_passed_windows']} "
          f"({info['gate_passed_pct']}%) at motion_energy_threshold")
    print(f"  all-finite              {info['finite']}   "
          f"all-zero windows={info['all_zero_rows_pct']}%")
    if per_feature:
        print(f"  {'#':>2} {'feature':<18}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'nonzero%':>10}")
        for f in per_feature:
            print(f"  {f['index']:>2} {f['name']:<18}{f['mean']:>9.4f}{f['std']:>9.4f}"
                  f"{f['min']:>9.4f}{f['max']:>9.4f}{f['nonzero_pct']:>10.2f}")
    return info


# --------------------------------------------------------------------------
# Step 14 — checkpoint round-trip through the REAL inference path
# --------------------------------------------------------------------------

def checkpoint_roundtrip(ckpt_path: Path, normalization: dict, tcfg,
                         sample_window: np.ndarray | None) -> dict:
    """Prove the trained artifact is loadable by the shipped inference code."""
    from suraksha.fight.recognizer import FightRecognizer
    from suraksha.fight.temporal_classifier import (
        HeuristicTemporalScorer, TorchTemporalClassifier, build_scorer,
    )

    checks: dict[str, bool] = {}
    notes: list[str] = []
    info = inspect_checkpoint(ckpt_path)

    scorer = build_scorer(str(ckpt_path), device=tcfg.training.device)
    checks["build_scorer_returns_torch_classifier"] = isinstance(
        scorer, TorchTemporalClassifier)
    if not checks["build_scorer_returns_torch_classifier"]:
        notes.append("build_scorer silently fell back to HeuristicTemporalScorer")

    checks["feature_dim_matches"] = int(info["feature_dim"] or 0) == FEATURE_DIM
    checks["hidden_size_matches"] = int(info["hidden_size"] or 0) == HIDDEN_SIZE
    checks["normalization_persisted"] = (info["normalization"] or {}).get("method") \
        == normalization.get("method")

    torch_ref = None
    if normalization.get("method") == "standard":
        got_mean = np.asarray(getattr(scorer, "norm_mean", []), dtype=np.float32)
        want_mean = np.asarray(normalization["mean"], dtype=np.float32)
        got_std = np.asarray(getattr(scorer, "norm_std", []), dtype=np.float32)
        want_std = np.asarray(normalization["std"], dtype=np.float32)
        checks["normalization_roundtrips"] = bool(
            got_mean.shape == want_mean.shape and got_std.shape == want_std.shape
            and np.allclose(got_mean, want_mean, atol=1e-6)
            and np.allclose(got_std, want_std, atol=1e-6))
    else:
        checks["normalization_roundtrips"] = True
        notes.append("normalization method is not 'standard'; the inference "
                     "classifier only re-applies 'standard' — see notes")
        if normalization.get("method") not in ("none", None):
            checks["normalization_roundtrips"] = False
            notes.append("MISMATCH: fitted method "
                         f"{normalization.get('method')!r} cannot be applied at inference")

    arch_ok = True
    if sample_window is not None and len(sample_window):
        w = np.ascontiguousarray(sample_window, dtype=np.float32)
        s_inf = float(scorer.score(w))
        checks["score_in_unit_interval"] = 0.0 <= s_inf <= 1.0

        # Recompute the same score with the TRAINING-side code path. Agreement
        # proves training and inference apply identical weights and an identical
        # normalization transform — the property Step 14 exists to establish.
        #
        # The comparison must run on the SAME device as the scorer. A GRU
        # evaluated on CUDA and on CPU differs by ~1e-4 in float32 because the
        # reduction order differs; that is real numerical behaviour, not a
        # pipeline defect, so it is measured and reported separately rather than
        # being allowed to fail the compatibility check.
        import torch

        def score_with(device) -> float:
            m = build_gru(FEATURE_DIM, HIDDEN_SIZE).to(device)
            st = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            m.load_state_dict(st["model_state"])
            m.eval()
            xn = apply_normalization(w[None, ...], normalization)
            with torch.no_grad():
                return float(torch.sigmoid(
                    m(torch.from_numpy(xn).to(device))).item())

        s_same_device = score_with(scorer.device)
        s_cpu = score_with(torch.device("cpu"))
        same_diff = abs(s_inf - s_same_device)
        cross_diff = abs(s_inf - s_cpu)
        torch_ref = {
            "inference_score": s_inf,
            "training_side_score_same_device": s_same_device,
            "training_side_score_cpu": s_cpu,
            "same_device_abs_diff": same_diff,
            "cpu_vs_inference_device_abs_diff": cross_diff,
            "compared_on": str(scorer.device),
        }
        checks["train_and_inference_scores_agree"] = same_diff < 1e-6
        arch_ok = checks["train_and_inference_scores_agree"]
        notes.append(f"same-device diff {same_diff:.2e}; CPU vs {scorer.device} diff "
                     f"{cross_diff:.2e} (float32 reduction order — scores are not "
                     "bit-identical across devices)")
        if cross_diff > 1e-2:
            notes.append(f"WARNING: CPU/GPU drift {cross_diff:.2e} is large enough to "
                         "flip a decision near the threshold")

    # The recognizer must accept the checkpoint through the normal config path.
    cfg = load_config()
    cfg.fight.temporal.model_weights = str(ckpt_path)
    rec = FightRecognizer(cfg.fight, (360, 640), device=tcfg.training.device)
    checks["recognizer_uses_trained_scorer"] = isinstance(
        rec.scorer, TorchTemporalClassifier) and not isinstance(
        rec.scorer, HeuristicTemporalScorer)
    checks["architecture_recorded"] = bool(info.get("architecture"))

    result = {
        "checkpoint": str(ckpt_path),
        "size_kib": round(ckpt_path.stat().st_size / 1024, 1),
        "n_parameters": info["n_parameters"],
        "top_level_keys": info["top_level_keys"],
        "param_tensors": {k: list(v) for k, v in (info["param_tensors"] or {}).items()},
        "checks": checks,
        "score_crosscheck": torch_ref,
        "passed": all(checks.values()) and arch_ok,
        "notes": notes,
    }

    print(f"\n{RULE}\nSTEP 14 — CHECKPOINT -> INFERENCE ROUND-TRIP (mandatory)\n{RULE}")
    print(f"  checkpoint              {result['checkpoint']} ({result['size_kib']} KiB)")
    print(f"  parameters              {result['n_parameters']:,}")
    print(f"  top-level keys          {result['top_level_keys']}")
    for k, v in (info["param_tensors"] or {}).items():
        print(f"    {k:<28}{list(v)}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if torch_ref:
        print(f"  inference score         {torch_ref['inference_score']:.6f}")
        print(f"  training-side (same dev){torch_ref['training_side_score_same_device']:.6f}"
              f"  diff {torch_ref['same_device_abs_diff']:.2e}  "
              f"[{torch_ref['compared_on']}]")
        print(f"  training-side (cpu)     {torch_ref['training_side_score_cpu']:.6f}"
              f"  diff {torch_ref['cpu_vs_inference_device_abs_diff']:.2e}")
    for n in notes:
        print(f"  note: {n}")
    print(f"  RESULT                  {'PASS' if result['passed'] else 'FAIL'}")
    return result


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def load_split_entries(entries, split: str, tcfg, per_cell: int | None = None,
                       cells_labels=None) -> list:
    labels = cells_labels or list(tcfg.dataset.labels)
    cells = [(split, l) for l in labels]
    if per_cell:
        return select_cells(entries, per_cell, cells)
    return sorted((e for e in entries if e.split == split and e.label in labels),
                  key=lambda e: e.clip_id)


def with_leak_pairs(entries, train_pool, val_pool, registry) -> tuple[list, list]:
    """Force the known duplicate pairs INTO the verification sample.

    The default sample is the first N clip_ids per cell, and all 6 leaked val
    clips sit in the rwf2000_018xx-019xx range, so a small sample would never
    contain them — the leakage-adjusted metric would then exclude zero windows
    and prove nothing. Adding both halves of every straddling pair makes the
    mechanism demonstrable: the model trains on one copy and is scored on the
    byte-identical other.
    """
    wanted_val = set(registry.leaked_val_clip_ids)
    wanted_train = set(registry.leaked_train_clip_ids)
    have_val = {e.clip_id for e in val_pool}
    have_train = {e.clip_id for e in train_pool}

    extra_val = [e for e in entries
                 if e.clip_id in wanted_val and e.clip_id not in have_val]
    extra_train = [e for e in entries
                   if e.clip_id in wanted_train and e.clip_id not in have_train]

    train_pool = sorted(train_pool + extra_train, key=lambda e: e.clip_id)
    val_pool = sorted(val_pool + extra_val, key=lambda e: e.clip_id)
    return train_pool, val_pool


def cmd_verify(args) -> int:
    t0 = time.time()
    tcfg = load_training_config(args.config or None)
    cfg = load_config()

    problems = tcfg.validate_against_app(cfg)
    print(f"\n{RULE}\nSTEP 2/7 — CONFIG CONSISTENCY WITH THE SHIPPED INFERENCE PATH\n{RULE}")
    print(f"  training config         {_resolve('configs/training.yaml')}")
    print(f"  window/stride/feature   {tcfg.dataset.window_frames}/"
          f"{tcfg.dataset.stride_frames}/{tcfg.dataset.feature_dim}")
    print(f"  app fight.temporal      {cfg.fight.temporal.window_frames}/"
          f"{cfg.fight.temporal.stride_frames}  score_threshold="
          f"{cfg.fight.temporal.score_threshold} (NOT tuned here)")
    print(f"  model                   GRU(in={tcfg.model.feature_dim}, "
          f"hidden={tcfg.model.hidden_size}, layers={tcfg.model.num_layers}, "
          f"bidir={tcfg.model.bidirectional}, dropout={tcfg.model.dropout}) "
          f"head={tcfg.model.head_hidden}")
    print(f"  RESULT                  {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print(f"    ! {p}")
    if problems:
        return 2

    entries = load_manifest(tcfg.dataset.manifest).entries
    audit = audit_labels_and_splits(entries, tcfg)
    if not audit["passed"]:
        print("\nrefusing to continue: dataset invariants failed")
        return 2

    # ---- leakage registry (must exist before any val metric is quoted) ----
    registry, reg_path = load_or_build_registry(
        tcfg.dataset.manifest, _resolve(tcfg.leakage.registry_path))
    print(f"\n{RULE}\nKNOWN DUPLICATE-CONTENT LEAKAGE IN THE OFFICIAL SPLIT\n{RULE}")
    print(f"  registry                {reg_path}")
    for k, v in registry.summary().items():
        if k not in ("note", "leaked_val_clip_ids"):
            print(f"  {k:<28}{v}")
    print(f"  leaked val clips        {registry.leaked_val_clip_ids}")
    print(f"  leaked train clips      {registry.leaked_train_clip_ids}")
    for g in registry.straddling_groups:
        print(f"    {g.sha256[:12]}  {g.label:<10}{g.clip_ids}  splits={g.splits}")
    print("  NOTE: the official val set is NOT content-disjoint from train. Every")
    print("        validation number below is reported twice and neither is clean.")

    # ---- Step 10: deterministic sample + cache ----
    per_cell = args.per_cell or tcfg.dataset.sample.survey_videos_per_cell
    cache_dir = _resolve(tcfg.dataset.feature_cache_dir)
    ex = tcfg.dataset.extraction
    train_entries = load_split_entries(entries, tcfg.dataset.train_split, tcfg, per_cell)
    val_entries = load_split_entries(entries, tcfg.dataset.val_split, tcfg, per_cell)
    if not args.no_leak_pairs:
        n_tr, n_val = len(train_entries), len(val_entries)
        train_entries, val_entries = with_leak_pairs(
            entries, train_entries, val_entries, registry)
        added = (len(train_entries) - n_tr, len(val_entries) - n_val)
        if any(added):
            print(f"  + forced the known duplicate pairs into the sample: "
                  f"{added[0]} train clip(s), {added[1]} val clip(s)")
    sample = train_entries + val_entries

    print(f"\n{RULE}\nSTEP 10 — FEATURE EXTRACTION ON A DETERMINISTIC SAMPLE\n{RULE}")
    print(f"  sample                  {len(sample)} clips "
          f"({per_cell} per split/label cell), first clip_ids in sorted order")
    build = build_cache(
        sample, cache_dir, manifest=tcfg.dataset.manifest,
        project_root=PROJECT_ROOT, app_config=cfg,
        frame_shape=(ex.frame_height, ex.frame_width), device=ex.device,
        force=args.force_extract, max_frames=ex.max_frames_per_clip,
    )
    bs = build.summary()
    for k in ("requested", "built", "reused", "failed", "windows",
              "zero_yield_clips", "zero_yield_pct", "seconds_per_clip"):
        print(f"  {k:<22}{bs[k]}")
    note_counts = Counter(r.note or "ok" for r in build.results)
    print(f"  notes                   {dict(note_counts)}")
    zero_by_cell = defaultdict(lambda: [0, 0])
    for r in build.results:
        z = zero_by_cell[(r.split, r.label)]
        z[0] += 1
        z[1] += 1 if r.zero_yield else 0
    print(f"  {'cell':22s}{'clips':>7s}{'zero-yield':>12s}{'windows':>9s}")
    for key in sorted(zero_by_cell):
        c = zero_by_cell[key]
        wn = sum(r.n_windows for r in build.results
                 if (r.split, r.label) == key)
        print(f"  {key[0] + '/' + key[1]:22s}{c[0]:7d}"
              f"{c[1]:7d} ({100 * c[1] / max(c[0], 1):4.1f}%){wn:9d}")

    # ---- corpora ----
    train_corpus = build_corpus(train_entries, cache_dir, tcfg.dataset.manifest,
                                split=tcfg.dataset.train_split)
    val_corpus = build_corpus(val_entries, cache_dir, tcfg.dataset.manifest,
                              split=tcfg.dataset.val_split)
    if len(train_corpus) == 0 or len(val_corpus) == 0:
        print("\nFAIL: empty corpus — run scripts/extract_features.py first")
        return 2

    train_shapes = document_sequences(train_corpus, tcfg, cache_dir)
    print(f"\n  (val corpus: {len(val_corpus)} windows, "
          f"{len(val_corpus.clip_ids_present())} clips, "
          f"ratio {val_corpus.class_ratio():.2f}:1)")

    # ---- Step 5: normalization fitted on TRAIN ONLY ----
    print(f"\n{RULE}\nSTEP 5 — NORMALIZATION FITTED ON TRAINING DATA ONLY\n{RULE}")
    norm = fit_and_report_normalization(train_corpus.windows, tcfg.normalization.method)
    val_stats = fit_normalization(val_corpus.windows, tcfg.normalization.method)
    drift = None
    if norm.get("method") == "standard":
        drift = {
            "mean_max_abs_diff": float(np.max(np.abs(
                np.asarray(norm["mean"]) - np.asarray(val_stats["mean"])))),
            "std_max_abs_diff": float(np.max(np.abs(
                np.asarray(norm["std"]) - np.asarray(val_stats["std"])))),
        }
        print(f"  val-fitted stats computed ONLY to quantify distribution drift;")
        print(f"  they are discarded and never used to transform any data.")
        print(f"  mean max|train-val| = {drift['mean_max_abs_diff']:.4f}   "
              f"std max|train-val| = {drift['std_max_abs_diff']:.4f}")
    print(f"  persisted with checkpoint: {norm.get('method')} "
          f"(fitted_on={norm.get('fitted_on', 'n/a')}, n_windows={norm.get('n_windows')})")

    # ---- Step 6: augmentation ----
    print(f"\n{RULE}\nSTEP 6 — AUGMENTATION\n{RULE}")
    print(f"  enabled                 {tcfg.augmentation.enabled}")
    print("  NO feature-space augmentation is implemented, and none is invented.")
    print("  data/sequences.py provides horizontal flip + brightness jitter, but")
    print("  those are PIXEL transforms for the unused (T,128,128,3) clip path.")
    print("  The GRU consumes 8 kinematic features per frame: flipping a bbox pair")
    print("  leaves relative distance / IoU / flow magnitude unchanged, and")
    print("  brightness does not exist in this representation.")
    print("  Validation is deterministic and unaugmented by construction.")

    # ---- Step 9: class weighting from the MEASURED distribution ----
    print(f"\n{RULE}\nSTEP 9 — CLASS DISTRIBUTION AND WEIGHTING\n{RULE}")
    weights = class_weights(train_corpus.labels) if tcfg.training.balanced_class_weights else None
    tc = train_corpus.counts()
    vc = val_corpus.counts()
    print(f"  video-level  train      {audit['per_split_video_counts'][tcfg.dataset.train_split]}")
    print(f"  video-level  val        {audit['per_split_video_counts'][tcfg.dataset.val_split]}")
    print(f"  WINDOW-level train      {tc}  ratio {train_corpus.class_ratio():.2f}:1")
    print(f"  WINDOW-level val        {vc}  ratio {val_corpus.class_ratio():.2f}:1")
    print("  The imbalance is structural: emitting a window requires two tracked")
    print("  persons overlapping by proximity_iou>=0.05, which is itself correlated")
    print("  with fighting. Balanced videos do NOT give balanced windows.")
    if weights is not None:
        print(f"  class weights APPLIED   {np.round(weights, 4).tolist()} "
              f"(justified by the measured ratio above)")
    else:
        print("  class weights NOT applied (ratio under 1.25:1, or disabled)")

    # ---- Step 12: smoke test ----
    print(f"\n{RULE}")
    smoke = smoke_test(batch_size=min(8, max(2, args.smoke_batch)),
                       window_frames=tcfg.dataset.window_frames,
                       seed=tcfg.training.seed, device=tcfg.training.device)
    print(smoke.table())
    if not smoke.passed:
        print("\nFAIL: smoke test did not pass — stopping before any training")
        return 2

    # ---- Step 13: tiny overfit on a small deterministic subset ----
    print(f"\n{RULE}")
    ov_cell = args.overfit_per_cell or tcfg.dataset.sample.overfit_videos_per_cell
    ov_train_entries = load_split_entries(entries, tcfg.dataset.train_split, tcfg, ov_cell)
    ov_val_entries = load_split_entries(entries, tcfg.dataset.val_split, tcfg,
                                        tcfg.dataset.sample.overfit_val_videos_per_cell)
    ov_train = build_corpus(ov_train_entries, cache_dir, tcfg.dataset.manifest,
                            split=tcfg.dataset.train_split)
    ov_val = build_corpus(ov_val_entries, cache_dir, tcfg.dataset.manifest,
                          split=tcfg.dataset.val_split)
    overfit = None
    ckpt_path = _resolve(tcfg.outputs.checkpoint_dir) / "verification_overfit.pt"
    ov_sample_window = None
    if len(ov_train) >= 4:
        ov_ds = TemporalFightDataset(ov_train, norm)
        ov_val_ds = TemporalFightDataset(ov_val, norm) if len(ov_val) else None
        overfit = tiny_overfit(
            ov_ds, epochs=args.overfit_epochs, batch_size=args.overfit_batch,
            lr=args.overfit_lr, device=tcfg.training.device,
            seed=tcfg.training.seed, n_videos=len(ov_train.clip_ids_present()),
            val_ds=ov_val_ds,
        )
        print(overfit.table())
        if not overfit.passed:
            print("\nSTOP: the model failed to overfit a tiny sample. Per PROMPT 4")
            print("      Step 13 this must be diagnosed before any full training.")
            return 3

        # Checkpoint the artifact Step 13 just produced, so Step 14 round-trips a
        # REAL trained checkpoint rather than a randomly initialized one.
        ov_sample_window = ov_val.windows[0] if len(ov_val) else ov_train.windows[0]
        save_checkpoint(
            ckpt_path, overfit.model, norm,
            architecture=overfit.model.architecture(),
            training={"purpose": "verification tiny-overfit artifact — NOT a trained model",
                      "epochs": overfit.epochs, "batch_size": args.overfit_batch,
                      "lr": args.overfit_lr, "seed": tcfg.training.seed,
                      "loss": "BCEWithLogitsLoss", "optimizer": "adam",
                      "class_weights": overfit.class_weights},
            metrics={"overfit_final_loss": overfit.final_loss,
                     "overfit_final_accuracy": overfit.final_accuracy,
                     "warning": "memorization of <=32 clips; not a performance claim"},
            dataset_info={"manifest": tcfg.dataset.manifest,
                          "clips": ov_train.clip_ids_present(),
                          "windows": len(ov_train),
                          "window_frames": tcfg.dataset.window_frames,
                          "stride_frames": tcfg.dataset.stride_frames,
                          "normalization_fitted_on": "train sample of this verification run"},
        )
    else:
        print("TINY OVERFIT TEST — SKIPPED")
        print(f"  only {len(ov_train)} windows cached for {ov_cell} clips/cell; "
              "increase --overfit-per-cell or run scripts/extract_features.py")

    # ---- Step 14: checkpoint round-trip ----
    roundtrip = None
    if ckpt_path.exists():
        roundtrip = checkpoint_roundtrip(
            ckpt_path, norm, tcfg,
            ov_sample_window if ov_sample_window is not None else val_corpus.windows[0])
        if not roundtrip["passed"]:
            return 4

    # ---- Step 8: evaluation protocol actually executed on the sample ----
    print(f"\n{RULE}\nSTEP 8 — EVALUATION PROTOCOL EXECUTED ON THE SAMPLE\n{RULE}")
    print(f"  This is a PROTOCOL-EXECUTION check on a {len(sample)}-clip deterministic")
    print("  sample with a model that was overfit on <=32 clips. It is NOT validation")
    print("  performance, NOT production accuracy, and NOT a model claim.")
    eval_block = {}
    if roundtrip and roundtrip["passed"]:
        import torch
        device = resolve_device(tcfg.training.device)
        model = build_gru(FEATURE_DIM, tcfg.model.hidden_size).to(device)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        model.eval()

        val_ds = TemporalFightDataset(val_corpus, norm)
        logits = _all_logits(model, val_ds, device)
        thr = tcfg.evaluation.threshold
        m_full = compute_metrics(logits, val_corpus.labels,
                                 loss=_mean_bce(logits, val_corpus.labels), threshold=thr)

        keep = val_corpus.mask_excluding_clips(registry.leaked_val_clip_ids)
        n_leaked = int((~keep).sum())
        m_adj = compute_metrics(logits[keep], val_corpus.labels[keep],
                                loss=_mean_bce(logits[keep], val_corpus.labels[keep]),
                                threshold=thr)

        print(m_full.table("official val sample — ALL windows"))
        print()
        print(m_adj.table(f"official val sample — EXCLUDING {n_leaked} window(s) from "
                          "known duplicate-content clips"))

        # Show the leaked clips individually so the effect is visible rather than
        # absorbed into an aggregate.
        leaked_rows = []
        if n_leaked:
            print(f"\n  per-clip detail for the {len(registry.leaked_val_clip_ids)} "
                  "known duplicate val clips present in this sample:")
            print(f"    {'clip':<16}{'label':<11}{'wins':>5}{'mean p':>9}"
                  f"{'mean logit':>12}{'pred@0.5':>10}")
            for cid in registry.leaked_val_clip_ids:
                sel = np.array([c == cid for c in val_corpus.clip_ids])
                if not sel.any():
                    continue
                p = sigmoid(logits[sel])
                lab = "FIGHT" if val_corpus.labels[sel][0] == 1 else "NON_FIGHT"
                pred = "FIGHT" if float((logits[sel] > thr).mean()) >= 0.5 else "NON_FIGHT"
                leaked_rows.append({"clip_id": cid, "label": lab,
                                    "windows": int(sel.sum()),
                                    "mean_probability": round(float(p.mean()), 4),
                                    "mean_logit": round(float(logits[sel].mean()), 4),
                                    "majority_prediction": pred})
                print(f"    {cid:<16}{lab:<11}{int(sel.sum()):5d}{p.mean():9.4f}"
                      f"{logits[sel].mean():12.4f}{pred:>10}")
        else:
            print("\n  no known duplicate val clip is present in this sample "
                  "(use the default --no-leak-pairs=off behaviour to include them)")

        print("\n  Neither number is leakage-free: excluding the known duplicate")
        print("  clips removes only the byte-identical overlap that sha256 can detect.")
        print("  Near-duplicate (not identical) content would remain undetected.")
        print("  And this model memorized <=32 clips, so these figures describe the")
        print("  evaluation PROTOCOL, not the classifier. A roc_auc near or below 0.5")
        print(f"  is the expected signature of memorization under this sample's")
        print(f"  {val_corpus.class_ratio():.1f}:1 val window imbalance — not evidence of a")
        print("  wiring fault, which the Step 14 round-trip above establishes directly.")
        eval_block = {
            "sample_val_all": m_full.as_dict(),
            "sample_val_excluding_known_duplicates": m_adj.as_dict(),
            "windows_excluded": n_leaked,
            "leaked_val_clips_present": leaked_rows,
            "leaked_val_clips_with_windows": len(leaked_rows),
            "leaked_val_clips_zero_yield": len(registry.leaked_val_clip_ids) - len(leaked_rows),
            "threshold": thr,
            "val_window_class_ratio": round(val_corpus.class_ratio(), 4),
            "caveat": (f"{len(sample)}-clip sample, model overfit on <=32 clips. "
                       "Protocol-execution check only. Not validation performance, not "
                       "production accuracy. The official val split is leaky; neither "
                       "figure is leakage-free."),
        }

    # ---- Step 17: benchmark ----
    print(f"\n{RULE}")
    bench = run_benchmark(batch_sizes=tuple(args.bench_batches or tcfg.benchmark.batch_sizes),
                          window_frames=tcfg.dataset.window_frames,
                          iters=args.bench_iters, warmup=tcfg.benchmark.warmup,
                          device=tcfg.training.device,
                          corpus_sizes=tuple(tcfg.benchmark.corpus_sizes))
    for b in bench:
        print(b.table())
        print()

    # ---- Step 15: reproducibility statement ----
    det = set_seed(tcfg.training.seed, tcfg.training.deterministic_cuda)
    print(f"\n{RULE}\nSTEP 15 — REPRODUCIBILITY\n{RULE}")
    for k, v in det.items():
        print(f"  {k:<34}{v}")
    print("  Bit-for-bit reproducibility is NOT claimed on CUDA unless")
    print("  torch_deterministic_algorithms is True above; cudnn non-deterministic")
    print("  kernels and atomicAdd reductions remain a source of run-to-run drift.")

    # ---- Step 19: the full-training command, printed and NOT executed ----
    full_cmd = print_full_training_command(tcfg, args)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": "PROMPT 4 — pipeline verification (no full training performed)",
        "elapsed_seconds": round(time.time() - t0, 1),
        "config_consistency": {"problems": problems},
        "label_split_audit": audit,
        "leakage_registry": registry.to_json(),
        "feature_cache": bs,
        "train_sequences": train_shapes,
        "val_windows": len(val_corpus),
        "val_clips": len(val_corpus.clip_ids_present()),
        "normalization": norm,
        "normalization_drift_train_vs_val": drift,
        "augmentation": {"enabled": tcfg.augmentation.enabled,
                         "implemented": False,
                         "reason": "no feature-space augmentation exists; pixel-space "
                                   "augmentation in data/sequences.py does not apply"},
        "class_distribution": {"train_windows": tc, "val_windows": vc,
                               "train_ratio": train_corpus.class_ratio(),
                               "val_ratio": val_corpus.class_ratio(),
                               "weights_applied": (np.round(weights, 4).tolist()
                                                   if weights is not None else None)},
        "smoke_test": smoke.as_dict(),
        "tiny_overfit": ({"passed": overfit.passed, "n_videos": overfit.n_videos,
                          "n_windows": overfit.n_windows, "epochs": overfit.epochs,
                          "initial_loss": overfit.initial_loss,
                          "final_loss": overfit.final_loss,
                          "loss_reduction": overfit.loss_reduction,
                          "final_accuracy": overfit.final_accuracy,
                          "notes": overfit.notes} if overfit else {"skipped": True}),
        "checkpoint_roundtrip": roundtrip,
        "evaluation_protocol_check": eval_block,
        "benchmark": [b.as_dict() for b in bench],
        "reproducibility": det,
        "full_training_command": full_cmd,
        "full_training_executed": False,
    }
    out = _resolve(tcfg.outputs.report_dir) / tcfg.outputs.verification_report
    write_json(out, report)
    print(f"\n{RULE}\nwrote {out}")
    print("FULL TRAINING WAS NOT STARTED.")
    return 0


def _all_logits(model, ds, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            xb = ds.x[i:i + 512].to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _mean_bce(logits: np.ndarray, labels: np.ndarray) -> float:
    """Unweighted BCE from raw logits — the honest evaluation loss.

    Training may use class weights; the reported loss must not, or the number
    stops being comparable across runs with different weighting.
    """
    if len(labels) == 0:
        return float("nan")
    return binary_cross_entropy_with_logits(np.asarray(logits), np.asarray(labels))


def print_full_training_command(tcfg, args) -> list[str]:
    """Step 19 — print the command that WOULD start full training. Do not run it."""
    py = ".venv/Scripts/python.exe"
    cmds = [
        f"# 1) build the feature cache for the whole official split (~150 min, resumable)",
        f"{py} scripts/extract_features.py --all --json-out "
        f"{tcfg.outputs.report_dir}/rwf2000_feature_cache_full.json",
        f"",
        f"# 2) train (guarded: refuses without --confirm-full-training)",
        f"{py} scripts/train_temporal.py train --confirm-full-training "
        f"--epochs {tcfg.training.epochs} --batch-size {tcfg.training.batch_size} "
        f"--lr {tcfg.training.lr} --seed {tcfg.training.seed}",
    ]
    print(f"\n{RULE}\nSTEP 19 — FULL-TRAINING COMMAND (PRINTED, NOT EXECUTED)\n{RULE}")
    for c in cmds:
        print(f"  {c}")
    print("\n  PROMPT 4 forbids running these at this stage. Nothing was executed.")
    return [c for c in cmds if c and not c.startswith("#")]


def cmd_smoke(args) -> int:
    tcfg = load_training_config(args.config or None)
    r = smoke_test(batch_size=args.smoke_batch, window_frames=tcfg.dataset.window_frames,
                   seed=tcfg.training.seed, device=args.device or tcfg.training.device)
    print(RULE)
    print(r.table())
    return 0 if r.passed else 2


def cmd_benchmark(args) -> int:
    tcfg = load_training_config(args.config or None)
    res = run_benchmark(batch_sizes=tuple(args.bench_batches or tcfg.benchmark.batch_sizes),
                        window_frames=tcfg.dataset.window_frames,
                        iters=args.bench_iters, warmup=tcfg.benchmark.warmup,
                        device=args.device or tcfg.training.device,
                        corpus_sizes=tuple(tcfg.benchmark.corpus_sizes))
    print(RULE)
    for b in res:
        print(b.table())
        print()
    if args.json_out:
        write_json(_resolve(args.json_out), {"benchmark": [b.as_dict() for b in res]})
    return 0


def cmd_checkpoint(args) -> int:
    tcfg = load_training_config(args.config or None)
    ckpt = _resolve(args.checkpoint)
    if not ckpt.exists():
        print(f"no such checkpoint: {ckpt}")
        return 2
    state = inspect_checkpoint(ckpt)
    norm = state.get("normalization") or {"method": "none"}
    import torch
    w = torch.randn(tcfg.dataset.window_frames, FEATURE_DIM).numpy().astype(np.float32)
    r = checkpoint_roundtrip(ckpt, norm, tcfg, w)
    return 0 if r["passed"] else 4


def cmd_overfit(args) -> int:
    tcfg = load_training_config(args.config or None)
    entries = load_manifest(tcfg.dataset.manifest).entries
    cache_dir = _resolve(tcfg.dataset.feature_cache_dir)
    per_cell = args.overfit_per_cell or tcfg.dataset.sample.overfit_videos_per_cell
    tr = build_corpus(load_split_entries(entries, tcfg.dataset.train_split, tcfg, per_cell),
                      cache_dir, tcfg.dataset.manifest, split=tcfg.dataset.train_split)
    if len(tr) < 4:
        print(f"only {len(tr)} windows cached — run scripts/extract_features.py first")
        return 2
    norm = fit_normalization(tr.windows, tcfg.normalization.method)
    ds = TemporalFightDataset(tr, norm)
    r = tiny_overfit(ds, epochs=args.overfit_epochs, batch_size=args.overfit_batch,
                     lr=args.overfit_lr, device=args.device or tcfg.training.device,
                     seed=tcfg.training.seed, n_videos=len(tr.clip_ids_present()))
    print(RULE)
    print(r.table())
    return 0 if r.passed else 3


def cmd_train(args) -> int:
    """The full run. Refuses to start without explicit confirmation."""
    tcfg = load_training_config(args.config or None)
    cfg = load_config()

    problems = tcfg.validate_against_app(cfg)
    if problems:
        print("config inconsistent with the shipped inference path:")
        for p in problems:
            print(f"  ! {p}")
        return 2

    if not args.confirm_full_training:
        print(RULE)
        print("FULL TRAINING NOT STARTED")
        print(RULE)
        print("  PROMPT 4 forbids executing a full RWF-2000 training run at this")
        print("  stage. Re-run with --confirm-full-training once you have decided.")
        print()
        print("  The plan that WOULD execute:")
        print_full_training_command(tcfg, args)
        return 0

    entries = load_manifest(tcfg.dataset.manifest).entries
    audit = audit_labels_and_splits(entries, tcfg)
    if not audit["passed"]:
        return 2

    registry, reg_path = load_or_build_registry(
        tcfg.dataset.manifest, _resolve(tcfg.leakage.registry_path))
    cache_dir = _resolve(tcfg.dataset.feature_cache_dir)
    ex = tcfg.dataset.extraction

    train_entries = load_split_entries(entries, tcfg.dataset.train_split, tcfg)
    val_entries = load_split_entries(entries, tcfg.dataset.val_split, tcfg)

    print(f"\n{RULE}\nFEATURE CACHE\n{RULE}")
    build = build_cache(train_entries + val_entries, cache_dir,
                        manifest=tcfg.dataset.manifest, project_root=PROJECT_ROOT,
                        app_config=cfg,
                        frame_shape=(ex.frame_height, ex.frame_width),
                        device=ex.device, force=args.force_extract,
                        max_frames=ex.max_frames_per_clip, progress_every=50)
    print(f"  {build.summary()}")

    train_corpus = build_corpus(train_entries, cache_dir, tcfg.dataset.manifest,
                                split=tcfg.dataset.train_split)
    val_corpus = build_corpus(val_entries, cache_dir, tcfg.dataset.manifest,
                              split=tcfg.dataset.val_split)
    if len(train_corpus) == 0 or len(val_corpus) == 0:
        print("empty corpus — nothing to train on")
        return 2
    document_sequences(train_corpus, tcfg, cache_dir)

    # Step 5: fit BEFORE the val corpus is transformed with anything
    norm = fit_and_report_normalization(train_corpus.windows, tcfg.normalization.method)

    weights = class_weights(train_corpus.labels) if tcfg.training.balanced_class_weights else None
    device = resolve_device(args.device or tcfg.training.device)
    det = set_seed(args.seed if args.seed is not None else tcfg.training.seed,
                  tcfg.training.deterministic_cuda)

    train_ds = TemporalFightDataset(train_corpus, norm)
    val_ds = TemporalFightDataset(val_corpus, norm)
    train_loader = train_ds.make_loader(args.batch_size, shuffle=True,
                                        num_workers=tcfg.training.num_workers)
    val_loader = val_ds.make_loader(tcfg.training.eval_batch_size, shuffle=False,
                                    num_workers=tcfg.training.num_workers)

    import torch
    model = build_gru(FEATURE_DIM, tcfg.model.hidden_size).to(device)
    loss_fn = make_loss(weights)
    opt = (torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=tcfg.training.weight_decay)
           if tcfg.training.optimizer == "adam" else
           torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                           weight_decay=tcfg.training.weight_decay))

    epochs = args.epochs or tcfg.training.epochs
    history = []
    best = None
    print(f"\n{RULE}\nTRAINING  ({epochs} epochs, {len(train_ds)} train windows, "
          f"{len(val_ds)} val windows)\n{RULE}")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tm, _ = train_one_epoch(model, train_loader, opt, loss_fn, device,
                                grad_clip=tcfg.training.grad_clip)
        vm = evaluate(model, val_loader, device, loss_fn)

        keep = val_corpus.mask_excluding_clips(registry.leaked_val_clip_ids)
        logits = _all_logits(model, val_ds, device)
        vm_adj = compute_metrics(logits[keep], val_corpus.labels[keep],
                                 loss=_mean_bce(logits[keep], val_corpus.labels[keep]),
                                 threshold=tcfg.evaluation.threshold)

        history.append({"epoch": ep, "train": tm.as_dict(), "val": vm.as_dict(),
                        "val_excl_duplicates": vm_adj.as_dict(),
                        "seconds": round(time.time() - t0, 2)})
        print(f"  ep {ep:3d}/{epochs}  train loss {tm.loss:.4f} acc {tm.accuracy:.4f} | "
              f"val loss {vm.loss:.4f} acc {vm.accuracy:.4f} auc "
              f"{('%.4f' % vm.roc_auc) if vm.roc_auc is not None else 'n/a'} | "
              f"val-dupfree acc {vm_adj.accuracy:.4f}  ({time.time() - t0:.1f}s)")

        score = vm.roc_auc if vm.roc_auc is not None else vm.accuracy
        if best is None or score > best[0]:
            best = (score, ep)
            ck = _resolve(tcfg.outputs.checkpoint_dir) / "rwf2000_gru_best.pt"
            save_checkpoint(
                ck, model, norm, architecture=model.architecture(),
                training={"epochs_run": ep, "batch_size": args.batch_size, "lr": args.lr,
                          "optimizer": tcfg.training.optimizer,
                          "weight_decay": tcfg.training.weight_decay,
                          "grad_clip": tcfg.training.grad_clip, "loss": "BCEWithLogitsLoss",
                          "class_weights": (np.round(weights, 4).tolist()
                                            if weights is not None else None),
                          "seed": det["seed"], "determinism": det},
                metrics={"val": vm.as_dict(), "val_excluding_known_duplicates": vm_adj.as_dict()},
                dataset_info={"manifest": tcfg.dataset.manifest,
                              "train_videos": len(train_corpus.clip_ids_present()),
                              "train_windows": len(train_corpus),
                              "val_videos": len(val_corpus.clip_ids_present()),
                              "val_windows": len(val_corpus),
                              "window_frames": tcfg.dataset.window_frames,
                              "stride_frames": tcfg.dataset.stride_frames,
                              "leaked_val_clips": registry.leaked_val_clip_ids},
            )

    final_ck = _resolve(tcfg.outputs.checkpoint_dir) / "rwf2000_gru_final.pt"
    save_checkpoint(final_ck, model, norm, architecture=model.architecture(),
                    training={"epochs_run": epochs, "batch_size": args.batch_size,
                              "lr": args.lr, "optimizer": tcfg.training.optimizer,
                              "loss": "BCEWithLogitsLoss",
                              "class_weights": (np.round(weights, 4).tolist()
                                                if weights is not None else None),
                              "seed": det["seed"], "determinism": det},
                    metrics={"val": history[-1]["val"],
                             "val_excluding_known_duplicates": history[-1]["val_excl_duplicates"]},
                    dataset_info={"manifest": tcfg.dataset.manifest,
                                  "train_windows": len(train_corpus),
                                  "val_windows": len(val_corpus)})

    out = _resolve(tcfg.outputs.report_dir) / "rwf2000_training_run.json"
    write_json(out, {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "audit": audit, "leakage": registry.to_json(),
        "normalization": norm, "class_weights": (np.round(weights, 4).tolist()
                                                 if weights is not None else None),
        "determinism": det, "history": history,
        "best_epoch": best[1] if best else None,
        "checkpoints": {"best": str(_resolve(tcfg.outputs.checkpoint_dir) / "rwf2000_gru_best.pt"),
                        "final": str(final_ck)},
        "leakage_caveat": ("The official RWF-2000 val split contains clips that are "
                           "byte-identical to training clips. val_* metrics include them; "
                           "val_excluding_known_duplicates removes only the sha256-detectable "
                           "overlap. NEITHER is a leakage-free estimate."),
    })
    print(f"\n  best epoch {best[1] if best else None}  report -> {out}")
    print("  These are validation numbers on a leaky official split. They are not")
    print("  production accuracy and do not establish production readiness.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="", help="path to configs/training.yaml")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--device", default="")
        p.add_argument("--smoke-batch", type=int, default=8)
        p.add_argument("--overfit-per-cell", type=int, default=0)
        p.add_argument("--overfit-epochs", type=int, default=200)
        p.add_argument("--overfit-batch", type=int, default=8)
        p.add_argument("--overfit-lr", type=float, default=3e-3)
        p.add_argument("--bench-batches", nargs="*", type=int, default=None)
        p.add_argument("--bench-iters", type=int, default=30)
        p.add_argument("--force-extract", action="store_true")

    v = sub.add_parser("verify", help="run all verification stages (default)")
    common(v)
    v.add_argument("--per-cell", type=int, default=0)
    v.add_argument("--no-leak-pairs", action="store_true",
                   help="do not force the known duplicate train/val pairs into the sample")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("smoke", help="Step 12 forward/backward/step only")
    common(s)
    s.set_defaults(func=cmd_smoke)

    o = sub.add_parser("overfit", help="Step 13 tiny overfit only")
    common(o)
    o.set_defaults(func=cmd_overfit)

    b = sub.add_parser("benchmark", help="Step 17 throughput/VRAM only")
    common(b)
    b.add_argument("--json-out", default="")
    b.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("checkpoint", help="Step 14 inference round-trip only")
    common(c)
    c.add_argument("--checkpoint", required=True)
    c.set_defaults(func=cmd_checkpoint)

    t = sub.add_parser("train", help="FULL training run (guarded)")
    common(t)
    t.add_argument("--confirm-full-training", action="store_true",
                   help="required to actually start; omitted = print the plan only")
    t.add_argument("--epochs", type=int, default=0)
    t.add_argument("--batch-size", type=int, default=0)
    t.add_argument("--lr", type=float, default=0.0)
    t.add_argument("--seed", type=int, default=None)
    t.set_defaults(func=cmd_train)

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    if not getattr(args, "cmd", None):
        args = ap.parse_args((argv or sys.argv[1:]) + ["verify"])
        setup_logging(args.log_level)
    # defaults that fall back to the YAML
    tcfg = load_training_config(args.config or None)
    if getattr(args, "batch_size", 0) == 0:
        args.batch_size = tcfg.training.batch_size
    if getattr(args, "lr", 0.0) == 0.0:
        args.lr = tcfg.training.lr
    if getattr(args, "epochs", 0) == 0:
        args.epochs = tcfg.training.epochs
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
