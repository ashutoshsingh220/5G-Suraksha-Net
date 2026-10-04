"""Verify and compare Corpus v1 (old) vs Corpus v2 (new relaxed gating)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from suraksha.config import load_training_config
from suraksha.data.manifest import load_manifest
from suraksha.training.dataset import (
    FEATURE_DIM,
    build_corpus,
    class_weights,
    fit_normalization,
)

WINDOW_FRAMES = 32


def analyze_corpus(cache_dir: Path, manifest_name: str = "rwf2000_v1") -> dict:
    tcfg = load_training_config()
    manifest = load_manifest(manifest_name)

    train_entries = [e for e in manifest.entries if e.split == "train" and e.label in tcfg.dataset.labels]
    val_entries = [e for e in manifest.entries if e.split == "val" and e.label in tcfg.dataset.labels]

    train_corpus = build_corpus(train_entries, cache_dir, manifest=manifest_name, split="train")
    val_corpus = build_corpus(val_entries, cache_dir, manifest=manifest_name, split="val")

    norm_stats = fit_normalization(train_corpus.windows, method="standard")

    # Check finiteness
    assert np.isfinite(train_corpus.windows).all(), "Train windows contain NaN/Inf"
    assert np.isfinite(val_corpus.windows).all(), "Val windows contain NaN/Inf"
    assert train_corpus.windows.shape[1:] == (WINDOW_FRAMES, FEATURE_DIM)
    assert val_corpus.windows.shape[1:] == (WINDOW_FRAMES, FEATURE_DIM)

    # Label counts
    train_counts = train_corpus.counts()
    val_counts = val_corpus.counts()

    train_weights = class_weights(train_corpus.labels)

    return {
        "cache_dir": str(cache_dir),
        "total_windows": len(train_corpus) + len(val_corpus),
        "train": {
            "total_windows": len(train_corpus),
            "fights": int(train_counts.get("FIGHT", 0)),
            "non_fights": int(train_counts.get("NON_FIGHT", 0)),
            "fight_ratio": round(train_corpus.class_ratio(), 2),
            "class_weights": [round(float(w), 4) for w in train_weights.tolist()],
            "zero_yield_clips": len(train_corpus.stats.clips_zero_yield),
            "zero_yield_pct": round(100.0 * len(train_corpus.stats.clips_zero_yield) / len(train_entries), 1),
        },
        "val": {
            "total_windows": len(val_corpus),
            "fights": int(val_counts.get("FIGHT", 0)),
            "non_fights": int(val_counts.get("NON_FIGHT", 0)),
            "fight_ratio": round(val_corpus.class_ratio(), 2),
            "zero_yield_clips": len(val_corpus.stats.clips_zero_yield),
            "zero_yield_pct": round(100.0 * len(val_corpus.stats.clips_zero_yield) / len(val_entries), 1),
        },
        "normalization": {
            "mean": [round(float(m), 4) for m in norm_stats["mean"]],
            "std": [round(float(s), 4) for s in norm_stats["std"]],
        },
    }


def main():
    print("=" * 70)
    print("COMPARING CORPUS V1 (OLD) VS CORPUS V2 (NEW RELAXED GATING)")
    print("=" * 70)

    v1_dir = ROOT / "datasets" / "processed" / "features"
    v2_dir = ROOT / "datasets" / "processed" / "features_v2"

    print("\n1. Loading and verifying Corpus v1...")
    s1 = analyze_corpus(v1_dir)
    print(f"   Corpus v1 total windows: {s1['total_windows']} (Train: {s1['train']['total_windows']}, Val: {s1['val']['total_windows']})")

    print("\n2. Loading and verifying Corpus v2...")
    s2 = analyze_corpus(v2_dir)
    print(f"   Corpus v2 total windows: {s2['total_windows']} (Train: {s2['train']['total_windows']}, Val: {s2['val']['total_windows']})")

    comparison = {
        "v1_baseline": s1,
        "v2_relaxed": s2,
        "delta": {
            "total_windows_gain": s2["total_windows"] - s1["total_windows"],
            "total_windows_gain_pct": round(100.0 * (s2["total_windows"] - s1["total_windows"]) / s1["total_windows"], 1),
            "train_windows_gain": s2["train"]["total_windows"] - s1["train"]["total_windows"],
            "val_windows_gain": s2["val"]["total_windows"] - s1["val"]["total_windows"],
            "train_nonfight_gain_pct": round(100.0 * (s2["train"]["non_fights"] - s1["train"]["non_fights"]) / max(s1["train"]["non_fights"], 1), 1),
            "val_nonfight_gain_pct": round(100.0 * (s2["val"]["non_fights"] - s1["val"]["non_fights"]) / max(s1["val"]["non_fights"], 1), 1),
        }
    }

    out_json = ROOT / "datasets" / "reports" / "corpus_v2_verification.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    md_lines = [
        "# Corpus v2 Verification & Comparison Report",
        "",
        "## 1. Corpus Window Counts",
        "",
        "| Split / Class | Corpus v1 (Old Baseline) | Corpus v2 (New Relaxed Gating) | Absolute Gain | Relative Gain |",
        "|---|:---:|:---:|:---:|:---:|",
        f"| **Train Total** | {s1['train']['total_windows']} | **{s2['train']['total_windows']}** | +{comparison['delta']['train_windows_gain']} | +{round(100.0 * comparison['delta']['train_windows_gain'] / s1['train']['total_windows'], 1)}% |",
        f"|   Train FIGHT | {s1['train']['fights']} | **{s2['train']['fights']}** | +{s2['train']['fights'] - s1['train']['fights']} | +{round(100.0 * (s2['train']['fights'] - s1['train']['fights']) / s1['train']['fights'], 1)}% |",
        f"|   Train NON_FIGHT | {s1['train']['non_fights']} | **{s2['train']['non_fights']}** | +{s2['train']['non_fights'] - s1['train']['non_fights']} | **+{comparison['delta']['train_nonfight_gain_pct']}%** |",
        f"|   Train Zero-Yield | {s1['train']['zero_yield_clips']} ({s1['train']['zero_yield_pct']}%) | **{s2['train']['zero_yield_clips']} ({s2['train']['zero_yield_pct']}%)** | -{s1['train']['zero_yield_clips'] - s2['train']['zero_yield_clips']} | **{s2['train']['zero_yield_pct'] - s1['train']['zero_yield_pct']}% abs** |",
        f"| **Val Total** | {s1['val']['total_windows']} | **{s2['val']['total_windows']}** | +{comparison['delta']['val_windows_gain']} | +{round(100.0 * comparison['delta']['val_windows_gain'] / s1['val']['total_windows'], 1)}% |",
        f"|   Val FIGHT | {s1['val']['fights']} | **{s2['val']['fights']}** | +{s2['val']['fights'] - s1['val']['fights']} | +{round(100.0 * (s2['val']['fights'] - s1['val']['fights']) / s1['val']['fights'], 1)}% |",
        f"|   Val NON_FIGHT | {s1['val']['non_fights']} | **{s2['val']['non_fights']}** | +{s2['val']['non_fights'] - s1['val']['non_fights']} | **+{comparison['delta']['val_nonfight_gain_pct']}%** |",
        f"|   Val Zero-Yield | {s1['val']['zero_yield_clips']} ({s1['val']['zero_yield_pct']}%) | **{s2['val']['zero_yield_clips']} ({s2['val']['zero_yield_pct']}%)** | -{s1['val']['zero_yield_clips'] - s2['val']['zero_yield_clips']} | **{s2['val']['zero_yield_pct'] - s1['val']['zero_yield_pct']}% abs** |",
        f"| **FULL CORPUS** | {s1['total_windows']} | **{s2['total_windows']}** | **+{comparison['delta']['total_windows_gain']}** | **+{comparison['delta']['total_windows_gain_pct']}%** |",
        "",
        "## 2. Class Balance",
        f"- Train Fight:NonFight Ratio: **{s1['train']['fight_ratio']}:1** (v1) -> **{s2['train']['fight_ratio']}:1** (v2)",
        f"- Val Fight:NonFight Ratio: **{s1['val']['fight_ratio']}:1** (v1) -> **{s2['val']['fight_ratio']}:1** (v2)",
        f"- Train Class Weights (inverse frequency): `{s2['train']['class_weights']}`",
        "",
        "## 3. Train-Only Normalization Statistics (Fitted on Train Set Only)",
        "| Feature | Mean (v1) | Mean (v2) | Std (v1) | Std (v2) |",
        "|---|:---:|:---:|:---:|:---:|",
    ]

    feat_names = [
        "rel_dist", "iou", "norm_motion", "conf_a", "conf_b", "width_a/W", "width_b/W", "bias"
    ]
    for i, name in enumerate(feat_names):
        md_lines.append(
            f"| `{name}` | {s1['normalization']['mean'][i]:.4f} | {s2['normalization']['mean'][i]:.4f} | "
            f"{s1['normalization']['std'][i]:.4f} | {s2['normalization']['std'][i]:.4f} |"
        )

    md_lines.extend([
        "",
        "## 4. Verification Check",
        "- All features finite (no NaN, no Inf): **PASS**",
        "- Tensor dimensions match (N, 32, 8): **PASS**",
        "- Zero-leakage: Train-only normalization strictly observed: **PASS**",
    ])

    out_md = ROOT / "datasets" / "reports" / "corpus_v2_verification.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nVerification reports written to:\n  - {out_json}\n  - {out_md}\n")


if __name__ == "__main__":
    main()
