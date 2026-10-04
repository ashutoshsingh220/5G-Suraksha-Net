"""Tests for the RWF-2000 temporal training pipeline (PROMPT 4).

These cover the properties the stage exists to guarantee: the sequence shape the
GRU actually consumes, split isolation, train-only normalization, the loss/metric
implementations (sklearn is not installed, so AUC is ours), and — most
importantly — that a checkpoint written by training loads and scores identically
through the shipped inference path.

Nothing here touches the real dataset or the GPU-heavy extractor; corpora are
synthetic .npz files written to tmp_path, so the suite stays fast and hermetic.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from suraksha.data.manifest import ClipEntry
from suraksha.training.dataset import (
    FEATURE_DIM,
    LABEL_FIGHT,
    LABEL_NON_FIGHT,
    NORM_MINMAX,
    NORM_NONE,
    NORM_STANDARD,
    TemporalFightDataset,
    apply_normalization,
    build_corpus,
    class_weights,
    fit_normalization,
    label_to_int,
)
from suraksha.training.features import VideoFeatures, _pad_window, load_features, save_features
from suraksha.training.leakage import build_leakage_registry
from suraksha.training.metrics import (
    binary_cross_entropy_with_logits,
    compute_metrics,
    compute_metrics_from_probs,
    confusion_matrix,
    pr_auc,
    roc_auc,
    sigmoid,
    video_level_aggregate,
    video_level_metrics,
)
from suraksha.training.model import (
    BIDIRECTIONAL,
    DROPOUT,
    HEAD_HIDDEN,
    HIDDEN_SIZE,
    NUM_LAYERS,
    build_gru,
)
from suraksha.training.trainer import (
    inspect_checkpoint,
    save_checkpoint,
    set_seed,
    smoke_test,
    tiny_overfit,
)

MANIFEST = "test_manifest"
WINDOW = 32
STRIDE = 8


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _entry(clip_id: str, label: str, split: str, sha: str | None = None) -> ClipEntry:
    folder = "Fight" if label == LABEL_FIGHT else "NonFight"
    return ClipEntry(
        clip_id=clip_id,
        path=f"datasets/raw/RWF-2000/{split}/{folder}/{clip_id}.avi",
        label=label, split=split, dataset_name="rwf2000",
        video_id=clip_id, group_id=clip_id, sha256=sha,
        frame_count=150, fps=30.0, width=640, height=360,
    )


def _write_clip(cache_dir, clip_id, label, split, n_windows, seed=0,
                signature=0.0, window_frames=WINDOW):
    """Write a synthetic cached clip; `signature` shifts the flow feature."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n_windows, window_frames, FEATURE_DIM)).astype(np.float32)
    w[:, :, 2] = np.abs(w[:, :, 2]) + signature      # flow magnitude is >= 0
    w[:, :, 7] = 1.0                                  # bias feature is constant
    vf = VideoFeatures(
        clip_id=clip_id, path="", label=label, split=split, fps=30.0, frame_count=150,
        windows=w,
        gate_passed=np.zeros(n_windows, dtype=bool),
        mean_flow=np.full(n_windows, signature, dtype=np.float32),
        tracks_per_frame=np.full(150, 2, dtype=np.int16),
        engaged_pairs_per_frame=np.full(150, 1, dtype=np.int16),
        frames_decoded=150, note="",
    )
    save_features(vf, cache_dir / MANIFEST / f"{clip_id}.npz")
    return _entry(clip_id, label, split)


def _corpus(tmp_path, per_class=2, windows=12, seed=0):
    """A balanced synthetic train corpus with a separable flow signature."""
    entries = []
    for i in range(per_class):
        entries.append(_write_clip(tmp_path, f"fight_{i}", LABEL_FIGHT, "train",
                                   windows, seed=seed + i, signature=6.0))
        entries.append(_write_clip(tmp_path, f"nonfight_{i}", LABEL_NON_FIGHT, "train",
                                   windows, seed=seed + 100 + i, signature=0.2))
    return build_corpus(entries, tmp_path, MANIFEST, split="train"), entries


# --------------------------------------------------------------------------
# architecture — read off the shipped network, never redesigned
# --------------------------------------------------------------------------

def test_feature_dim_matches_the_live_detector():
    from suraksha.fight.candidate import FightCandidateDetector

    assert FEATURE_DIM == FightCandidateDetector.FEATURE_DIM == 8


def test_model_constants_match_shipped_inference_network():
    assert (HIDDEN_SIZE, HEAD_HIDDEN, NUM_LAYERS, BIDIRECTIONAL, DROPOUT) == \
        (64, 32, 1, False, 0.0)


def test_build_gru_emits_one_raw_logit_per_window():
    import torch
    from torch import nn

    model = build_gru()
    out = model(torch.randn(5, WINDOW, FEATURE_DIM))
    assert out.shape == (5,)

    # The head must end in Linear, not Sigmoid: callers apply sigmoid and the
    # loss is BCEWithLogitsLoss, so a built-in sigmoid would double-apply it.
    assert isinstance(model.head[-1], nn.Linear)
    assert not any(isinstance(m, nn.Sigmoid) for m in model.modules())

    # A raw logit is unbounded; over enough random windows some fall outside
    # [0, 1], which a probability output could never do.
    many = model(torch.randn(512, WINDOW, FEATURE_DIM) * 4.0)
    assert bool(((many < 0) | (many > 1)).any())


def test_build_gru_rejects_pixel_shaped_input():
    """The (T,128,128,3) clip path cannot be fed to this network — by design."""
    import torch

    model = build_gru()
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, WINDOW, 128 * 128 * 3))


def test_architecture_dict_is_self_describing():
    arch = build_gru().architecture()
    assert arch["input_size"] == FEATURE_DIM
    assert arch["hidden_size"] == HIDDEN_SIZE
    assert arch["head"] == [HIDDEN_SIZE, HEAD_HIDDEN, 1]
    assert "logit" in arch["output"]


def test_parameter_count_is_small_enough_for_6gb_gpu():
    n = sum(p.numel() for p in build_gru().parameters())
    assert n < 100_000, f"GRU grew to {n} params — re-check the VRAM budget"


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def test_label_to_int_maps_canonical_labels():
    assert label_to_int("FIGHT") == 1
    assert label_to_int("NON_FIGHT") == 0
    assert label_to_int(" fight ") == 1


@pytest.mark.parametrize("bad", ["UNKNOWN", "", "ambiguous", "Fight?", None])
def test_label_to_int_rejects_anything_untrainable(bad):
    with pytest.raises(ValueError):
        label_to_int(bad)


def test_corpus_skips_unknown_labels_instead_of_training_on_them(tmp_path):
    good = _write_clip(tmp_path, "c_good", LABEL_FIGHT, "train", 6, seed=1)
    bad = _entry("c_bad", "UNKNOWN", "train")
    save_features(
        VideoFeatures(clip_id="c_bad", path="", label="UNKNOWN", split="train",
                      fps=30.0, frame_count=150,
                      windows=np.zeros((3, WINDOW, FEATURE_DIM), np.float32),
                      gate_passed=np.zeros(3, bool), mean_flow=np.zeros(3, np.float32),
                      tracks_per_frame=np.zeros(0, np.int16),
                      engaged_pairs_per_frame=np.zeros(0, np.int16),
                      frames_decoded=150, note=""),
        tmp_path / MANIFEST / "c_bad.npz")
    corpus = build_corpus([good, bad], tmp_path, MANIFEST, split="train")
    assert corpus.stats.labels_rejected == {"UNKNOWN": 1}
    assert set(corpus.clip_ids) == {"c_good"}
    assert len(corpus) == 6


# --------------------------------------------------------------------------
# split isolation (Step 4)
# --------------------------------------------------------------------------

def test_build_corpus_refuses_to_mix_splits(tmp_path):
    a = _write_clip(tmp_path, "a", LABEL_FIGHT, "train", 4, seed=1)
    b = _write_clip(tmp_path, "b", LABEL_FIGHT, "val", 4, seed=2)
    with pytest.raises(ValueError, match="mixed splits"):
        build_corpus([a, b], tmp_path, MANIFEST)


def test_windows_never_cross_the_split_boundary(tmp_path):
    """A clip's windows all carry that clip's split — no window-level reshuffle."""
    tr, _ = _corpus(tmp_path, per_class=2, windows=10)
    val_entries = [
        _write_clip(tmp_path, "v_fight", LABEL_FIGHT, "val", 7, seed=50, signature=6.0),
        _write_clip(tmp_path, "v_non", LABEL_NON_FIGHT, "val", 5, seed=51, signature=0.2),
    ]
    va = build_corpus(val_entries, tmp_path, MANIFEST, split="val")

    assert tr.split == "train" and va.split == "val"
    assert set(tr.clip_ids).isdisjoint(set(va.clip_ids))
    assert len(va) == 12
    # every window of a clip is present, contiguously, in clip order
    assert va.clip_ids == ["v_fight"] * 7 + ["v_non"] * 5
    assert va.window_index.tolist() == list(range(7)) + list(range(5))


def test_clip_provenance_is_not_truncated(tmp_path):
    """Regression: np.full(n, id, dtype=np.str_) yields '<U1' and loses the id."""
    corpus, entries = _corpus(tmp_path, per_class=2, windows=5)
    ids = set(corpus.clip_ids)
    assert ids == {"fight_0", "fight_1", "nonfight_0", "nonfight_1"}
    assert all(len(c) > 1 for c in corpus.clip_ids)
    assert corpus.clip_ids_present() == sorted(ids)


def test_corpus_labels_align_with_their_source_clip(tmp_path):
    """Direct check that features and labels cannot get shuffled relative to each other."""
    corpus, _ = _corpus(tmp_path, per_class=2, windows=8)
    for cid in corpus.clip_ids_present():
        rows = [i for i, c in enumerate(corpus.clip_ids) if c == cid]
        labs = set(corpus.labels[rows].tolist())
        assert len(labs) == 1, f"{cid} carries mixed labels {labs}"
        expected = 1 if cid.startswith("fight") else 0
        assert labs == {expected}


def test_subset_preserves_provenance_alignment(tmp_path):
    corpus, _ = _corpus(tmp_path, per_class=2, windows=6)
    mask = corpus.mask_excluding_clips(["fight_0", "nonfight_1"])
    sub = corpus.subset(mask)
    assert len(sub) == int(mask.sum())
    assert len(sub.clip_ids) == len(sub)
    assert set(sub.clip_ids) == {"fight_1", "nonfight_0"}
    assert np.array_equal(sub.windows, corpus.windows[mask])


def test_empty_corpus_is_well_formed(tmp_path):
    empty = build_corpus([_entry("nope", LABEL_FIGHT, "train")], tmp_path, MANIFEST,
                         split="train")
    assert len(empty) == 0
    assert empty.windows.shape[2] == FEATURE_DIM
    assert empty.clip_ids == []
    assert empty.stats.clips_missing


# --------------------------------------------------------------------------
# sequence shape (Steps 10 / 11)
# --------------------------------------------------------------------------

def test_window_tensor_shape_is_n_t_featuredim(tmp_path):
    corpus, _ = _corpus(tmp_path, per_class=2, windows=9)
    assert corpus.windows.ndim == 3
    assert corpus.windows.shape == (4 * 9, WINDOW, FEATURE_DIM)
    assert corpus.windows.dtype == np.float32
    assert np.isfinite(corpus.windows).all()


def test_dataset_tensor_shape_matches(tmp_path):
    corpus, _ = _corpus(tmp_path, per_class=1, windows=4)
    ds = TemporalFightDataset(corpus, {"method": NORM_NONE})
    assert ds.tensor_shape == (8, WINDOW, FEATURE_DIM)
    x, y = ds[0]
    assert tuple(x.shape) == (WINDOW, FEATURE_DIM)
    assert int(y) in (0, 1)


def test_pad_window_matches_the_recognizer_rule():
    """Training windows must be built exactly as FightRecognizer._make_window builds them."""
    from suraksha.fight.recognizer import FightRecognizer

    rng = np.random.default_rng(0)
    feats = [rng.normal(size=FEATURE_DIM).astype(np.float32) for _ in range(10)]

    fake_self = SimpleNamespace(cfg=SimpleNamespace(
        temporal=SimpleNamespace(window_frames=WINDOW, stride_frames=STRIDE)))
    cand = SimpleNamespace(features=list(feats))
    expected = FightRecognizer._make_window(fake_self, cand)

    got = _pad_window(feats, WINDOW)
    assert got.shape == (WINDOW, FEATURE_DIM)
    assert np.array_equal(got, expected)
    # front-padding repeats the first frame
    assert np.array_equal(got[0], got[WINDOW - len(feats)])


def test_pad_window_returns_none_when_too_short():
    assert _pad_window([np.zeros(FEATURE_DIM, np.float32)] * 3, WINDOW) is None


def test_features_roundtrip_through_the_npz_cache(tmp_path):
    entry = _write_clip(tmp_path, "rt", LABEL_FIGHT, "train", 5, seed=7, signature=3.0)
    vf = load_features(tmp_path / MANIFEST / "rt.npz")
    assert vf.clip_id == "rt" and vf.label == LABEL_FIGHT and vf.split == "train"
    assert vf.n_windows == 5
    assert vf.windows.shape == (5, WINDOW, FEATURE_DIM)
    assert vf.fps == 30.0 and vf.frame_count == 150
    assert entry.clip_id == "rt"


# --------------------------------------------------------------------------
# normalization — TRAIN ONLY (Step 5)
# --------------------------------------------------------------------------

def test_fit_normalization_is_standard_and_finite(tmp_path):
    corpus, _ = _corpus(tmp_path)
    p = fit_normalization(corpus.windows, NORM_STANDARD)
    assert p["method"] == NORM_STANDARD
    assert p["fitted_on"] == "train"
    assert p["n_windows"] == len(corpus)
    assert len(p["mean"]) == len(p["std"]) == FEATURE_DIM
    assert np.all(np.asarray(p["std"]) > 0)


def test_normalization_never_divides_by_zero_on_constant_features(tmp_path):
    corpus, _ = _corpus(tmp_path)
    p = fit_normalization(corpus.windows, NORM_STANDARD)
    # feature 7 is the constant bias term
    assert p["std"][7] == pytest.approx(1.0)
    out = apply_normalization(corpus.windows, p)
    assert np.isfinite(out).all()
    assert np.allclose(out[:, :, 7], 0.0, atol=1e-6)


def test_apply_normalization_is_the_inverse_of_fit(tmp_path):
    corpus, _ = _corpus(tmp_path)
    p = fit_normalization(corpus.windows, NORM_STANDARD)
    out = apply_normalization(corpus.windows, p)
    flat = out.reshape(-1, FEATURE_DIM)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-5)

    # Constant features must be identified from the RAW data: the eps guard sets
    # their fitted std to 1.0, which is indistinguishable from a real std there.
    constant = corpus.windows.reshape(-1, FEATURE_DIM).std(axis=0) < 1e-6
    assert constant[7], "the bias feature should be constant"
    assert np.allclose(flat.std(axis=0)[~constant], 1.0, atol=0.05)
    # a constant feature normalizes to exactly zero, it is not scaled up
    assert np.allclose(flat.std(axis=0)[constant], 0.0, atol=1e-6)
    assert np.allclose(np.asarray(p["std"])[constant], 1.0)


def test_normalization_none_is_identity():
    x = np.random.default_rng(0).normal(size=(3, WINDOW, FEATURE_DIM)).astype(np.float32)
    assert np.array_equal(apply_normalization(x, {"method": NORM_NONE}), x)
    assert np.array_equal(apply_normalization(x, None), x)


def test_minmax_normalization_maps_into_unit_range(tmp_path):
    corpus, _ = _corpus(tmp_path)
    p = fit_normalization(corpus.windows, NORM_MINMAX)
    out = apply_normalization(corpus.windows, p)
    assert out.min() >= -1e-5 and out.max() <= 1 + 1e-5


def test_fit_normalization_rejects_empty_and_bad_shapes():
    with pytest.raises(ValueError):
        fit_normalization(np.zeros((0, WINDOW, FEATURE_DIM), np.float32))
    with pytest.raises(ValueError):
        fit_normalization(np.zeros((4, WINDOW, 3), np.float32))
    with pytest.raises(ValueError):
        fit_normalization(np.zeros((4, WINDOW, FEATURE_DIM), np.float32), method="bogus")


def test_inference_classifier_applies_the_same_normalization(tmp_path):
    """The fitted stats must survive the checkpoint and be re-applied identically."""
    from suraksha.fight.temporal_classifier import TorchTemporalClassifier

    corpus, _ = _corpus(tmp_path, per_class=1, windows=4)
    p = fit_normalization(corpus.windows, NORM_STANDARD)
    model = build_gru()
    ck = save_checkpoint(tmp_path / "norm.pt", model, p)

    scorer = TorchTemporalClassifier(ck, device="cpu")
    assert scorer.norm_method == NORM_STANDARD
    assert np.allclose(scorer.norm_mean, np.asarray(p["mean"], np.float32), atol=1e-6)
    assert np.allclose(scorer.norm_std, np.asarray(p["std"], np.float32), atol=1e-6)

    w = corpus.windows[0]
    assert np.allclose(scorer._normalize(w), apply_normalization(w, p), atol=1e-6)


# --------------------------------------------------------------------------
# class weighting (Step 9)
# --------------------------------------------------------------------------

def test_class_weights_are_none_when_balanced():
    labels = np.array([0, 1] * 50)
    assert class_weights(labels) is None


def test_class_weights_appear_only_when_actually_imbalanced():
    labels = np.array([1] * 90 + [0] * 10)
    w = class_weights(labels)
    assert w is not None
    assert w[0] > w[1], "the minority class must carry the larger weight"
    assert w.mean() == pytest.approx(1.0, abs=1e-6)


def test_class_weights_raise_when_a_class_is_absent():
    with pytest.raises(ValueError):
        class_weights(np.ones(10, dtype=np.int64))


def test_class_weights_match_inverse_frequency_formula():
    """Pin w_c = N/(2*n_c), renormalized to sum=2 — computed from the labels
    passed in (the TRAIN split), never from val/test."""
    labels = np.array([1] * 96 + [0] * 51)   # FIGHT=96, NON_FIGHT=51
    w = class_weights(labels)
    counts = np.array([51.0, 96.0])          # [n_neg, n_pos]
    total = counts.sum()
    manual = total / (2.0 * counts)
    manual = manual / manual.sum() * 2.0
    assert np.allclose(w, manual, atol=1e-5)
    assert w.sum() == pytest.approx(2.0, abs=1e-5)



# --------------------------------------------------------------------------
# loss and metrics (Step 8)
# --------------------------------------------------------------------------

def test_bce_with_logits_matches_an_independent_reference():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500) * 3
    y = (rng.random(500) > 0.5).astype(np.float64)
    ref = -np.mean(y * np.log(sigmoid(z) + 1e-12) + (1 - y) * np.log(1 - sigmoid(z) + 1e-12))
    assert binary_cross_entropy_with_logits(z, y) == pytest.approx(ref, abs=1e-9)


def test_torch_bcewithlogits_matches_the_numpy_reference():
    import torch

    rng = np.random.default_rng(1)
    z = rng.normal(size=(16,)).astype(np.float32)
    y = (rng.random(16) > 0.5).astype(np.float32)
    t = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(z), torch.from_numpy(y)))
    assert t == pytest.approx(binary_cross_entropy_with_logits(z, y), abs=1e-6)


def test_sigmoid_is_numerically_stable_at_extremes():
    assert sigmoid(np.array([-1000.0]))[0] == pytest.approx(0.0)
    assert sigmoid(np.array([1000.0]))[0] == pytest.approx(1.0)


def test_roc_auc_matches_hand_computed_values():
    # perfect separation
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    # perfectly inverted
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    # all tied -> 0.5
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1])) == 0.5
    # negatives 0.1/0.2, positives 0.15/0.9: 3 of 4 (pos, neg) pairs ordered -> 0.75
    assert roc_auc(np.array([0.1, 0.2, 0.15, 0.9]), np.array([0, 0, 1, 1])) == 0.75
    # 2 of 4 ordered -> 0.5
    assert roc_auc(np.array([0.1, 0.9, 0.4, 0.6]), np.array([0, 0, 1, 1])) == 0.5


def test_roc_auc_handles_ties_with_average_ranks():
    s = np.array([0.2, 0.2, 0.7, 0.7])
    y = np.array([0, 1, 0, 1])
    assert roc_auc(s, y) == 0.5


def test_pr_auc_is_average_precision_not_trapezoid():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 1, 0])
    # AP = 1/2 * (1/1) + 1/2 * (2/3) = 0.8333...
    assert pr_auc(s, y) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3), abs=1e-9)
    assert pr_auc(np.array([0.9, 0.8]), np.array([1, 1])) == 1.0


def test_auc_returns_none_for_single_class_input():
    assert roc_auc(np.array([0.1, 0.2]), np.array([1, 1])) is None
    assert pr_auc(np.array([0.1, 0.2]), np.array([0, 0])) is None


def test_confusion_matrix_orientation():
    pred = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    tn, fp, fn, tp = confusion_matrix(pred, y)
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)


def test_compute_metrics_on_a_known_case():
    logits = np.array([-2.0, -1.0, 1.0, 2.0])
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(logits, y, loss=0.1, threshold=0.5)
    assert m.accuracy == 1.0 and m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    assert m.roc_auc == 1.0 and m.pr_auc == 1.0
    assert m.confusion == (2, 0, 0, 2)
    assert m.loss == 0.1


def test_compute_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(3), np.zeros(4))


def test_metrics_table_and_dict_are_serializable():
    import json

    m = compute_metrics(np.array([-1.0, 1.0]), np.array([0, 1]), loss=0.2)
    assert "accuracy" in m.table("t")
    json.dumps(m.as_dict())


def test_balanced_accuracy_is_mean_of_class_recalls():
    # 2 tp, 1 fn (recall 2/3); 1 tn, 1 fp (specificity 1/2) -> balacc 0.5833
    logits = np.array([2.0, 2.0, -2.0, 2.0, -2.0])
    y = np.array([1, 1, 1, 0, 0])
    m = compute_metrics(logits, y, threshold=0.5)
    assert m.recall == pytest.approx(2 / 3)
    assert m.specificity == pytest.approx(0.5)
    assert m.balanced_accuracy == pytest.approx((2 / 3 + 0.5) / 2)
    assert "balanced_accuracy" in m.as_dict()


def test_compute_metrics_from_probs_uses_probability_threshold():
    # the production 0.60 boundary lives in probability space, not logit space
    probs = np.array([0.55, 0.65, 0.10, 0.90])
    y = np.array([1, 1, 0, 0])
    m50 = compute_metrics_from_probs(probs, y, 0.5)
    m60 = compute_metrics_from_probs(probs, y, 0.60)
    assert m50.tp == 2 and m50.fp == 1          # 0.55 and 0.90 clear 0.5
    assert m60.tp == 1 and m60.fp == 1          # only 0.65 and 0.90 clear 0.6; 0.90 is the fp
    assert m50.roc_auc == m60.roc_auc           # AUC is threshold-independent
    with pytest.raises(ValueError):
        compute_metrics_from_probs(np.zeros(3), np.zeros(4))


def test_video_level_aggregate_mean_and_max():
    probs = np.array([0.9, 0.8, 0.1, 0.2])
    clip_ids = ["a", "a", "b", "b"]
    labels = np.array([1, 1, 0, 0])
    ids, scores, lab = video_level_aggregate(probs, clip_ids, labels, "mean")
    assert ids == ["a", "b"]
    assert np.allclose(scores, [0.85, 0.15])
    assert lab.tolist() == [1, 0]
    _, smax, _ = video_level_aggregate(probs, clip_ids, labels, "max")
    assert np.allclose(smax, [0.9, 0.2])
    with pytest.raises(ValueError):
        video_level_aggregate(probs, clip_ids, labels, "median")


def test_video_level_metrics_collapses_windows_to_clips():
    # 3 windows of clip 'a' (all fight) and 1 window of clip 'b' (non-fight):
    # window-level n would be 4, video-level n must be 2.
    probs = np.array([0.9, 0.4, 0.5, 0.1])
    clip_ids = ["a", "a", "a", "b"]
    labels = np.array([1, 1, 1, 0])
    vm = video_level_metrics(probs, clip_ids, labels, 0.5, "mean")
    assert vm.n == 2
    assert vm.extra["level"] == "video" and vm.extra["aggregation"] == "mean"
    # clip 'a' mean = 0.6 -> fight (correct); clip 'b' = 0.1 -> non-fight (correct)
    assert vm.accuracy == 1.0
    assert vm.balanced_accuracy == 1.0


# --------------------------------------------------------------------------
# smoke test, overfit, checkpoints (Steps 12-14)
# --------------------------------------------------------------------------

def test_smoke_test_passes_and_reports_shapes():
    r = smoke_test(batch_size=4, window_frames=WINDOW, device="cpu")
    assert r.passed, r.notes
    assert r.input_shape == (4, WINDOW, FEATURE_DIM)
    assert r.output_shape == (4,)
    assert r.grads_finite and r.grad_norm > 0 and r.params_changed
    assert r.loss == pytest.approx(r.loss_reference, abs=1e-5)


def test_smoke_test_is_seeded_reproducible_on_cpu():
    a = smoke_test(batch_size=4, seed=123, device="cpu")
    b = smoke_test(batch_size=4, seed=123, device="cpu")
    assert a.loss == pytest.approx(b.loss, abs=1e-9)


def test_tiny_overfit_learns_separable_synthetic_windows(tmp_path):
    corpus, _ = _corpus(tmp_path, per_class=2, windows=8)
    p = fit_normalization(corpus.windows, NORM_STANDARD)
    ds = TemporalFightDataset(corpus, p)
    r = tiny_overfit(ds, epochs=60, batch_size=8, lr=5e-3, device="cpu", seed=0,
                     n_videos=len(corpus.clip_ids_present()))
    assert r.passed, r.notes
    assert r.final_loss < r.initial_loss
    assert r.final_accuracy >= 0.9
    assert r.model is not None


def test_checkpoint_roundtrip_through_the_shipped_inference_path(tmp_path):
    from suraksha.fight.temporal_classifier import (
        HeuristicTemporalScorer, TorchTemporalClassifier, build_scorer,
    )

    corpus, _ = _corpus(tmp_path, per_class=1, windows=4)
    p = fit_normalization(corpus.windows, NORM_STANDARD)
    ds = TemporalFightDataset(corpus, p)
    r = tiny_overfit(ds, epochs=40, batch_size=8, lr=5e-3, device="cpu", seed=0)
    ck = save_checkpoint(tmp_path / "rt.pt", r.model, p,
                         architecture=r.model.architecture(),
                         training={"loss": "BCEWithLogitsLoss"},
                         metrics={"final_loss": r.final_loss},
                         dataset_info={"manifest": MANIFEST})

    info = inspect_checkpoint(ck)
    assert info["format"] == "wrapped"
    assert info["feature_dim"] == FEATURE_DIM
    assert info["hidden_size"] == HIDDEN_SIZE
    assert info["normalization"]["method"] == NORM_STANDARD
    assert info["n_parameters"] == sum(q.numel() for q in r.model.parameters())

    scorer = build_scorer(str(ck), device="cpu")
    assert isinstance(scorer, TorchTemporalClassifier)
    assert not isinstance(scorer, HeuristicTemporalScorer)

    w = corpus.windows[0]
    s = scorer.score(w)
    assert 0.0 <= s <= 1.0

    # independent recomputation on the training side must agree exactly
    import torch

    m2 = build_gru()
    m2.load_state_dict(torch.load(str(ck), map_location="cpu",
                                  weights_only=False)["model_state"])
    m2.eval()
    with torch.no_grad():
        ref = float(torch.sigmoid(
            m2(torch.from_numpy(apply_normalization(w[None, ...], p)))).item())
    assert s == pytest.approx(ref, abs=1e-6)


def test_recognizer_accepts_a_trained_checkpoint(tmp_path):
    from suraksha.config import FightConfig
    from suraksha.fight.recognizer import FightRecognizer
    from suraksha.fight.temporal_classifier import TorchTemporalClassifier

    corpus, _ = _corpus(tmp_path, per_class=1, windows=3)
    p = fit_normalization(corpus.windows, NORM_STANDARD)
    ck = save_checkpoint(tmp_path / "rec.pt", build_gru(), p)

    cfg = FightConfig()
    cfg.temporal.model_weights = str(ck)
    rec = FightRecognizer(cfg, (360, 640), device="cpu")
    assert isinstance(rec.scorer, TorchTemporalClassifier)
    assert rec.cfg.temporal.window_frames == WINDOW


def test_build_scorer_falls_back_when_the_checkpoint_is_absent(tmp_path):
    from suraksha.fight.temporal_classifier import HeuristicTemporalScorer, build_scorer

    assert isinstance(build_scorer(str(tmp_path / "missing.pt")), HeuristicTemporalScorer)
    assert isinstance(build_scorer(""), HeuristicTemporalScorer)


def test_save_checkpoint_creates_parent_directories(tmp_path):
    ck = save_checkpoint(tmp_path / "deep" / "nested" / "m.pt", build_gru(), None)
    assert ck.exists()
    assert inspect_checkpoint(ck)["normalization"] == {"method": NORM_NONE}


def test_set_seed_records_honest_reproducibility_state():
    d = set_seed(7, deterministic_cuda=False)
    assert d["seed"] == 7
    assert d["cudnn_benchmark"] is True
    assert d["bitwise_reproducible_on_cuda"] is False, \
        "must not claim bitwise reproducibility without deterministic algorithms"


# --------------------------------------------------------------------------
# leakage registry
# --------------------------------------------------------------------------

def test_registry_flags_duplicates_that_straddle_the_split():
    entries = [
        _entry("t1", LABEL_NON_FIGHT, "train", sha="aaa"),
        _entry("v1", LABEL_NON_FIGHT, "val", sha="aaa"),     # byte-identical, other split
        _entry("t2", LABEL_FIGHT, "train", sha="bbb"),
        _entry("v2", LABEL_FIGHT, "val", sha="ccc"),
    ]
    r = build_leakage_registry(entries, "test")
    assert r.leaked_val_clip_ids == ["v1"]
    assert r.leaked_train_clip_ids == ["t1"]
    assert len(r.straddling_groups) == 1
    assert r.clean_val_ids(["v1", "v2"]) == ["v2"]
    assert r.val_split_of("v1") and not r.val_split_of("v2")


def test_registry_ignores_within_split_duplicates_for_leakage():
    entries = [
        _entry("t1", LABEL_NON_FIGHT, "train", sha="aaa"),
        _entry("t2", LABEL_NON_FIGHT, "train", sha="aaa"),   # redundant, not leaking
        _entry("v1", LABEL_FIGHT, "val", sha="ddd"),
    ]
    r = build_leakage_registry(entries, "test")
    assert len(r.duplicate_groups) == 1
    assert r.straddling_groups == []
    assert r.leaked_val_clip_ids == []
    assert r.clean_val_ids(["v1"]) == ["v1"]


def test_registry_tolerates_missing_hashes():
    entries = [_entry("t1", LABEL_FIGHT, "train"), _entry("v1", LABEL_FIGHT, "val")]
    r = build_leakage_registry(entries, "test")
    assert r.duplicate_groups == []
    assert r.total_clips == 2


def test_registry_summary_states_the_split_is_not_clean():
    entries = [_entry("t1", LABEL_NON_FIGHT, "train", sha="x"),
               _entry("v1", LABEL_NON_FIGHT, "val", sha="x")]
    s = build_leakage_registry(entries, "test").summary()
    assert "NOT content-disjoint" in s["note"]
    assert "leakage-free" in s["note"]


def test_registry_loads_and_persists_from_the_real_manifest(tmp_path):
    import json

    from suraksha.data.manifest import load_manifest
    from suraksha.training.leakage import load_or_build_registry

    try:
        load_manifest("rwf2000_v1")
    except Exception:
        pytest.skip("rwf2000_v1 manifest not available")

    report, dest = load_or_build_registry("rwf2000_v1", tmp_path / "leak.json")
    assert dest is not None and dest.exists()

    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["total_clips"] == 2000
    # measured facts about the official split — if these change, the audit is stale
    assert loaded["groups_straddling_train_val"] == 6
    assert loaded["leaked_val_clips"] == 6
    assert len(loaded["straddling_groups"]) == 6
    assert all(g["label"] == LABEL_NON_FIGHT for g in loaded["straddling_groups"])
    assert "NOT content-disjoint" in loaded["note"]
    assert report.leaked_val_set.isdisjoint(report.leaked_train_clip_ids)


# --------------------------------------------------------------------------
# configuration (Step 18)
# --------------------------------------------------------------------------

def test_training_config_loads_and_agrees_with_the_app_config():
    from suraksha.config import load_config, load_training_config

    tcfg = load_training_config()
    app = load_config()
    assert tcfg.validate_against_app(app) == []
    assert tcfg.dataset.window_frames == app.fight.temporal.window_frames
    assert tcfg.dataset.stride_frames == app.fight.temporal.stride_frames
    assert tcfg.dataset.feature_dim == FEATURE_DIM
    assert tcfg.augmentation.enabled is False


def test_training_config_detects_window_geometry_drift():
    from suraksha.config import TrainingConfig, load_config

    tcfg = TrainingConfig()
    tcfg.dataset.window_frames = 16          # would silently break inference
    problems = tcfg.validate_against_app(load_config())
    assert any("window_frames" in p for p in problems)


def test_training_config_detects_architecture_drift():
    from suraksha.config import TrainingConfig, load_config

    tcfg = TrainingConfig()
    tcfg.model.hidden_size = 128
    tcfg.model.bidirectional = True
    problems = tcfg.validate_against_app(load_config())
    assert any("bidirectional" in p for p in problems)


def test_training_config_is_rwf2000_only():
    from suraksha.config import load_training_config

    tcfg = load_training_config()
    assert tcfg.dataset.manifest == "rwf2000_v1"
    assert set(tcfg.dataset.labels) == {LABEL_FIGHT, LABEL_NON_FIGHT}
    assert tcfg.dataset.val_split == "val"
    assert "test" not in (tcfg.dataset.train_split, tcfg.dataset.val_split)


# --------------------------------------------------------------------------
# the full-training guard
# --------------------------------------------------------------------------

def _load_cli():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_temporal.py"
    spec = importlib.util.spec_from_file_location("_train_temporal_cli", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_full_training_is_off_by_default():
    """`train` must not start unless --confirm-full-training is passed explicitly."""
    cli = _load_cli()
    args = cli.build_parser().parse_args(["train"])
    assert args.confirm_full_training is False


def test_full_training_requires_the_explicit_flag():
    cli = _load_cli()
    args = cli.build_parser().parse_args(["train", "--confirm-full-training"])
    assert args.confirm_full_training is True


def test_verify_is_the_default_subcommand():
    cli = _load_cli()
    args = cli.build_parser().parse_args([])
    # no subcommand -> main() rewrites to "verify", never to "train"
    assert getattr(args, "cmd", None) in (None, "verify")
    assert cli.build_parser().parse_args(["verify"]).cmd == "verify"
