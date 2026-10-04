"""Temporal sequence tests: dims, determinism, window counting, padding."""
import numpy as np
import pytest

from suraksha.data.sequences import (
    SequenceConfig, SequenceSampler, count_sequences, sample_window_indices,
)
from tests.video_helpers import make_tiny_video


def test_count_sequences():
    assert count_sequences(64, 16, 8) == 7
    assert count_sequences(64, 32, 8) == 5
    assert count_sequences(10, 32, 8) == 1     # short video -> 1 padded window
    assert count_sequences(0, 16, 8) == 0


def test_window_indices_deterministic_and_in_range():
    a = sample_window_indices(100, 32, 8, window_index=3)
    b = sample_window_indices(100, 32, 8, window_index=3)
    assert a == b and len(a) == 32
    assert 0 <= min(a) and max(a) < 100

    short = sample_window_indices(10, 32, 8)
    assert len(short) == 32 and short[-1] == 9  # edge-padded


@pytest.mark.parametrize("window", [16, 32])
def test_extract_dims_and_normalization(tmp_path, window):
    v = make_tiny_video(tmp_path / "seq.mp4", n_frames=40, size=(64, 48))
    sampler = SequenceSampler(SequenceConfig(window_frames=window, frame_size=(32, 32)))
    seq = sampler.extract(v, window_index=0)
    assert seq is not None
    assert seq.frames.shape == (window, 32, 32, 3)
    assert seq.frames.dtype == np.float32
    assert 0.0 <= seq.frames.min() and seq.frames.max() <= 1.0  # unit normalization
    assert len(seq.indices) == window


def test_extract_is_deterministic(tmp_path):
    v = make_tiny_video(tmp_path / "det.mp4", n_frames=40, seed=9)
    s = SequenceSampler(SequenceConfig(window_frames=16, augment=False))
    a = s.extract(v, 0)
    b = s.extract(v, 0)
    assert np.array_equal(a.frames, b.frames)


def test_imagenet_normalization_range(tmp_path):
    v = make_tiny_video(tmp_path / "in.mp4", n_frames=20)
    s = SequenceSampler(SequenceConfig(window_frames=16, normalization="imagenet"))
    seq = s.extract(v, 0)
    assert seq.frames.min() < 0.0 and seq.frames.max() > 1.0  # mean/std scaled


def test_iter_windows_lazy(tmp_path):
    v = make_tiny_video(tmp_path / "it.mp4", n_frames=40)
    s = SequenceSampler(SequenceConfig(window_frames=16, stride=8))
    n = sum(1 for _ in s.iter_windows(v, frame_count=40))
    assert n == count_sequences(40, 16, 8)


def test_missing_file_returns_none(tmp_path):
    s = SequenceSampler(SequenceConfig())
    assert s.extract(tmp_path / "nope.mp4") is None
