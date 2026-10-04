"""Validation tests: metadata, sha256, corrupt/missing/duplicate handling."""
import shutil

import pytest

from suraksha.data.manifest import ClipEntry, Manifest, file_sha256
from suraksha.data.validation import (
    CORRUPTED, DUPLICATE, MISSING, VALID, probe_video, validate_manifest,
)
from tests.video_helpers import make_corrupt_video, make_tiny_video


def test_probe_metadata(tmp_path):
    v = make_tiny_video(tmp_path / "ok.mp4", n_frames=24, fps=12.0)
    meta = probe_video(v, decode_check="all")
    assert meta.exists and meta.readable
    assert meta.frame_count == 24
    assert meta.fps == pytest.approx(12.0, abs=0.5)
    assert meta.width, meta.height == (64, 48)
    assert meta.duration_sec == pytest.approx(2.0, abs=0.2)
    assert meta.corrupt_frames == 0
    assert meta.sha256 and len(meta.sha256) == 64


def test_sha256_stable_and_distinct(tmp_path):
    a = make_tiny_video(tmp_path / "a.mp4", seed=1)
    b = make_tiny_video(tmp_path / "b.mp4", seed=2)
    assert file_sha256(a) == file_sha256(a)
    assert file_sha256(a) != file_sha256(b)
    assert file_sha256(tmp_path / "nope.mp4") is None


def _manifest(entries):
    m = Manifest(name="_t")
    for e in entries:
        m.add(e)
    return m


def test_validate_statuses(tmp_path):
    good = make_tiny_video(tmp_path / "good.mp4", seed=3)
    dup = tmp_path / "dup.mp4"
    shutil.copyfile(good, dup)
    bad = make_corrupt_video(tmp_path / "bad.mp4")

    m = _manifest([
        ClipEntry(clip_id="good", path=str(good)),
        ClipEntry(clip_id="dup", path=str(dup)),
        ClipEntry(clip_id="bad", path=str(bad)),
        ClipEntry(clip_id="gone", path=str(tmp_path / "missing.mp4")),
    ])
    report = validate_manifest(m, decode_check="sample", with_sha256=True)
    by_id = {e.clip_id: e.status for e in report.entries}

    assert by_id["good"] == VALID
    assert by_id["dup"] == DUPLICATE       # identical bytes flagged, not discarded
    assert by_id["bad"] == CORRUPTED
    assert by_id["gone"] == MISSING
    assert report.summary[VALID] >= 1


def test_validate_backfills_metadata(tmp_path):
    v = make_tiny_video(tmp_path / "m.mp4", n_frames=20, seed=4)
    e = ClipEntry(clip_id="m", path=str(v))
    m = _manifest([e])
    validate_manifest(m, decode_check="none", with_sha256=True)
    assert e.frame_count == 20
    assert e.fps and e.duration_sec and e.sha256
