"""Summarizer: measured statistics must come from probed metadata, never guesses.

Guards the audit numbers (durations, FPS/resolution distributions, duplicate
groups, temporal-window capacity) and the rule that missing metadata is
reported as missing rather than filled in.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "summarize_datasets", ROOT / "scripts" / "summarize_datasets.py")
summ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summ)


def _write_manifest(reports: Path, manifests: Path, entries: list[dict], probes: list[dict]):
    manifests.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    mpath = manifests / "tiny_v1.json"
    mpath.write_text(json.dumps({
        "name": "tiny_v1", "version": "1.1", "task": "fight_temporal", "group": "A",
        "license": "research-use-only", "created_at": "2026-09-27T00:00:00+00:00",
        "entries": entries,
    }), encoding="utf-8")
    (reports / "tiny_v1_validation_20260927.json").write_text(json.dumps({
        "manifest": "tiny_v1", "created_at": "2026-09-27",
        "summary": {"VALID": 2, "DUPLICATE": 1, "CORRUPTED": 1},
        "entries": probes,
    }), encoding="utf-8")
    return mpath


def _entry(cid, label, split, path, sha, **kw):
    e = {"clip_id": cid, "path": path, "label": label, "split": split,
         "dataset_name": "tiny", "video_id": cid, "group_id": path,
         "license": "research-use-only", "source": "kaggle_mirror",
         "sha256": sha,
         "notes": {"source_class": "Fight", "proposed_label": "FIGHT",
                   "provenance": "manual download"}}
    e.update(kw)
    return e


def _probe(cid, path, status, **meta):
    return {"clip_id": cid, "path": path, "status": status, "details": "",
            "meta": {"exists": True, "readable": status != "CORRUPTED",
                     "corrupt_frames": 0, **meta}}


def test_summary_counts_durations_and_duplicates(tmp_path):
    rep, man = tmp_path / "reports", tmp_path / "manifests"
    entries = [
        _entry("a", "FIGHT", "train", "raw/a.mp4", "aa", duration_sec=2.0, fps=30.0,
               frame_count=60, width=320, height=240),
        _entry("b", "FIGHT", "train", "raw/b.mp4", "aa", duration_sec=4.0, fps=25.0,
               frame_count=100, width=1280, height=720),   # same sha => duplicate content
        _entry("c", "NON_FIGHT", "val", "raw/c.mp4", "cc", duration_sec=6.0, fps=25.0,
               frame_count=150, width=320, height=240),
        _entry("d", "UNKNOWN", "test", "raw/d.mp4", None, duration_sec=None, fps=None,
               frame_count=None, width=None, height=None),  # probe failed
    ]
    probes = [
        _probe("a", "raw/a.mp4", "VALID", fourcc="h264", frame_count=60, fps=30.0,
               duration_sec=2.0, width=320, height=240, sha256="aa"),
        _probe("b", "raw/b.mp4", "DUPLICATE", fourcc="h264", frame_count=100, fps=25.0,
               duration_sec=4.0, width=1280, height=720, sha256="aa"),
        _probe("c", "raw/c.mp4", "VALID", fourcc="MJPG", frame_count=150, fps=25.0,
               duration_sec=6.0, width=320, height=240, sha256="cc"),
        _probe("d", "raw/d.mp4", "CORRUPTED", corrupt_frames=7),
    ]
    mpath = _write_manifest(rep, man, entries, probes)

    s = summ.summarize_manifest(mpath, rep, [16, 32])

    assert s["videos"] == 4
    assert s["labels"] == {"FIGHT": 2, "NON_FIGHT": 1, "UNKNOWN": 1}
    assert s["splits"] == {"train": 2, "val": 1, "test": 1}
    assert s["validation_statuses"] == {"VALID": 2, "DUPLICATE": 1, "CORRUPTED": 1}
    assert s["corrupt_frames_total"] == 7

    d = s["duration_sec"]
    assert d["total"] == 12.0 and d["mean"] == 4.0 and d["median"] == 4.0
    assert d["min"] == 2.0 and d["max"] == 6.0 and d["missing"] == 1

    assert s["fps_distribution"] == {"25": 2, "30": 1}
    assert s["resolution_distribution"] == {"320x240": 2, "1280x720": 1}
    assert s["distinct_resolutions"] == 2
    assert s["fourcc_distribution"] == {"h264": 2, "MJPG": 1}
    assert s["container_distribution"] == {".mp4": 4}

    # duplicate content is REPORTED as a group, never collapsed or deleted
    assert s["duplicate_content_groups"] == 1
    assert s["duplicate_content_files"] == 2
    assert list(s["duplicates"].values()) == [["a", "b"]]

    # unknown metadata is counted, not fabricated
    assert s["entries_with_unknown_metadata"] == 1
    assert s["frame_count"]["missing"] == 1


def test_window_capacity_matches_sliding_window_math(tmp_path):
    rep, man = tmp_path / "reports", tmp_path / "manifests"
    # 100 frames: (100-32)//8+1 = 9 windows @32, (100-16)//8+1 = 11 @16
    # 20 frames: fits one 16-frame window, too short for 32 -> edge-padded
    entries = [
        _entry("long", "FIGHT", "train", "raw/long.mp4", "l1", frame_count=100,
               duration_sec=4.0, fps=25.0, width=64, height=48),
        _entry("short", "FIGHT", "train", "raw/short.mp4", "s1", frame_count=20,
               duration_sec=0.8, fps=25.0, width=64, height=48),
    ]
    probes = [
        _probe("long", "raw/long.mp4", "VALID", frame_count=100, fps=25.0,
               duration_sec=4.0, width=64, height=48, sha256="l1"),
        _probe("short", "raw/short.mp4", "VALID", frame_count=20, fps=25.0,
               duration_sec=0.8, width=64, height=48, sha256="s1"),
    ]
    s = summ.summarize_manifest(_write_manifest(rep, man, entries, probes), rep, [16, 32])
    tw = s["temporal_windows"]
    assert tw["stride"] == 8
    assert tw["totals"] == {"16": 12, "32": 10}
    assert tw["by_label"]["32"] == {"FIGHT": 10}
    assert tw["mean_per_video"]["32"] == 5.0
    assert tw["videos_shorter_than_window"] == {"16": 0, "32": 1}
    # capacity is computed from metadata only: nothing was extracted to disk
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifests", "reports"]


def test_markdown_render_reports_and_never_claims_clean_when_flagged(tmp_path):
    rep, man = tmp_path / "reports", tmp_path / "manifests"
    entries = [_entry("x", "FIGHT", "train", "raw/x.mp4", None, duration_sec=None,
                      fps=None, frame_count=None, width=None, height=None)]
    probes = [_probe("x", "raw/x.mp4", "CORRUPTED", corrupt_frames=3)]
    s = summ.summarize_manifest(_write_manifest(rep, man, entries, probes), rep, [16, 32])
    md = summ.render_markdown([s])
    assert "tiny_v1" in md
    assert "`x` CORRUPTED" in md
    assert "none in any manifest" not in md
