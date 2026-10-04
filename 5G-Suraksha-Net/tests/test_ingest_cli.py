"""End-to-end CLI test for scripts/ingest_dataset.py on the synthetic fixture.

Uses datasets/raw/_demo (8 tiny generated clips, one deliberate byte-identical
copy) so the whole scan -> probe -> sha256 -> split -> leakage -> stats chain is
exercised without touching any real downloaded dataset.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ingest_dataset.py"
MANIFEST_NAME = "_it_ingest_cli"


def _cleanup():
    (ROOT / "datasets" / "manifests" / f"{MANIFEST_NAME}.json").unlink(missing_ok=True)
    for p in (ROOT / "datasets" / "reports").glob(f"{MANIFEST_NAME}*"):
        p.unlink(missing_ok=True)


@pytest.fixture
def ingest_run():
    _cleanup()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--root", "datasets/raw/_demo",
         "--dataset-name", "it_demo",
         "--manifest-name", MANIFEST_NAME,
         "--group", "A",
         "--license", "n/a (synthetic)",
         "--source", "synthetic",
         "--provenance", "integration-test fixture",
         "--decode-check", "sample"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    yield proc
    _cleanup()


def test_cli_ingest_produces_manifest(ingest_run):
    assert ingest_run.returncode == 0, ingest_run.stdout + ingest_run.stderr
    path = ROOT / "datasets" / "manifests" / f"{MANIFEST_NAME}.json"
    assert path.exists()
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["group"] == "A"
    assert len(m["entries"]) == 8

    labels = {}
    for e in m["entries"]:
        labels[e["label"]] = labels.get(e["label"], 0) + 1
    assert labels == {"FIGHT": 4, "NON_FIGHT": 4}


def test_cli_records_metadata_and_provenance(ingest_run):
    m = json.loads((ROOT / "datasets" / "manifests" / f"{MANIFEST_NAME}.json")
                   .read_text(encoding="utf-8"))
    for e in m["entries"]:
        assert e["notes"]["provenance"] == "integration-test fixture"
        assert e["notes"]["source_class"] in ("fight", "nonfight")
        assert e["frame_count"] and e["fps"] and e["width"] and e["height"]
        assert e["duration_sec"] > 0
        assert len(e["sha256"]) == 64


def test_cli_detects_duplicate_without_deleting(ingest_run):
    """The byte-identical copy must be REPORTED as DUPLICATE, never removed."""
    m = json.loads((ROOT / "datasets" / "manifests" / f"{MANIFEST_NAME}.json")
                   .read_text(encoding="utf-8"))
    assert (ROOT / "datasets" / "raw" / "_demo" / "fight" / "clip_00_copy.mp4").exists()

    hashes = [e["sha256"] for e in m["entries"] if e["sha256"]]
    assert len(hashes) != len(set(hashes)), "expected a duplicate sha256 pair"

    reports = sorted((ROOT / "datasets" / "reports").glob(f"{MANIFEST_NAME}_validation_*.json"))
    assert reports, "no validation report written"
    rep = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert rep["summary"].get("DUPLICATE", 0) >= 1
    assert "DUP-CONTENT" in ingest_run.stdout


def test_cli_assigns_video_level_splits_when_none_predefined(ingest_run):
    m = json.loads((ROOT / "datasets" / "manifests" / f"{MANIFEST_NAME}.json")
                   .read_text(encoding="utf-8"))
    splits = {e["split"] for e in m["entries"]}
    assert splits <= {"train", "val", "test"}
    assert len(splits) >= 2, "fixture should be split across at least two sets"
    # no official split folders in _demo -> nothing may claim a predefined split
    assert all("predefined_split" not in e["notes"] for e in m["entries"])


def test_cli_writes_stats_with_sequence_counts(ingest_run):
    stats_path = sorted((ROOT / "datasets" / "reports").glob(f"{MANIFEST_NAME}_stats.json"))
    assert stats_path, "no stats file written"
    s = json.loads(stats_path[-1].read_text(encoding="utf-8"))
    assert set(s) >= {"class_stats", "validation_summary", "leakage_violations", "split_counts"}
    seqs = s["class_stats"]["by_split"]["train"]["sequences"]
    for _label, per_window in seqs.items():
        # JSON round-trip turns the int window sizes into string keys
        assert "16" in per_window and "32" in per_window
        assert per_window["16"] >= per_window["32"]   # shorter window -> more sequences
