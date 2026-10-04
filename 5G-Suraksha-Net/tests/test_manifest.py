import pytest

from suraksha.data.manifest import (
    ClipEntry,
    Manifest,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def _entry(i=0, label="fight", split="train"):
    return ClipEntry(clip_id=f"c{i:03d}", path=f"datasets/raw/{i}.mp4",
                     label=label, split=split, source="synthetic")


def test_manifest_roundtrip():
    m = Manifest(name="_test_manifest", task="fight_temporal")
    m.add(_entry(0))
    m.add(_entry(1, label="no_fight", split="val"))
    path = save_manifest(m)
    try:
        back = load_manifest("_test_manifest")
        assert len(back.entries) == 2
        assert back.by_label() == {"fight": 1, "no_fight": 1}
        assert back.split_counts() == {"train": 1, "val": 1}
        assert validate_manifest(back) == []
    finally:
        path.unlink(missing_ok=True)


def test_duplicate_clip_id_rejected():
    m = Manifest(name="_dup")
    m.add(_entry(0))
    with pytest.raises(ValueError):
        m.add(_entry(0))


def test_validate_flags_bad_split():
    m = Manifest(name="_bad")
    m.add(ClipEntry(clip_id="x", path="a.mp4", label="fight", split="holdout"))
    problems = validate_manifest(m)
    assert any("split" in p for p in problems)


def test_sample_manifest_valid():
    m = load_manifest("example_fight_v0")
    assert validate_manifest(m) == []
    assert m.task == "fight_temporal"


def test_label_normalization():
    from suraksha.data.manifest import LABEL_FIGHT, LABEL_NON_FIGHT, LABEL_UNKNOWN, normalize_label

    assert normalize_label("fight") == LABEL_FIGHT
    assert normalize_label("nonfight") == LABEL_NON_FIGHT   # RWF-2000 folder name
    assert normalize_label("Normal_Videos") == LABEL_UNKNOWN  # unmapped raw -> never invented
    assert normalize_label("no_fight") == LABEL_NON_FIGHT
    assert normalize_label("Fighting") == LABEL_FIGHT


def test_new_schema_fields_and_grouping():
    e = ClipEntry(clip_id="x", path="a.mp4", label="FIGHT", dataset_name="rwf2000",
                  video_id="0001", license="research-use-only", duration_sec=5.0,
                  width=720, height=480, frame_count=150, fps=30.0, sha256="ab" * 32)
    assert e.effective_group == "0001"          # video_id used when group_id empty
    assert e.effective_frames == 150
    e2 = ClipEntry(clip_id="y", path="b.mp4", num_frames=42)
    assert e2.effective_group == "y"            # falls back to clip_id
    assert e2.effective_frames == 42            # legacy alias still works
