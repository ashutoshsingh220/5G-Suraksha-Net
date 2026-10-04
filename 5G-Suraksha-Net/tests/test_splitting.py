"""Video-level split tests: determinism, leakage rules, predefined preservation."""
from suraksha.data.manifest import ClipEntry, Manifest
from suraksha.data.splitting import assign_splits, check_leakage, split_class_stats


def _manifest(n_videos=40, clips_per_video=3, predefined=None):
    m = Manifest(name="_split_t")
    for v in range(n_videos):
        for c in range(clips_per_video):
            split = "train"
            if predefined and v in predefined:
                split = predefined[v]
            m.add(ClipEntry(
                clip_id=f"v{v:03d}_c{c}",
                path=f"raw/v{v:03d}/clip{c}.mp4",
                label="FIGHT" if v % 2 == 0 else "NON_FIGHT",
                split=split,
                group_id=f"v{v:03d}",   # 3 clips share one source video
            ))
    return m


def test_splits_are_video_level():
    m = _manifest()
    assign_splits(m, seed=42)
    per_group = {}
    for e in m.entries:
        per_group.setdefault(e.group_id, set()).add(e.split)
    assert all(len(s) == 1 for s in per_group.values()), "clips of one video crossed splits"
    assert check_leakage(m) == []


def test_deterministic_with_seed():
    m1, m2 = _manifest(), _manifest()
    assign_splits(m1, seed=7)
    assign_splits(m2, seed=7)
    assert [e.split for e in m1.entries] == [e.split for e in m2.entries]

    m3 = _manifest()
    assign_splits(m3, seed=8)
    assert [e.split for e in m3.entries] != [e.split for e in m1.entries]


def test_predefined_test_split_preserved_and_not_rehashed():
    predefined = {v: "test" for v in range(30, 40)}  # official test videos
    m = _manifest(predefined=predefined)
    assign_splits(m, seed=42, respect_predefined=True)

    for e in m.entries:
        if e.group_id in {f"v{v:03d}" for v in range(30, 40)}:
            assert e.split == "test"
        else:
            assert e.split in ("train", "val")  # never hashed into official test
    assert check_leakage(m) == []


def test_leakage_detected_when_group_spans_splits():
    m = _manifest(n_videos=2, clips_per_video=3)
    m.entries[0].split = "train"
    m.entries[1].split = "test"   # same group_id v000 -> violation
    m.entries[2].split = "train"
    violations = check_leakage(m)
    assert any("LEAK" in v and "v000" in v for v in violations)


def test_official_train_split_is_never_reshuffled():
    """RWF-2000 ships official train/val folders — both must be frozen.

    'train' is also the default split for datasets with NO official split, so it
    only counts as official when the ingester marked it in notes.
    """
    m = Manifest(name="_official_split")
    for v in range(30):
        official = "train" if v < 24 else "val"     # 80/20 official split
        m.add(ClipEntry(clip_id=f"c{v:03d}", path=f"raw/{official}/v{v:03d}.avi",
                        label="FIGHT", split=official, group_id=f"v{v:03d}",
                        notes={"predefined_split": official}))
    before = {e.clip_id: e.split for e in m.entries}
    assign_splits(m, seed=42, respect_predefined=True)
    after = {e.clip_id: e.split for e in m.entries}
    assert before == after, "official train/val split was reshuffled"
    assert m.split_counts() == {"train": 24, "val": 6}
    assert check_leakage(m) == []


def test_unmarked_train_default_is_still_hashed():
    """Datasets with no official split must still get a real train/val/test split."""
    m = Manifest(name="_no_official_split")
    for v in range(40):
        m.add(ClipEntry(clip_id=f"c{v:03d}", path=f"raw/v{v:03d}.avi",
                        label="FIGHT", group_id=f"v{v:03d}"))   # split defaults to train
    counts = assign_splits(m, seed=42)
    assert set(counts) == {"train", "val", "test"}
    assert counts["val"] > 0 and counts["test"] > 0


def test_duplicate_content_never_spans_splits():
    """Same bytes under two filenames = one leakage unit, whatever the grouping."""
    m = Manifest(name="_dup_split")
    same = "ab" * 32
    for i in range(30):
        m.add(ClipEntry(clip_id=f"u{i:03d}", path=f"raw/u{i:03d}.mp4", label="FIGHT",
                        group_id=f"u{i:03d}", sha256=f"{i:064d}"))
    # two unrelated groups holding byte-identical content
    m.add(ClipEntry(clip_id="dupA", path="raw/dupA.mp4", label="FIGHT",
                    group_id="dupA", sha256=same))
    m.add(ClipEntry(clip_id="dupB", path="other/dupB.mp4", label="FIGHT",
                    group_id="dupB", sha256=same))

    assign_splits(m, seed=42)
    by_id = {e.clip_id: e.split for e in m.entries}
    assert by_id["dupA"] == by_id["dupB"], "identical content landed in two splits"

    violations = check_leakage(m)
    assert not any(v.startswith("LEAK") for v in violations)
    assert any("DUP-CONTENT" in v for v in violations)   # still reported, not deleted


def test_official_split_outranks_duplicate_merging():
    """RWF-2000 regression: identical bytes sitting in official train AND val.

    Merging duplicates must never move an officially-split file — not into the
    other official split, and not into a 'test' split the dataset does not have.
    The straddle is disclosed by check_leakage() instead of silently repaired.
    """
    m = Manifest(name="_official_dup")
    same = "cd" * 32
    for v in range(30):
        official = "train" if v < 24 else "val"     # every file officially split
        m.add(ClipEntry(clip_id=f"c{v:03d}", path=f"raw/{official}/v{v:03d}.avi",
                        label="FIGHT", split=official, group_id=f"v{v:03d}",
                        sha256=f"{v:064d}", notes={"predefined_split": official}))
    # ...except one byte-identical pair straddling the official train/val boundary
    m.add(ClipEntry(clip_id="trainTwin", path="raw/train/Fight/t_0.avi", label="FIGHT",
                    group_id="t", sha256=same, split="train",
                    notes={"predefined_split": "train"}))
    m.add(ClipEntry(clip_id="valTwin", path="raw/val/Fight/u_0.avi", label="FIGHT",
                    group_id="u", sha256=same, split="val",
                    notes={"predefined_split": "val"}))
    before = {e.clip_id: e.split for e in m.entries}

    counts = assign_splits(m, seed=42)
    after = {e.clip_id: e.split for e in m.entries}

    assert before == after, "an officially-split file was moved"
    assert after["trainTwin"] == "train" and after["valTwin"] == "val"
    assert counts == {"train": 25, "val": 7}
    assert "test" not in counts, f"fabricated a test split the dataset does not have: {counts}"

    violations = check_leakage(m)
    assert any(v.startswith("LEAK") and "spans splits ['train', 'val']" in v
               for v in violations), "official-split duplicate straddle was not reported"


def test_unmarked_duplicate_follows_its_official_twin():
    """A file with no official split adopts the split of its identical twin."""
    m = Manifest(name="_follow_twin")
    same = "ef" * 32
    m.add(ClipEntry(clip_id="official", path="test/NonFight/c_0.avi", label="NON_FIGHT",
                    group_id="c", sha256=same, split="test",
                    notes={"predefined_split": "test"}))
    m.add(ClipEntry(clip_id="copyNoSplit", path="extra/copy.mp4", label="NON_FIGHT",
                    group_id="copy", sha256=same))

    assign_splits(m, seed=42)
    by_id = {e.clip_id: e.split for e in m.entries}
    assert by_id["official"] == "test"
    assert by_id["copyNoSplit"] == "test", "unmarked duplicate did not follow its official twin"
    assert not any(v.startswith("LEAK") for v in check_leakage(m))


def test_class_stats_shape():
    m = _manifest()
    assign_splits(m, seed=42)
    for e in m.entries:
        e.frame_count, e.duration_sec = 64, 5.0
    stats = split_class_stats(m, windows=[16, 32], stride=8)
    assert set(stats["by_split"]) == {"train", "val", "test"}
    assert stats["by_label"] == {"FIGHT": 60, "NON_FIGHT": 60}
    train = stats["by_split"]["train"]
    assert train["videos"] == 120 - stats["by_split"]["val"]["videos"] - stats["by_split"]["test"]["videos"]
    # sequences calculable per clip: 64 frames, w=16, s=8 -> 7 windows per clip
    for lab, seqs in train["sequences"].items():
        assert seqs[16] == train["labels"][lab] * 7
        assert seqs[32] == train["labels"][lab] * 5
