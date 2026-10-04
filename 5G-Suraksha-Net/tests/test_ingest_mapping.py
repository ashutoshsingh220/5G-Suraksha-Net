"""Ingestion helpers: source-class detection, leakage grouping, label mapping.

Guards the audit rules that keep semantic mistakes out of training data:
  - a class name is recorded, never silently turned into a label
  - combat-like sport classes stay AMBIGUOUS, never NON_FIGHT
  - clips cut from one source video share a leakage group
"""
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ingest_dataset", ROOT / "scripts" / "ingest_dataset.py")
ingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest)


@pytest.fixture(scope="module")
def data_cfg():
    with open(ROOT / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_source_class_from_parent_skips_split_dirs():
    rel = Path("train") / "Fight" / "a2qLzOu_n0s_0.avi"
    assert ingest.source_class(rel, "parent") == "Fight"
    rel2 = Path("test") / "Fencing" / "v_Fencing_g01_c01.avi"
    assert ingest.source_class(rel2, "parent") == "Fencing"


def test_source_class_from_filename_prefix():
    # UCF-Crime style: class lives in the filename, folders are only splits
    assert ingest.source_class(Path("train/Abuse018_x264.mp4"), "prefix") == "Abuse"
    assert ingest.source_class(Path("valid/Clapping_(107).mp4"), "prefix") == "Clapping"
    assert ingest.source_class(Path("train/Walking_While_Using_Phone004_x264.mp4"),
                              "prefix") == "Walking_While_Using_Phone"
    assert ingest.source_class(Path("train/Normal_Videos_001_x264.mp4"),
                              "prefix") == "Normal_Videos"


def test_group_key_groups_clips_of_one_source_video():
    """RWF-2000 / UCF101 clips from one source video must never cross splits."""
    import re
    rx = re.compile(r"^(?P<grp>.+)_\d+$")
    keys = {ingest.group_key(Path("train/Fight") / f"vid123_{i}.avi", "file", "vid123")
            for i in range(4)}
    assert keys == {"vid123"}
    assert rx.search("a2qLzOu_n0s_7").group("grp") == "a2qLzOu_n0s"

    ucf_rx = re.compile(r"^v_(?P<grp>.+_g\d+)_c\d+$")
    assert ucf_rx.search("v_Fencing_g01_c02").group("grp") == "Fencing_g01"
    # no match -> fall back to the file itself (never a bogus shared group)
    assert ingest.group_key(Path("a/b/weird_name.mp4"), "file", None) == str(Path("a/b/weird_name.mp4"))


def test_name_regex_separates_mixed_folders():
    """RLVS and Hockey Fight ship in the SAME folders; only names tell them apart."""
    import re
    hockey = re.compile(r"^(fi|no)[0-9]")
    rlvs = re.compile(r"^(V_|NV_)")
    names = ["V_1.mp4", "NV_602.avi", "fi1_xvid.mp4", "no100_xvid.mp4"]
    assert [n for n in names if hockey.search(n)] == ["fi1_xvid.mp4", "no100_xvid.mp4"]
    assert [n for n in names if rlvs.search(n)] == ["V_1.mp4", "NV_602.avi"]


def test_label_mapping_never_calls_combat_sport_non_fight(data_cfg):
    mapping = data_cfg["label_mapping"]
    ambiguous = ["Fencing", "BoxingPunchingBag", "BoxingSpeedBag", "Punch",
                 "Wrestling", "SumoWrestling", "Nunchucks"]
    for cls in ambiguous:
        assert mapping["ucf101"][cls] == "AMBIGUOUS_COMBAT_LIKE", \
            f"{cls} must stay ambiguous, not NON_FIGHT"
    # hard negatives stay hard negatives
    for cls in ["WalkingWithDog", "SalsaSpin", "IceDancing"]:
        assert mapping["ucf101"][cls] == "HARD_NEGATIVE"
    # real CCTV violence is the only thing proposed as FIGHT in Group B
    assert mapping["ucf_crime_subset"]["Fighting"] == "FIGHT"
    assert mapping["ucf_crime_subset"]["Assault"] == "FIGHT"
    assert mapping["ucf_crime_subset"]["Clapping"] == "HARD_NEGATIVE"


def test_label_mapping_covers_every_ingested_dataset(data_cfg):
    mapping = data_cfg["label_mapping"]
    for ds in ["rwf2000", "rlvs", "hockey_fight", "movies_fight",
               "ucf101", "ucf_crime_subset", "crowd_abnormal"]:
        assert ds in mapping, f"no label-mapping proposal for {ds}"


def test_rwf2000_label_aliases_are_unambiguous():
    from suraksha.data.manifest import LABEL_FIGHT, LABEL_NON_FIGHT, normalize_label

    assert normalize_label("Fight") == LABEL_FIGHT
    assert normalize_label("NonFight") == LABEL_NON_FIGHT
    assert normalize_label("fights") == LABEL_FIGHT        # movies_fight folder
    assert normalize_label("noFights") == LABEL_NON_FIGHT
    assert normalize_label("Violence") == LABEL_FIGHT      # RLVS folder
    assert normalize_label("NonViolence") == LABEL_NON_FIGHT
    # still never invents a label for unmapped raw names
    assert normalize_label("Clapping") == "UNKNOWN"
    assert normalize_label("Meet_and_Split") == "UNKNOWN"
