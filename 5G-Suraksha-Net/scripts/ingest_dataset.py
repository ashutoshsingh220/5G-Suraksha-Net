#!/usr/bin/env python
"""Ingest a raw video dataset folder into the manifest system.

Pipeline: scan -> probe metadata -> sha256 -> manifest -> validate
          -> video-level split -> leakage check -> class stats -> reports

Folder conventions (auto-detected):
  - a path component matching a label alias (fight/nonfight/normal/...) sets the label
  - a path component 'train'/'val'/'test' marks a PREDEFINED official split
    (preserved by the splitter)
  - each file is treated as its own source video (group_id = relative path);
    for clip-collections cut from longer videos, pass --group-by parent

Usage:
  python scripts/ingest_dataset.py --root datasets/raw/rwf2000 ^
      --dataset-name rwf2000 --group A --license "research-use-only (RWF-2000 terms)" ^
      --source kaggle_mirror --manifest-name rwf2000_v1

  # Group B (evaluation, hard negatives in per-category subfolders):
  python scripts/ingest_dataset.py --root datasets/evaluation/raw ^
      --dataset-name eval_cctv_v1 --group B --license "per-source" ^
      --source mixed --manifest-name eval_cctv_v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}
SPLIT_DIRS = {"train": "train", "val": "val", "valid": "val", "validation": "val",
              "test": "test", "testing": "test"}
# Abuse018_x264 / Clapping_(107) / Normal_Videos_001 -> class prefix
PREFIX_RE = re.compile(r"^(?P<cls>[A-Za-z_]+?)(?=[0-9(])")


def source_class(rel: Path, mode: str) -> str | None:
    """Best-effort original class name, without inventing a label from it."""
    if mode == "prefix":
        m = PREFIX_RE.match(rel.stem)
        return m.group("cls").rstrip("_") if m else rel.stem
    dirs = [p for p in rel.parts[:-1] if p.lower() not in SPLIT_DIRS]
    return dirs[-1] if dirs else None


def group_key(rel: Path, mode: str, src_video: str | None = None) -> str:
    """Leakage group id. Clips of one original video MUST return the same key."""
    if src_video is not None:
        return src_video
    return str(rel.parent) if mode == "parent" else str(rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder containing the videos")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--manifest-name", required=True)
    ap.add_argument("--group", default="A", choices=["A", "B"],
                    help="A=training data, B=independent evaluation (never mixed)")
    ap.add_argument("--license", default="unknown", help="dataset license/terms string")
    ap.add_argument("--source", default="unknown", help="e.g. kaggle_mirror | official | self_recorded")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-check", default="sample", choices=["all", "sample", "none"])
    ap.add_argument("--no-sha", action="store_true", help="skip SHA256 (faster, weaker dup check)")
    ap.add_argument("--group-by", default="file", choices=["file", "parent"],
                    help="leakage grouping: each file, or all clips in one folder")
    ap.add_argument("--group-regex", default=None,
                    help="regex with named group 'grp' applied to the filename stem; "
                         "the match becomes group_id AND video_id. Use for clips cut "
                         "from longer videos, e.g. RWF-2000 '^(?P<grp>.+)_\\d+$' or "
                         "UCF101 '^v_(?P<grp>.+_g\\d+)_c\\d+$'")
    ap.add_argument("--name-regex", default=None,
                    help="only ingest files whose NAME matches this regex "
                         "(e.g. '^fi|^no' to pull Hockey out of a mixed folder)")
    ap.add_argument("--source-class-from", default="parent", choices=["parent", "prefix"],
                    help="where the original class name lives: parent folder "
                         "(UCF101) or filename prefix (UCF-Crime style)")
    ap.add_argument("--provenance", default=None,
                    help="acquisition note stamped into every entry's "
                         "notes.provenance (how WE got the files, not who owns them)")
    ap.add_argument("--skip-split", action="store_true",
                    help="keep existing/predefined splits only")
    args = ap.parse_args()

    import yaml

    from suraksha.config import PROJECT_ROOT
    from suraksha.data.manifest import (
        ClipEntry, Manifest, normalize_label, save_manifest,
    )
    from suraksha.data.splitting import assign_splits, check_leakage, split_class_stats
    from suraksha.data.validation import probe_video, validate_manifest, write_report
    from suraksha.logging_utils import setup_logging

    setup_logging("INFO")

    data_cfg = {}
    cfg_path = PROJECT_ROOT / "configs" / "data.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}
    windows = data_cfg.get("sequences", {}).get("windows", [16, 32])
    stride = data_cfg.get("sequences", {}).get("stride", 8)
    reports_dir = data_cfg.get("validation", {}).get("reports_dir", "datasets/reports")

    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if args.name_regex:
        rx = re.compile(args.name_regex)
        files = [p for p in files if rx.search(p.name)]
        print(f"  name filter /{args.name_regex}/ -> {len(files)} files")
    if not files:
        print(f"No videos found under {root}")
        return 1
    print(f"Found {len(files)} video files under {root}")

    m = Manifest(name=args.manifest_name, group=args.group,
                 license=args.license, task="fight_temporal")

    hard_neg_cats = set(data_cfg.get("hard_negatives", {}).get("categories", []))
    # Label-mapping PROPOSAL (configs/data.yaml). Recorded in notes only — the
    # canonical `label` field keeps just what the source dataset actually states.
    label_map = (data_cfg.get("label_mapping", {}) or {}).get(args.dataset_name, {}) or {}
    group_rx = re.compile(args.group_regex) if args.group_regex else None
    for i, f in enumerate(files):
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = Path(f.name)
        parts = [p.lower() for p in rel.parts[:-1]]

        label = next((normalize_label(p) for p in parts
                      if normalize_label(p) != "UNKNOWN"), "UNKNOWN")
        split_dir = next((SPLIT_DIRS[p] for p in parts if p in SPLIT_DIRS), None)
        split = split_dir or "train"

        category = next((p for p in parts if p in hard_neg_cats), None)
        notes: dict = {}
        if args.provenance:
            notes["provenance"] = args.provenance
        if category:
            notes["hard_negative_category"] = category
        if split_dir:
            # marks an OFFICIAL split so assign_splits() never reshuffles it
            notes["predefined_split"] = split_dir

        src_class = source_class(rel, args.source_class_from)
        if src_class:
            notes["source_class"] = src_class
        proposed = (label_map.get(src_class or "")
                    or label_map.get((src_class or "").lower())
                    or label_map.get("__default__"))
        if proposed:
            notes["proposed_label"] = proposed

        match = group_rx.search(rel.stem) if group_rx is not None else None
        src_video = match.group("grp") if match else None
        group_id = group_key(rel, args.group_by, src_video)
        # video_id = ORIGINAL source video (leakage grouping); the clip stem is
        # kept in notes so derived clips stay individually identifiable.
        video_id = src_video or rel.stem
        if src_video and src_video != rel.stem:
            notes["clip_stem"] = rel.stem

        e = ClipEntry(
            clip_id=f"{args.dataset_name}_{i:05d}",
            path=str((root / rel).relative_to(PROJECT_ROOT)).replace("\\", "/")
                 if PROJECT_ROOT in (root / rel).parents else str(f).replace("\\", "/"),
            label=label,
            split=split,
            dataset_name=args.dataset_name,
            video_id=video_id,
            group_id=group_id.replace("\\", "/"),
            license=args.license,
            source=args.source,
            notes=notes,
        )
        m.add(e)
        if (i + 1) % 200 == 0:
            print(f"  ingested {i + 1}/{len(files)}")

    # ---- validate (probes metadata, sha256, corruption, duplicates) ----
    print("Validating files (decode_check=%s, sha256=%s)..." % (args.decode_check, not args.no_sha))
    report = validate_manifest(m, decode_check=args.decode_check,
                               with_sha256=not args.no_sha)
    json_path, md_path = write_report(report, reports_dir)
    print(f"Validation summary: {report.summary}")
    print(f"Reports: {json_path}\n         {md_path}")

    # ---- splits ----
    if not args.skip_split:
        ratios = data_cfg.get("splitting", {}).get("ratios",
                                                   {"train": 0.7, "val": 0.15, "test": 0.15})
        counts = assign_splits(m, ratios=ratios, seed=args.seed,
                               respect_predefined=data_cfg.get("splitting", {})
                               .get("respect_predefined", True))
        print(f"Split counts: {counts}")

    violations = check_leakage(m)
    if violations:
        print(f"LEAKAGE/DUPLICATE PROBLEMS ({len(violations)}):")
        for v in violations[:20]:
            print(" -", v)
    else:
        print("Leakage check: CLEAN (no group or duplicate content spans splits)")

    # ---- class stats ----
    stats = split_class_stats(m, windows=windows, stride=stride)
    print("\nCLASS STATISTICS")
    print(json.dumps(stats, indent=2))

    save_manifest(m)
    stats_path = Path(md_path).with_name(f"{m.name}_stats.json")
    stats_path.write_text(json.dumps({
        "class_stats": stats,
        "validation_summary": report.summary,
        "leakage_violations": violations,
        "split_counts": m.split_counts(),
    }, indent=2), encoding="utf-8")
    print(f"\nManifest saved. Stats: {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
