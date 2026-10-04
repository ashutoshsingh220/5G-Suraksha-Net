"""Known duplicate-content leakage in the OFFICIAL RWF-2000 train/val split.

RWF-2000's own split is not content-disjoint. Measured on the ingested
manifest (sha256 over the raw files, `datasets/manifests/rwf2000_v1.json`):

    11 sha256 groups contain more than one clip
     6 of those groups straddle the train/val boundary
    -> 6 val clips are byte-identical to a clip the model also trains on

Every straddling group is NON_FIGHT, so the leak inflates *specificity*, not
sensitivity.

This module does not repair the split. PROMPT 4 forbids moving, renaming or
modifying the official files, and inventing a private "clean" split would make
our numbers incomparable with every published RWF-2000 result. Instead the leak
is made explicit: the registry is computed from the manifest, written to
`datasets/reports/rwf2000_duplicate_leakage.json`, and validation metrics are
always reported twice — once on the full official val set, once with the
straddling val clips excluded. Neither number may be described as a
leakage-free estimate; the honest statement is that the official split is
leaky by this much.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from suraksha.logging_utils import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "rwf2000_v1"
REPORT_PATH = Path("datasets/reports/rwf2000_duplicate_leakage.json")


@dataclass
class DuplicateGroup:
    sha256: str
    clip_ids: list[str]
    paths: list[str]
    label: str
    splits: list[str]

    @property
    def straddles_split(self) -> bool:
        return len(set(self.splits)) > 1


@dataclass
class LeakageReport:
    manifest: str = MANIFEST_NAME
    total_clips: int = 0
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    straddling_groups: list[DuplicateGroup] = field(default_factory=list)
    leaked_val_clip_ids: list[str] = field(default_factory=list)
    leaked_train_clip_ids: list[str] = field(default_factory=list)

    @property
    def leaked_val_set(self) -> frozenset[str]:
        return frozenset(self.leaked_val_clip_ids)

    def val_split_of(self, clip_id: str) -> bool:
        return clip_id in self.leaked_val_set

    def clean_val_ids(self, val_clip_ids: Iterable[str]) -> list[str]:
        """Official val clips whose content does NOT also appear in train."""
        leaked = self.leaked_val_set
        return [c for c in val_clip_ids if c not in leaked]

    def summary(self) -> dict:
        return {
            "manifest": self.manifest,
            "total_clips": self.total_clips,
            "duplicate_groups": len(self.duplicate_groups),
            "groups_straddling_train_val": len(self.straddling_groups),
            "leaked_val_clips": len(self.leaked_val_clip_ids),
            "leaked_train_clips": len(self.leaked_train_clip_ids),
            "leaked_val_clip_ids": list(self.leaked_val_clip_ids),
            "note": (
                "Official RWF-2000 val is NOT content-disjoint from train. "
                "Validation metrics must be reported both including and "
                "excluding these val clips, and never described as leakage-free."
            ),
        }

    def to_json(self) -> dict:
        return {
            **self.summary(),
            "straddling_groups": [asdict(g) for g in self.straddling_groups],
            "within_split_duplicate_groups": [
                asdict(g) for g in self.duplicate_groups if not g.straddles_split
            ],
        }


def build_leakage_registry(entries: Sequence, manifest: str = MANIFEST_NAME) -> LeakageReport:
    """Group manifest entries by sha256 and flag groups crossing the split line."""
    report = LeakageReport(manifest=manifest, total_clips=len(entries))

    by_hash: dict[str, list] = defaultdict(list)
    for e in entries:
        if getattr(e, "sha256", None):
            by_hash[e.sha256].append(e)

    for sha, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        splits = sorted({e.split for e in group})
        dg = DuplicateGroup(
            sha256=sha,
            clip_ids=sorted(e.clip_id for e in group),
            paths=sorted(e.path for e in group),
            label=group[0].label,
            splits=splits,
        )
        report.duplicate_groups.append(dg)
        if not dg.straddles_split:
            continue
        report.straddling_groups.append(dg)
        for e in group:
            if e.split == "val":
                report.leaked_val_clip_ids.append(e.clip_id)
            elif e.split == "train":
                report.leaked_train_clip_ids.append(e.clip_id)

    report.leaked_val_clip_ids.sort()
    report.leaked_train_clip_ids.sort()
    log.info(
        "RWF-2000 leakage registry: %d duplicate groups, %d straddle train/val "
        "(%d val clips affected)",
        len(report.duplicate_groups),
        len(report.straddling_groups),
        len(report.leaked_val_clip_ids),
    )
    return report


def load_or_build_registry(
    manifest_name: str = MANIFEST_NAME, report_path: Path | str | None = None
) -> tuple[LeakageReport, Path | None]:
    """Build the registry from the manifest and persist it as an audit artifact."""
    from suraksha.data.manifest import load_manifest

    manifest = load_manifest(manifest_name)
    report = build_leakage_registry(manifest.entries, manifest_name)

    dest = Path(report_path) if report_path else REPORT_PATH
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    except OSError as exc:  # read-only checkout, CI sandbox, ...
        log.warning("could not persist leakage registry to %s: %s", dest, exc)
        return report, None
    return report, dest
