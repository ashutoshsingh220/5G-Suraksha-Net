"""Shared feature-cache builder used by the survey, the extractor CLI and training.

One code path builds the `.npz` cache so a survey run, a full extraction and a
training run cannot disagree about how features were produced. Clips already in
the cache are reused unless `force=True`, which means an 80-clip survey and a
later 2000-clip extraction share their work instead of repeating it.

Extraction is CPU/GPU-bound on YOLO+ByteTrack (~4.5 s per 150-frame clip at
640x360 on the development machine), so this is the expensive part of the
pipeline — roughly 150 minutes for all of RWF-2000 single-process, versus
seconds per epoch for the 16k-parameter GRU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from suraksha.logging_utils import get_logger
from suraksha.training.features import load_features, save_features

log = get_logger(__name__)


@dataclass
class ExtractionResult:
    clip_id: str
    label: str
    split: str
    n_windows: int = 0
    gate_passed: int = 0
    frames_decoded: int = 0
    note: str = ""
    seconds: float = 0.0
    cached: bool = False
    path: str = ""

    @property
    def zero_yield(self) -> bool:
        return self.n_windows == 0


@dataclass
class CacheBuildReport:
    requested: int = 0
    built: int = 0
    reused: int = 0
    failed: int = 0
    windows: int = 0
    zero_yield: int = 0
    seconds: float = 0.0
    results: list[ExtractionResult] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "requested": self.requested, "built": self.built, "reused": self.reused,
            "failed": self.failed, "windows": self.windows,
            "zero_yield_clips": self.zero_yield,
            "zero_yield_pct": round(100 * self.zero_yield / max(self.requested, 1), 1),
            "seconds": round(self.seconds, 1),
            "seconds_per_clip": round(self.seconds / max(self.built, 1), 2),
        }


def cache_path(cache_dir: Path | str, manifest: str, clip_id: str) -> Path:
    return Path(cache_dir) / manifest / f"{clip_id}.npz"


def build_cache(
    entries: Sequence,
    cache_dir: Path | str,
    *,
    manifest: str = "rwf2000_v1",
    project_root: Path | None = None,
    app_config=None,
    frame_shape: tuple[int, int] = (360, 640),
    device: str = "auto",
    force: bool = False,
    max_frames: int = 0,
    progress_every: int = 25,
) -> CacheBuildReport:
    """Extract + cache 8-dim pair features for `entries`, reusing what exists.

    `app_config` supplies detection/tracking/fight settings so the cache is built
    with exactly the parameters the live pipeline uses.
    """
    from suraksha.config import load_config
    from suraksha.fight.candidate import FightCandidateDetector
    from suraksha.training.features import PairFeatureExtractor

    cfg = app_config or load_config()
    root = Path(project_root) if project_root else Path.cwd()
    cache_dir = Path(cache_dir)
    report = CacheBuildReport(requested=len(entries))

    todo: list[tuple] = []
    for e in entries:
        dest = cache_path(cache_dir, manifest, e.clip_id)
        if dest.exists() and not force:
            vf = load_features(dest)
            if vf is not None:
                report.reused += 1
                report.windows += vf.n_windows
                report.zero_yield += 1 if vf.n_windows == 0 else 0
                report.results.append(ExtractionResult(
                    clip_id=e.clip_id, label=e.label, split=e.split,
                    n_windows=vf.n_windows,
                    gate_passed=int(vf.gate_passed.sum()) if vf.n_windows else 0,
                    frames_decoded=vf.frames_decoded, note=vf.note,
                    cached=True, path=e.path,
                ))
                continue
        todo.append((e, dest))

    if not todo:
        log.info("feature cache complete: %d clips reused, 0 to extract", report.reused)
        return report

    # Heavy imports stay behind the cache check: a fully cached run never loads
    # YOLO weights or touches the GPU.
    from suraksha.detection.tracker import MultiObjectTracker

    log.info("extracting %d clips (%d reused from cache)", len(todo), report.reused)
    tracker = MultiObjectTracker(cfg.detection, cfg.tracking, device=device)
    detector = FightCandidateDetector(cfg.fight.candidate, frame_shape)
    extractor = PairFeatureExtractor(
        tracker, detector,
        window_frames=cfg.fight.temporal.window_frames,
        stride_frames=cfg.fight.temporal.stride_frames,
        motion_energy_threshold=cfg.fight.candidate.motion_energy_threshold,
    )

    t_start = time.time()
    for i, (e, dest) in enumerate(todo, 1):
        src = Path(e.path)
        if not src.is_absolute():
            src = root / src
        t0 = time.time()
        try:
            vf = extractor.extract(
                e.clip_id, src, e.label, e.split,
                e.fps or 0.0, e.effective_frames or 0, max_frames=max_frames,
            )
        except Exception as exc:
            report.failed += 1
            report.results.append(ExtractionResult(
                clip_id=e.clip_id, label=e.label, split=e.split,
                note=f"EXCEPTION:{type(exc).__name__}:{exc}", path=e.path,
                seconds=time.time() - t0,
            ))
            log.warning("extraction failed for %s: %s", e.clip_id, exc)
            continue
        dt = time.time() - t0
        vf.path = e.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_features(vf, dest)
        report.built += 1
        report.seconds += dt
        report.windows += vf.n_windows
        report.zero_yield += 1 if vf.n_windows == 0 else 0
        report.results.append(ExtractionResult(
            clip_id=e.clip_id, label=e.label, split=e.split,
            n_windows=vf.n_windows,
            gate_passed=int(vf.gate_passed.sum()) if vf.n_windows else 0,
            frames_decoded=vf.frames_decoded, note=vf.note,
            seconds=dt, path=e.path,
        ))
        if progress_every and (i % progress_every == 0 or i == len(todo)):
            log.info("  %d/%d clips  windows=%d  zero-yield=%d  %.0fs elapsed",
                     i, len(todo), report.windows, report.zero_yield,
                     time.time() - t_start)

    return report


def select_cells(entries, per_cell: int,
                 cells: Sequence[tuple[str, str]]) -> list:
    """Deterministic sample: the first `per_cell` clip_ids of each (split, label).

    Sorting by clip_id (not by filesystem order) is what makes the sample
    reproducible across machines and runs.
    """
    picked: list = []
    for split, label in cells:
        pool = sorted((e for e in entries if e.split == split and e.label == label),
                      key=lambda e: e.clip_id)
        picked.extend(pool[:per_cell])
    return picked


def default_cells(train_split: str = "train", val_split: str = "val",
                  labels: Sequence[str] = ("FIGHT", "NON_FIGHT")) -> list[tuple[str, str]]:
    return [(s, l) for s in (train_split, val_split) for l in labels]
