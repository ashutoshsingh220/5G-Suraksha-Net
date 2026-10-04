"""Quantitatively measure fight candidate gating behavior across datasets.

Measures bounding box IoU distributions, pairwise relative distance, and motion
energy across RWF-2000 and UCF-Crime surveillance video to diagnose candidate
starvation and evaluate relaxed proximity gating.

Outputs:
  - datasets/reports/candidate_gating_analysis.json
  - datasets/reports/candidate_gating_analysis.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suraksha.config import load_config
from suraksha.data.manifest import load_manifest
from suraksha.detection.tracker import MultiObjectTracker, TrackedPerson
from suraksha.fight.candidate import FightCandidateDetector, _iou, _union
from suraksha.logging_utils import setup_logging


@dataclass
class VideoGatingStats:
    clip_id: str
    manifest: str
    group_name: str
    source_class: str
    ground_truth: str  # FIGHT or NON_FIGHT
    duration_s: float
    fps: float
    resolution: str
    frames_processed: int
    frames_with_ge2_persons: int
    max_persons: int
    mean_persons: float
    max_pair_iou: float
    min_pair_rel_dist: float
    frames_iou_ge_005: int
    frames_dist_le_10: int
    frames_dist_le_15: int
    frames_dist_le_20: int
    candidate_emitted_current: bool
    candidate_emitted_dist_15: bool
    candidate_emitted_dist_20: bool
    iou_distribution: list[float] = field(default_factory=list)
    dist_distribution: list[float] = field(default_factory=list)


def select_eval_corpus(max_per_cat: int = 25) -> list[dict]:
    """Build balanced sample across RWF-2000 and UCF-Crime."""
    rwf = load_manifest("rwf2000_v1")
    ucf_crime = load_manifest("ucf_crime_subset_eval_v1")

    # 1. RWF-2000 val FIGHT
    rwf_fights = [
        {"entry": e, "manifest": "rwf2000_v1", "group": "rwf2000_fight", "gt": "FIGHT", "cls": "FIGHT"}
        for e in rwf.entries if e.split == "val" and e.label == "FIGHT"
    ][:max_per_cat]

    # 2. RWF-2000 val NON_FIGHT
    rwf_non_fights = [
        {"entry": e, "manifest": "rwf2000_v1", "group": "rwf2000_non_fight", "gt": "NON_FIGHT", "cls": "NON_FIGHT"}
        for e in rwf.entries if e.split == "val" and e.label == "NON_FIGHT"
    ][:max_per_cat]

    # 3. UCF-Crime FIGHT (explicitly sample Fighting clips, 320x240 CCTV)
    ucf_fight_entries = [
        e for e in ucf_crime.entries
        if e.notes.get("source_class") == "Fighting" and e.width == 320
    ]
    ucf_fight_entries.sort(key=lambda x: x.clip_id)
    ucf_fights = [
        {"entry": e, "manifest": "ucf_crime_subset_eval_v1", "group": "ucf_crime_fight", "gt": "FIGHT", "cls": "Fighting"}
        for e in ucf_fight_entries[:max_per_cat]
    ]

    # 4. UCF-Crime Normal (Normal_Videos, 320x240 CCTV)
    ucf_normal_entries = [
        e for e in ucf_crime.entries
        if e.notes.get("source_class") == "Normal_Videos" and e.width == 320
    ]
    ucf_normal_entries.sort(key=lambda x: x.clip_id)
    ucf_normals = [
        {"entry": e, "manifest": "ucf_crime_subset_eval_v1", "group": "ucf_crime_normal", "gt": "NON_FIGHT", "cls": "Normal_Videos"}
        for e in ucf_normal_entries[:max_per_cat]
    ]

    corpus = rwf_fights + rwf_non_fights + ucf_fights + ucf_normals
    return corpus


def compute_histograms(values: list[float], bins: list[float]) -> dict[str, int]:
    """Bucket values into discrete bins."""
    counts = defaultdict(int)
    for v in values:
        placed = False
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            if low <= v < high:
                counts[f"[{low:.2f}, {high:.2f})"] += 1
                placed = True
                break
        if not placed:
            if v >= bins[-1]:
                counts[f">={bins[-1]:.2f}"] += 1
            else:
                counts[f"<{bins[0]:.2f}"] += 1
    return dict(counts)


def analyze_video(
    item: dict,
    tracker: MultiObjectTracker,
    detector: FightCandidateDetector,
    max_frames: int = 150,
) -> VideoGatingStats:
    e = item["entry"]
    cap = cv2.VideoCapture(str(ROOT / e.path))
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(e.path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res_str = f"{raw_w}x{raw_h}"

    frame_w, frame_h = 640, 360
    if callable(getattr(tracker, "reset", None)):
        tracker.reset()
    detector.reset()
    detector.fps = fps

    # Tracking simulation with 3 gating variants
    # Current gate: iou >= 0.05
    # Relaxed gate 1: iou >= 0.05 OR rel_dist <= 1.5
    # Relaxed gate 2: iou >= 0.05 OR rel_dist <= 2.0
    active_current: dict[tuple[int, int], int] = defaultdict(int)
    active_dist15: dict[tuple[int, int], int] = defaultdict(int)
    active_dist20: dict[tuple[int, int], int] = defaultdict(int)

    cand_emitted_current = False
    cand_emitted_dist15 = False
    cand_emitted_dist20 = False

    frames_processed = 0
    frames_ge2_persons = 0
    person_counts: list[int] = []

    all_ious: list[float] = []
    all_dists: list[float] = []

    max_pair_iou = 0.0
    min_pair_rel_dist = 999.0

    frames_iou_ge_005 = 0
    frames_dist_le_10 = 0
    frames_dist_le_15 = 0
    frames_dist_le_20 = 0

    while True:
        ret, frame = cap.read()
        if not ret or frames_processed >= max_frames:
            break

        if (frame.shape[1], frame.shape[0]) != (frame_w, frame_h):
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        detector._frame_seq += 1
        tracks = tracker.update(frame)
        num_persons = len(tracks)
        person_counts.append(num_persons)

        frame_has_iou_ge_005 = False
        frame_has_dist_le_10 = False
        frame_has_dist_le_15 = False
        frame_has_dist_le_20 = False

        if num_persons >= 2:
            frames_ge2_persons += 1
            pairs = list(itertools.combinations(tracks, 2))

            for ta, tb in pairs:
                key = (min(ta.track_id, tb.track_id), max(ta.track_id, tb.track_id))
                iou = _iou(ta.bbox_xyxy, tb.bbox_xyxy)
                dist = float(np.linalg.norm(ta.center - tb.center))
                scale = max((ta.width + tb.width) / 2.0, 1e-3)
                rel_dist = dist / scale

                all_ious.append(iou)
                all_dists.append(rel_dist)

                if iou > max_pair_iou:
                    max_pair_iou = iou
                if rel_dist < min_pair_rel_dist:
                    min_pair_rel_dist = rel_dist

                if iou >= 0.05:
                    frame_has_iou_ge_005 = True
                if rel_dist <= 1.0:
                    frame_has_dist_le_10 = True
                if rel_dist <= 1.5:
                    frame_has_dist_le_15 = True
                if rel_dist <= 2.0:
                    frame_has_dist_le_20 = True

                # Compute motion
                roi = _union(ta.bbox_xyxy, tb.bbox_xyxy)
                flow = detector._flow_energy(frame, roi)
                motion = detector._normalize_motion(flow, scale)

                # Variant A: Current (iou >= 0.05)
                if iou >= 0.05:
                    active_current[key] += 1
                    if active_current[key] >= 5 and motion >= 0.05:
                        cand_emitted_current = True
                else:
                    active_current[key] = 0

                # Variant B: Relaxed (iou >= 0.05 OR rel_dist <= 1.5)
                if iou >= 0.05 or rel_dist <= 1.5:
                    active_dist15[key] += 1
                    if active_dist15[key] >= 5 and motion >= 0.05:
                        cand_emitted_dist15 = True
                else:
                    active_dist15[key] = 0

                # Variant C: Relaxed (iou >= 0.05 OR rel_dist <= 2.0)
                if iou >= 0.05 or rel_dist <= 2.0:
                    active_dist20[key] += 1
                    if active_dist20[key] >= 5 and motion >= 0.05:
                        cand_emitted_dist20 = True
                else:
                    active_dist20[key] = 0

        if frame_has_iou_ge_005:
            frames_iou_ge_005 += 1
        if frame_has_dist_le_10:
            frames_dist_le_10 += 1
        if frame_has_dist_le_15:
            frames_dist_le_15 += 1
        if frame_has_dist_le_20:
            frames_dist_le_20 += 1

        frames_processed += 1

    cap.release()

    if min_pair_rel_dist == 999.0:
        min_pair_rel_dist = float("inf")

    return VideoGatingStats(
        clip_id=e.clip_id,
        manifest=item["manifest"],
        group_name=item["group"],
        source_class=item["cls"],
        ground_truth=item["gt"],
        duration_s=float(e.duration_sec or (frames_processed / fps)),
        fps=float(fps),
        resolution=res_str,
        frames_processed=frames_processed,
        frames_with_ge2_persons=frames_ge2_persons,
        max_persons=max(person_counts) if person_counts else 0,
        mean_persons=float(np.mean(person_counts)) if person_counts else 0.0,
        max_pair_iou=float(max_pair_iou),
        min_pair_rel_dist=float(min_pair_rel_dist),
        frames_iou_ge_005=frames_iou_ge_005,
        frames_dist_le_10=frames_dist_le_10,
        frames_dist_le_15=frames_dist_le_15,
        frames_dist_le_20=frames_dist_le_20,
        candidate_emitted_current=cand_emitted_current,
        candidate_emitted_dist_15=cand_emitted_dist15,
        candidate_emitted_dist_20=cand_emitted_dist20,
        iou_distribution=all_ious,
        dist_distribution=all_dists,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-cat", type=int, default=25, help="Clips per evaluation category")
    parser.add_argument("--max-frames", type=int, default=120, help="Max frames to process per clip")
    parser.add_argument("--out-json", default="datasets/reports/candidate_gating_analysis.json")
    parser.add_argument("--out-md", default="datasets/reports/candidate_gating_analysis.md")
    args = parser.parse_args()

    setup_logging("WARNING")
    cfg = load_config()

    print(f"Loading tracker on device: {cfg.detection.weights}...")
    tracker = MultiObjectTracker(cfg.detection, cfg.tracking, device="auto")
    detector = FightCandidateDetector(cfg.fight.candidate, (360, 640))

    corpus = select_eval_corpus(max_per_cat=args.max_per_cat)
    print(f"Total videos selected for candidate gating analysis: {len(corpus)}")
    counts = Counter(x["group"] for x in corpus)
    for g, cnt in counts.items():
        print(f"  - {g}: {cnt} clips")

    results: list[VideoGatingStats] = []
    t_start = time.time()

    for idx, item in enumerate(corpus, 1):
        t0 = time.time()
        res = analyze_video(item, tracker, detector, max_frames=args.max_frames)
        results.append(res)
        dt = time.time() - t0
        print(f"[{idx:03d}/{len(corpus):03d}] {res.group_name:<18} {res.clip_id:<22} "
              f"max_iou={res.max_pair_iou:.3f} min_dist={res.min_pair_rel_dist:.2f} "
              f"curr_cand={'YES' if res.candidate_emitted_current else 'no '} "
              f"dist15_cand={'YES' if res.candidate_emitted_dist_15 else 'no '} "
              f"({dt:.2f}s)")

    total_time = time.time() - t_start
    print(f"\nAnalysis completed in {total_time:.1f}s ({total_time / len(results):.2f}s/video)\n")

    # Aggregations by group
    groups = sorted(list(set(r.group_name for r in results)))
    group_summaries = {}

    iou_bins = [0.0, 0.001, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0]
    dist_bins = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    for g in groups:
        group_res = [r for r in results if r.group_name == g]
        n = len(group_res)
        all_ious = [v for r in group_res for v in r.iou_distribution]
        all_dists = [v for r in group_res for v in r.dist_distribution if np.isfinite(v)]

        ge2_persons_pct = 100.0 * sum(1 for r in group_res if r.frames_with_ge2_persons > 0) / n
        max_iou_ge_005_pct = 100.0 * sum(1 for r in group_res if r.max_pair_iou >= 0.05) / n
        max_iou_gt_0_pct = 100.0 * sum(1 for r in group_res if r.max_pair_iou > 0.0) / n
        min_dist_le_10_pct = 100.0 * sum(1 for r in group_res if r.min_pair_rel_dist <= 1.0) / n
        min_dist_le_15_pct = 100.0 * sum(1 for r in group_res if r.min_pair_rel_dist <= 1.5) / n
        min_dist_le_20_pct = 100.0 * sum(1 for r in group_res if r.min_pair_rel_dist <= 2.0) / n

        cand_curr_pct = 100.0 * sum(1 for r in group_res if r.candidate_emitted_current) / n
        cand_dist15_pct = 100.0 * sum(1 for r in group_res if r.candidate_emitted_dist_15) / n
        cand_dist20_pct = 100.0 * sum(1 for r in group_res if r.candidate_emitted_dist_20) / n

        iou_hist = compute_histograms(all_ious, iou_bins)
        dist_hist = compute_histograms(all_dists, dist_bins)

        group_summaries[g] = {
            "n_clips": n,
            "pct_videos_ge2_persons": round(ge2_persons_pct, 1),
            "pct_videos_max_iou_ge_005": round(max_iou_ge_005_pct, 1),
            "pct_videos_max_iou_gt_0": round(max_iou_gt_0_pct, 1),
            "pct_videos_min_dist_le_10": round(min_dist_le_10_pct, 1),
            "pct_videos_min_dist_le_15": round(min_dist_le_15_pct, 1),
            "pct_videos_min_dist_le_20": round(min_dist_le_20_pct, 1),
            "candidate_yield_current_gate": round(cand_curr_pct, 1),
            "candidate_yield_dist_15_gate": round(cand_dist15_pct, 1),
            "candidate_yield_dist_20_gate": round(cand_dist20_pct, 1),
            "iou_histogram": iou_hist,
            "dist_histogram": dist_hist,
        }

    # Save JSON report
    out_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_clips": len(results),
            "max_frames_per_clip": args.max_frames,
            "total_runtime_s": round(total_time, 2),
        },
        "group_summaries": group_summaries,
        "video_results": [
            {k: v for k, v in asdict(r).items() if k not in ("iou_distribution", "dist_distribution")}
            for r in results
        ],
    }

    out_json_path = ROOT / args.out_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved JSON report to: {out_json_path}")

    # Build Markdown summary
    md_lines = [
        "# Candidate Gating Quantitative Analysis",
        "",
        f"_Evaluated {len(results)} clips across 4 categories · Total runtime {total_time:.1f}s_",
        "",
        "## 1. Summary of Gating and Candidate Yield",
        "",
        "| Category | Clips | >=2 Persons | Max IoU >= 0.05 (Current) | Min Dist <= 1.5 bw | Current Cand Yield | Relaxed Cand Yield (Dist <= 1.5) | Relaxed Cand Yield (Dist <= 2.0) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for g, s in group_summaries.items():
        md_lines.append(
            f"| `{g}` | {s['n_clips']} | {s['pct_videos_ge2_persons']}% | "
            f"**{s['pct_videos_max_iou_ge_005']}%** | {s['pct_videos_min_dist_le_15']}% | "
            f"**{s['candidate_yield_current_gate']}%** | **{s['candidate_yield_dist_15_gate']}%** | "
            f"**{s['candidate_yield_dist_20_gate']}%** |"
        )

    md_lines.extend([
        "",
        "## 2. Key Findings",
        "",
        "### A. Real Surveillance CCTV (UCF-Crime Fight)",
        f"- Under current rigid `IoU >= 0.05` gating: only **{group_summaries.get('ucf_crime_fight', {}).get('pct_videos_max_iou_ge_005', 0)}%** of fights achieve bbox overlap.",
        f"- Current candidate yield on UCF-Crime fight: **{group_summaries.get('ucf_crime_fight', {}).get('candidate_yield_current_gate', 0)}%** (the exact root cause of 0% recall).",
        f"- With distance-based relaxation (`rel_dist <= 1.5` body-widths): candidate yield increases to **{group_summaries.get('ucf_crime_fight', {}).get('candidate_yield_dist_15_gate', 0)}%**.",
        f"- With distance-based relaxation (`rel_dist <= 2.0` body-widths): candidate yield increases to **{group_summaries.get('ucf_crime_fight', {}).get('candidate_yield_dist_20_gate', 0)}%**.",
        "",
        "### B. False Positive Control on Normal Videos",
        f"- On UCF-Crime Normal_Videos, candidate yield under current gate: **{group_summaries.get('ucf_crime_normal', {}).get('candidate_yield_current_gate', 0)}%**.",
        f"- Under `rel_dist <= 1.5`: **{group_summaries.get('ucf_crime_normal', {}).get('candidate_yield_dist_15_gate', 0)}%**.",
        f"- Under `rel_dist <= 2.0`: **{group_summaries.get('ucf_crime_normal', {}).get('candidate_yield_dist_20_gate', 0)}%**.",
        "",
        "## 3. Recommendation for Task 2",
        "- Implement dual gating in `FightCandidateDetector`: a pair is engaged if `iou >= proximity_iou` OR `rel_dist <= proximity_distance` (recommended: `1.5` body-widths).",
        "- This preserves proximity requirement while unlocking standing-distance and low-resolution surveillance footage.",
    ])

    out_md_path = ROOT / args.out_md
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown report to: {out_md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
