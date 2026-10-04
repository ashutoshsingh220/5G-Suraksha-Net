"""Stage 2 & 3: Benchmark Detection Improvements for Low-Resolution CCTV.

Evaluates candidate detection improvements on the representative 320x240
surveillance benchmark:
- 35 Real CCTV combat clips (25 Fighting, 10 Assault) from UCF-Crime
- 15 Real CCTV normal clips (Normal_Videos) from UCF-Crime
Total: 50 clips, all native 320x240, 90 frames each (4,500 frames per experiment).

Systematically tests:
1. Baseline: YOLO11s, imgsz=640, conf=0.35, iou=0.50
2. imgsz=960: YOLO11s, imgsz=960, conf=0.35, iou=0.50
3. imgsz=1280: YOLO11s, imgsz=1280, conf=0.35, iou=0.50
4. imgsz=320: YOLO11s, imgsz=320, conf=0.35, iou=0.50
5. conf=0.25: YOLO11s, imgsz=640, conf=0.25, iou=0.50
6. conf=0.15: YOLO11s, imgsz=640, conf=0.15, iou=0.50
7. iou=0.70: YOLO11s, imgsz=640, conf=0.35, iou=0.70
8. Upscaling (2x bicubic): YOLO11s, imgsz=640, conf=0.35, iou=0.50, bicubic 2x
9. YOLO11m: YOLO11m, imgsz=640, conf=0.35, iou=0.50
10. Promising Combinations (if warranted)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from suraksha.config import AppConfig, DetectionConfig, FightConfig, TrackingConfig, load_config
from suraksha.detection.tracker import MultiObjectTracker, TrackedPerson
from suraksha.fight.recognizer import FightRecognizer
from suraksha.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


@dataclass
class BenchmarkExperiment:
    name: str
    weights: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    preprocessing: str  # "none" | "bicubic_2x"
    description: str


@dataclass
class ExperimentResult:
    experiment_name: str
    model: str
    imgsz: int
    conf: float
    iou: float
    preprocessing: str
    total_clips: int
    fight_clips_count: int
    normal_clips_count: int
    total_frames_sampled: int
    total_persons_detected: int
    frames_ge2_persons: int
    pct_frames_ge2_persons: float
    # Upstream pipeline yield
    fight_clips_reaching_candidates: int
    fight_candidate_reachability_pct: float
    fight_clips_reaching_windows: int
    fight_window_reachability_pct: float
    fight_clips_verified: int
    fight_verified_pct: float
    # False positives on normal CCTV
    normal_clips_reaching_candidates: int
    normal_clips_reaching_windows: int
    normal_clips_verified: int
    normal_false_positive_rate_pct: float
    # Performance & System Resources
    avg_detect_track_latency_ms: float
    avg_total_pipeline_latency_ms: float
    processing_fps: float
    peak_vram_mb: float
    elapsed_seconds: float


class BenchmarkTracker:
    """Wrapper around YOLO + ByteTrack allowing custom preprocessing (e.g. 2x upscaling)."""

    def __init__(self, det_cfg: DetectionConfig, trk_cfg: TrackingConfig, device: str = "cuda", preprocessing: str = "none"):
        self.det_cfg = det_cfg
        self.trk_cfg = trk_cfg
        self.preprocessing = preprocessing
        self.model = YOLO(det_cfg.weights)
        self.device_arg = 0 if device == "cuda" else "cpu"

    def reset(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        for t in trackers:
            reset = getattr(t, "reset", None)
            if callable(reset):
                reset()

    def update(self, frame: np.ndarray) -> list[TrackedPerson]:
        h, w = frame.shape[:2]
        proc_frame = frame
        scale_x, scale_y = 1.0, 1.0

        if self.preprocessing == "bicubic_2x":
            proc_frame = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            scale_x, scale_y = 2.0, 2.0

        kwargs = dict(
            conf=self.det_cfg.conf_threshold,
            iou=self.det_cfg.iou_threshold,
            imgsz=self.det_cfg.imgsz,
            classes=[self.det_cfg.person_class_id],
            tracker=self.trk_cfg.tracker_config,
            persist=self.trk_cfg.persist,
            verbose=False,
            device=self.device_arg,
        )
        results = self.model.track(proc_frame, **kwargs)

        tracks: list[TrackedPerson] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.cpu().numpy()
            for tid, box, c in zip(ids, xyxy, conf):
                if scale_x != 1.0 or scale_y != 1.0:
                    box = box.copy()
                    box[0] /= scale_x
                    box[1] /= scale_y
                    box[2] /= scale_x
                    box[3] /= scale_y
                tracks.append(TrackedPerson(track_id=int(tid), bbox_xyxy=box, confidence=float(c)))
        return tracks


def load_benchmark_corpus() -> list[dict]:
    """Loads the 50-clip 320x240 real CCTV benchmark corpus."""
    with open(ROOT / "datasets" / "manifests" / "ucf_crime_subset_eval_v1.json", "r", encoding="utf-8") as f:
        entries = json.load(f)["entries"]

    fighting = [e for e in entries if e["notes"].get("source_class") == "Fighting" and e.get("width") == 320][:25]
    assault = [e for e in entries if e["notes"].get("source_class") == "Assault"][:10]
    normal = [e for e in entries if e["notes"].get("source_class") == "Normal_Videos" and e.get("width") == 320][:15]

    corpus = []
    for e in fighting:
        corpus.append({
            "clip_id": e["clip_id"],
            "video_id": e["video_id"],
            "path": ROOT / e["path"],
            "ground_truth": 1,
            "source_class": "Fighting",
        })
    for e in assault:
        corpus.append({
            "clip_id": e["clip_id"],
            "video_id": e["video_id"],
            "path": ROOT / e["path"],
            "ground_truth": 1,
            "source_class": "Assault",
        })
    for e in normal:
        corpus.append({
            "clip_id": e["clip_id"],
            "video_id": e["video_id"],
            "path": ROOT / e["path"],
            "ground_truth": 0,
            "source_class": "Normal_Videos",
        })
    return corpus


def run_single_experiment(
    exp: BenchmarkExperiment,
    corpus: list[dict],
    base_cfg: AppConfig,
    max_frames_per_clip: int = 90,
    device: str = "cuda",
) -> ExperimentResult:
    det_cfg = copy.deepcopy(base_cfg.detection)
    det_cfg.weights = exp.weights
    det_cfg.imgsz = exp.imgsz
    det_cfg.conf_threshold = exp.conf_threshold
    det_cfg.iou_threshold = exp.iou_threshold

    trk_cfg = copy.deepcopy(base_cfg.tracking)
    fight_cfg = copy.deepcopy(base_cfg.fight)

    tracker = BenchmarkTracker(det_cfg, trk_cfg, device=device, preprocessing=exp.preprocessing)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_exp_start = time.time()

    total_frames = 0
    total_persons = 0
    frames_ge2 = 0
    detect_track_times: list[float] = []
    total_pipeline_times: list[float] = []

    fight_reached_cands = 0
    fight_reached_windows = 0
    fight_verified = 0

    normal_reached_cands = 0
    normal_reached_windows = 0
    normal_verified = 0

    fight_clips_count = sum(1 for c in corpus if c["ground_truth"] == 1)
    normal_clips_count = sum(1 for c in corpus if c["ground_truth"] == 0)

    for c in corpus:
        vpath = c["path"]
        gt = c["ground_truth"]

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            log.warning("Could not open %s", vpath)
            continue

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 30.0

        tracker.reset()
        recognizer = FightRecognizer(fight_cfg, (orig_h, orig_w), device=device)

        clip_frames = 0
        clip_candidates_formed = 0
        clip_windows_formed = 0
        clip_verified_fights = 0

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames_per_clip > 0 and clip_frames >= max_frames_per_clip):
                break

            ts = clip_frames / source_fps

            t0 = time.perf_counter()
            tracks = tracker.update(frame)
            t1 = time.perf_counter()

            verified = recognizer.update(frame, tracks, ts, fps=source_fps)
            t2 = time.perf_counter()

            detect_track_times.append((t1 - t0) * 1000.0)
            total_pipeline_times.append((t2 - t0) * 1000.0)

            n_p = len(tracks)
            total_persons += n_p
            if n_p >= 2:
                frames_ge2 += 1

            # Candidate checks
            active_cands = [p for p in recognizer.detector._pairs.values() if p.engaged_frames >= fight_cfg.candidate.proximity_min_frames]
            if active_cands:
                clip_candidates_formed += len(active_cands)

            for af in recognizer._active.values():
                if len(af.candidate.features) >= fight_cfg.temporal.window_frames:
                    clip_windows_formed += 1

            if verified:
                clip_verified_fights += len(verified)

            clip_frames += 1
            total_frames += 1

        cap.release()

        if gt == 1:
            if clip_candidates_formed > 0:
                fight_reached_cands += 1
            if clip_windows_formed > 0:
                fight_reached_windows += 1
            if clip_verified_fights > 0:
                fight_verified += 1
        else:
            if clip_candidates_formed > 0:
                normal_reached_cands += 1
            if clip_windows_formed > 0:
                normal_reached_windows += 1
            if clip_verified_fights > 0:
                normal_verified += 1

    elapsed = time.time() - t_exp_start
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    fps = total_frames / max(elapsed, 1e-3)
    avg_det_ms = float(np.mean(detect_track_times)) if detect_track_times else 0.0
    avg_tot_ms = float(np.mean(total_pipeline_times)) if total_pipeline_times else 0.0

    pct_ge2 = round(100.0 * frames_ge2 / max(1, total_frames), 2)
    fight_cand_pct = round(100.0 * fight_reached_cands / max(1, fight_clips_count), 2)
    fight_win_pct = round(100.0 * fight_reached_windows / max(1, fight_clips_count), 2)
    fight_ver_pct = round(100.0 * fight_verified / max(1, fight_clips_count), 2)
    normal_fp_pct = round(100.0 * normal_verified / max(1, normal_clips_count), 2)

    return ExperimentResult(
        experiment_name=exp.name,
        model=Path(exp.weights).stem,
        imgsz=exp.imgsz,
        conf=exp.conf_threshold,
        iou=exp.iou_threshold,
        preprocessing=exp.preprocessing,
        total_clips=len(corpus),
        fight_clips_count=fight_clips_count,
        normal_clips_count=normal_clips_count,
        total_frames_sampled=total_frames,
        total_persons_detected=total_persons,
        frames_ge2_persons=frames_ge2,
        pct_frames_ge2_persons=pct_ge2,
        fight_clips_reaching_candidates=fight_reached_cands,
        fight_candidate_reachability_pct=fight_cand_pct,
        fight_clips_reaching_windows=fight_reached_windows,
        fight_window_reachability_pct=fight_win_pct,
        fight_clips_verified=fight_verified,
        fight_verified_pct=fight_ver_pct,
        normal_clips_reaching_candidates=normal_reached_cands,
        normal_clips_reaching_windows=normal_reached_windows,
        normal_clips_verified=normal_verified,
        normal_false_positive_rate_pct=normal_fp_pct,
        avg_detect_track_latency_ms=round(avg_det_ms, 2),
        avg_total_pipeline_latency_ms=round(avg_tot_ms, 2),
        processing_fps=round(fps, 1),
        peak_vram_mb=round(peak_vram, 1),
        elapsed_seconds=round(elapsed, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--out-json", default="datasets/reports/low_res_detection_benchmark.json")
    ap.add_argument("--out-md", default="datasets/reports/low_res_detection_benchmark.md")
    ap.add_argument("--single-exp", default=None, help="Run only a single experiment name")
    args = ap.parse_args()
    setup_logging("INFO")

    base_cfg = load_config()
    corpus = load_benchmark_corpus()
    print(f"\n========================================================")
    print(f"STAGE 2 & 3: BENCHMARKING DETECTION IMPROVEMENTS")
    print(f"Benchmark Corpus: {len(corpus)} clips (35 Fight, 15 Normal, all 320x240)")
    print(f"========================================================\n")

    experiments: list[BenchmarkExperiment] = [
        # 1. Baseline
        BenchmarkExperiment(
            name="1_Baseline_YOLO11s_640_conf035",
            weights="models/yolo11s.pt",
            imgsz=640,
            conf_threshold=0.35,
            iou_threshold=0.50,
            preprocessing="none",
            description="Current active production baseline",
        ),
        # 2. Higher imgsz: 960
        BenchmarkExperiment(
            name="2_YOLO11s_imgsz960",
            weights="models/yolo11s.pt",
            imgsz=960,
            conf_threshold=0.35,
            iou_threshold=0.50,
            preprocessing="none",
            description="Higher inference resolution (imgsz=960)",
        ),
        # 3. Higher imgsz: 1280
        BenchmarkExperiment(
            name="3_YOLO11s_imgsz1280",
            weights="models/yolo11s.pt",
            imgsz=1280,
            conf_threshold=0.35,
            iou_threshold=0.50,
            preprocessing="none",
            description="Ultra-high inference resolution (imgsz=1280)",
        ),
        # 4. Canvas-matching imgsz: 320
        BenchmarkExperiment(
            name="4_YOLO11s_imgsz320",
            weights="models/yolo11s.pt",
            imgsz=320,
            conf_threshold=0.35,
            iou_threshold=0.50,
            preprocessing="none",
            description="Native canvas matching resolution (imgsz=320)",
        ),
        # 5. Adjusted confidence: 0.25
        BenchmarkExperiment(
            name="5_YOLO11s_conf025",
            weights="models/yolo11s.pt",
            imgsz=640,
            conf_threshold=0.25,
            iou_threshold=0.50,
            preprocessing="none",
            description="Moderate confidence reduction (conf=0.25)",
        ),
        # 6. Adjusted confidence: 0.15
        BenchmarkExperiment(
            name="6_YOLO11s_conf015",
            weights="models/yolo11s.pt",
            imgsz=640,
            conf_threshold=0.15,
            iou_threshold=0.50,
            preprocessing="none",
            description="Aggressive confidence reduction (conf=0.15)",
        ),
        # 7. Adjusted NMS/IoU threshold: 0.70
        BenchmarkExperiment(
            name="7_YOLO11s_iou070",
            weights="models/yolo11s.pt",
            imgsz=640,
            conf_threshold=0.35,
            iou_threshold=0.70,
            preprocessing="none",
            description="Relaxed NMS IoU threshold for overlapping combatants (iou=0.70)",
        ),
        # 8. Frame upscaling: 2x bicubic
        BenchmarkExperiment(
            name="8_YOLO11s_bicubic2x",
            weights="models/yolo11s.pt",
            imgsz=640,
            conf_threshold=0.35,
            iou_threshold=0.50,
            preprocessing="bicubic_2x",
            description="2x Bicubic frame upscaling before YOLO (320x240 -> 640x480)",
        ),
        # 9. YOLO11m comparison
        BenchmarkExperiment(
            name="9_YOLO11m_imgsz640",
            weights="models/yolo11m.pt",
            imgsz=640,
            conf_threshold=0.35,
            iou_threshold=0.50,
            preprocessing="none",
            description="Larger capacity YOLO11m detector at baseline settings",
        ),
        # 10. Synergistic Combination: YOLO11s imgsz=960 + conf=0.25
        BenchmarkExperiment(
            name="10_YOLO11s_imgsz960_conf025",
            weights="models/yolo11s.pt",
            imgsz=960,
            conf_threshold=0.25,
            iou_threshold=0.50,
            preprocessing="none",
            description="Synergistic combination: imgsz=960 with conf=0.25",
        ),
    ]

    if args.single_exp:
        experiments = [e for e in experiments if e.name == args.single_exp]
        if not experiments:
            print(f"Error: experiment '{args.single_exp}' not found.")
            return 1

    all_results: list[ExperimentResult] = []

    print(f"{'#':<3} | {'Experiment Name':<30} | {'imgsz':<5} | {'conf':<4} | {'iou':<4} | {'Prep':<10}")
    print("-" * 65)
    for i, e in enumerate(experiments):
        print(f"{i+1:<3} | {e.name:<30} | {e.imgsz:<5} | {e.conf_threshold:<4.2f} | {e.iou_threshold:<4.2f} | {e.preprocessing:<10}")

    for idx, exp in enumerate(experiments):
        print(f"\n[{idx+1}/{len(experiments)}] Running {exp.name}...")
        t0 = time.time()
        res = run_single_experiment(exp, corpus, base_cfg, max_frames_per_clip=args.max_frames, device=args.device)
        elapsed = time.time() - t0
        all_results.append(res)
        print(f"  -> Persons: {res.total_persons_detected} | >=2 Frames: {res.pct_frames_ge2_persons}% | Fight Cand: {res.fight_candidate_reachability_pct}% | Fight Win: {res.fight_window_reachability_pct}% | Fight Ver: {res.fight_verified_pct}% | Normal FP: {res.normal_false_positive_rate_pct}% | Lat: {res.avg_detect_track_latency_ms:.1f}ms ({res.processing_fps:.1f} FPS) | VRAM: {res.peak_vram_mb:.1f}MB ({elapsed:.1f}s)")

    # Save JSON results
    out_json_p = Path(args.out_json)
    out_json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_p, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Generate Markdown Table
    out_md_p = Path(args.out_md)
    lines = [
        "# Benchmark Results: Low-Resolution (320x240) CCTV Detection Improvements",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Device:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.1)  ",
        f"**Benchmark Dataset:** 50 real CCTV clips from UCF-Crime (35 Fight / Assault, 15 Normal_Videos, all native 320x240, 90 frames each = 4,500 frames/run)  ",
        "",
        "## Summary Results Table",
        "",
        "| # | Experiment | Model | imgsz | conf | IoU | Prep | >=2-Pers % | Cand Reach % | Win Reach % | Fight Ver % | Normal FP % | Latency (ms) | FPS | Peak VRAM |",
        "|:--|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for i, r in enumerate(all_results):
        lines.append(
            f"| {i+1} | {r.experiment_name} | {r.model} | {r.imgsz} | {r.conf:.2f} | {r.iou:.2f} | {r.preprocessing} | "
            f"{r.pct_frames_ge2_persons:.1f}% | {r.fight_candidate_reachability_pct:.1f}% | {r.fight_window_reachability_pct:.1f}% | "
            f"{r.fight_verified_pct:.1f}% | {r.normal_false_positive_rate_pct:.1f}% | {r.avg_detect_track_latency_ms:.1f} ms | "
            f"{r.processing_fps:.1f} | {r.peak_vram_mb:.0f} MB |"
        )

    with open(out_md_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nAll benchmark results saved to:")
    print(f"  JSON: {out_json_p}")
    print(f"  MD:   {out_md_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
