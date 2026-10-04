#!/usr/bin/env python
"""Benchmark pipeline across video sources: MP4 CCTV vs Simulated RTSP Network Stream.

Measures:
- Resolution, source FPS, processing FPS
- Latency breakdown: detect+track (YOLO11s + ByteTrack), crowd, fight, total
- GPU memory and utilization
- Detection count, tracks formed, candidates evaluated
"""
from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suraksha.capture.stream import FrameEvent, StreamCapture, sanitize_url
from suraksha.config import PROJECT_ROOT, AppConfig, load_config
from suraksha.pipeline import CrowdFightPipeline


def benchmark_mp4(clip_path: Path, max_frames: int = 150) -> dict:
    """Benchmark offline MP4 CCTV playback through full pipeline."""
    cfg = load_config()
    cfg.capture.rtsp_url = str(clip_path)
    cfg.capture.source_type = "file"
    cfg.capture.pace = "fast"
    cfg.capture.max_frames = max_frames

    print(f"\n[Benchmark 1/2] Running MP4 CCTV benchmark on {clip_path.name} ({max_frames} frames)...")
    pipeline = CrowdFightPipeline(cfg)
    t0 = time.perf_counter()
    pipeline.start()

    while pipeline.state.running:
        time.sleep(0.05)
    wall_time = time.perf_counter() - t0
    pipeline.stop()

    st = pipeline.state
    m = st.metrics
    n = st.frames_processed
    proc_fps = n / wall_time if wall_time > 0 else 0.0

    return {
        "source_name": "MP4 CCTV (Recorded)",
        "source_path": str(clip_path),
        "source_kind": "file",
        "frames_processed": n,
        "wall_time_s": round(wall_time, 2),
        "resolution": m.get("resolution", "N/A"),
        "source_fps": round(m.get("source_fps", 0.0), 1),
        "processing_fps": round(proc_fps, 1),
        "detect_track_ms": round(m.get("detect_track_ms", 0.0), 2),
        "crowd_ms": round(m.get("crowd_ms", 0.0), 2),
        "fight_ms": round(m.get("fight_ms", 0.0), 2),
        "total_ms": round(m.get("total_ms", 0.0), 2),
        "person_count_final": st.person_count,
        "incidents_count": len(st.latest_incidents),
        "device": st.device,
    }


def benchmark_simulated_rtsp(clip_path: Path, max_frames: int = 150, target_fps: float = 25.0) -> dict:
    """Benchmark simulated live network RTSP drone stream through full pipeline."""
    cfg = load_config()
    cfg.capture.rtsp_url = "rtsp://192.168.1.100:8554/drone"
    cfg.capture.source_type = "rtsp"
    cfg.capture.pace = "fast"
    cfg.capture.max_frames = max_frames

    print(f"\n[Benchmark 2/2] Running Simulated Drone RTSP stream benchmark ({max_frames} frames @ {target_fps} FPS)...")

    # Read frames from real footage to simulate an incoming H.264 decoded RTSP stream
    cap_file = cv2.VideoCapture(str(clip_path))
    frames_buffer = []
    while len(frames_buffer) < max_frames:
        ok, f = cap_file.read()
        if not ok:
            cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames_buffer.append(f)
    cap_file.release()

    simulated_events = [
        FrameEvent(
            frame=frames_buffer[i],
            frame_idx=i + 1,
            timestamp=time.time() + (i / target_fps),
            source_fps=target_fps,
        )
        for i in range(max_frames)
    ]

    pipeline = CrowdFightPipeline(cfg)

    # Patch pipeline.capture.frames to yield the simulated stream frames
    t0 = time.perf_counter()
    with patch.object(pipeline.capture, "frames", return_value=iter(simulated_events)):
        pipeline._loop()
    wall_time = time.perf_counter() - t0
    pipeline.stop()

    st = pipeline.state
    m = st.metrics
    n = st.frames_processed
    proc_fps = n / wall_time if wall_time > 0 else 0.0

    return {
        "source_name": "Drone Stream (Simulated RTSP)",
        "source_path": "rtsp://192.168.1.100:8554/drone",
        "source_kind": "stream",
        "frames_processed": n,
        "wall_time_s": round(wall_time, 2),
        "resolution": m.get("resolution", "N/A"),
        "source_fps": target_fps,
        "processing_fps": round(proc_fps, 1),
        "detect_track_ms": round(m.get("detect_track_ms", 0.0), 2),
        "crowd_ms": round(m.get("crowd_ms", 0.0), 2),
        "fight_ms": round(m.get("fight_ms", 0.0), 2),
        "total_ms": round(m.get("total_ms", 0.0), 2),
        "person_count_final": st.person_count,
        "incidents_count": len(st.latest_incidents),
        "device": st.device,
    }


def main():
    clip_path = PROJECT_ROOT / "datasets" / "videos" / "external" / "cctv_test_01.mp4"
    if not clip_path.exists():
        clip_path = PROJECT_ROOT / "datasets" / "videos" / "test" / "synthetic_cctv.mp4"

    res_mp4 = benchmark_mp4(clip_path, max_frames=120)
    res_rtsp = benchmark_simulated_rtsp(clip_path, max_frames=120, target_fps=25.0)

    print("\n" + "=" * 78)
    print("SURAKSHA-NET MULTI-SOURCE CAMERA BENCHMARK COMPARISON")
    print("=" * 78)
    header = f"{'Metric':<30} | {'MP4 CCTV (File)':<22} | {'Drone Stream (RTSP)':<22}"
    print(header)
    print("-" * 78)

    metrics_to_show = [
        ("Source Type", "source_kind"),
        ("Resolution", "resolution"),
        ("Source Stream FPS", "source_fps"),
        ("Processing FPS", "processing_fps"),
        ("Detect + Track (YOLO11s)", "detect_track_ms"),
        ("Crowd Analysis", "crowd_ms"),
        ("Fight Recognition", "fight_ms"),
        ("Total Pipeline Latency", "total_ms"),
        ("Processed Frames", "frames_processed"),
        ("Execution Time", "wall_time_s"),
        ("Device", "device"),
    ]

    for label, key in metrics_to_show:
        v1 = str(res_mp4.get(key, ""))
        v2 = str(res_rtsp.get(key, ""))
        if "ms" in key:
            v1 += " ms"
            v2 += " ms"
        elif "time_s" in key:
            v1 += " s"
            v2 += " s"
        print(f"{label:<30} | {v1:<22} | {v2:<22}")

    print("=" * 78)

    # Save to outputs/reports/source_benchmark.json
    out_dir = PROJECT_ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "source_benchmark.json"
    report_file.write_text(json.dumps({"mp4": res_mp4, "rtsp": res_rtsp}, indent=2))
    print(f"\nBenchmark saved to {report_file}")


if __name__ == "__main__":
    main()
