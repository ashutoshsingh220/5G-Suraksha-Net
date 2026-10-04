#!/usr/bin/env python
"""Run the crowd+fight pipeline standalone (no API server).

Sources: RTSP stream (default from config/.env), local video file, or webcam.
The downstream pipeline is identical for all three.

Usage:
  # configured stream (configs/app.yaml + .env)
  python scripts/run_pipeline.py

  # OFFLINE: local video file (no camera, no internet needed)
  python scripts/run_pipeline.py --source file --path datasets/videos/test/synthetic_cctv.mp4

  # offline, paced like a live camera (realtime simulation)
  python scripts/run_pipeline.py --source file --path clip.mp4 --realtime

  # offline benchmark: decode+process as fast as possible
  python scripts/run_pipeline.py --source file --path clip.mp4 --fast

  # offline with interactive visual preview
  python scripts/run_pipeline.py --source file --path clip.mp4 --realtime --preview

  # offline with annotated video export
  python scripts/run_pipeline.py --source file --path clip.mp4 --save-video output.mp4

  # preview and save video simultaneously
  python scripts/run_pipeline.py --source file --path clip.mp4 --realtime --preview --save-video output.mp4

  # webcam (via --camera-index or --path or default index 0)
  python scripts/run_pipeline.py --source webcam --camera-index 0 --display
  python scripts/run_pipeline.py --source webcam --path 0 --display
  python scripts/run_pipeline.py --source webcam --display

  # limit frames (smoke tests)
  python scripts/run_pipeline.py --source file --path clip.mp4 --fast --max-frames 50
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _sample_gpu() -> tuple[float, float] | None:
    """One nvidia-smi sample: (gpu_util %, vram_used MB)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        parts = out.stdout.strip().splitlines()[0].split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run crowd+fight pipeline",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to app.yaml override")
    parser.add_argument("--source", choices=["file", "webcam", "rtsp"], default=None,
                        help="video source kind (default: from config/.env)")
    parser.add_argument("--path", default=None,
                        help="file path, webcam index, or RTSP URL for --source")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="camera device index for --source webcam (default: 0)")
    parser.add_argument("--realtime", action="store_true",
                        help="pace file/webcam playback at source FPS (live simulation)")
    parser.add_argument("--fast", action="store_true",
                        help="no pacing — process as fast as possible (benchmark)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = unlimited)")
    parser.add_argument("--preview", "--display", dest="preview", action="store_true",
                        help="open interactive OpenCV preview window with detections, tracks, fight alerts, and HUD (alias: --display)")
    parser.add_argument("--save-video", "--save-annotated", dest="save_video", default=None, metavar="PATH",
                        help="path to write annotated output video (.mp4) (alias: --save-annotated)")
    parser.add_argument("--weapon", dest="weapon_enabled", action="store_true", default=None,
                        help="explicitly enable weapon detection branch")
    parser.add_argument("--no-weapon", dest="weapon_enabled", action="store_false",
                        help="disable weapon detection branch")
    parser.add_argument("--weapon-conf", type=float, default=None,
                        help="override weapon detection confidence threshold")
    parser.add_argument("--weapon-interval", type=int, default=None,
                        help="override weapon inference frame interval (1 = every frame)")
    parser.add_argument("--save-high-confidence", action="store_true",
                        help="enable read-only false-positive auditor (saves frame/crop/metadata for conf >= 0.60)")
    args = parser.parse_args()

    from suraksha.config import PROJECT_ROOT, load_config
    from suraksha.logging_utils import setup_logging

    cfg = load_config(args.config)

    # ---- weapon override ----
    if args.weapon_enabled is not None:
        cfg.weapon.enabled = args.weapon_enabled
    if args.weapon_conf is not None:
        cfg.weapon.conf_threshold = args.weapon_conf
    if args.weapon_interval is not None:
        cfg.weapon.inference_interval = args.weapon_interval

    # ---- source override ----
    if args.source:
        cfg.capture.source_type = args.source
        if args.source == "webcam":
            cam_idx = 0
            if args.camera_index is not None:
                cam_idx = args.camera_index
            elif args.path is not None:
                try:
                    cam_idx = int(args.path)
                except ValueError:
                    parser.error(f"Invalid webcam index '{args.path}': must be an integer")
            cfg.capture.camera_index = cam_idx
            cfg.capture.rtsp_url = str(cam_idx)
        elif args.source == "file":
            if not args.path:
                parser.error("--source file requires --path")
            p = Path(args.path)
            cfg.capture.rtsp_url = str(p if p.is_absolute() else PROJECT_ROOT / p)
        elif args.source == "rtsp":
            if args.path:
                cfg.capture.rtsp_url = str(args.path)
    if args.fast:
        cfg.capture.pace = "fast"
    elif args.realtime:
        cfg.capture.pace = "realtime"
    elif args.source in ("file", "webcam"):
        cfg.capture.pace = "throttle"
    if args.max_frames:
        cfg.capture.max_frames = args.max_frames

    save_video_path = None
    if args.save_video:
        svp = Path(args.save_video)
        save_video_path = str(svp if svp.is_absolute() else PROJECT_ROOT / svp)

    setup_logging(cfg.log_level)

    from suraksha.pipeline import CrowdFightPipeline

    auditor = None
    if args.save_high_confidence:
        from suraksha.detection.audit import FalsePositiveAuditor
        source_label = f"webcam:{cfg.capture.camera_index}" if cfg.capture.source_type == "webcam" else cfg.capture.source_type
        auditor = FalsePositiveAuditor(source_name=source_label, save_high_confidence=True)
        print(f"[AUDIT] False-positive diagnostic auditor enabled for source '{source_label}'.")
        print(f"[AUDIT] Detections will be logged to {auditor.log_file}")
        print(f"[AUDIT] High-confidence (>= 0.60) artifacts will be saved to {auditor.output_dir}")

    preview_q: queue.Queue | None = queue.Queue(maxsize=15) if args.preview else None
    pipeline = CrowdFightPipeline(
        cfg,
        preview_queue=preview_q,
        save_video_path=save_video_path,
        auditor=auditor,
    )
    gpu_samples: list[tuple[float, float]] = []
    pipeline.start()

    window_title = "5G Suraksha-Net CCTV Preview"
    if args.preview:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 540)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
        print(f"\n[Preview] Visual display window '{window_title}' active. Bring to front if behind terminal.")
        print("[Preview] Controls: SPACE = Pause/Resume | 'q' or ESC = Quit.\n")

    last_gpu_check = time.time()
    user_requested_quit = False
    last_frame = None
    try:
        if args.preview and preview_q is not None:
            first_frame_shown = False
            while pipeline.state.running or not preview_q.empty():
                try:
                    frame = preview_q.get(timeout=0.03)
                except queue.Empty:
                    cv2.waitKey(1)
                    continue

                if frame is None:
                    break

                last_frame = frame
                cv2.imshow(window_title, frame)
                if not first_frame_shown:
                    # After first frame is painted, remove topmost lock so user can interact normally
                    cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 0)
                    first_frame_shown = True

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    print("\n[Preview] User requested quit ('q'/ESC). Stopping...")
                    user_requested_quit = True
                    pipeline.stop()
                    break
                elif key == 32:  # SPACE = pause / resume
                    if pipeline.is_paused:
                        pipeline.resume()
                        print("[Preview] RESUMED")
                    else:
                        pipeline.pause()
                        print("[Preview] PAUSED (press SPACE to resume)")
                        if last_frame is not None:
                            paused_frame = last_frame.copy()
                            ph, pw = paused_frame.shape[:2]
                            cv2.rectangle(paused_frame, (pw // 2 - 180, ph // 2 - 25), (pw // 2 + 180, ph // 2 + 25), (20, 20, 20), -1)
                            cv2.rectangle(paused_frame, (pw // 2 - 180, ph // 2 - 25), (pw // 2 + 180, ph // 2 + 25), (0, 215, 255), 2)
                            cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", (pw // 2 - 160, ph // 2 + 7),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
                            cv2.imshow(window_title, paused_frame)
                        while pipeline.is_paused and not pipeline._stop.is_set():
                            k = cv2.waitKey(30) & 0xFF
                            if k in (ord("q"), ord("Q"), 27):
                                user_requested_quit = True
                                pipeline.stop()
                                break
                            elif k == 32:
                                pipeline.resume()
                                print("[Preview] RESUMED")
                                break

                now = time.time()
                if now - last_gpu_check >= 0.5:
                    g = _sample_gpu()
                    if g:
                        gpu_samples.append(g)
                    last_gpu_check = now

            # If finished normally at EOF, pause briefly on the final annotated frame
            if not user_requested_quit and last_frame is not None:
                banner = last_frame.copy()
                bh, bw = banner.shape[:2]
                cv2.putText(
                    banner,
                    "END OF VIDEO - Press any key to close",
                    (max(20, bw // 2 - 200), bh - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_title, banner)
                cv2.waitKey(3000)
        else:
            while pipeline.state.running:
                time.sleep(0.5)
                g = _sample_gpu()
                if g:
                    gpu_samples.append(g)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()

    # ---- end-of-run summary ----
    st = pipeline.state
    m = st.metrics
    elapsed = (st.last_frame_time - st.started_at) if st.last_frame_time and st.started_at else 0.0
    from suraksha.capture.stream import sanitize_url

    print("\n" + "=" * 62)
    print("PIPELINE RUN SUMMARY")
    print("=" * 62)
    print(f"source             : {sanitize_url(cfg.capture.rtsp_url)}")
    print(f"source kind / pace : {pipeline.capture.source_kind if hasattr(pipeline, 'capture') else '?'} / {cfg.capture.pace}")
    print(f"device             : {st.device} ({pipeline.device_info.gpu_name or 'CPU'})")
    print(f"resolution         : {m['resolution']}")
    print(f"source fps         : {m['source_fps']:.1f}")
    print(f"frames processed   : {st.frames_processed}")
    print(f"wall time          : {elapsed:.1f}s")
    print(f"processing fps     : {st.frames_processed / elapsed:.1f}" if elapsed > 0 else "processing fps     : n/a")
    print(f"latency detect+track (YOLO11s + ByteTrack, single call): {m['detect_track_ms']:.1f} ms/frame")
    print(f"latency crowd      : {m['crowd_ms']:.1f} ms/frame")
    print(f"latency fight      : {m['fight_ms']:.1f} ms/frame")
    if cfg.weapon.enabled and hasattr(pipeline, "weapon_detector"):
        print(f"latency weapon     : {m.get('weapon_ms', 0.0):.1f} ms/frame")
        w_met = pipeline.weapon_detector.get_metrics()
        print(f"weapon detector    : state={w_met['state']} | candidates={w_met['candidate_count']} | confirmed={w_met['confirmed_count']}")
        print(f"weapon detections  : {w_met['per_class_detections']} (forward passes: {w_met['inference_count']})")
    print(f"latency total      : {m['total_ms']:.1f} ms/frame")
    if gpu_samples:
        util = sum(g[0] for g in gpu_samples) / len(gpu_samples)
        vram = max(g[1] for g in gpu_samples)
        print(f"gpu util (avg)     : {util:.0f}%   vram used (peak): {vram:.0f} MB")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"torch cuda peak mem: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    except ImportError:
        pass
    print(f"final person_count : {st.person_count}")
    print(f"incidents raised   : {len(st.latest_incidents)}")
    for inc in st.latest_incidents[-5:]:
        print(f"  - {inc['incident_id']} {inc['incident_type']} sev={inc['severity']} conf={inc['confidence']}")
    if st.latest_crowd:
        print(f"last crowd snapshot: {st.latest_crowd}")
    print("=" * 62)


if __name__ == "__main__":
    main()
