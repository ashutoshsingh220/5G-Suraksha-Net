import os
import sys
import time
import cv2
import numpy as np
import argparse

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acc_det.pipeline import HierarchicalAccidentPipeline, PipelineResult
from acc_det.config import ROOT_DIR, OUTPUT_DIR, TARGET_FPS, FRAME_INTERVAL_MS


def benchmark_video(video_path: str, max_frames: int = 150, save_output: bool = True):
    if not os.path.exists(video_path):
        print(f"[!] Error: Video file not found: {video_path}")
        return None

    video_name = os.path.basename(video_path)
    print("\n" + "=" * 65)
    print(f"   BENCHMARKING 15 FPS PIPELINE ON: {video_name}   ")
    print("=" * 65)
    print(f"[*] Stream Target FPS : {TARGET_FPS} FPS")
    print(f"[*] Max Frame Budget  : {FRAME_INTERVAL_MS:.2f} ms per frame")

    pipeline = HierarchicalAccidentPipeline(conf_thres=0.35)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_writer = None
    if save_output:
        out_dir = os.path.join(OUTPUT_DIR, "benchmark_videos")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"annotated_{video_name}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (w, h))
        print(f"[*] Output Recording  : {out_path}")

    frame_times = []
    t1_times = []
    t2_times = []
    accident_frames = 0
    moderate_count = 0
    severe_count = 0
    processed_count = 0

    print("\nProcessing frames...")
    while True:
        ret, frame = cap.read()
        if not ret or processed_count >= max_frames:
            break

        processed_count += 1
        res: PipelineResult = pipeline.process_frame(frame, frame_idx=processed_count)

        frame_times.append(res.inference_time_ms)
        t1_times.append(res.tier1_time_ms)
        t2_times.append(res.tier2_time_ms)

        if res.has_accident:
            accident_frames += 1
            if res.highest_severity == 3:
                severe_count += 1
            elif res.highest_severity == 2:
                moderate_count += 1

        # Draw visual annotations on frame
        if out_writer is not None:
            annotated = frame.copy()

            # Draw HUD card with safe dimension handling
            card_w = min(420, max(10, w - 20))
            card_h = min(95, max(10, h - 20))
            if card_w > 60 and card_h > 30:
                sub = annotated[10 : 10 + card_h, 10 : 10 + card_w]
                hud_bg = np.zeros_like(sub)
                cv2.addWeighted(hud_bg, 0.75, sub, 0.25, 0, sub)
                annotated[10 : 10 + card_h, 10 : 10 + card_w] = sub
                cv2.rectangle(
                    annotated, (10, 10), (10 + card_w, 10 + card_h), (100, 100, 100), 1
                )

            status_str = "NORMAL TRAFFIC"
            status_color = (0, 210, 60)
            if res.has_accident:
                if res.highest_severity == 3:
                    status_str = "SEVERE ACCIDENT ALERT"
                    status_color = (0, 40, 240)
                else:
                    status_str = "MODERATE ACCIDENT WARNING"
                    status_color = (0, 165, 255)

            cv2.putText(
                annotated,
                f"CCTV SERVER [15 FPS CAP]",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                annotated,
                f"STATUS: {status_str}",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
            )
            cv2.putText(
                annotated,
                f"Latency: {res.inference_time_ms:.1f}ms / Budget: {FRAME_INTERVAL_MS:.1f}ms",
                (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            # Draw detections
            for d in res.detections:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), d.color, 2)
                badge_text = f"{d.display_name} ({d.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(
                    badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                by1 = max(0, y1 - th - 6)
                cv2.rectangle(annotated, (x1, by1), (x1 + tw + 6, y1), d.color, -1)
                cv2.putText(
                    annotated,
                    badge_text,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            out_writer.write(annotated)

    cap.release()
    if out_writer is not None:
        out_writer.release()

    avg_latency = np.mean(frame_times) if frame_times else 0.0
    p95_latency = np.percentile(frame_times, 95) if frame_times else 0.0
    avg_t1 = np.mean(t1_times) if t1_times else 0.0
    avg_t2 = np.mean(t2_times) if t2_times else 0.0

    print("\n" + "-" * 40)
    print("   15 FPS PERFORMANCE BENCHMARK REPORT   ")
    print("-" * 40)
    print(f"Total Frames Processed: {processed_count}")
    print(
        f"Accident Detected     : {accident_frames} frames ({accident_frames / max(1, processed_count):.1%})"
    )
    print(f"  - Moderate Accidents: {moderate_count}")
    print(f"  - Severe Accidents  : {severe_count}")
    print(f"Average Total Latency : {avg_latency:.2f} ms per frame")
    print(f"  - Tier 1 Detection  : {avg_t1:.2f} ms")
    print(f"  - Tier 2 Severity   : {avg_t2:.2f} ms")
    print(f"95th Percentile Latency: {p95_latency:.2f} ms")
    print(f"15 FPS Time Budget    : {FRAME_INTERVAL_MS:.2f} ms")
    print(f"Budget Utilization    : {(avg_latency / FRAME_INTERVAL_MS):.1%}")
    print(
        f"Real-Time Headroom    : {(FRAME_INTERVAL_MS - avg_latency):.2f} ms spare per frame"
    )
    print("-" * 40)

    return {
        "video": video_name,
        "frames": processed_count,
        "accidents": accident_frames,
        "moderate": moderate_count,
        "severe": severe_count,
        "avg_latency": avg_latency,
        "budget_pct": (avg_latency / FRAME_INTERVAL_MS) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Hierarchical 15 FPS Accident Detection"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Specific video to benchmark. If None, runs standard suite.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=120, help="Frames per video (default: 120)"
    )
    args = parser.parse_args()

    videos_dir = os.path.join(ROOT_DIR, "data", "videos")
    acc_dir = os.path.join(videos_dir, "accident")
    noacc_dir = os.path.join(videos_dir, "no accident")

    if args.source:
        test_videos = [args.source]
    else:
        # Search candidate videos across root, accident/, and no accident/
        test_videos = []
        # Selected accident feeds
        for vid in [
            "v10.mp4",
            "v1.mp4",
            "v77.mp4",
            "v28.mp4",
            "crash.mp4",
            "bikeacc.mp4",
            "realcrash1.mp4",
        ]:
            for d in [videos_dir, acc_dir]:
                p = os.path.join(d, vid)
                if os.path.exists(p) and p not in test_videos:
                    test_videos.append(p)
                    break
            if len(test_videos) >= 3:
                break

        # Selected no-accident feeds
        for vid in ["v8.mp4", "v3.mp4", "v11.mp4"]:
            for d in [videos_dir, noacc_dir]:
                p = os.path.join(d, vid)
                if os.path.exists(p) and p not in test_videos:
                    test_videos.append(p)
                    break

    print("=" * 65)
    print("   SERVER-SIDE 15 FPS SURVEILLANCE ACCIDENT BENCHMARK SUITE   ")
    print("=" * 65)
    print(f"Found {len(test_videos)} benchmark video feeds.")

    results = []
    for vid in test_videos:
        res = benchmark_video(vid, max_frames=args.max_frames, save_output=True)
        if res:
            results.append(res)

    print("\n" + "=" * 65)
    print("   BENCHMARK SUITE SUMMARY   ")
    print("=" * 65)
    for r in results:
        print(
            f"  {r['video']:<15} | Acc: {r['accidents']:>3}/{r['frames']} | Mod: {r['moderate']:>2} | Sev: {r['severe']:>2} | Latency: {r['avg_latency']:>5.1f}ms ({r['budget_pct']:.1f}% budget)"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
