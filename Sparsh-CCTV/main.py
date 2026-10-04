import argparse
import os
import sys
import threading
import time
import cv2
import torch
import uvicorn

from config import (
    get_settings,
    update_settings,
    DEFAULT_TEST_VIDEO,
    SPARSH_RTSP_URL,
    DEFAULT_CRASH_CONF,
    DEFAULT_FIRE_CONF,
    RECORDING_DIR,
    ACCIDENT_DETECTOR_BACKEND,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_LOCATION,
    EMERGENCY_SEARCH_RADIUS_KM,
    DEFAULT_RECIPIENT_EMAIL,
    EMAIL_ALERTS_ENABLED,
)
from stream_capture import StreamCapture
from detectors.accident_detector import AccidentDetector
from detectors.yolo_hierarchical_detector import YOLOHierarchicalAccidentDetector
from detectors.fire_detector import FireDetector
from visualizer import CCTVVisualizer
from incident_logger import IncidentLogger
from services.alert_dispatcher import EmergencyAlertDispatcher
from services.email_service import EmailService
from surveillance_engine import get_surveillance_engine
from api.app import app, create_app


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sparsh CCTV Real-Time Accident Severity & Fire Detection FastAPI Server"
    )
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="FastAPI server binding host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI server binding port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn hot reloading for development",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        default=True,
        help="Automatically launch surveillance stream on server startup (default: True)",
    )
    parser.add_argument(
        "--no-auto-start",
        dest="auto_start",
        action="store_false",
        help="Start server with stream in idle mode",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run legacy OpenCV desktop window loop instead of FastAPI server",
    )

    # Surveillance and detection flags
    parser.add_argument(
        "--source",
        type=str,
        default="rtsp",
        help="Path to video file, webcam index (e.g. 0), or RTSP URL. Defaults to 'rtsp' (Sparsh live camera) if not specified.",
    )
    parser.add_argument(
        "--rtsp",
        action="store_true",
        help="Stream directly from Sparsh CCTV RTSP camera feed.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CRASH_CONF,
        help=f"Confidence threshold for accident/vehicle detection (default: {DEFAULT_CRASH_CONF})",
    )
    parser.add_argument(
        "--fire-conf",
        type=float,
        default=DEFAULT_FIRE_CONF,
        help=f"Confidence threshold for fire detection (default: {DEFAULT_FIRE_CONF})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--save",
        type=str,
        nargs="?",
        const=os.path.join(RECORDING_DIR, "surveillance_output.mp4"),
        default=None,
        help="Path to save output video with HUD. If passed without value, saves to output/recordings/surveillance_output.mp4",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="In CLI mode, run headless without opening OpenCV preview window.",
    )
    parser.add_argument(
        "--detector-backend",
        type=str,
        default=ACCIDENT_DETECTOR_BACKEND,
        choices=["yolo", "detr"],
        help=f"Accident detector backend: 'yolo' (Hierarchical 15 FPS) or 'detr' (legacy) (default: {ACCIDENT_DETECTOR_BACKEND})",
    )
    parser.add_argument(
        "--fps-cap",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help=f"Target camera stream FPS cap for real-time synchronization (default: {DEFAULT_CAMERA_FPS})",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=DEFAULT_CAMERA_ID,
        help=f"Camera identifier string (default: {DEFAULT_CAMERA_ID})",
    )
    parser.add_argument(
        "--camera-location",
        type=str,
        default=DEFAULT_CAMERA_LOCATION,
        help=f"Statically typed camera location or Plus Code (default: '{DEFAULT_CAMERA_LOCATION}')",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=EMERGENCY_SEARCH_RADIUS_KM,
        help=f"Emergency services search radius in km (default: {EMERGENCY_SEARCH_RADIUS_KM} km)",
    )
    parser.add_argument(
        "--disable-dispatch",
        action="store_true",
        help="Disable automated emergency alert dispatching.",
    )
    parser.add_argument(
        "--recipient-email",
        type=str,
        default=DEFAULT_RECIPIENT_EMAIL,
        help=f"Recipient email address for emergency alerts (default: {DEFAULT_RECIPIENT_EMAIL})",
    )
    parser.add_argument(
        "--disable-email",
        action="store_true",
        help="Disable automated emergency email dispatching.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process in CLI mode.",
    )
    return parser.parse_args()


def run_standalone_cli(args):
    """Legacy standalone OpenCV desktop preview loop."""
    if args.rtsp or args.source == "rtsp":
        video_source = SPARSH_RTSP_URL
        source_label = "Sparsh CCTV (Live RTSP)"
    elif args.source:
        if args.source.isdigit():
            video_source = int(args.source)
            source_label = f"Webcam {video_source}"
        else:
            video_source = args.source
            source_label = os.path.basename(args.source)
    else:
        video_source = SPARSH_RTSP_URL
        source_label = "Sparsh CCTV (Live RTSP)"

    print("=" * 65)
    print("   SPARSH CCTV REAL-TIME ACCIDENT SEVERITY & FIRE DETECTOR (CLI)   ")
    print("=" * 65)
    print(f"[*] Input Source    : {video_source} ({source_label})")
    print(f"[*] Detector Backend: {args.detector_backend.upper()}")
    print(f"[*] Target Stream   : {args.fps_cap} FPS Cap")
    print(f"[*] Inference Device: {args.device}")
    print("=" * 65)

    if args.detector_backend == "yolo":
        accident_detector = YOLOHierarchicalAccidentDetector(
            conf_thres=args.conf, device=args.device
        )
    else:
        accident_detector = AccidentDetector(conf_thres=args.conf, device=args.device)

    fire_detector = FireDetector(conf_thres=args.fire_conf, device=args.device)

    dispatcher = None
    if not args.disable_dispatch:
        enable_email = not args.disable_email and EMAIL_ALERTS_ENABLED
        email_service = None
        if enable_email:
            custom_dept_emails = {
                "police": args.recipient_email,
                "hospital": args.recipient_email,
                "fire": args.recipient_email,
                "traffic_control": args.recipient_email,
            }
            email_service = EmailService(
                enabled=True, department_emails=custom_dept_emails
            )

        dispatcher = EmergencyAlertDispatcher(
            camera_id=args.camera_id,
            camera_location=args.camera_location,
            email_service=email_service,
            enable_email=enable_email,
        )

    visualizer = CCTVVisualizer(
        camera_name=f"{args.camera_id} ({args.camera_location.split(',')[0]})"
    )
    incident_logger = IncidentLogger(
        dispatcher=dispatcher, enable_dispatch=not args.disable_dispatch
    )

    try:
        stream = StreamCapture(video_source, name=source_label, target_fps=args.fps_cap)
    except Exception as e:
        print(f"\n[ERROR] No live stream running: Could not connect to '{video_source}'.")
        print(f"        Details: {e}")
        print("        Ensure your RTSP camera is connected and reachable on the network,")
        print("        or specify a video file via: --source data/videos/test/crash.mp4\n")
        return
    video_writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save, fourcc, stream.fps, (stream.width, stream.height)
        )

    window_name = "Sparsh CCTV - Real-Time Accident & Fire Surveillance"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, stream.width), min(720, stream.height))

    start_time = time.time()
    processed_count = 0
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = stream.read()
                if not ret:
                    break

                processed_count += 1
                if args.max_frames and processed_count > args.max_frames:
                    break

                detections = accident_detector.detect(frame)
                highest_sev, highest_det = accident_detector.get_highest_severity(
                    detections
                )
                fire_result = fire_detector.detect(frame)

                incident_logger.check_and_log(
                    frame=frame,
                    highest_severity=highest_sev,
                    highest_det=highest_det,
                    fire_result=fire_result,
                    detections=detections,
                )

                annotated_frame = visualizer.draw(
                    frame, detections, fire_result, source_label
                )
                if video_writer is not None:
                    video_writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("s") or key == ord("S"):
                    incident_logger.save_snapshot(annotated_frame, tag="manual")
                elif key == ord("h") or key == ord("H"):
                    visualizer.show_hud = not visualizer.show_hud

    finally:
        stream.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


def main():
    args = parse_arguments()

    if args.cli:
        run_standalone_cli(args)
        return

    # Update system settings from CLI overrides
    updates = {}
    if args.conf is not None:
        updates["crash_conf"] = args.conf
    if args.fire_conf is not None:
        updates["fire_conf"] = args.fire_conf
    if args.camera_id is not None:
        updates["camera_id"] = args.camera_id
    if args.camera_location is not None:
        updates["camera_location"] = args.camera_location
    if args.search_radius is not None:
        updates["emergency_search_radius_km"] = args.search_radius
    if args.fps_cap is not None:
        updates["camera_fps"] = args.fps_cap
    if args.recipient_email is not None:
        updates["default_recipient_email"] = args.recipient_email
    if args.disable_dispatch:
        updates["enable_dispatch"] = False
    if args.disable_email:
        updates["smtp_enabled"] = False
    if args.detector_backend is not None:
        updates["detector_backend"] = args.detector_backend

    if args.rtsp:
        updates["video_source"] = "rtsp"
    elif args.source:
        updates["video_source"] = args.source

    update_settings(updates)
    current_settings = get_settings()

    print("=" * 72)
    print("   SPARSH CCTV REAL-TIME ACCIDENT SEVERITY & FIRE SURVEILLANCE GATEWAY   ")
    print("=" * 72)
    print(f"[*] API Base URL    : http://{args.host}:{args.port}")
    print(f"[*] Web Dashboard   : http://{args.host}:{args.port}/")
    print(f"[*] Swagger Docs    : http://{args.host}:{args.port}/docs")
    print(
        f"[*] Live Stream Feed: http://{args.host}:{args.port}/api/surveillance/stream"
    )
    print(f"[*] Dynamic Settings: http://{args.host}:{args.port}/api/settings")
    print(
        f"[*] Camera Node     : {current_settings.camera_id} ({current_settings.camera_location})"
    )
    print(f"[*] Video Source    : {current_settings.video_source}")
    print(f"[*] Stream Target   : {current_settings.camera_fps} FPS Cap")
    print(f"[*] Crash Conf Thres: {current_settings.crash_conf}")
    print(f"[*] Fire Conf Thres : {current_settings.fire_conf}")
    print("=" * 72)

    # Automatically start surveillance background worker in non-blocking thread
    if args.auto_start:
        def _background_startup():
            print("[+] Launching background surveillance coordinator...")
            engine = get_surveillance_engine()
            try:
                status = engine.start(
                    source=current_settings.video_source,
                    fps_cap=current_settings.camera_fps,
                    save_recording=bool(args.save),
                    recording_path=args.save,
                )
                if status.get("stream_connected", False):
                    print(
                        f"[+] Surveillance online: {status['frame_width']}x{status['frame_height']} from {status['source_label']}"
                    )
                else:
                    print(
                        f"[!] RTSP camera offline or unreachable. Surveillance coordinator running in NO_STREAM mode (NO LIVE STREAM RUNNING)."
                    )
                    print(
                        f"    (Connect camera to {SPARSH_RTSP_URL} or switch source via dashboard to a test video)"
                    )
            except Exception as e:
                print(f"[!] Warning: Could not auto-start stream: {e}")
                print(
                    "    (You can start it any time via POST /api/surveillance/start or Web Dashboard)"
                )

        threading.Thread(
            target=_background_startup,
            name="SparshAutoStartWorker",
            daemon=True,
        ).start()

    print(
        f"\n[+] Serving FastAPI HTTP server on http://{args.host}:{args.port} (Press Ctrl+C to terminate)...\n"
    )

    uvicorn.run(
        "main:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
