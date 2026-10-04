import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Generator, Tuple
import cv2
import numpy as np

from config import (
    get_settings,
    register_settings_listener,
    SystemSettings,
    DEFAULT_TEST_VIDEO,
    SPARSH_RTSP_URL,
    RECORDING_DIR,
    SNAPSHOT_DIR,
    SEVERITY_CONFIG,
)
from stream_capture import StreamCapture, probe_rtsp_reachability
from detectors.accident_detector import AccidentDetector, VehicleDetection
from detectors.yolo_hierarchical_detector import YOLOHierarchicalAccidentDetector
from detectors.fire_detector import FireDetector, FireResult
from visualizer import CCTVVisualizer
from incident_logger import IncidentLogger
from services.alert_dispatcher import EmergencyAlertDispatcher
from services.email_service import EmailService


class SurveillanceEngine:
    """
    Central background surveillance coordinator managing the real-time AI processing pipeline,
    stream capture pacing, MJPEG streaming feed, frame buffering, and dynamic settings updates.
    """

    def __init__(
        self,
        accident_detector=None,
        fire_detector=None,
        visualizer=None,
        incident_logger=None,
        dispatcher=None,
        auto_init_detectors: bool = True,
    ):
        self.settings: SystemSettings = get_settings()

        # Threading & Control
        self._lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # State
        self.is_running: bool = False
        self.is_paused: bool = False
        self.stream_connected: bool = False
        self.stream_status: str = "IDLE"  # "STREAMING", "NO_STREAM", "PAUSED", "IDLE", "STOPPED"
        self.stream_error: Optional[str] = None
        self._last_reconnect_attempt: float = 0.0
        self.current_source: Optional[str] = None
        self.source_label: str = "Idle"
        self.current_fps: float = 0.0
        self.target_fps: float = self.settings.camera_fps
        self.processed_frames: int = 0
        self.start_time: float = 0.0
        self.frame_width: int = 1280
        self.frame_height: int = 720
        self.last_alert_time: Optional[str] = None

        # Live Frame & Detection Cache
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_detections: List[VehicleDetection] = []
        self.latest_fire_result: Optional[FireResult] = None
        self.latest_highest_severity: int = -1
        self.latest_highest_det: Optional[VehicleDetection] = None

        # Stream & Recording handles
        self.stream: Optional[StreamCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None

        # Components
        self.accident_detector = accident_detector
        self.fire_detector = fire_detector
        self.visualizer = visualizer or CCTVVisualizer(
            camera_name=f"{self.settings.camera_id} ({self.settings.camera_location.split(',')[0]})"
        )
        self.email_service = None
        self.dispatcher = dispatcher
        self.incident_logger = incident_logger

        if auto_init_detectors:
            self._init_components()

        # Register for dynamic settings synchronization
        register_settings_listener(self._on_settings_updated)

    def _init_components(self):
        """Initializes detectors, dispatcher, email service, and incident logger if not provided."""
        # Email Service
        if self.email_service is None:
            self.email_service = EmailService(
                host=self.settings.smtp_host,
                port=self.settings.smtp_port,
                use_tls=self.settings.smtp_use_tls,
                sender_email=self.settings.smtp_sender_email,
                password=self.settings.smtp_app_password,
                timeout=self.settings.smtp_timeout_seconds,
                enabled=self.settings.smtp_enabled,
                department_emails=self.settings.department_emails,
                incident_routing=self.settings.incident_department_routing,
                attach_snapshot=self.settings.attach_snapshot_in_email,
            )

        # Dispatcher
        if self.dispatcher is None:
            self.dispatcher = EmergencyAlertDispatcher(
                camera_id=self.settings.camera_id,
                camera_location=self.settings.camera_location,
                email_service=self.email_service,
                enable_email=self.settings.smtp_enabled,
                enable_console=self.settings.enable_console_alerts,
                webhook_url=self.settings.webhook_url,
            )
            # Update Twilio credentials from settings
            self.dispatcher.update_twilio(
                sid=self.settings.twilio_account_sid,
                auth=self.settings.twilio_auth_token,
                phone_from=self.settings.twilio_phone_number,
                dispatch_to=self.settings.emergency_dispatch_phone,
            )

        # Incident Logger
        if self.incident_logger is None:
            self.incident_logger = IncidentLogger(
                dispatcher=self.dispatcher,
                enable_dispatch=self.settings.enable_dispatch,
            )
            self.incident_logger.set_cooldown(self.settings.alert_cooldown_sec)

        # Detectors
        if self.accident_detector is None:
            try:
                if self.settings.detector_backend == "yolo":
                    self.accident_detector = YOLOHierarchicalAccidentDetector(
                        conf_thres=self.settings.crash_conf,
                        iou_thres=self.settings.crash_iou,
                        device=self.settings.inference_device,
                    )
                else:
                    self.accident_detector = AccidentDetector(
                        conf_thres=self.settings.crash_conf,
                        iou_thres=self.settings.crash_iou,
                        device=self.settings.inference_device,
                    )
            except Exception as e:
                print(
                    f"[SurveillanceEngine] Warning: Could not initialize accident detector: {e}"
                )

        if self.fire_detector is None:
            try:
                self.fire_detector = FireDetector(
                    conf_thres=self.settings.fire_conf,
                    device=self.settings.inference_device,
                )
            except Exception as e:
                print(
                    f"[SurveillanceEngine] Warning: Could not initialize fire detector: {e}"
                )

    def _on_settings_updated(self, settings: SystemSettings, updates: Dict[str, Any]):
        """Observer callback invoked whenever settings are dynamically updated via API."""
        with self._lock:
            self.settings = settings
            self.target_fps = settings.camera_fps
            if self.stream is not None:
                self.stream.target_fps = settings.camera_fps
                self.stream.frame_interval = (
                    (1.0 / settings.camera_fps) if settings.camera_fps > 0 else 0.0
                )

            # Reload or update Accident Detector
            if "detector_backend" in updates or "inference_device" in updates:
                try:
                    if settings.detector_backend == "yolo":
                        self.accident_detector = YOLOHierarchicalAccidentDetector(
                            conf_thres=settings.crash_conf,
                            iou_thres=settings.crash_iou,
                            device=settings.inference_device,
                        )
                    else:
                        self.accident_detector = AccidentDetector(
                            conf_thres=settings.crash_conf,
                            iou_thres=settings.crash_iou,
                            device=settings.inference_device,
                        )
                except Exception as e:
                    print(
                        f"[SurveillanceEngine] Warning: Could not reconfigure accident detector: {e}"
                    )
            elif self.accident_detector is not None:
                if "crash_conf" in updates:
                    self.accident_detector.set_confidence(settings.crash_conf)
                if "crash_iou" in updates and hasattr(
                    self.accident_detector, "iou_thres"
                ):
                    self.accident_detector.iou_thres = settings.crash_iou

            # Reload or update Fire Detector
            if "inference_device" in updates:
                try:
                    self.fire_detector = FireDetector(
                        conf_thres=settings.fire_conf, device=settings.inference_device
                    )
                except Exception as e:
                    print(
                        f"[SurveillanceEngine] Warning: Could not reconfigure fire detector: {e}"
                    )
            elif self.fire_detector is not None:
                if "fire_conf" in updates:
                    self.fire_detector.set_confidence(settings.fire_conf)

            # Update Visualizer
            if self.visualizer is not None and (
                "camera_id" in updates or "camera_location" in updates
            ):
                loc_short = settings.camera_location.split(",")[0]
                self.visualizer.camera_name = f"{settings.camera_id} ({loc_short})"

            # Update Email Service
            if self.email_service is not None:
                email_fields = [
                    "smtp_host",
                    "smtp_port",
                    "smtp_use_tls",
                    "smtp_sender_email",
                    "smtp_app_password",
                    "smtp_timeout_seconds",
                    "smtp_enabled",
                    "department_emails",
                    "incident_department_routing",
                    "attach_snapshot_in_email",
                ]
                if any(f in updates for f in email_fields):
                    self.email_service.update_config(
                        host=settings.smtp_host,
                        port=settings.smtp_port,
                        use_tls=settings.smtp_use_tls,
                        sender_email=settings.smtp_sender_email,
                        password=settings.smtp_app_password,
                        timeout=settings.smtp_timeout_seconds,
                        enabled=settings.smtp_enabled,
                        department_emails=settings.department_emails,
                        incident_routing=settings.incident_department_routing,
                        attach_snapshot=settings.attach_snapshot_in_email,
                    )

            # Update Emergency Dispatcher
            if self.dispatcher is not None:
                if "camera_id" in updates:
                    self.dispatcher.camera_id = settings.camera_id
                if any(
                    k in updates
                    for k in (
                        "camera_location",
                        "emergency_search_radius_km",
                        "camera_latitude",
                        "camera_longitude",
                    )
                ):
                    self.dispatcher.update_location(
                        new_location=settings.camera_location,
                        new_radius_km=settings.emergency_search_radius_km,
                        latitude=settings.camera_latitude,
                        longitude=settings.camera_longitude,
                    )
                if "max_dispatch_facilities_per_type" in updates:
                    self.dispatcher.max_dispatch_facilities_per_type = (
                        settings.max_dispatch_facilities_per_type
                    )
                if any(
                    k.startswith("twilio") or k == "emergency_dispatch_phone"
                    for k in updates
                ):
                    self.dispatcher.update_twilio(
                        sid=settings.twilio_account_sid,
                        auth=settings.twilio_auth_token,
                        phone_from=settings.twilio_phone_number,
                        dispatch_to=settings.emergency_dispatch_phone,
                    )
                if "enable_console_alerts" in updates:
                    self.dispatcher.enable_console = settings.enable_console_alerts
                if "webhook_url" in updates:
                    self.dispatcher.webhook_url = settings.webhook_url

            # Update Incident Logger
            if self.incident_logger is not None:
                if "alert_cooldown_sec" in updates:
                    self.incident_logger.set_cooldown(settings.alert_cooldown_sec)
                if "enable_dispatch" in updates:
                    self.incident_logger.set_enable_dispatch(settings.enable_dispatch)

            # Hot-switch video source if stream is currently active
            if "video_source" in updates and self.is_running:
                try:
                    self.start(source=settings.video_source)
                except Exception as e:
                    print(
                        f"[SurveillanceEngine] Could not hot-switch video source: {e}"
                    )

            print(
                f"[SurveillanceEngine] Dynamically reconfigured runtime settings: {list(updates.keys())}"
            )

    def start(
        self,
        source: Optional[str] = None,
        fps_cap: Optional[float] = None,
        save_recording: bool = False,
        recording_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Starts the background surveillance stream capture and real-time detection thread.
        """
        with self._lock:
            if self.is_running:
                # Stop existing stream first to switch clean
                self.stop()

            # Ensure detectors are initialized
            self._init_components()

            # Resolve video source
            chosen_source = source or self.settings.video_source
            is_rtsp = False
            if isinstance(chosen_source, str) and chosen_source.isdigit():
                video_source = int(chosen_source)
                self.source_label = f"Webcam {video_source}"
            elif chosen_source in ("rtsp", "sparsh", "live", "camera") or (
                isinstance(chosen_source, str)
                and (
                    chosen_source.startswith("rtsp://")
                    or chosen_source.startswith("rtsps://")
                )
            ):
                video_source = (
                    self.settings.rtsp_url
                    if chosen_source in ("rtsp", "sparsh", "live", "camera")
                    else chosen_source
                )
                self.source_label = "Sparsh CCTV (Live RTSP)"
                is_rtsp = True
            elif chosen_source:
                video_source = chosen_source
                self.source_label = os.path.basename(str(chosen_source))
            else:
                video_source = self.settings.rtsp_url
                self.source_label = "Sparsh CCTV (Live RTSP)"
                is_rtsp = True

            self.current_source = str(video_source)
            self.target_fps = fps_cap or self.settings.camera_fps

            # Clean all previous frame and detection buffers
            with self._frame_lock:
                self.latest_raw_frame = None
                self.latest_annotated_frame = None
                self.latest_detections = []
                self.latest_fire_result = None
                self.latest_highest_severity = -1
                self.latest_highest_det = None
                self.current_fps = 0.0

            # Attempt stream opening
            self.stream = None
            self.stream_connected = False
            self.stream_error = None

            try:
                self.stream = StreamCapture(
                    source=video_source,
                    name=self.source_label,
                    target_fps=self.target_fps,
                )
                self.stream_connected = True
                self.stream_status = "STREAMING"
                self.frame_width = self.stream.width
                self.frame_height = self.stream.height
            except Exception as e:
                if is_rtsp:
                    # RTSP streaming device offline/unreachable: do NOT crash or fallback to video!
                    print(f"[SurveillanceEngine] RTSP camera offline or unreachable: {e}")
                    self.stream = None
                    self.stream_connected = False
                    self.stream_status = "NO_STREAM"
                    self.stream_error = f"No streaming device found at {video_source}"
                else:
                    raise RuntimeError(
                        f"Failed to connect to video stream '{video_source}': {e}"
                    )

            # Optional Recording
            if save_recording and self.stream is not None:
                rec_file = recording_path or os.path.join(
                    RECORDING_DIR,
                    f"surveillance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                )
                os.makedirs(os.path.dirname(os.path.abspath(rec_file)), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(
                    rec_file,
                    fourcc,
                    self.stream.fps or self.target_fps,
                    (self.frame_width, self.frame_height),
                )

            self._stop_event.clear()
            self.is_running = True
            self.is_paused = False
            self.processed_frames = 0
            self.start_time = time.time()

            self._thread = threading.Thread(
                target=self._surveillance_loop,
                name="SparshSurveillanceWorker",
                daemon=True,
            )
            self._thread.start()

            return self.get_status()

    def stop(self) -> Dict[str, Any]:
        """Gracefully halts the surveillance worker thread and releases media streams."""
        with self._lock:
            if not self.is_running:
                return self.get_status()

            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3.0)
            self._thread = None

            if self.stream:
                try:
                    self.stream.release()
                except Exception:
                    pass
                self.stream = None

            if self.video_writer:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None

            self.is_running = False
            self.is_paused = False
            self.stream_connected = False
            self.stream_status = "STOPPED"
            self.current_fps = 0.0

            # Wipe all frame and detection buffers so no stale frames/alerts linger
            with self._frame_lock:
                self.latest_raw_frame = None
                self.latest_annotated_frame = None
                self.latest_detections = []
                self.latest_fire_result = None
                self.latest_highest_severity = -1
                self.latest_highest_det = None

            print("[SurveillanceEngine] Background surveillance stream stopped.")
            return self.get_status()

    def pause(self) -> Dict[str, Any]:
        """Pauses frame ingestion and inference processing."""
        with self._lock:
            if self.is_running:
                self.is_paused = True
                self.stream_status = "PAUSED"
            return self.get_status()

    def resume(self) -> Dict[str, Any]:
        """Resumes frame ingestion and inference processing."""
        with self._lock:
            if self.is_running:
                self.is_paused = False
                self.stream_status = "STREAMING" if self.stream_connected else "NO_STREAM"
            return self.get_status()

    def _surveillance_loop(self):
        """Main background real-time processing loop."""
        print(
            f"[SurveillanceEngine] Stream running on {self.source_label} (target {self.target_fps} FPS)"
        )

        while not self._stop_event.is_set():
            if self.is_paused:
                time.sleep(0.05)
                continue

            # If no active stream is connected (e.g. RTSP camera offline)
            if not self.stream_connected or self.stream is None:
                slate = self._generate_no_stream_frame()
                with self._frame_lock:
                    self.latest_raw_frame = None
                    self.latest_annotated_frame = slate
                    self.latest_detections = []
                    self.latest_fire_result = None
                    self.latest_highest_severity = -1
                    self.latest_highest_det = None
                    self.current_fps = 0.0

                # Periodically attempt reconnection if in RTSP mode (every 8s)
                now = time.time()
                is_rtsp_target = (
                    self.current_source == "rtsp"
                    or self.current_source == self.settings.rtsp_url
                    or (
                        isinstance(self.current_source, str)
                        and (
                            self.current_source.startswith("rtsp://")
                            or self.current_source.startswith("rtsps://")
                        )
                    )
                )
                if is_rtsp_target and (now - self._last_reconnect_attempt > 8.0):
                    self._last_reconnect_attempt = now
                    try:
                        probe_url = (
                            self.settings.rtsp_url
                            if self.current_source in ("rtsp", "sparsh")
                            else self.current_source
                        )
                        reachable, _ = probe_rtsp_reachability(probe_url, timeout=1.0)
                        if reachable:
                            new_stream = StreamCapture(
                                source=probe_url,
                                name=self.source_label,
                                target_fps=self.target_fps,
                            )
                            self.stream = new_stream
                            self.stream_connected = True
                            self.stream_status = "STREAMING"
                            self.stream_error = None
                            self.frame_width = new_stream.width
                            self.frame_height = new_stream.height
                            print(
                                f"[SurveillanceEngine] RTSP camera reconnected successfully at {probe_url}!"
                            )
                            continue
                    except Exception:
                        pass

                time.sleep(0.066)  # ~15 FPS slate refresh
                continue

            ret, frame = self.stream.read()
            if not ret or frame is None:
                # Video file reached end -> Loop video for continuous surveillance feed
                if not self.stream.is_live and self.stream.cap is not None:
                    self.stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.stream.read()

                if not ret or frame is None:
                    if self.stream.is_live:
                        print(
                            f"[SurveillanceEngine] Live stream interrupted on {self.source_label}"
                        )
                        self.stream_connected = False
                        self.stream_status = "NO_STREAM"
                        self.stream_error = f"RTSP connection lost to {self.current_source}"
                    time.sleep(0.05)
                    continue

            self.processed_frames += 1

            # 1. Accident Detection & Severity Classification
            detections: List[VehicleDetection] = []
            highest_sev: int = -1
            highest_det: Optional[VehicleDetection] = None

            if self.accident_detector is not None:
                try:
                    detections = self.accident_detector.detect(frame)
                    highest_sev, highest_det = (
                        self.accident_detector.get_highest_severity(detections)
                    )
                except Exception as e:
                    print(f"[SurveillanceEngine] Accident detector error: {e}")

            # 2. Fire & Smoke Classification
            fire_result: Optional[FireResult] = None
            if self.fire_detector is not None:
                try:
                    fire_result = self.fire_detector.detect(frame)
                except Exception as e:
                    print(f"[SurveillanceEngine] Fire detector error: {e}")

            # 3. Incident Logging & Automated Emergency Alert Dispatch
            if self.incident_logger is not None:
                try:
                    snap_saved = self.incident_logger.check_and_log(
                        frame=frame,
                        highest_severity=highest_sev,
                        highest_det=highest_det,
                        fire_result=fire_result,
                        detections=detections,
                    )
                    if snap_saved:
                        self.last_alert_time = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                except Exception as e:
                    print(f"[SurveillanceEngine] Incident logger error: {e}")

            # 4. Render HUD Overlay
            annotated_frame = frame
            if self.visualizer is not None:
                try:
                    annotated_frame = self.visualizer.draw(
                        frame=frame,
                        detections=detections,
                        fire_result=fire_result,
                        source_label=self.source_label,
                    )
                except Exception as e:
                    print(f"[SurveillanceEngine] Visualizer error: {e}")
                    annotated_frame = frame

            # 5. Record frame if writer is active
            if self.video_writer is not None:
                try:
                    self.video_writer.write(annotated_frame)
                except Exception as e:
                    print(f"[SurveillanceEngine] Video writer error: {e}")

            # 6. Update thread-safe latest frame buffer
            with self._frame_lock:
                self.latest_raw_frame = frame
                self.latest_annotated_frame = annotated_frame
                self.latest_detections = detections
                self.latest_fire_result = fire_result
                self.latest_highest_severity = highest_sev
                self.latest_highest_det = highest_det
                if self.visualizer is not None:
                    self.current_fps = self.visualizer.fps

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive real-time surveillance status."""
        uptime = (
            (time.time() - self.start_time)
            if (self.is_running and self.start_time > 0 and self.stream_connected)
            else 0.0
        )

        is_no_stream = (not self.stream_connected) or (self.stream_status == "NO_STREAM")

        if is_no_stream:
            current_status = "NO_STREAM" if self.is_running else "IDLE"
            highest_label = "No Feed"
            fire_label = "No Feed"
            has_fire = False
            has_smoke = False
            fps_val = 0.0
            veh_count = 0
            highest_sev_val = -1
        else:
            has_fire = bool(self.latest_fire_result and self.latest_fire_result.has_fire)
            has_smoke = bool(
                self.latest_fire_result and self.latest_fire_result.has_smoke
            )
            current_status = (
                "INCIDENT"
                if (self.latest_highest_severity >= 2 or has_fire or has_smoke)
                else "NORMAL"
            )
            highest_label = "None"
            if self.latest_highest_det:
                highest_label = self.latest_highest_det.display_name
            elif (
                self.latest_highest_severity >= 0
                and self.latest_highest_severity in SEVERITY_CONFIG
            ):
                highest_label = SEVERITY_CONFIG[self.latest_highest_severity][
                    "display_name"
                ]
            fire_label = (
                self.latest_fire_result.predicted_label
                if self.latest_fire_result
                else "Normal"
            )
            fps_val = round(self.current_fps, 1)
            veh_count = len(self.latest_detections)
            highest_sev_val = self.latest_highest_severity

        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "stream_connected": self.stream_connected,
            "stream_status": self.stream_status,
            "stream_error": self.stream_error,
            "source": self.current_source or "None",
            "source_label": self.source_label,
            "fps": fps_val,
            "target_fps": self.target_fps,
            "processed_frames": self.processed_frames,
            "uptime_seconds": round(uptime, 1),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "camera_id": self.settings.camera_id,
            "camera_location": self.settings.camera_location,
            "detector_backend": self.settings.detector_backend,
            "current_status": current_status,
            "highest_severity": highest_sev_val,
            "highest_severity_label": highest_label,
            "monitored_vehicles": veh_count,
            "has_fire": has_fire,
            "has_smoke": has_smoke,
            "fire_status_label": fire_label,
            "last_alert_time": self.last_alert_time,
        }

    def _generate_no_stream_frame(self) -> np.ndarray:
        """Generates a high-contrast 'NO LIVE STREAM RUNNING' slate when camera is offline."""
        h, w = self.frame_height, self.frame_width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 22)  # Dark surveillance background

        # Grid lines
        for y in range(0, h, 60):
            cv2.line(canvas, (0, y), (w, y), (28, 28, 34), 1)
        for x in range(0, w, 60):
            cv2.line(canvas, (x, 0), (x, h), (28, 28, 34), 1)

        # Center Alert Box Outline (Amber / Crimson Red)
        box_w, box_h = 760, 240
        box_x = max(20, (w - box_w) // 2)
        box_y = max(20, (h - box_h) // 2 - 20)
        cv2.rectangle(
            canvas,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (30, 30, 42),
            -1,
        )
        cv2.rectangle(
            canvas,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (0, 140, 255),  # Vibrant Amber-Orange
            2,
        )

        font_header = cv2.FONT_HERSHEY_DUPLEX
        font_sub = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Main Header: "NO LIVE STREAM RUNNING"
        cv2.putText(
            canvas,
            "NO LIVE STREAM RUNNING",
            (box_x + 110, box_y + 55),
            font_header,
            1.0,
            (0, 69, 255),  # Red-Orange Alert
            2,
            cv2.LINE_AA,
        )

        # 2. Subheader
        cv2.putText(
            canvas,
            "RTSP Stream Offline (No streaming device detected)",
            (box_x + 90, box_y + 95),
            font_sub,
            0.62,
            (0, 220, 255),  # Yellow-Amber
            1,
            cv2.LINE_AA,
        )

        # 3. Connection details
        target_url = (
            self.settings.rtsp_url
            if self.current_source in ("rtsp", "sparsh")
            else (self.current_source or "RTSP")
        )
        if len(target_url) > 65:
            display_url = target_url[:62] + "..."
        else:
            display_url = target_url

        cv2.putText(
            canvas,
            f"Target URL: {display_url}",
            (box_x + 40, box_y + 135),
            font_sub,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Camera ID : {self.settings.camera_id} ({self.settings.camera_location.split(',')[0]})",
            (box_x + 40, box_y + 165),
            font_sub,
            0.48,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Action: Connect your Sparsh camera or select a test video from the dropdown above.",
            (box_x + 40, box_y + 205),
            font_sub,
            0.44,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )

        # Timestamp at bottom of screen
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            canvas,
            f"● OFFLINE | Server Time: {ts}",
            (w // 2 - 140, box_y + box_h + 40),
            font_sub,
            0.46,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def get_latest_frame(self, annotated: bool = True) -> Optional[np.ndarray]:
        """Returns the latest available frame as a numpy array."""
        with self._frame_lock:
            frame = self.latest_annotated_frame if annotated else self.latest_raw_frame
            if frame is not None:
                return frame.copy()
            if self.stream_status == "NO_STREAM" or not self.stream_connected:
                return self._generate_no_stream_frame()
            return self._generate_standby_frame()

    def get_latest_jpeg(self, annotated: bool = True, quality: int = 85) -> bytes:
        """Encodes and returns the latest available frame as JPEG bytes."""
        frame = self.get_latest_frame(annotated=annotated)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, quality))]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            raise RuntimeError("Failed to encode frame to JPEG format")
        return buffer.tobytes()

    def generate_mjpeg(
        self, annotated: bool = True, fps_limit: float = 15.0
    ) -> Generator[bytes, None, None]:
        """
        Yields multipart MJPEG chunks for continuous browser video streaming.
        """
        interval = 1.0 / max(1.0, fps_limit)
        while True:
            t0 = time.time()
            try:
                frame_bytes = self.get_latest_jpeg(annotated=annotated)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(frame_bytes)).encode()
                    + b"\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            except Exception as e:
                print(f"[SurveillanceEngine] Stream frame error: {e}")

            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _generate_standby_frame(self) -> np.ndarray:
        """Generates a styled standby slate when stream is paused or stopped."""
        h, w = self.frame_height, self.frame_width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (22, 22, 26)  # Dark surveillance background

        # Grid lines
        for y in range(0, h, 60):
            cv2.line(canvas, (0, y), (w, y), (32, 32, 38), 1)
        for x in range(0, w, 60):
            cv2.line(canvas, (x, 0), (x, h), (32, 32, 38), 1)

        # Center banner
        header = "SPARSH CCTV - AI SURVEILLANCE & EMERGENCY RESPONSE"
        status_msg = (
            "STREAM IDLE - STANDBY MODE"
            if not self.is_running
            else "SURVEILLANCE PAUSED"
        )
        cam_info = f"Camera: {self.settings.camera_id} | Location: {self.settings.camera_location}"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            canvas,
            header,
            (w // 2 - 320, h // 2 - 40),
            font,
            0.75,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            status_msg,
            (w // 2 - 180, h // 2 + 10),
            font,
            0.65,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            cam_info,
            (w // 2 - 280, h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Server Time: {ts}",
            (w // 2 - 120, h // 2 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def capture_snapshot(
        self,
        tag: str = "manual",
        dispatch_alert: bool = False,
        incident_type: str = "MANUAL_ALERT",
        severity_label: str = "Manual Inspection",
    ) -> Dict[str, Any]:
        """
        Captures and saves a snapshot JPEG image to disk with optional emergency dispatch.
        """
        frame = self.get_latest_frame(annotated=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{tag}_{ts_str}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)

        cv2.imwrite(filepath, frame)

        dispatch_res = None
        if dispatch_alert and self.dispatcher:
            try:
                dispatch_res = self.dispatcher.dispatch(
                    incident_type=incident_type,
                    severity_label=severity_label,
                    confidence=1.0,
                    snapshot_path=filepath,
                    vehicles_count=len(self.latest_detections),
                )
            except Exception as e:
                print(f"[SurveillanceEngine] Manual dispatch error: {e}")

        return {
            "status": "SUCCESS",
            "filename": filename,
            "filepath": filepath,
            "relative_url": f"/api/snapshots/{filename}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dispatched": bool(dispatch_res is not None),
            "dispatch_id": dispatch_res.get("dispatch_id") if dispatch_res else None,
        }

    def get_latest_detections_payload(self) -> Dict[str, Any]:
        """Returns structured JSON serialization of current frame detections."""
        with self._frame_lock:
            vehicles_data = []
            for d in self.latest_detections:
                vehicles_data.append(
                    {
                        "bbox": list(d.bbox),
                        "confidence": round(float(d.confidence), 3),
                        "class_id": d.class_id,
                        "raw_name": d.raw_name,
                        "display_name": d.display_name,
                        "short_name": d.short_name,
                        "is_accident": d.is_accident,
                        "severity_level": d.severity_level,
                    }
                )

            fire_data = {
                "has_fire": False,
                "has_smoke": False,
                "fire_confidence": 0.0,
                "smoke_confidence": 0.0,
                "predicted_label": "Normal",
                "detections": [],
            }
            if self.latest_fire_result:
                fire_boxes = []
                for fb in self.latest_fire_result.detections:
                    fire_boxes.append(
                        {
                            "bbox": list(fb.bbox),
                            "confidence": round(float(fb.confidence), 3),
                            "class_id": fb.class_id,
                            "label": fb.label,
                        }
                    )
                fire_data = {
                    "has_fire": self.latest_fire_result.has_fire,
                    "has_smoke": self.latest_fire_result.has_smoke,
                    "fire_confidence": round(
                        float(self.latest_fire_result.fire_confidence), 3
                    ),
                    "smoke_confidence": round(
                        float(self.latest_fire_result.smoke_confidence), 3
                    ),
                    "predicted_label": self.latest_fire_result.predicted_label,
                    "detections": fire_boxes,
                }

            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "monitored_vehicles": len(self.latest_detections),
                "highest_severity": self.latest_highest_severity,
                "highest_severity_label": self.latest_highest_det.display_name
                if self.latest_highest_det
                else "None",
                "vehicles": vehicles_data,
                "fire_hazard": fire_data,
            }


_engine_instance: Optional[SurveillanceEngine] = None
_engine_lock = threading.Lock()


def get_surveillance_engine(auto_init_detectors: bool = True) -> SurveillanceEngine:
    """Returns the singleton SurveillanceEngine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = SurveillanceEngine(
                auto_init_detectors=auto_init_detectors
            )
        return _engine_instance
