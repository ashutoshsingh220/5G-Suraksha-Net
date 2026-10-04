import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
RECORDING_DIR = os.path.join(OUTPUT_DIR, "recordings")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)

# Default Model Paths
CRASH_MODEL_DIR = os.path.join(MODELS_DIR, "crash_classifier")
CRASH_MODEL_PATH = os.path.join(CRASH_MODEL_DIR, "model.safetensors")
FIRE_MODEL_PATH = os.path.join(MODELS_DIR, "fire_classifier", "fire_classifier.pt")

# YOLO Hierarchical Accident & Severity Detector
ACCIDENT_DETECTOR_BACKEND = "yolo"  # "yolo" or "detr"
YOLO_DETECTOR_PATH = os.path.join(MODELS_DIR, "accident_classifier", "yolo_detector.pt")
YOLO_SEVERITY_PATH = os.path.join(MODELS_DIR, "accident_classifier", "yolo_severity.pt")

# Camera Stream & Server FPS Configuration
DEFAULT_CAMERA_FPS = 15.0  # Camera stream runs on 15 FPS cap
FRAME_TIME_MS = 1000.0 / DEFAULT_CAMERA_FPS  # 66.66 ms budget per frame

# Default Video & Camera Settings
DEFAULT_VIDEO_SOURCE = "rtsp"
DEFAULT_TEST_VIDEO = os.path.join(BASE_DIR, "data", "videos", "accident", "v3.mp4")
TEST_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "videos", "test")
LEGACY_TEST_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "test")
os.makedirs(TEST_VIDEOS_DIR, exist_ok=True)


def get_available_test_videos() -> List[Dict[str, Any]]:
    """
    Discovers all available test video files in data/videos/test/ (and data/test/).
    Returns a sorted list of video metadata dictionaries.
    """
    video_exts = {".mp4", ".avi", ".mkv", ".mov"}
    found_videos: Dict[str, Dict[str, Any]] = {}

    search_dirs = [TEST_VIDEOS_DIR, LEGACY_TEST_VIDEOS_DIR]
    for directory in search_dirs:
        if os.path.isdir(directory):
            try:
                for entry in sorted(os.listdir(directory)):
                    full_path = os.path.join(directory, entry)
                    if os.path.isfile(full_path):
                        ext = os.path.splitext(entry)[1].lower()
                        if ext in video_exts and entry not in found_videos:
                            try:
                                size_bytes = os.path.getsize(full_path)
                            except Exception:
                                size_bytes = 0
                            found_videos[entry] = {
                                "id": entry,
                                "filename": entry,
                                "label": entry,
                                "path": full_path.replace("\\", "/"),
                                "size_mb": round(size_bytes / (1024 * 1024), 2),
                            }
            except Exception as e:
                print(f"[config] Warning scanning test videos directory {directory}: {e}")

    return sorted(list(found_videos.values()), key=lambda x: x["filename"].lower())

# Sparsh CCTV RTSP Configuration
RTSP_USERNAME = "admin"
RTSP_PASSWORD = "admin123"
RTSP_CAMERA_IP = "192.168.128.10"
RTSP_PORT = 554
RTSP_CHANNEL = 1
RTSP_STREAM = 1
SPARSH_RTSP_URL = (
    f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_CAMERA_IP}:{RTSP_PORT}/"
    f"avstream/channel={RTSP_CHANNEL}/stream={RTSP_STREAM}.sdp"
)

# Severity Configuration for Accident Detection
# BGR Colors for OpenCV
# DETR Class indices: 0: accident, 1: accident, 2: vehicle
SEVERITY_CONFIG: Dict[int, Dict[str, any]] = {
    0: {
        "raw_name": "accident",
        "display_name": "Traffic Accident",
        "short_name": "ACCIDENT",
        "color": (0, 40, 240),  # Bright Red
        "is_accident": True,
        "level": 3,
    },
    1: {
        "raw_name": "accident",
        "display_name": "Traffic Accident",
        "short_name": "ACCIDENT",
        "color": (0, 40, 240),  # Bright Red
        "is_accident": True,
        "level": 3,
    },
    2: {
        "raw_name": "vehicle",
        "display_name": "Normal Vehicle",
        "short_name": "NORMAL",
        "color": (0, 210, 60),  # Bright Green
        "is_accident": False,
        "level": 0,
    },
    3: {
        "raw_name": "accident_severe",
        "display_name": "Severe Accident",
        "short_name": "SEVERE",
        "color": (0, 40, 240),  # Bright Red
        "is_accident": True,
        "level": 3,
    },
    4: {
        "raw_name": "accident_100",
        "display_name": "Totaled Vehicle",
        "short_name": "TOTALED",
        "color": (160, 0, 210),  # Magenta
        "is_accident": True,
        "level": 4,
    },
}

# Fire & Hazard Detection Classes and Colors
FIRE_CLASSES = ["fire", "other", "smoke"]
FIRE_COLORS = {
    "fire": (0, 69, 255),  # Vibrant Red-Orange
    "smoke": (180, 180, 180),  # Light Gray
    "other": (150, 150, 150),  # Gray
}
FIRE_ALERT_COLOR = (0, 0, 255)  # Red
NORMAL_STATUS_COLOR = (0, 200, 0)  # Green

# Thresholds
DEFAULT_CRASH_CONF = 0.70
DEFAULT_CRASH_IOU = 0.65
DEFAULT_FIRE_CONF = 0.75

# Snapshot cooldown (seconds) to prevent flooding disk during continuous alerts
ALERT_SNAPSHOT_COOLDOWN_SEC = 5.0

# Camera Geolocation & Emergency Dispatch Configuration
DEFAULT_CAMERA_ID = "SPARSH-CAM-01"
DEFAULT_CAMERA_LOCATION = "H23V+F8X, Sector 25 Dwarka, Dwarka, Delhi, 110077"
DEFAULT_CAMERA_LATITUDE = 28.5537375
DEFAULT_CAMERA_LONGITUDE = 77.0433594
EMERGENCY_SEARCH_RADIUS_KM = 3.0
MAX_DISPATCH_FACILITIES_PER_TYPE = 2

# Dispatch Paths
DISPATCH_LOG_PATH = os.path.join(OUTPUT_DIR, "emergency_dispatch.json")
EMERGENCY_CACHE_PATH = os.path.join(BASE_DIR, "data", "emergency_facilities_cache.json")

# SMTP Email Dispatch Configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")
SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL", "vinayak.varshney7@gmail.com")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "")
SMTP_TIMEOUT_SECONDS = int(os.getenv("SMTP_TIMEOUT_SECONDS", 15))
EMAIL_ALERTS_ENABLED = os.getenv("SMTP_ENABLED", "true").lower() in ("true", "1", "yes")

# Designated Department Email Recipients (Configurable)
DEFAULT_RECIPIENT_EMAIL = os.getenv(
    "EMERGENCY_RECEIVER_EMAIL", "vinayak.varshney.btech2024@sitpune.edu.in"
)
DEPARTMENT_EMAILS = {
    "police": os.getenv("POLICE_DEPARTMENT_EMAIL", DEFAULT_RECIPIENT_EMAIL),
    "hospital": os.getenv("HOSPITAL_DEPARTMENT_EMAIL", DEFAULT_RECIPIENT_EMAIL),
    "fire": os.getenv("FIRE_DEPARTMENT_EMAIL", DEFAULT_RECIPIENT_EMAIL),
    "traffic_control": os.getenv("TRAFFIC_CONTROL_EMAIL", DEFAULT_RECIPIENT_EMAIL),
}

# Incident-to-Department Routing Matrix
INCIDENT_DEPARTMENT_ROUTING = {
    "accident": ["police", "hospital", "traffic_control"],
    "fire": ["fire", "hospital", "police"],
    "smoke": ["fire", "police"],
}

ATTACH_SNAPSHOT_IN_EMAIL = True

# Twilio SMS Alert Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
EMERGENCY_DISPATCH_PHONE = os.getenv("EMERGENCY_DISPATCH_PHONE", "")


import threading
from typing import Callable, Any, Optional


@dataclass
class SystemSettings:
    """Thread-safe, dynamically reconfigurable system settings model."""

    # Camera Stream & RTSP
    camera_id: str = DEFAULT_CAMERA_ID
    camera_location: str = DEFAULT_CAMERA_LOCATION
    camera_latitude: float = DEFAULT_CAMERA_LATITUDE
    camera_longitude: float = DEFAULT_CAMERA_LONGITUDE
    camera_fps: float = DEFAULT_CAMERA_FPS
    video_source: str = DEFAULT_VIDEO_SOURCE
    rtsp_username: str = RTSP_USERNAME
    rtsp_password: str = RTSP_PASSWORD
    rtsp_camera_ip: str = RTSP_CAMERA_IP
    rtsp_port: int = RTSP_PORT
    rtsp_channel: int = RTSP_CHANNEL
    rtsp_stream: int = RTSP_STREAM
    custom_rtsp_url: Optional[str] = None

    # Detector Thresholds
    detector_backend: str = ACCIDENT_DETECTOR_BACKEND
    crash_conf: float = DEFAULT_CRASH_CONF
    crash_iou: float = DEFAULT_CRASH_IOU
    fire_conf: float = DEFAULT_FIRE_CONF
    inference_device: str = "cpu"

    # Incident Logging & Dispatch
    alert_cooldown_sec: float = ALERT_SNAPSHOT_COOLDOWN_SEC
    emergency_search_radius_km: float = EMERGENCY_SEARCH_RADIUS_KM
    max_dispatch_facilities_per_type: int = MAX_DISPATCH_FACILITIES_PER_TYPE
    enable_dispatch: bool = True
    enable_console_alerts: bool = True
    webhook_url: Optional[str] = None

    # SMTP Email Alerts
    smtp_enabled: bool = EMAIL_ALERTS_ENABLED
    smtp_host: str = SMTP_HOST
    smtp_port: int = SMTP_PORT
    smtp_use_tls: bool = SMTP_USE_TLS
    smtp_sender_email: str = SMTP_SENDER_EMAIL
    smtp_app_password: str = SMTP_APP_PASSWORD
    smtp_timeout_seconds: int = SMTP_TIMEOUT_SECONDS
    default_recipient_email: str = DEFAULT_RECIPIENT_EMAIL
    department_emails: Dict[str, str] = field(
        default_factory=lambda: dict(DEPARTMENT_EMAILS)
    )
    incident_department_routing: Dict[str, List[str]] = field(
        default_factory=lambda: dict(INCIDENT_DEPARTMENT_ROUTING)
    )
    attach_snapshot_in_email: bool = ATTACH_SNAPSHOT_IN_EMAIL

    # Twilio SMS
    twilio_account_sid: str = TWILIO_ACCOUNT_SID
    twilio_auth_token: str = TWILIO_AUTH_TOKEN
    twilio_phone_number: str = TWILIO_PHONE_NUMBER
    emergency_dispatch_phone: str = EMERGENCY_DISPATCH_PHONE

    def __post_init__(self):
        try:
            import torch

            self.inference_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.inference_device = "cpu"

    @property
    def rtsp_url(self) -> str:
        if self.custom_rtsp_url:
            return self.custom_rtsp_url
        return (
            f"rtsp://{self.rtsp_username}:{self.rtsp_password}@{self.rtsp_camera_ip}:{self.rtsp_port}/"
            f"avstream/channel={self.rtsp_channel}/stream={self.rtsp_stream}.sdp"
        )

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        """Converts settings to a JSON-serializable dictionary."""
        data = {
            "camera": {
                "camera_id": self.camera_id,
                "camera_location": self.camera_location,
                "camera_latitude": self.camera_latitude,
                "camera_longitude": self.camera_longitude,
                "camera_fps": self.camera_fps,
                "video_source": self.video_source,
                "rtsp_username": self.rtsp_username,
                "rtsp_camera_ip": self.rtsp_camera_ip,
                "rtsp_port": self.rtsp_port,
                "rtsp_channel": self.rtsp_channel,
                "rtsp_stream": self.rtsp_stream,
                "custom_rtsp_url": self.custom_rtsp_url,
                "resolved_rtsp_url": self.rtsp_url,
            },
            "detectors": {
                "detector_backend": self.detector_backend,
                "crash_conf": self.crash_conf,
                "crash_iou": self.crash_iou,
                "fire_conf": self.fire_conf,
                "inference_device": self.inference_device,
            },
            "alerts": {
                "alert_cooldown_sec": self.alert_cooldown_sec,
                "emergency_search_radius_km": self.emergency_search_radius_km,
                "max_dispatch_facilities_per_type": self.max_dispatch_facilities_per_type,
                "enable_dispatch": self.enable_dispatch,
                "enable_console_alerts": self.enable_console_alerts,
                "webhook_url": self.webhook_url,
            },
            "email": {
                "smtp_enabled": self.smtp_enabled,
                "smtp_host": self.smtp_host,
                "smtp_port": self.smtp_port,
                "smtp_use_tls": self.smtp_use_tls,
                "smtp_sender_email": self.smtp_sender_email,
                "smtp_timeout_seconds": self.smtp_timeout_seconds,
                "default_recipient_email": self.default_recipient_email,
                "department_emails": self.department_emails,
                "incident_department_routing": self.incident_department_routing,
                "attach_snapshot_in_email": self.attach_snapshot_in_email,
            },
            "twilio": {
                "twilio_account_sid": self.twilio_account_sid
                if not mask_secrets
                else ("***" if self.twilio_account_sid else ""),
                "twilio_phone_number": self.twilio_phone_number,
                "emergency_dispatch_phone": self.emergency_dispatch_phone,
            },
        }
        if mask_secrets:
            data["camera"]["rtsp_password"] = "***" if self.rtsp_password else ""
            data["email"]["smtp_app_password"] = "***" if self.smtp_app_password else ""
            data["twilio"]["twilio_auth_token"] = (
                "***" if self.twilio_auth_token else ""
            )
        else:
            data["camera"]["rtsp_password"] = self.rtsp_password
            data["email"]["smtp_app_password"] = self.smtp_app_password
            data["twilio"]["twilio_auth_token"] = self.twilio_auth_token
        return data


_settings_lock = threading.RLock()
_settings_instance = SystemSettings()
_settings_listeners: List[Callable[[SystemSettings, Dict[str, Any]], None]] = []


def get_settings() -> SystemSettings:
    """Returns global SystemSettings instance."""
    with _settings_lock:
        return _settings_instance


def register_settings_listener(
    listener: Callable[[SystemSettings, Dict[str, Any]], None],
):
    """Registers callback invoked when dynamic settings are updated."""
    with _settings_lock:
        if listener not in _settings_listeners:
            _settings_listeners.append(listener)


def unregister_settings_listener(
    listener: Callable[[SystemSettings, Dict[str, Any]], None],
):
    """Removes registered settings update callback."""
    with _settings_lock:
        if listener in _settings_listeners:
            _settings_listeners.remove(listener)


def update_settings(updates: Dict[str, Any]) -> SystemSettings:
    """
    Dynamically updates system settings at runtime without requiring a restart.
    Propagates changes to legacy module globals and notifies all registered observers.
    """
    global \
        DEFAULT_CAMERA_ID, \
        DEFAULT_CAMERA_LOCATION, \
        DEFAULT_CAMERA_LATITUDE, \
        DEFAULT_CAMERA_LONGITUDE
    global \
        DEFAULT_CAMERA_FPS, \
        FRAME_TIME_MS, \
        DEFAULT_CRASH_CONF, \
        DEFAULT_CRASH_IOU, \
        DEFAULT_FIRE_CONF
    global \
        ALERT_SNAPSHOT_COOLDOWN_SEC, \
        EMERGENCY_SEARCH_RADIUS_KM, \
        MAX_DISPATCH_FACILITIES_PER_TYPE
    global \
        SMTP_HOST, \
        SMTP_PORT, \
        SMTP_USE_TLS, \
        SMTP_SENDER_EMAIL, \
        SMTP_APP_PASSWORD, \
        SMTP_TIMEOUT_SECONDS
    global \
        EMAIL_ALERTS_ENABLED, \
        DEFAULT_RECIPIENT_EMAIL, \
        DEPARTMENT_EMAILS, \
        INCIDENT_DEPARTMENT_ROUTING
    global \
        ATTACH_SNAPSHOT_IN_EMAIL, \
        SPARSH_RTSP_URL, \
        RTSP_USERNAME, \
        RTSP_PASSWORD, \
        RTSP_CAMERA_IP
    global RTSP_PORT, RTSP_CHANNEL, RTSP_STREAM, ACCIDENT_DETECTOR_BACKEND
    global \
        TWILIO_ACCOUNT_SID, \
        TWILIO_AUTH_TOKEN, \
        TWILIO_PHONE_NUMBER, \
        EMERGENCY_DISPATCH_PHONE

    with _settings_lock:
        s = _settings_instance
        # Flatten nested updates if structured dictionaries were passed
        flat_updates = {}
        for k, v in updates.items():
            if isinstance(v, dict) and k in (
                "camera",
                "detectors",
                "alerts",
                "email",
                "twilio",
            ):
                flat_updates.update(v)
            else:
                flat_updates[k] = v

        secret_fields = {"rtsp_password", "smtp_app_password", "twilio_auth_token"}
        nullable_fields = {"custom_rtsp_url", "webhook_url"}

        for key, val in flat_updates.items():
            if not hasattr(s, key):
                continue

            # Protect against overwriting real credentials with mask placeholder
            if key in secret_fields and val == "***":
                continue

            # Handle nullable fields explicitly
            if key in nullable_fields and (val is None or val == ""):
                setattr(s, key, None)
                continue

            if val is not None:
                orig = getattr(s, key)
                # Merge dictionary updates rather than wiping sibling keys
                if isinstance(orig, dict) and isinstance(val, dict):
                    merged = dict(orig)
                    merged.update(val)
                    val = merged
                elif orig is not None:
                    target_type = type(orig)
                    try:
                        if target_type == bool and isinstance(val, str):
                            val = val.lower() in ("true", "1", "yes")
                        elif target_type in (int, float, str):
                            val = target_type(val)
                    except ValueError, TypeError:
                        pass
                setattr(s, key, val)

        # Automatic coordinate resolution if location changed without explicit coordinates
        if "camera_location" in flat_updates and "camera_latitude" not in flat_updates:
            try:
                from services.geo_service import GeoService

                geo = GeoService()
                loc = geo.resolve_location(s.camera_location)
                s.camera_latitude = loc.latitude
                s.camera_longitude = loc.longitude
                flat_updates["camera_latitude"] = loc.latitude
                flat_updates["camera_longitude"] = loc.longitude
            except Exception as e:
                print(
                    f"[Config] Warning: Could not auto-resolve coordinates for {s.camera_location}: {e}"
                )

        # Synchronize module globals
        DEFAULT_CAMERA_ID = s.camera_id
        DEFAULT_CAMERA_LOCATION = s.camera_location
        DEFAULT_CAMERA_LATITUDE = s.camera_latitude
        DEFAULT_CAMERA_LONGITUDE = s.camera_longitude
        DEFAULT_CAMERA_FPS = s.camera_fps
        FRAME_TIME_MS = 1000.0 / s.camera_fps if s.camera_fps > 0 else 66.66
        DEFAULT_CRASH_CONF = s.crash_conf
        DEFAULT_CRASH_IOU = s.crash_iou
        DEFAULT_FIRE_CONF = s.fire_conf
        ALERT_SNAPSHOT_COOLDOWN_SEC = s.alert_cooldown_sec
        EMERGENCY_SEARCH_RADIUS_KM = s.emergency_search_radius_km
        MAX_DISPATCH_FACILITIES_PER_TYPE = s.max_dispatch_facilities_per_type
        SMTP_HOST = s.smtp_host
        SMTP_PORT = s.smtp_port
        SMTP_USE_TLS = s.smtp_use_tls
        SMTP_SENDER_EMAIL = s.smtp_sender_email
        SMTP_APP_PASSWORD = s.smtp_app_password
        SMTP_TIMEOUT_SECONDS = s.smtp_timeout_seconds
        EMAIL_ALERTS_ENABLED = s.smtp_enabled
        DEFAULT_RECIPIENT_EMAIL = s.default_recipient_email
        DEPARTMENT_EMAILS = s.department_emails
        INCIDENT_DEPARTMENT_ROUTING = s.incident_department_routing
        ATTACH_SNAPSHOT_IN_EMAIL = s.attach_snapshot_in_email
        RTSP_USERNAME = s.rtsp_username
        RTSP_PASSWORD = s.rtsp_password
        RTSP_CAMERA_IP = s.rtsp_camera_ip
        RTSP_PORT = s.rtsp_port
        RTSP_CHANNEL = s.rtsp_channel
        RTSP_STREAM = s.rtsp_stream
        SPARSH_RTSP_URL = s.rtsp_url
        ACCIDENT_DETECTOR_BACKEND = s.detector_backend
        TWILIO_ACCOUNT_SID = s.twilio_account_sid
        TWILIO_AUTH_TOKEN = s.twilio_auth_token
        TWILIO_PHONE_NUMBER = s.twilio_phone_number
        EMERGENCY_DISPATCH_PHONE = s.emergency_dispatch_phone

        # Notify listeners
        for listener in list(_settings_listeners):
            try:
                listener(s, flat_updates)
            except Exception as e:
                print(f"[Config] Error notifying settings listener {listener}: {e}")

        return s


def reset_settings_to_defaults() -> SystemSettings:
    """Resets system settings instance and module globals back to default values."""
    default_s = SystemSettings()
    return update_settings(
        {
            "camera_id": default_s.camera_id,
            "camera_location": default_s.camera_location,
            "camera_latitude": default_s.camera_latitude,
            "camera_longitude": default_s.camera_longitude,
            "camera_fps": default_s.camera_fps,
            "video_source": default_s.video_source,
            "rtsp_username": default_s.rtsp_username,
            "rtsp_password": default_s.rtsp_password,
            "rtsp_camera_ip": default_s.rtsp_camera_ip,
            "rtsp_port": default_s.rtsp_port,
            "rtsp_channel": default_s.rtsp_channel,
            "rtsp_stream": default_s.rtsp_stream,
            "custom_rtsp_url": None,
            "detector_backend": default_s.detector_backend,
            "crash_conf": default_s.crash_conf,
            "crash_iou": default_s.crash_iou,
            "fire_conf": default_s.fire_conf,
            "inference_device": default_s.inference_device,
            "alert_cooldown_sec": default_s.alert_cooldown_sec,
            "emergency_search_radius_km": default_s.emergency_search_radius_km,
            "max_dispatch_facilities_per_type": default_s.max_dispatch_facilities_per_type,
            "enable_dispatch": default_s.enable_dispatch,
            "enable_console_alerts": default_s.enable_console_alerts,
            "webhook_url": None,
            "smtp_enabled": default_s.smtp_enabled,
            "smtp_host": default_s.smtp_host,
            "smtp_port": default_s.smtp_port,
            "smtp_use_tls": default_s.smtp_use_tls,
            "smtp_sender_email": default_s.smtp_sender_email,
            "smtp_app_password": default_s.smtp_app_password,
            "smtp_timeout_seconds": default_s.smtp_timeout_seconds,
            "default_recipient_email": default_s.default_recipient_email,
            "department_emails": dict(DEPARTMENT_EMAILS),
            "incident_department_routing": dict(INCIDENT_DEPARTMENT_ROUTING),
            "attach_snapshot_in_email": default_s.attach_snapshot_in_email,
            "twilio_account_sid": default_s.twilio_account_sid,
            "twilio_auth_token": default_s.twilio_auth_token,
            "twilio_phone_number": default_s.twilio_phone_number,
            "emergency_dispatch_phone": default_s.emergency_dispatch_phone,
        }
    )
