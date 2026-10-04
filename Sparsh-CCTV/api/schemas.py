from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


# ==========================================
# Dynamic Settings Schemas
# ==========================================


class CameraSettingsUpdate(BaseModel):
    camera_id: Optional[str] = Field(None, description="Camera identifier string")
    camera_location: Optional[str] = Field(
        None, description="Camera address or Plus Code"
    )
    camera_latitude: Optional[float] = Field(None, description="Latitude coordinate")
    camera_longitude: Optional[float] = Field(None, description="Longitude coordinate")
    camera_fps: Optional[float] = Field(
        None, ge=1.0, le=60.0, description="Camera FPS cap"
    )
    video_source: Optional[str] = Field(
        None, description="Path to video, webcam index, or rtsp"
    )
    rtsp_username: Optional[str] = Field(
        None, description="RTSP HTTP Digest / Basic Auth username"
    )
    rtsp_password: Optional[str] = Field(None, description="RTSP password")
    rtsp_camera_ip: Optional[str] = Field(None, description="RTSP camera host IP")
    rtsp_port: Optional[int] = Field(
        None, description="RTSP streaming port (default 554)"
    )
    rtsp_channel: Optional[int] = Field(None, description="Camera channel number")
    rtsp_stream: Optional[int] = Field(
        None, description="Camera stream profile (1=main, 2=sub)"
    )
    custom_rtsp_url: Optional[str] = Field(None, description="Override full RTSP URL")


class DetectorsSettingsUpdate(BaseModel):
    detector_backend: Optional[str] = Field(
        None, description="Accident detector backend: 'yolo' or 'detr'"
    )
    crash_conf: Optional[float] = Field(
        None, ge=0.01, le=1.0, description="Confidence threshold for crash detector"
    )
    crash_iou: Optional[float] = Field(
        None, ge=0.01, le=1.0, description="NMS IoU threshold for crash detector"
    )
    fire_conf: Optional[float] = Field(
        None, ge=0.01, le=1.0, description="Confidence threshold for fire detector"
    )
    inference_device: Optional[str] = Field(
        None, description="Execution device: 'cuda', 'cuda:0', or 'cpu'"
    )


class AlertsSettingsUpdate(BaseModel):
    alert_cooldown_sec: Optional[float] = Field(
        None, ge=0.1, le=3600.0, description="Cooldown between snapshots in seconds"
    )
    emergency_search_radius_km: Optional[float] = Field(
        None, ge=0.5, le=50.0, description="Emergency responders radius in km"
    )
    max_dispatch_facilities_per_type: Optional[int] = Field(
        None, ge=1, le=10, description="Max facilities to notify per category"
    )
    enable_dispatch: Optional[bool] = Field(
        None, description="Toggle emergency dispatch router"
    )
    enable_console_alerts: Optional[bool] = Field(
        None, description="Toggle console alert banner"
    )
    webhook_url: Optional[str] = Field(
        None, description="Optional HTTP webhook URL for alerts"
    )


class EmailSettingsUpdate(BaseModel):
    smtp_enabled: Optional[bool] = Field(
        None, description="Enable automated SMTP emergency emails"
    )
    smtp_host: Optional[str] = Field(None, description="SMTP server hostname")
    smtp_port: Optional[int] = Field(None, description="SMTP server port")
    smtp_use_tls: Optional[bool] = Field(None, description="Enable STARTTLS encryption")
    smtp_sender_email: Optional[str] = Field(None, description="Sender email address")
    smtp_app_password: Optional[str] = Field(
        None, description="Sender email password / app password"
    )
    smtp_timeout_seconds: Optional[int] = Field(
        None, ge=1, le=120, description="SMTP timeout"
    )
    default_recipient_email: Optional[str] = Field(
        None, description="Default recipient email"
    )
    department_emails: Optional[Dict[str, str]] = Field(
        None, description="Mapping of department name to email"
    )
    incident_department_routing: Optional[Dict[str, List[str]]] = Field(
        None, description="Incident to department routing matrix"
    )
    attach_snapshot_in_email: Optional[bool] = Field(
        None, description="Include inline JPEG snapshot in email"
    )


class TwilioSettingsUpdate(BaseModel):
    twilio_account_sid: Optional[str] = Field(None, description="Twilio Account SID")
    twilio_auth_token: Optional[str] = Field(None, description="Twilio Auth Token")
    twilio_phone_number: Optional[str] = Field(
        None, description="Twilio sender phone number"
    )
    emergency_dispatch_phone: Optional[str] = Field(
        None, description="Emergency dispatch receiver phone"
    )


class SystemSettingsUpdate(BaseModel):
    camera: Optional[CameraSettingsUpdate] = None
    detectors: Optional[DetectorsSettingsUpdate] = None
    alerts: Optional[AlertsSettingsUpdate] = None
    email: Optional[EmailSettingsUpdate] = None
    twilio: Optional[TwilioSettingsUpdate] = None

    # Support flat properties as well
    camera_id: Optional[str] = None
    camera_location: Optional[str] = None
    camera_latitude: Optional[float] = None
    camera_longitude: Optional[float] = None
    camera_fps: Optional[float] = None
    video_source: Optional[str] = None
    rtsp_username: Optional[str] = None
    rtsp_password: Optional[str] = None
    rtsp_camera_ip: Optional[str] = None
    rtsp_port: Optional[int] = None
    rtsp_channel: Optional[int] = None
    rtsp_stream: Optional[int] = None
    custom_rtsp_url: Optional[str] = None
    detector_backend: Optional[str] = None
    crash_conf: Optional[float] = None
    crash_iou: Optional[float] = None
    fire_conf: Optional[float] = None
    inference_device: Optional[str] = None
    alert_cooldown_sec: Optional[float] = None
    emergency_search_radius_km: Optional[float] = None
    max_dispatch_facilities_per_type: Optional[int] = None
    enable_dispatch: Optional[bool] = None
    enable_console_alerts: Optional[bool] = None
    webhook_url: Optional[str] = None
    smtp_enabled: Optional[bool] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_use_tls: Optional[bool] = None
    smtp_sender_email: Optional[str] = None
    smtp_app_password: Optional[str] = None
    smtp_timeout_seconds: Optional[int] = None
    default_recipient_email: Optional[str] = None
    department_emails: Optional[Dict[str, str]] = None
    incident_department_routing: Optional[Dict[str, List[str]]] = None
    attach_snapshot_in_email: Optional[bool] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    emergency_dispatch_phone: Optional[str] = None


# ==========================================
# Surveillance Operation Schemas
# ==========================================


class SurveillanceStartRequest(BaseModel):
    source: Optional[str] = Field(
        None, description="Video file path, RTSP URL, or webcam index"
    )
    fps_cap: Optional[float] = Field(
        None, ge=1.0, le=60.0, description="Stream FPS cap"
    )
    save_recording: Optional[bool] = Field(
        False, description="Record annotated stream to MP4"
    )
    recording_path: Optional[str] = Field(None, description="Custom output MP4 path")


class SwitchSourceRequest(BaseModel):
    source: str = Field(
        ...,
        description="Video source identifier: 'rtsp'/'live' for the live RTSP stream, or a test video filename (e.g. 'bikeacc.mp4') or path.",
    )


class SnapshotRequest(BaseModel):
    tag: str = Field("manual", description="Prefix tag for the snapshot filename")
    dispatch_alert: bool = Field(
        False, description="Dispatch emergency alert with this snapshot"
    )


class TriggerAlertRequest(BaseModel):
    incident_type: str = Field(
        "ACCIDENT_LVL_3", description="Event category e.g. ACCIDENT_LVL_3, FIRE_ALERT"
    )
    severity_label: str = Field(
        "Severe Accident", description="Human-readable severity title"
    )
    confidence: float = Field(
        0.95, ge=0.0, le=1.0, description="Detection confidence score"
    )
    dispatch_emergency: bool = Field(
        True, description="Trigger notification dispatch across active channels"
    )


class TriggerAccidentAlertRequest(BaseModel):
    severity_label: Optional[str] = Field(
        "Severe Accident (Manual Dispatch)", description="Accident severity label"
    )
    confidence: Optional[float] = Field(
        0.98, ge=0.0, le=1.0, description="Detection confidence score"
    )
    dispatch_emergency: Optional[bool] = Field(
        True, description="Trigger notification dispatch to Police, Hospitals, and Traffic Control"
    )


class TriggerFireAlertRequest(BaseModel):
    severity_label: Optional[str] = Field(
        "Active Fire Hazard (Manual Dispatch)", description="Fire hazard label"
    )
    confidence: Optional[float] = Field(
        0.99, ge=0.0, le=1.0, description="Detection confidence score"
    )
    dispatch_emergency: Optional[bool] = Field(
        True, description="Trigger notification dispatch to Fire Dept, Hospitals, and Police"
    )


class GeocodeRequest(BaseModel):
    location_query: str = Field(
        ..., description="Address or Plus Code string to geocode"
    )


class TestEmailRequest(BaseModel):
    recipient: Optional[str] = Field(None, description="Test email recipient address")
    subject: Optional[str] = Field(
        "Sparsh CCTV Surveillance Test Alert", description="Email subject"
    )


class TestSmsRequest(BaseModel):
    to_phone: Optional[str] = Field(None, description="Recipient phone number")
    message: Optional[str] = Field(
        "Sparsh CCTV Test Emergency SMS", description="SMS message body"
    )
