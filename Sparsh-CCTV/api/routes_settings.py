from typing import Dict, Any
from fastapi import APIRouter, Depends, Query, Body

from api.dependencies import get_sys_settings
from api.schemas import (
    SystemSettingsUpdate,
    CameraSettingsUpdate,
    DetectorsSettingsUpdate,
    AlertsSettingsUpdate,
    EmailSettingsUpdate,
    TwilioSettingsUpdate,
)
from config import update_settings, SystemSettings

router = APIRouter(prefix="/api/settings", tags=["Dynamic System Settings"])


@router.get("", summary="Get all current system settings")
def get_all_settings(
    mask_secrets: bool = Query(
        False, description="Mask sensitive credentials (passwords, tokens)"
    ),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Returns the comprehensive real-time system configuration dictionary."""
    return settings.to_dict(mask_secrets=mask_secrets)


@router.patch("", summary="Dynamically update system settings")
@router.put("", summary="Dynamically update system settings")
def update_all_settings(
    updates: SystemSettingsUpdate, settings: SystemSettings = Depends(get_sys_settings)
):
    """
    Dynamically reconfigures system settings at runtime without requiring process restart.
    Propagates updates to active detectors, stream capture, visualizers, and notification routers.
    """
    dumped = updates.model_dump(exclude_unset=True)
    new_settings = update_settings(dumped)
    return {
        "status": "SUCCESS",
        "message": "System settings dynamically updated at runtime",
        "settings": new_settings.to_dict(mask_secrets=True),
    }


# ==========================================
# Granular Settings Sub-Routes
# ==========================================


@router.get("/camera", summary="Get camera & stream configuration")
def get_camera_settings(
    mask_secrets: bool = Query(False, description="Mask RTSP password"),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Returns camera identification, RTSP credentials, location, and stream configuration."""
    return settings.to_dict(mask_secrets=mask_secrets)["camera"]


@router.patch("/camera", summary="Dynamically update camera & stream configuration")
def update_camera_settings(
    updates: CameraSettingsUpdate, settings: SystemSettings = Depends(get_sys_settings)
):
    """Dynamically updates camera ID, RTSP parameters, or geolocation string."""
    dumped = updates.model_dump(exclude_unset=True)
    new_settings = update_settings({"camera": dumped})
    return {
        "status": "SUCCESS",
        "message": "Camera settings updated dynamically",
        "camera": new_settings.to_dict(mask_secrets=True)["camera"],
    }


@router.get("/detectors", summary="Get detector thresholds & backend")
def get_detector_settings(settings: SystemSettings = Depends(get_sys_settings)):
    """Returns accident and fire confidence thresholds, IoU parameters, and backend device."""
    return settings.to_dict()["detectors"]


@router.patch("/detectors", summary="Dynamically update detector thresholds")
def update_detector_settings(
    updates: DetectorsSettingsUpdate,
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Dynamically modifies detection sensitivity and thresholds without pipeline restart."""
    dumped = updates.model_dump(exclude_unset=True)
    new_settings = update_settings({"detectors": dumped})
    return {
        "status": "SUCCESS",
        "message": "Detector thresholds updated dynamically",
        "detectors": new_settings.to_dict()["detectors"],
    }


@router.get("/alerts", summary="Get incident alert & dispatch settings")
def get_alert_settings(settings: SystemSettings = Depends(get_sys_settings)):
    """Returns cooldown interval, 3km emergency radius, and dispatch configuration."""
    return settings.to_dict()["alerts"]


@router.patch("/alerts", summary="Dynamically update incident alert settings")
def update_alert_settings(
    updates: AlertsSettingsUpdate, settings: SystemSettings = Depends(get_sys_settings)
):
    """Dynamically updates snapshot cooldown, search radius, or webhook destination."""
    dumped = updates.model_dump(exclude_unset=True)
    new_settings = update_settings({"alerts": dumped})
    return {
        "status": "SUCCESS",
        "message": "Alert settings updated dynamically",
        "alerts": new_settings.to_dict()["alerts"],
    }


@router.get("/email", summary="Get SMTP & department email recipient settings")
def get_email_settings(
    mask_secrets: bool = Query(False, description="Mask SMTP password"),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Returns SMTP credentials, sender email, and department recipient routing."""
    return settings.to_dict(mask_secrets=mask_secrets)["email"]


@router.patch("/email", summary="Dynamically update SMTP & recipient settings")
def update_email_settings(
    updates: EmailSettingsUpdate, settings: SystemSettings = Depends(get_sys_settings)
):
    """Dynamically updates SMTP host/port/credentials or department recipient emails."""
    dumped = updates.model_dump(exclude_unset=True)
    new_settings = update_settings({"email": dumped})
    return {
        "status": "SUCCESS",
        "message": "Email settings updated dynamically",
        "email": new_settings.to_dict(mask_secrets=True)["email"],
    }


@router.get("/twilio", summary="Get Twilio SMS notification settings")
def get_twilio_settings(
    mask_secrets: bool = Query(False, description="Mask Twilio auth token"),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Returns Twilio SMS credentials and emergency recipient phone number."""
    return settings.to_dict(mask_secrets=mask_secrets)["twilio"]


@router.patch("/twilio", summary="Dynamically update Twilio SMS settings")
def update_twilio_settings(
    updates: TwilioSettingsUpdate, settings: SystemSettings = Depends(get_sys_settings)
):
    """Dynamically updates Twilio Account SID, Auth Token, or recipient phone."""
    dumped = updates.model_dump(exclude_unset=True)
    new_settings = update_settings({"twilio": dumped})
    return {
        "status": "SUCCESS",
        "message": "Twilio settings updated dynamically",
        "twilio": new_settings.to_dict(mask_secrets=True)["twilio"],
    }
