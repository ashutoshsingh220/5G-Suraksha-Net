from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body

from api.dependencies import get_engine, get_sys_settings
from api.schemas import GeocodeRequest, TestEmailRequest, TestSmsRequest
from config import SystemSettings
from surveillance_engine import SurveillanceEngine
from services.geo_service import GeoService

router = APIRouter(prefix="/api/emergency", tags=["Emergency Services & Geolocation"])


@router.get("/location", summary="Get resolved camera location & coordinates")
def get_camera_location(
    engine: SurveillanceEngine = Depends(get_engine),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Returns geocoded coordinates, formatted address, and Google Maps search link."""
    if engine.dispatcher and getattr(engine.dispatcher, "location", None):
        loc = engine.dispatcher.location
        return {
            "camera_id": engine.dispatcher.camera_id,
            "location_string": engine.dispatcher.camera_location_str,
            "latitude": round(loc.latitude, 7),
            "longitude": round(loc.longitude, 7),
            "formatted_address": loc.formatted_address,
            "maps_url": loc.maps_url,
            "geocoding_source": loc.source,
        }
    # Fallback to configured settings coordinates directly so frontend never stalls
    return {
        "camera_id": settings.camera_id,
        "location_string": settings.camera_location,
        "latitude": round(settings.camera_latitude, 7),
        "longitude": round(settings.camera_longitude, 7),
        "formatted_address": settings.camera_location,
        "maps_url": f"https://www.google.com/maps/search/?api=1&query={settings.camera_latitude}%2C{settings.camera_longitude}",
        "geocoding_source": "settings_default",
    }


@router.get("/facilities", summary="List nearby emergency responder facilities")
def get_nearby_facilities(
    limit: int = Query(5, ge=1, le=20, description="Max facilities per category"),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """Returns hospitals, police stations, and fire stations within the configured emergency radius."""
    if engine.dispatcher and engine.dispatcher.registry:
        reg = engine.dispatcher.registry
        hospitals = [f.to_dict() for f in reg.get_nearest_hospitals(limit=limit)]
        police = [f.to_dict() for f in reg.get_nearest_police(limit=limit)]
        fire = [f.to_dict() for f in reg.get_nearest_fire_stations(limit=limit)]

        return {
            "search_radius_km": reg.max_radius_km,
            "total_facilities": len(reg.facilities),
            "hospitals": hospitals,
            "police_stations": police,
            "fire_stations": fire,
        }
    return {
        "search_radius_km": 0,
        "total_facilities": 0,
        "hospitals": [],
        "police_stations": [],
        "fire_stations": [],
    }


@router.post("/resolve-location", summary="Test geocode any address or Plus Code")
def resolve_arbitrary_location(req: GeocodeRequest):
    """Geocodes an address or Plus Code without altering camera configuration."""
    geo = GeoService()
    loc = geo.resolve_location(req.location_query)
    return {
        "query": req.location_query,
        "latitude": round(loc.latitude, 7),
        "longitude": round(loc.longitude, 7),
        "formatted_address": loc.formatted_address,
        "maps_url": loc.maps_url,
        "source": loc.source,
    }


@router.post("/test-email", summary="Send a test emergency notification email")
def send_test_email(
    req: Optional[TestEmailRequest] = Body(None),
    engine: SurveillanceEngine = Depends(get_engine),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Sends a test alert email through the active SMTP service."""
    if not engine.email_service:
        raise HTTPException(status_code=400, detail="Email service not initialized")

    req_data = req or TestEmailRequest()
    test_payload = {
        "dispatch_id": "DISPATCH-TEST-VERIFY",
        "timestamp": "2026-10-03 12:00:00",
        "camera": {
            "camera_id": settings.camera_id,
            "location_string": settings.camera_location,
            "coordinates": {
                "latitude": settings.camera_latitude,
                "longitude": settings.camera_longitude,
            },
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={settings.camera_latitude},{settings.camera_longitude}",
        },
        "incident": {
            "type": "TEST_ALERT",
            "severity_label": req_data.subject or "Test Surveillance Alert",
            "confidence": 1.0,
            "vehicles_involved": 1,
            "snapshot_path": "N/A",
        },
        "dispatched_services": {
            "hospitals": [
                {"name": "Test Hospital", "distance_km": 1.2, "phone": "102"}
            ],
            "police_stations": [
                {"name": "Test Police", "distance_km": 1.5, "phone": "112"}
            ],
        },
    }

    recipients = [req_data.recipient] if req_data.recipient else None
    res = engine.email_service.send_incident_alert(
        test_payload, override_recipients=recipients
    )
    return res


@router.post("/test-sms", summary="Send a test emergency notification SMS")
def send_test_sms(
    req: Optional[TestSmsRequest] = Body(None),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """Sends or simulates a test emergency alert SMS via Twilio."""
    if not engine.dispatcher:
        raise HTTPException(status_code=400, detail="Dispatcher not initialized")

    req_data = req or TestSmsRequest()
    test_payload = {
        "incident": {
            "type": "TEST_SMS",
            "severity_label": req_data.message or "Test SMS Alert",
        },
        "camera": {
            "location_string": engine.dispatcher.camera_location_str,
            "maps_url": engine.dispatcher.location.maps_url
            if engine.dispatcher.location
            else "",
        },
    }
    routed = (
        engine.dispatcher.registry.get_facilities_for_incident("TEST")
        if engine.dispatcher.registry
        else {}
    )
    res = engine.dispatcher._send_twilio_alert(
        payload=test_payload,
        routed=routed,
        to_override=req_data.to_phone,
        message_override=req_data.message,
    )
    return {"status": "SUCCESS", "twilio_response": res}
