import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, Body
from fastapi.responses import StreamingResponse

from api.dependencies import get_engine
from api.schemas import (
    SurveillanceStartRequest,
    SwitchSourceRequest,
    SnapshotRequest,
    TriggerAlertRequest,
    TriggerAccidentAlertRequest,
    TriggerFireAlertRequest,
)
from config import (
    get_available_test_videos,
    TEST_VIDEOS_DIR,
    LEGACY_TEST_VIDEOS_DIR,
    get_settings,
    update_settings,
)
from surveillance_engine import SurveillanceEngine

router = APIRouter(prefix="/api/surveillance", tags=["Surveillance Operations"])


@router.post("/start", summary="Start video surveillance pipeline")
def start_surveillance(
    req: Optional[SurveillanceStartRequest] = Body(None),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """Starts the real-time AI surveillance processing loop in the background."""
    req_data = req or SurveillanceStartRequest()
    try:
        status = engine.start(
            source=req_data.source,
            fps_cap=req_data.fps_cap,
            save_recording=req_data.save_recording or False,
            recording_path=req_data.recording_path,
        )
        return {
            "status": "SUCCESS",
            "message": "Surveillance stream started",
            "details": status,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop", summary="Stop video surveillance pipeline")
def stop_surveillance(engine: SurveillanceEngine = Depends(get_engine)):
    """Halts surveillance capture and frees camera/stream resources."""
    status = engine.stop()
    return {
        "status": "SUCCESS",
        "message": "Surveillance stream stopped",
        "details": status,
    }


@router.post("/pause", summary="Pause video surveillance pipeline")
def pause_surveillance(engine: SurveillanceEngine = Depends(get_engine)):
    """Temporarily pauses frame ingestion and detection inference."""
    status = engine.pause()
    return {
        "status": "SUCCESS",
        "message": "Surveillance playback paused",
        "details": status,
    }


@router.post("/resume", summary="Resume video surveillance pipeline")
def resume_surveillance(engine: SurveillanceEngine = Depends(get_engine)):
    """Resumes paused surveillance processing."""
    status = engine.resume()
    return {
        "status": "SUCCESS",
        "message": "Surveillance playback resumed",
        "details": status,
    }


@router.get("/available-sources", summary="List available video stream sources")
def get_available_sources(engine: SurveillanceEngine = Depends(get_engine)):
    """
    Returns available video feeds (Sparsh RTSP live camera stream and test videos in data/videos/test/).
    """
    settings = get_settings()
    current_src = engine.current_source or settings.video_source
    is_rtsp = (
        current_src == "rtsp"
        or current_src == settings.rtsp_url
        or (isinstance(current_src, str) and current_src.startswith("rtsp://"))
    )
    current_type = "rtsp" if is_rtsp else "test_video"
    current_label = engine.source_label or (
        "Sparsh CCTV (Live RTSP)" if is_rtsp else os.path.basename(str(current_src))
    )

    test_videos = get_available_test_videos()

    return {
        "status": "SUCCESS",
        "current_source": str(current_src),
        "current_source_type": current_type,
        "current_source_label": current_label,
        "is_rtsp": is_rtsp,
        "stream_connected": engine.stream_connected,
        "stream_status": engine.stream_status,
        "rtsp": {
            "id": "rtsp",
            "label": f"Sparsh CCTV ({settings.rtsp_camera_ip})",
            "url": settings.rtsp_url,
            "camera_ip": settings.rtsp_camera_ip,
        },
        "test_videos": test_videos,
    }


@router.post("/switch-source", summary="Switch surveillance video feed source")
def switch_video_source(
    req: SwitchSourceRequest,
    engine: SurveillanceEngine = Depends(get_engine),
):
    """
    Hot-switches the surveillance stream between the live RTSP feed and test video files without restarting.
    """
    source_str = req.source.strip()
    if not source_str:
        raise HTTPException(status_code=400, detail="Video source cannot be empty")

    settings = get_settings()
    test_videos = get_available_test_videos()
    test_video_map = {v["filename"].lower(): v for v in test_videos}

    if source_str.lower() in ("rtsp", "sparsh", "live", "camera"):
        target_source = "rtsp"
        source_type = "rtsp"
        target_label = "Sparsh CCTV (Live RTSP)"
        update_settings({"video_source": "rtsp"})
    elif source_str.lower() in test_video_map:
        matched = test_video_map[source_str.lower()]
        target_source = matched["path"]
        source_type = "test_video"
        target_label = matched["filename"]
        update_settings({"video_source": target_source})
    elif os.path.isfile(source_str):
        target_source = os.path.abspath(source_str).replace("\\", "/")
        source_type = "test_video"
        target_label = os.path.basename(target_source)
        update_settings({"video_source": target_source})
    elif os.path.isfile(os.path.join(TEST_VIDEOS_DIR, source_str)):
        target_source = os.path.abspath(
            os.path.join(TEST_VIDEOS_DIR, source_str)
        ).replace("\\", "/")
        source_type = "test_video"
        target_label = os.path.basename(target_source)
        update_settings({"video_source": target_source})
    elif os.path.isfile(os.path.join(LEGACY_TEST_VIDEOS_DIR, source_str)):
        target_source = os.path.abspath(
            os.path.join(LEGACY_TEST_VIDEOS_DIR, source_str)
        ).replace("\\", "/")
        source_type = "test_video"
        target_label = os.path.basename(target_source)
        update_settings({"video_source": target_source})
    else:
        available_names = [v["filename"] for v in test_videos]
        raise HTTPException(
            status_code=400,
            detail=f"Video source '{source_str}' not found. Please specify 'rtsp' or one of: {available_names}",
        )

    try:
        status = engine.start(source=target_source)
        if source_type == "rtsp" and not status.get("stream_connected", False):
            return {
                "status": "WARNING",
                "message": f"Switched to RTSP mode. No streaming device detected at {target_source} - displaying 'NO LIVE STREAM RUNNING' slate.",
                "source": target_source,
                "source_type": source_type,
                "source_label": target_label,
                "stream_connected": False,
                "stream_status": status.get("stream_status", "NO_STREAM"),
                "details": status,
            }
        return {
            "status": "SUCCESS",
            "message": f"Successfully switched video source to '{target_label}'",
            "source": target_source,
            "source_type": source_type,
            "source_label": target_label,
            "stream_connected": status.get("stream_connected", True),
            "stream_status": status.get("stream_status", "ONLINE"),
            "details": status,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to switch video source: {e}"
        )


@router.get("/status", summary="Get real-time surveillance status")
def get_surveillance_status(engine: SurveillanceEngine = Depends(get_engine)):
    """Returns telemetry including FPS, uptime, active severity, vehicle counts, and hazard flags."""
    return engine.get_status()


@router.get("/stream", summary="Live MJPEG video stream")
def live_stream(
    annotated: bool = Query(
        True, description="Render HUD and bounding boxes on stream"
    ),
    fps: float = Query(
        15.0, ge=1.0, le=30.0, description="Target streaming frame rate"
    ),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """
    Continuous multipart MJPEG video stream for HTML <img src="..."> integration and dashboard preview.
    """
    return StreamingResponse(
        engine.generate_mjpeg(annotated=annotated, fps_limit=fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/frame", summary="Get current video frame as JPEG")
def get_current_frame(
    annotated: bool = Query(True, description="Render HUD and bounding boxes on frame"),
    quality: int = Query(85, ge=10, le=100, description="JPEG compression quality"),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """Captures and returns the latest available frame as a JPEG image."""
    try:
        jpeg_bytes = engine.get_latest_jpeg(annotated=annotated, quality=quality)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture frame: {e}")


@router.get("/detections", summary="Get active detections JSON")
def get_current_detections(engine: SurveillanceEngine = Depends(get_engine)):
    """Returns structured object detection and fire hazard telemetry for the latest frame."""
    return engine.get_latest_detections_payload()


@router.post("/snapshot", summary="Save manual incident snapshot")
def take_snapshot(
    req: Optional[SnapshotRequest] = Body(None),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """Saves a timestamped JPEG image snapshot of the current surveillance feed to disk."""
    req_data = req or SnapshotRequest()
    try:
        return engine.capture_snapshot(
            tag=req_data.tag, dispatch_alert=req_data.dispatch_alert
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save snapshot: {e}")


@router.post("/trigger-alert", summary="Manually trigger emergency alert")
def trigger_alert(
    req: TriggerAlertRequest, engine: SurveillanceEngine = Depends(get_engine)
):
    """Manually triggers an incident alert and dispatches to emergency services."""
    try:
        res = engine.capture_snapshot(
            tag="manual_alert",
            dispatch_alert=req.dispatch_emergency,
            incident_type=req.incident_type,
            severity_label=req.severity_label,
        )
        return {
            "status": "SUCCESS",
            "message": f"Alert {req.incident_type} triggered successfully",
            "snapshot": res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert dispatch failed: {e}")


@router.post("/trigger-accident-alert", summary="Manually trigger vehicle accident alert")
def trigger_accident_alert(
    req: Optional[TriggerAccidentAlertRequest] = Body(None),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """
    Manually triggers an accident emergency alert (ACCIDENT_LVL_3) and dispatches
    notifications to Police, Hospitals, and Traffic Control departments.
    """
    req_data = req or TriggerAccidentAlertRequest()
    severity_label = req_data.severity_label or "Severe Accident (Manual Dispatch)"
    dispatch = True if req_data.dispatch_emergency is None else req_data.dispatch_emergency

    try:
        res = engine.capture_snapshot(
            tag="manual_accident",
            dispatch_alert=dispatch,
            incident_type="ACCIDENT_LVL_3",
            severity_label=severity_label,
        )
        return {
            "status": "SUCCESS",
            "incident_type": "ACCIDENT_LVL_3",
            "severity_label": severity_label,
            "message": "Manual vehicle accident alert triggered successfully",
            "departments_notified": ["police", "hospital", "traffic_control"],
            "snapshot": res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Accident alert dispatch failed: {e}")


@router.post("/trigger-fire-alert", summary="Manually trigger fire & smoke alert")
def trigger_fire_alert(
    req: Optional[TriggerFireAlertRequest] = Body(None),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """
    Manually triggers a fire & smoke emergency alert (FIRE_ALERT) and dispatches
    notifications to Fire Stations, Hospitals, and Police departments.
    """
    req_data = req or TriggerFireAlertRequest()
    severity_label = req_data.severity_label or "Active Fire Hazard (Manual Dispatch)"
    dispatch = True if req_data.dispatch_emergency is None else req_data.dispatch_emergency

    try:
        res = engine.capture_snapshot(
            tag="manual_fire",
            dispatch_alert=dispatch,
            incident_type="FIRE_ALERT",
            severity_label=severity_label,
        )
        return {
            "status": "SUCCESS",
            "incident_type": "FIRE_ALERT",
            "severity_label": severity_label,
            "message": "Manual fire hazard alert triggered successfully",
            "departments_notified": ["fire", "hospital", "police"],
            "snapshot": res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fire alert dispatch failed: {e}")

