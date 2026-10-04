import os
import sys
import platform
import time
from datetime import datetime
from fastapi import APIRouter, Depends

from api.dependencies import get_engine, get_sys_settings
from config import (
    SystemSettings,
    CRASH_MODEL_PATH,
    FIRE_MODEL_PATH,
    YOLO_DETECTOR_PATH,
    YOLO_SEVERITY_PATH,
    SNAPSHOT_DIR,
    RECORDING_DIR,
)
from surveillance_engine import SurveillanceEngine

router = APIRouter(prefix="/api/system", tags=["System & Health"])

SERVER_START_TIME = time.time()


@router.get("/health", summary="System health check")
def health_check(
    engine: SurveillanceEngine = Depends(get_engine),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Returns uptime, surveillance worker status, detector status, and health flags."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "surveillance_running": engine.is_running,
        "surveillance_paused": engine.is_paused,
        "camera_id": settings.camera_id,
        "detector_backend": settings.detector_backend,
        "device": settings.inference_device,
        "email_alerts_enabled": settings.smtp_enabled,
    }


@router.get("/info", summary="Detailed platform, GPU, and model status")
def system_info(
    engine: SurveillanceEngine = Depends(get_engine),
    settings: SystemSettings = Depends(get_sys_settings),
):
    """Returns hardware acceleration info, CUDA availability, Python runtime, and model weights integrity."""
    cuda_available = False
    cuda_device_name = "N/A"
    cuda_device_count = 0
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "platform": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version.split()[0],
            "architecture": platform.machine(),
        },
        "hardware": {
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "gpu_name": cuda_device_name,
            "active_inference_device": settings.inference_device,
        },
        "model_weights": {
            "yolo_detector_pt": os.path.exists(YOLO_DETECTOR_PATH),
            "yolo_severity_pt": os.path.exists(YOLO_SEVERITY_PATH),
            "fire_classifier_pt": os.path.exists(FIRE_MODEL_PATH),
            "crash_classifier_safetensors": os.path.exists(CRASH_MODEL_PATH),
        },
        "storage": {
            "snapshots_dir": SNAPSHOT_DIR,
            "snapshots_count": len(os.listdir(SNAPSHOT_DIR))
            if os.path.exists(SNAPSHOT_DIR)
            else 0,
            "recordings_dir": RECORDING_DIR,
        },
    }
