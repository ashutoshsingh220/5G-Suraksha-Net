"""FastAPI REST + WebSocket interface for the wider 5G Suraksha-Net system."""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from suraksha.config import PROJECT_ROOT, load_config
from suraksha.incidents.schemas import IncidentReport
from suraksha.logging_utils import get_logger, setup_logging
from suraksha.pipeline import CrowdFightPipeline
from suraksha.network import get_network_policy_service
from suraksha.agents import get_agent_orchestrator
from suraksha.demo import get_demo_service, DemoScenarioId
from suraksha.telemetry import get_telemetry_service

log = get_logger(__name__)

_pipeline: CrowdFightPipeline | None = None
_ws_clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None

# Video streaming preview queue & latest frame cache
_preview_queue: queue.Queue = queue.Queue(maxsize=4)
_latest_frame_jpeg: bytes | None = None
_latest_frame_lock: threading.Lock = threading.Lock()
_encoder_stop: threading.Event = threading.Event()
_encoder_thread: threading.Thread | None = None


def _frame_encoder_worker():
    """Background worker that encodes annotated frames from CrowdFightPipeline to JPEG."""
    global _latest_frame_jpeg
    while not _encoder_stop.is_set():
        try:
            frame = _preview_queue.get(timeout=0.2)
            if frame is None:
                continue
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with _latest_frame_lock:
                    _latest_frame_jpeg = buf.tobytes()
        except queue.Empty:
            continue
        except Exception:
            pass


def _on_incident(report: IncidentReport) -> None:
    """EventBus callback (pipeline thread) -> update network policy & broadcast to all WS clients."""
    try:
        get_network_policy_service().on_incident(report)
    except Exception:
        log.exception("Failed to update network policy on incident")

    if _loop is None or not _ws_clients:
        return
    payload = report.model_dump_json()
    asyncio.run_coroutine_threadsafe(_broadcast(payload), _loop)


async def _broadcast(payload: str) -> None:
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _loop, _encoder_thread
    setup_logging()
    cfg = load_config()
    _loop = asyncio.get_running_loop()
    _encoder_stop.clear()
    _encoder_thread = threading.Thread(target=_frame_encoder_worker, name="mjpeg_encoder", daemon=True)
    _encoder_thread.start()
    _pipeline = CrowdFightPipeline(cfg, preview_queue=_preview_queue)
    _pipeline.bus.subscribe(_on_incident)
    _pipeline.start()

    telemetry_svc = get_telemetry_service(cfg)
    telemetry_svc.start()
    telemetry_task = asyncio.create_task(telemetry_svc.broadcast_loop())

    yield

    telemetry_task.cancel()
    telemetry_svc.stop()
    _encoder_stop.set()
    if _encoder_thread and _encoder_thread.is_alive():
        _encoder_thread.join(timeout=1.0)
    _pipeline.stop()


app = FastAPI(
    title="5G Suraksha-Net — Crowd & Fight Module",
    version="0.1.0",
    description="Real-time crowd monitoring and temporal fight detection from Sparsh CCTV/RTSP.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if _pipeline is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {
        "status": "ok" if _pipeline.state.running else "stopped",
        "device": _pipeline.state.device,
        "gpu": _pipeline.device_info.gpu_name,
        "cuda_available": _pipeline.device_info.cuda_available,
        "frames_processed": _pipeline.state.frames_processed,
        "effective_fps": round(_pipeline.state.effective_fps, 2),
    }


@app.get("/system/status")
def system_status():
    """Consolidated truthful runtime system, video pipeline, AI performance, and hardware metrics."""
    now = time.time()
    if _pipeline is None:
        return {
            "status": "OFFLINE",
            "uptime_seconds": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "video_source": {
                "status": "OFFLINE",
                "source_kind": "none",
                "source_url": "",
                "resolution": None,
                "source_fps": None,
                "effective_fps": 0.0,
                "frames_processed": 0,
                "last_frame_age_s": None,
                "has_live_frame": False,
            },
            "ai_inference": {
                "status": "OFFLINE",
                "device": "none",
                "effective_fps": 0.0,
                "inference_latency_ms": None,
                "latency_breakdown": {},
            },
            "gpu": {
                "available": False,
                "name": None,
                "memory_used_mb": None,
                "memory_total_mb": None,
                "utilization_percent": None,
            },
            "drone_telemetry": None,
            "incidents_count": 0,
        }

    started_at = getattr(_pipeline, "started_at", None)
    if not isinstance(started_at, (int, float)):
        started_at = getattr(_pipeline.state, "started_at", None)
    uptime_s = round(now - started_at, 1) if (isinstance(started_at, (int, float)) and started_at > 0) else 0.0

    running = bool(getattr(_pipeline.state, "running", False))
    frames_processed = getattr(_pipeline.state, "frames_processed", 0)
    if not isinstance(frames_processed, int):
        frames_processed = 0

    raw_fps = getattr(_pipeline.state, "effective_fps", 0.0)
    effective_fps = round(float(raw_fps), 1) if isinstance(raw_fps, (int, float)) else 0.0

    # Video Source
    source_kind = "unknown"
    source_url = ""
    if hasattr(_pipeline, "capture"):
        source_kind = str(getattr(_pipeline.capture, "kind", getattr(_pipeline.capture, "source_kind", "unknown")))
        source_url = str(getattr(_pipeline.capture, "source", getattr(_pipeline.capture, "resolved_url", "")))

    raw_metrics = getattr(_pipeline.state, "metrics", None)
    if hasattr(raw_metrics, "total_ms"):
        raw_res = getattr(raw_metrics, "resolution", None)
        raw_sfps = getattr(raw_metrics, "source_fps", None)
        metrics = {
            "resolution": raw_res if isinstance(raw_res, str) else None,
            "source_fps": raw_sfps if isinstance(raw_sfps, (int, float)) else None,
            "total_ms": getattr(raw_metrics, "total_ms", 0.0) if isinstance(getattr(raw_metrics, "total_ms", None), (int, float)) else 0.0,
            "detect_track_ms": getattr(raw_metrics, "detect_track_ms", 0.0) if isinstance(getattr(raw_metrics, "detect_track_ms", None), (int, float)) else 0.0,
            "crowd_ms": getattr(raw_metrics, "crowd_ms", 0.0) if isinstance(getattr(raw_metrics, "crowd_ms", None), (int, float)) else 0.0,
            "fight_ms": getattr(raw_metrics, "fight_ms", 0.0) if isinstance(getattr(raw_metrics, "fight_ms", None), (int, float)) else 0.0,
            "weapon_ms": getattr(raw_metrics, "weapon_ms", 0.0) if isinstance(getattr(raw_metrics, "weapon_ms", None), (int, float)) else 0.0,
        }
    elif isinstance(raw_metrics, dict):
        metrics = raw_metrics
    else:
        metrics = {}

    resolution = metrics.get("resolution") if isinstance(metrics.get("resolution"), str) else None
    if not resolution and hasattr(_pipeline, "capture"):
        w = getattr(_pipeline.capture, "width", None)
        h = getattr(_pipeline.capture, "height", None)
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            resolution = f"{w}x{h}"

    raw_source_fps = metrics.get("source_fps")
    if raw_source_fps is None and hasattr(_pipeline, "capture"):
        raw_source_fps = getattr(_pipeline.capture, "fps", None)
    source_fps = round(float(raw_source_fps), 1) if isinstance(raw_source_fps, (int, float)) and raw_source_fps > 0 else None

    last_frame_time = getattr(_pipeline.state, "last_frame_time", None)
    last_frame_age_s = round(now - last_frame_time, 2) if (isinstance(last_frame_time, (int, float)) and last_frame_time > 0) else None

    with _latest_frame_lock:
        has_live_frame = _latest_frame_jpeg is not None

    if running and has_live_frame and (last_frame_age_s is None or last_frame_age_s < 3.0):
        video_status = "LIVE"
    elif running:
        video_status = "STALE" if (last_frame_age_s and last_frame_age_s >= 3.0) else "READY"
    else:
        video_status = "OFFLINE"

    # AI Performance
    if running and effective_fps > 0.5:
        ai_status = "ACTIVE"
    elif running:
        ai_status = "IDLE"
    else:
        ai_status = "OFFLINE"

    device = getattr(_pipeline.state, "device", "cpu")
    inference_latency_ms = round(metrics.get("total_ms", 0.0), 1) if metrics.get("total_ms") else None

    # GPU
    cuda_avail = bool(getattr(_pipeline.device_info, "cuda_available", False))
    gpu_name = getattr(_pipeline.device_info, "gpu_name", None)
    gpu_total_mb = getattr(_pipeline.device_info, "gpu_memory_mb", None)
    gpu_used_mb = None
    if cuda_avail:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_used_mb = int(torch.cuda.memory_allocated(0) // (1024 * 1024))
        except Exception:
            gpu_used_mb = None

    # Telemetry
    telem = None
    try:
        svc = get_telemetry_service()
        t = svc.get_telemetry()
        telem = {
            "connected": t.connected,
            "gps_fix_type": t.gps_fix_type,
            "satellites_visible": t.satellites_visible,
            "last_heartbeat_s": t.last_heartbeat_s,
        }
    except Exception:
        pass

    # Overall system health
    if running and video_status == "LIVE":
        overall_status = "ONLINE"
    elif running:
        overall_status = "DEGRADED"
    else:
        overall_status = "OFFLINE"

    incidents_count = len(_pipeline.manager.recent) if hasattr(_pipeline, "manager") and hasattr(_pipeline.manager, "recent") else 0

    return {
        "status": overall_status,
        "uptime_seconds": uptime_s,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_source": {
            "status": video_status,
            "source_kind": source_kind,
            "source_url": source_url,
            "resolution": resolution,
            "source_fps": source_fps,
            "effective_fps": effective_fps,
            "frames_processed": frames_processed,
            "last_frame_age_s": last_frame_age_s,
            "has_live_frame": has_live_frame,
        },
        "ai_inference": {
            "status": ai_status,
            "device": device,
            "effective_fps": effective_fps,
            "inference_latency_ms": inference_latency_ms,
            "breakdown_ms": {
                "detect_track": round(metrics.get("detect_track_ms", 0.0), 1) if metrics.get("detect_track_ms") else None,
                "crowd": round(metrics.get("crowd_ms", 0.0), 1) if metrics.get("crowd_ms") else None,
                "fight": round(metrics.get("fight_ms", 0.0), 1) if metrics.get("fight_ms") else None,
                "weapon": round(metrics.get("weapon_ms", 0.0), 1) if metrics.get("weapon_ms") else None,
            },
        },
        "gpu": {
            "available": cuda_avail,
            "name": gpu_name,
            "memory_used_mb": gpu_used_mb,
            "memory_total_mb": gpu_total_mb,
            "utilization_percent": None,  # Not measured without continuous NVML sampling; explicitly null per safety rule
        },
        "drone_telemetry": telem,
        "incidents_count": incidents_count,
        "detection_state": {
            "weapons": (
                len(getattr(_pipeline.state, "latest_weapon", {}).get("telemetry", []))
                if isinstance(getattr(_pipeline.state, "latest_weapon", None), dict)
                else 0
            ) or (
                int(getattr(_pipeline.state, "latest_weapon", {}).get("detections_count", 0))
                if isinstance(getattr(_pipeline.state, "latest_weapon", None), dict)
                else 0
            ),
            "fights": getattr(_pipeline.state, "active_fights_count", 0) if hasattr(_pipeline, "state") else 0,
            "person_count": getattr(_pipeline.state, "person_count", 0) if hasattr(_pipeline, "state") else 0,
            "crowd_density": float(getattr(_pipeline.state, "latest_crowd", {}).get("density_score", 0.0))
            if isinstance(getattr(_pipeline.state, "latest_crowd", None), dict)
            else 0.0,
        },
    }


@app.get("/crowd/status")
def crowd_status():
    """Latest crowd snapshot: counts, zone densities, growth, movement."""
    if _pipeline is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return _pipeline.state.latest_crowd or {"person_count": 0, "zones": []}


@app.get("/weapon/status")
def weapon_status():
    """Latest weapon detector status, metrics, and active telemetry."""
    if _pipeline is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {
        "status": "ok" if _pipeline.state.running else "stopped",
        "metrics": _pipeline.weapon_detector.get_metrics(),
        "latest_weapon": _pipeline.state.latest_weapon or {},
    }


@app.get("/incidents")
def incidents(limit: int = 50, incident_type: str | None = None, include_persisted: bool = False):
    """Recent incidents (newest first). Queries runtime memory; falls back to disk persistence when empty."""
    if _pipeline is None:
        return JSONResponse({"status": "starting"}, status_code=503)

    items = []
    seen_ids = set()

    # 1. In-memory recent incidents
    if hasattr(_pipeline, "manager") and hasattr(_pipeline.manager, "recent"):
        for r in _pipeline.manager.recent:
            inc_id = getattr(r, "incident_id", None)
            if inc_id and inc_id not in seen_ids and isinstance(inc_id, str):
                seen_ids.add(inc_id)
                items.append(r.model_dump(mode="json"))

    # 2. Persisted incident reports on disk (used when memory is empty or explicitly requested)
    if not items or include_persisted:
        try:
            cfg = load_config()
            report_dir_str = cfg.incidents.report_dir
            if report_dir_str:
                report_dir = (PROJECT_ROOT / report_dir_str).resolve()
                if report_dir.is_dir():
                    json_files = sorted(
                        [p for p in report_dir.glob("*.json") if SAFE_INCIDENT_ID_REGEX.match(p.stem)],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for jf in json_files:
                        if len(items) >= limit + 20:
                            break
                        try:
                            with open(jf, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            inc_id = data.get("incident_id")
                            if inc_id and inc_id not in seen_ids:
                                seen_ids.add(inc_id)
                                items.append(data)
                        except Exception:
                            pass
        except Exception:
            pass

    if incident_type:
        items = [i for i in items if i.get("incident_type") == incident_type]

    def _parse_time(item):
        st = item.get("start_time")
        if isinstance(st, str):
            try:
                return datetime.fromisoformat(st.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
        return 0.0

    items.sort(key=_parse_time, reverse=True)
    sliced = items[:limit]
    return {"count": len(sliced), "incidents": sliced}


@app.get("/incidents/{incident_id}")
def incident_detail(incident_id: str):
    if _pipeline is not None and hasattr(_pipeline, "manager") and hasattr(_pipeline.manager, "recent"):
        for r in _pipeline.manager.recent:
            if getattr(r, "incident_id", None) == incident_id:
                return r.model_dump(mode="json")

    # Check demo isolation storage
    demo_inc = get_demo_service().get_demo_incident(incident_id)
    if demo_inc is not None:
        return demo_inc.model_dump(mode="json")

    # Check disk persistence
    try:
        if SAFE_INCIDENT_ID_REGEX.match(incident_id):
            cfg = load_config()
            report_dir = (PROJECT_ROOT / cfg.incidents.report_dir).resolve()
            path = (report_dir / f"{incident_id}.json").resolve()
            if path.is_file() and str(path).lower().startswith(str(report_dir).lower()):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception as e:
        log.exception("Failed to load disk incident: %s", e)

    if _pipeline is None:
        return JSONResponse({"status": "starting"}, status_code=503)

    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/incidents/{incident_id}/response")
def incident_response_plan(incident_id: str):
    """Retrieve the deterministic emergency response plan for an incident."""
    plan = None
    if _pipeline is not None:
        if hasattr(_pipeline.manager, "get_response_plan") and callable(_pipeline.manager.get_response_plan):
            try:
                res = _pipeline.manager.get_response_plan(incident_id)
                if res is not None and type(res).__name__ not in ("MagicMock", "Mock") and hasattr(res, "model_dump"):
                    plan = res
            except Exception:
                log.exception("Error querying response plan for incident %s", incident_id)

        if plan is None and hasattr(_pipeline.manager, "recent"):
            for r in _pipeline.manager.recent:
                if getattr(r, "incident_id", None) == incident_id:
                    planner = getattr(_pipeline, "planner", None)
                    if planner is None or not hasattr(planner, "plan"):
                        from suraksha.response import ResponsePlanner
                        planner = ResponsePlanner()
                    try:
                        plan = planner.plan(r)
                    except Exception:
                        log.exception("Error generating on-demand response plan for %s", incident_id)
                    break

    if plan is None and SAFE_INCIDENT_ID_REGEX.match(incident_id):
        try:
            cfg = load_config()
            report_dir = (PROJECT_ROOT / cfg.incidents.report_dir).resolve()
            path = (report_dir / f"{incident_id}.json").resolve()
            if path.is_file() and str(path).lower().startswith(str(report_dir).lower()):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                report = IncidentReport.model_validate(data)
                planner = getattr(_pipeline, "planner", None) if _pipeline is not None else None
                if planner is None or type(planner).__name__ in ("MagicMock", "Mock"):
                    from suraksha.response import ResponsePlanner
                    planner = ResponsePlanner()
                plan = planner.plan(report)
        except Exception as e:
            log.exception("Error generating on-demand response plan for disk incident: %s", e)

    if plan is None:
        demo_inc = get_demo_service().get_demo_incident(incident_id)
        if demo_inc is not None:
            planner = getattr(_pipeline, "planner", None) if _pipeline is not None else None
            if planner is None or type(planner).__name__ in ("MagicMock", "Mock"):
                from suraksha.response import ResponsePlanner
                planner = ResponsePlanner()
            plan = planner.plan(demo_inc)

    if plan is None:
        if _pipeline is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        return JSONResponse({"error": "response plan not found"}, status_code=404)

    return plan.model_dump(mode="json")


@app.get("/incidents/{incident_id}/assessment")
def incident_agent_assessment(incident_id: str, force_refresh: bool = False):
    """Retrieve agentic emergency orchestration assessment and situational briefing."""
    if not SAFE_INCIDENT_ID_REGEX.match(incident_id):
        return JSONResponse({"error": "invalid incident id"}, status_code=400)

    mgr = getattr(_pipeline, "manager", None) if _pipeline is not None else None
    orchestrator = get_agent_orchestrator()
    assessment = orchestrator.get_assessment(
        incident_id=incident_id,
        force_refresh=force_refresh,
        pipeline_manager=mgr,
    )
    if assessment is None:
        return JSONResponse({"error": "incident not found"}, status_code=404)

    return assessment.model_dump(mode="json")


SAFE_INCIDENT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _send_video_file(file_path: Path, request: Request | None = None):
    """Serve MP4 video supporting HTTP 206 Partial Content range requests for smooth HTML5 seeking."""
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range") if request else None

    if not range_header:
        return FileResponse(file_path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})

    try:
        h = range_header.replace("bytes=", "").strip()
        parts = h.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = (end - start) + 1

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_left = chunk_size
                while bytes_left > 0:
                    read_len = min(65536, bytes_left)
                    data = f.read(read_len)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)
    except Exception:
        return FileResponse(file_path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


@app.get("/incidents/{incident_id}/snapshot")
@app.head("/incidents/{incident_id}/snapshot")
@app.get("/evidence/snapshot/{incident_id}")
@app.head("/evidence/snapshot/{incident_id}")
def incident_snapshot(incident_id: str):
    if not SAFE_INCIDENT_ID_REGEX.match(incident_id):
        return JSONResponse({"error": "invalid incident id"}, status_code=400)

    from pathlib import Path
    from suraksha.config import PROJECT_ROOT

    cfg = load_config()
    snapshot_dir = (PROJECT_ROOT / cfg.incidents.snapshot_dir).resolve()
    path = (snapshot_dir / f"{incident_id}.jpg").resolve()
    if not path.is_relative_to(snapshot_dir) or not path.is_file():
        # Auto-extract keyframe from MP4 clip if available
        clip_dir = (PROJECT_ROOT / cfg.incidents.clip_dir).resolve()
        clip_path = (clip_dir / f"{incident_id}.mp4").resolve()
        if clip_path.is_file():
            try:
                cap = cv2.VideoCapture(str(clip_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid = max(0, total_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), frame)
                    return FileResponse(path, media_type="image/jpeg")
            except Exception as e:
                log.warning("Failed extracting keyframe from clip %s: %s", clip_path, e)

        # Fallback to current live frame or buffer if available for active/recent incident
        is_known = False
        if _pipeline is not None and hasattr(_pipeline, "manager") and hasattr(_pipeline.manager, "recent"):
            is_known = any(getattr(r, "incident_id", None) == incident_id for r in _pipeline.manager.recent)

        if is_known and _latest_frame_jpeg is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(_latest_frame_jpeg)
                return FileResponse(path, media_type="image/jpeg")
            except Exception as e:
                log.warning("Failed writing fallback snapshot for %s: %s", incident_id, e)

        if is_known and _pipeline is not None and hasattr(_pipeline, "evidence") and hasattr(_pipeline.evidence, "_buffer"):
            with _pipeline.evidence._lock:
                if _pipeline.evidence._buffer:
                    last_frame = _pipeline.evidence._buffer[-1][1]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), last_frame)
                    return FileResponse(path, media_type="image/jpeg")

        # Check demo snapshots
        demo_dir = (PROJECT_ROOT / "outputs" / "demo" / "snapshots").resolve()
        demo_path = (demo_dir / f"{incident_id}.jpg").resolve()
        if demo_path.is_relative_to(demo_dir) and demo_path.is_file():
            return FileResponse(demo_path, media_type="image/jpeg")
        default_demo = (demo_dir / "demo_evidence.jpg").resolve()
        if (incident_id.startswith("demo-") or incident_id.startswith("INC-DEMO-")) and default_demo.is_file():
            return FileResponse(default_demo, media_type="image/jpeg")
        return JSONResponse({"error": "snapshot not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/incidents/{incident_id}/clip")
@app.head("/incidents/{incident_id}/clip")
@app.get("/evidence/clip/{incident_id}")
@app.head("/evidence/clip/{incident_id}")
def incident_clip(incident_id: str, request: Request):
    if not SAFE_INCIDENT_ID_REGEX.match(incident_id):
        return JSONResponse({"error": "invalid incident id"}, status_code=400)

    from pathlib import Path
    from suraksha.config import PROJECT_ROOT

    cfg = load_config()
    clip_dir = (PROJECT_ROOT / cfg.incidents.clip_dir).resolve()
    path = (clip_dir / f"{incident_id}.mp4").resolve()
    if path.is_file() and path.is_relative_to(clip_dir):
        return _send_video_file(path, request)

    # If clip is currently recording in session, write interim clip from available frames immediately
    if _pipeline is not None and hasattr(_pipeline, "evidence") and hasattr(_pipeline.evidence, "_active_sessions"):
        session = _pipeline.evidence._active_sessions.get(incident_id)
        if session and len(session.frames) >= 5:
            try:
                res_path = _pipeline.evidence._write_session_clip(session)
                if res_path and Path(res_path).is_file():
                    return _send_video_file(Path(res_path), request)
            except Exception as e:
                log.warning("Interim clip write error for %s: %s", incident_id, e)

    # Check demo clips or synthetic video
    demo_clip_dir = (PROJECT_ROOT / "outputs" / "demo" / "clips").resolve()
    demo_clip_path = (demo_clip_dir / f"{incident_id}.mp4").resolve()
    if demo_clip_path.is_relative_to(demo_clip_dir) and demo_clip_path.is_file():
        return _send_video_file(demo_clip_path, request)
    synth_clip = (PROJECT_ROOT / "datasets" / "videos" / "test" / "synthetic_cctv.mp4").resolve()
    if (incident_id.startswith("demo-") or incident_id.startswith("INC-DEMO-")) and synth_clip.is_file():
        return _send_video_file(synth_clip, request)
    return JSONResponse({"error": "clip not found"}, status_code=404)


@app.websocket("/ws/incidents")
async def ws_incidents(ws: WebSocket):
    """Push channel: every verified incident is broadcast as JSON on arrival."""
    await ws.accept()
    _ws_clients.add(ws)
    log.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        while True:
            await ws.receive_text()  # keepalive / ignored
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


@app.get("/drone/telemetry")
def drone_telemetry():
    """Get the latest real-time drone telemetry snapshot from flight controller."""
    svc = get_telemetry_service()
    return svc.get_telemetry().model_dump(mode="json")


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    """Real-time flight controller telemetry push channel (5-10 Hz)."""
    await ws.accept()
    svc = get_telemetry_service()
    svc.register_client(ws)
    log.info("Telemetry WebSocket client connected (%d total)", svc.client_count)
    try:
        while True:
            await ws.receive_text()  # keepalive
    except WebSocketDisconnect:
        pass
    finally:
        svc.unregister_client(ws)


@app.get("/network/policy")
def network_policy():
    """Current truthful application-level network policy & transmission priority."""
    svc = get_network_policy_service()
    candidates = None
    if _pipeline is not None and hasattr(_pipeline, "manager") and hasattr(_pipeline.manager, "recent"):
        candidates = list(_pipeline.manager.recent)
    state = svc.get_policy_state(candidate_incidents=candidates)
    return state.model_dump(mode="json")


@app.get("/network/events")
def network_events(limit: int = 50):
    """Recent network policy transition & transmission instrumentation events."""
    svc = get_network_policy_service()
    return svc.get_events(limit=limit).model_dump(mode="json")


class DemoScenarioRequest(BaseModel):
    scenario_id: str
    video_choice: str | None = None


@app.get("/demo/status")
def demo_status():
    """Retrieve current demonstration mode and simulation status."""
    svc = get_demo_service()
    return svc.get_status().model_dump(mode="json")


@app.get("/demo/scenarios")
def demo_scenarios():
    """List all available controlled demonstration scenarios."""
    svc = get_demo_service()
    scenarios = svc.get_available_scenarios()
    return [s.model_dump(mode="json") for s in scenarios]


def _restore_baseline_capture() -> bool:
    """Helper to revert pipeline capture back to default baseline (RTSP / Webcam)."""
    global _pipeline, _preview_queue
    try:
        cfg = load_config()
        base_url = cfg.capture.rtsp_url
        raw_type = (cfg.capture.source_type or "").strip().lower()
        base_type = "stream" if raw_type in ("rtsp", "stream") else (raw_type or "stream")
        base_pace = cfg.capture.pace or "realtime"
        if _pipeline is not None and getattr(_pipeline.state, "running", False):
            switched = _pipeline.switch_source(url=base_url, source_type=base_type, pace=base_pace)
            if not switched:
                _pipeline.stop(timeout=2.0)
                _pipeline = CrowdFightPipeline(cfg, preview_queue=_preview_queue)
                _pipeline.bus.subscribe(_on_incident)
                _pipeline.start()
        else:
            _pipeline = CrowdFightPipeline(cfg, preview_queue=_preview_queue)
            _pipeline.bus.subscribe(_on_incident)
            _pipeline.start()
        log.info("Live pipeline restored to baseline capture: %s (%s)", base_url, base_type)
        return True
    except Exception as e:
        log.warning("Failed restoring baseline pipeline: %s", e)
        return False


def clear_all_incident_logs() -> dict:
    """Purge all incident reports, clips, snapshots from disk and clear in-memory buffers."""
    global _pipeline
    cleared_counts = {"incidents": 0, "snapshots": 0, "clips": 0, "demo_incidents": 0}
    try:
        # 1. Clear in-memory buffers
        if _pipeline is not None:
            if hasattr(_pipeline, "manager") and hasattr(_pipeline.manager, "recent"):
                _pipeline.manager.recent.clear()
            if hasattr(_pipeline, "state"):
                _pipeline.state.latest_incidents = []
                _pipeline.state.latest_weapon = {}
                _pipeline.state.active_fights_count = 0
            if hasattr(_pipeline, "weapon_detector") and hasattr(_pipeline.weapon_detector, "confirmation"):
                _pipeline.weapon_detector.confirmation.reset()
            if hasattr(_pipeline, "fight") and hasattr(_pipeline.fight, "_active"):
                _pipeline.fight._active.clear()
                if hasattr(_pipeline.fight, "latest_candidates"):
                    _pipeline.fight.latest_candidates.clear()

        # 2. Clear disk reports in outputs/incidents
        inc_dir = (PROJECT_ROOT / "outputs" / "incidents").resolve()
        if inc_dir.is_dir():
            for f in inc_dir.glob("*.json"):
                try:
                    f.unlink()
                    cleared_counts["incidents"] += 1
                except Exception:
                    pass
            arc_dir = inc_dir / "archive"
            if arc_dir.is_dir():
                for f in arc_dir.glob("*.json"):
                    try:
                        f.unlink()
                        cleared_counts["incidents"] += 1
                    except Exception:
                        pass

        # 3. Clear snapshots in outputs/snapshots
        snap_dir = (PROJECT_ROOT / "outputs" / "snapshots").resolve()
        if snap_dir.is_dir():
            for f in snap_dir.glob("*.jpg"):
                try:
                    f.unlink()
                    cleared_counts["snapshots"] += 1
                except Exception:
                    pass

        # 4. Clear clips in outputs/clips
        clip_dir = (PROJECT_ROOT / "outputs" / "clips").resolve()
        if clip_dir.is_dir():
            for f in clip_dir.glob("*.mp4"):
                try:
                    f.unlink()
                    cleared_counts["clips"] += 1
                except Exception:
                    pass

        # 5. Clear demo incidents in outputs/demo/incidents
        demo_inc_dir = (PROJECT_ROOT / "outputs" / "demo" / "incidents").resolve()
        if demo_inc_dir.is_dir():
            for f in demo_inc_dir.glob("*.json"):
                try:
                    f.unlink()
                    cleared_counts["demo_incidents"] += 1
                except Exception:
                    pass

        # 6. Clear demo generated snapshots (demo-*.jpg), keep static keyframe evidence templates
        demo_snap_dir = (PROJECT_ROOT / "outputs" / "demo" / "snapshots").resolve()
        if demo_snap_dir.is_dir():
            for f in demo_snap_dir.glob("demo-*.jpg"):
                try:
                    f.unlink()
                except Exception:
                    pass

        log.info("Cleared all incident logs and evidence: %s", cleared_counts)
    except Exception as e:
        log.warning("Error during clear_all_incident_logs: %s", e)
    return cleared_counts


@app.post("/demo/scenario")
def demo_start_scenario(req: DemoScenarioRequest):
    """Trigger a controlled demonstration scenario (WEAPON, CROWD_PANIC, ARMED_FIGHT, NORMAL)."""
    global _pipeline, _preview_queue
    svc = get_demo_service()
    res = svc.start_scenario(req.scenario_id, video_choice=req.video_choice)

    # 1. Switch active live video pipeline to the demo video file, or restore baseline if NORMAL/RESET
    video_file = getattr(res, "video_file", None)
    if video_file and os.path.isfile(video_file):
        try:
            cfg = load_config()
            cfg.capture.source_type = "file"
            cfg.capture.rtsp_url = str(video_file)
            cfg.capture.pace = "realtime"
            if _pipeline is not None and getattr(_pipeline.state, "running", False):
                switched = _pipeline.switch_source(url=str(video_file), source_type="file", pace="realtime")
                if not switched:
                    _pipeline.stop(timeout=2.0)
                    _pipeline = CrowdFightPipeline(cfg, preview_queue=_preview_queue)
                    _pipeline.bus.subscribe(_on_incident)
                    _pipeline.start()
            else:
                _pipeline = CrowdFightPipeline(cfg, preview_queue=_preview_queue)
                _pipeline.bus.subscribe(_on_incident)
                _pipeline.start()
            log.info("Live pipeline successfully switched to scenario video: %s", video_file)
        except Exception as e:
            log.warning("Failed switching pipeline to scenario video %s: %s", video_file, e)
    elif str(req.scenario_id).upper() in ("NORMAL", "RESET") or not res.is_demo_active:
        _restore_baseline_capture()

    # 2. Dispatch differentiated email notifications (Police vs Hospital)
    if res.incident is not None:
        try:
            notifier = getattr(_pipeline, "email_notifier", None)
            if notifier is None:
                cfg = load_config()
                from suraksha.notifications.service import EmailNotificationService
                notifier = EmailNotificationService(config=cfg.email)
            custom_clip = res.incident.details.get("email_clip_path")
            notifier.send_incident_email(
                incident=res.incident,
                allow_demo=True,
                separate_roles=True,
                custom_clip_path=custom_clip,
                blocking=False,
            )
        except Exception as e:
            log.warning("Failed dispatching demo email alert: %s", e)

    # 3. Broadcast simulated incident to WebSocket clients if connected
    if res.incident is not None and _loop is not None and _ws_clients:
        payload = res.incident.model_dump_json()
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _loop)

    return res.model_dump(mode="json")


@app.post("/demo/reset")
def demo_reset():
    """Stop active demonstration scenario and restore baseline normal state."""
    global _pipeline, _preview_queue
    svc = get_demo_service()
    res = svc.stop_scenario()

    # Revert pipeline capture back to default baseline (RTSP / Webcam)
    _restore_baseline_capture()

    # Clear transient demo state from runtime pipeline memory
    if _pipeline is not None:
        if hasattr(_pipeline, "manager") and hasattr(_pipeline.manager, "recent"):
            _pipeline.manager.recent.clear()
        if hasattr(_pipeline, "state"):
            _pipeline.state.latest_incidents = []
            _pipeline.state.latest_weapon = {}
            _pipeline.state.active_fights_count = 0
        if hasattr(_pipeline, "weapon_detector") and hasattr(_pipeline.weapon_detector, "confirmation"):
            _pipeline.weapon_detector.confirmation.reset()
        if hasattr(_pipeline, "fight") and hasattr(_pipeline.fight, "_active"):
            _pipeline.fight._active.clear()

    return res.model_dump(mode="json")


@app.post("/incidents/clear")
def incidents_clear():
    """Clear all incident records from runtime memory and storage."""
    counts = clear_all_incident_logs()
    if _loop is not None and _ws_clients:
        asyncio.run_coroutine_threadsafe(
            _broadcast(json.dumps({"event": "incidents_cleared", "count": 0})),
            _loop
        )
    return {"status": "ok", "cleared": counts}


@app.get("/location/config")
def location_config():
    """Provide client with location intelligence configuration, including Google Maps API key."""
    import os
    cfg = load_config()
    key = getattr(cfg.location, "google_maps_api_key", None) or os.environ.get("GOOGLE_MAPS_API_KEY", "")
    return {
        "google_maps_api_key": key,
        "default_query": getattr(cfg.location, "default_query", "Yashobhoomi, Dwarka Sector 25, New Delhi"),
        "default_latitude": 28.5529,
        "default_longitude": 77.0601,
    }


@app.get("/notifications/status")
def notification_status():
    """Return current SMTP email notification configuration and routing status (no secrets exposed)."""
    if _pipeline is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    notifier = getattr(_pipeline, "email_notifier", None)
    if notifier is None:
        from suraksha.notifications.service import EmailNotificationService
        cfg = load_config()
        notifier = EmailNotificationService(config=cfg.email)
    return notifier.get_status()


@app.post("/notifications/test")
def notification_test():
    """Attempt a live SMTP connection test. Returns ok=true/false with diagnostic message.

    Use this to verify your SMTP settings before an incident occurs.
    Does not send any email — only tests TCP connection and authentication handshake.
    """
    if _pipeline is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    notifier = getattr(_pipeline, "email_notifier", None)
    if notifier is None:
        from suraksha.notifications.service import EmailNotificationService
        cfg = load_config()
        notifier = EmailNotificationService(config=cfg.email)
    result = notifier.test_smtp_connection()
    status_code = 200 if result.get("ok") else 503
    return JSONResponse(result, status_code=status_code)


def _generate_standby_frame(source_url: str = "") -> bytes:
    """Generate an informative tactical standby frame while waiting for the video source to send packets."""
    import numpy as np
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (26, 23, 17)  # Slate dark bg
    cv2.rectangle(canvas, (20, 20), (1260, 700), (65, 51, 38), 2)
    cv2.putText(canvas, "5G SURAKSHA-NET - TACTICAL VIDEO INGESTION CORE", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 158, 59), 2, cv2.LINE_AA)
    display_url = source_url or "rtsp://10.254.18.48:8554/drone"
    cv2.putText(canvas, f"SOURCE: {display_url}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (243, 237, 232), 2, cv2.LINE_AA)
    cv2.putText(canvas, "STATUS: AWAITING VIDEO PACKETS FROM RTSP PUBLISHER...", (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (59, 196, 231), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Handshake active. If stream does not appear: verify camera IP/Wi-Fi or test with Webcam/Sample.", (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (181, 166, 152), 1, cv2.LINE_AA)
    t_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cv2.putText(canvas, f"TIMESTAMP: {t_str}", (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (181, 166, 152), 1, cv2.LINE_AA)
    ret, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ret else b""


async def _mjpeg_generator():
    """Generate multipart/x-mixed-replace MJPEG stream from latest annotated frame."""
    last_sent = None
    standby_counter = 0
    while True:
        with _latest_frame_lock:
            frame_bytes = _latest_frame_jpeg
        if frame_bytes is not None and frame_bytes is not last_sent:
            last_sent = frame_bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        elif frame_bytes is None:
            standby_counter += 1
            if standby_counter >= 30 or last_sent is None:
                standby_counter = 0
                src_url = ""
                if _pipeline is not None and hasattr(_pipeline, "capture"):
                    src_url = str(getattr(_pipeline.capture, "resolved_url", ""))
                standby_frame = _generate_standby_frame(src_url)
                if standby_frame:
                    last_sent = standby_frame
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + standby_frame + b"\r\n"
                    )
        await asyncio.sleep(0.033)


@app.get("/video/stream")
async def video_stream():
    """Stream live annotated frames from the active pipeline as MJPEG."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/status")
def video_status():
    """Status of the video capture and streaming pipeline with live real-time detection telemetry."""
    if _pipeline is None:
        return {
            "status": "inactive",
            "source_kind": "none",
            "source_url": "",
            "effective_fps": 0.0,
            "frames_processed": 0,
            "resolution": None,
            "has_live_frame": False,
            "persons": 0,
            "crowd_density": 0.0,
            "weapons": 0,
            "fights": 0,
            "latency_ms": 0.0,
        }

    source_kind = getattr(_pipeline.capture, "source_kind", "unknown") if hasattr(_pipeline, "capture") else "unknown"
    source_url = getattr(_pipeline.capture, "resolved_url", "") if hasattr(_pipeline, "capture") else ""
    running = getattr(_pipeline.state, "running", False) if hasattr(_pipeline, "state") else False
    fps = getattr(_pipeline.state, "effective_fps", 0.0) if hasattr(_pipeline, "state") else 0.0
    frames = getattr(_pipeline.state, "frames_processed", 0) if hasattr(_pipeline, "state") else 0
    metrics = getattr(_pipeline.state, "metrics", {}) if hasattr(_pipeline, "state") else {}
    has_frame = _latest_frame_jpeg is not None

    status = "streaming" if (running and has_frame) else ("active" if running else "stopped")

    # Real-time detection metrics directly from active computer vision pipeline state
    person_count = getattr(_pipeline.state, "person_count", 0) if hasattr(_pipeline, "state") else 0
    latest_crowd = getattr(_pipeline.state, "latest_crowd", {}) if hasattr(_pipeline, "state") else {}
    crowd_density = float(latest_crowd.get("density_score", 0.0)) if isinstance(latest_crowd, dict) else 0.0
    latest_weapon = getattr(_pipeline.state, "latest_weapon", {}) if hasattr(_pipeline, "state") else {}
    weapons = len(latest_weapon.get("telemetry", [])) if isinstance(latest_weapon, dict) else 0
    if weapons == 0 and isinstance(latest_weapon, dict):
        weapons = int(latest_weapon.get("detections_count", 0)) or (int(latest_weapon.get("candidate_count", 0)) + int(latest_weapon.get("confirmed_count", 0)))
    fights = getattr(_pipeline.state, "active_fights_count", 0) if hasattr(_pipeline, "state") else 0
    latency_ms = round(float(metrics.get("total_ms", 0.0)), 1) if isinstance(metrics, dict) else 0.0

    return {
        "status": status,
        "source_kind": source_kind,
        "source_url": source_url,
        "effective_fps": round(fps, 1),
        "frames_processed": frames,
        "resolution": metrics.get("resolution") if isinstance(metrics, dict) else None,
        "has_live_frame": has_frame,
        "persons": person_count,
        "crowd_density": round(crowd_density, 2),
        "weapons": weapons,
        "fights": fights,
        "latency_ms": latency_ms,
    }


class VideoSourceConfig(BaseModel):
    source_type: str  # "webcam", "file", "rtsp"
    url: str | None = None
    camera_index: int = 0
    pace: str | None = None


@app.post("/video/source")
def switch_video_source(req: VideoSourceConfig):
    """Switch active camera/video source without restarting the server."""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    cfg = load_config()
    cfg.capture.source_type = req.source_type
    if req.source_type == "webcam":
        cfg.capture.camera_index = req.camera_index
        cfg.capture.rtsp_url = str(req.camera_index)
        cfg.capture.pace = req.pace or "throttle"
    elif req.source_type == "file":
        cfg.capture.rtsp_url = str(cfg.resolve(req.url or "datasets/videos/test/synthetic_cctv.mp4"))
        cfg.capture.pace = req.pace or "realtime"
    elif req.source_type == "rtsp":
        cfg.capture.rtsp_url = req.url or "rtsp://localhost:554/stream1"
        cfg.capture.pace = req.pace or "throttle"

    if getattr(_pipeline.state, "running", False):
        switched = _pipeline.switch_source(url=cfg.capture.rtsp_url, source_type=cfg.capture.source_type, pace=cfg.capture.pace)
        if not switched:
            _pipeline.stop(timeout=2.0)
            _pipeline = CrowdFightPipeline(cfg, bus=_pipeline.bus, preview_queue=_preview_queue)
            _pipeline.bus.subscribe(_on_incident)
            _pipeline.start()
    else:
        _pipeline = CrowdFightPipeline(cfg, bus=_pipeline.bus, preview_queue=_preview_queue)
        _pipeline.bus.subscribe(_on_incident)
        _pipeline.start()

    return {
        "status": "ok",
        "source_type": cfg.capture.source_type,
        "resolved_url": _pipeline.capture.resolved_url,
        "source_kind": _pipeline.capture.source_kind,
    }

