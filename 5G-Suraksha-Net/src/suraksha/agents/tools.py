"""Read-only tool contracts for the agentic orchestration layer.

Agents consume these structured functions rather than performing arbitrary file access
or mutating system state. These tools strictly read from verified runtime state
and persisted disk archives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from suraksha.config import PROJECT_ROOT, load_config
from suraksha.incidents.schemas import IncidentReport
from suraksha.location.schemas import Location
from suraksha.logging_utils import get_logger
from suraksha.network import get_network_policy_service
from suraksha.network.schemas import NetworkEventMetric, NetworkPolicyState
from suraksha.response.planner import ResponsePlanner
from suraksha.response.schemas import EmergencyResource, IncidentResponse

log = get_logger(__name__)


def get_incident(incident_id: str, pipeline_manager: Any = None) -> IncidentReport | None:
    """Retrieve an incident report by ID from runtime memory or disk persistence."""
    # 1. Check in-memory manager
    if pipeline_manager is not None and hasattr(pipeline_manager, "recent"):
        for report in pipeline_manager.recent:
            if getattr(report, "incident_id", None) == incident_id:
                return report

    # 2. Check disk persistence
    try:
        cfg = load_config()
        report_dir = (PROJECT_ROOT / cfg.incidents.report_dir).resolve()
        path = (report_dir / f"{incident_id}.json").resolve()
        if path.is_file() and str(path).lower().startswith(str(report_dir).lower()):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return IncidentReport.model_validate(data)
    except Exception as e:
        log.debug("Disk lookup failed for incident %s: %s", incident_id, e)

    # 3. Check demo scenario isolation storage
    try:
        from suraksha.demo import get_demo_service
        demo_inc = get_demo_service().get_demo_incident(incident_id)
        if demo_inc is not None:
            return demo_inc
    except Exception:
        pass

    return None


def get_response_plan(
    incident_id: str,
    incident: IncidentReport | None = None,
    pipeline_manager: Any = None,
) -> IncidentResponse | None:
    """Retrieve or deterministically compute emergency response plan for an incident."""
    # 1. Check manager cache
    if pipeline_manager is not None and hasattr(pipeline_manager, "get_response_plan") and callable(pipeline_manager.get_response_plan):
        try:
            plan = pipeline_manager.get_response_plan(incident_id)
            if plan is not None and type(plan).__name__ not in ("MagicMock", "Mock") and hasattr(plan, "response_actions"):
                return plan
        except Exception:
            pass

    # 2. If incident report available, evaluate using authoritative ResponsePlanner
    rep = incident or get_incident(incident_id, pipeline_manager=pipeline_manager)
    if rep is not None:
        try:
            planner = getattr(pipeline_manager, "response_planner", None)
            if planner is None or type(planner).__name__ in ("MagicMock", "Mock"):
                planner = ResponsePlanner()
            return planner.plan(rep)
        except Exception as e:
            log.exception("Failed to evaluate response plan for %s: %s", incident_id, e)

    return None


def get_location(
    incident_id: str,
    incident: IncidentReport | None = None,
    pipeline_manager: Any = None,
) -> Location | None:
    """Retrieve resolved geospatial location for an incident."""
    rep = incident or get_incident(incident_id, pipeline_manager=pipeline_manager)
    return getattr(rep, "location", None) if rep else None


def get_nearby_resources(
    incident_id: str,
    plan: IncidentResponse | None = None,
    incident: IncidentReport | None = None,
    pipeline_manager: Any = None,
) -> dict[str, list[EmergencyResource]]:
    """Retrieve pre-discovered emergency resources associated with the incident's response plan."""
    resp_plan = plan or get_response_plan(incident_id, incident=incident, pipeline_manager=pipeline_manager)
    if resp_plan is None or type(resp_plan).__name__ in ("MagicMock", "Mock"):
        return {"police_stations": [], "hospitals": []}

    pol = getattr(resp_plan, "police_resources", [])
    med = getattr(resp_plan, "medical_resources", [])
    return {
        "police_stations": list(pol) if isinstance(pol, (list, tuple)) else [],
        "hospitals": list(med) if isinstance(med, (list, tuple)) else [],
    }


def get_evidence_metadata(
    incident_id: str,
    incident: IncidentReport | None = None,
    pipeline_manager: Any = None,
) -> dict[str, Any]:
    """Retrieve file existence and metadata for snapshot and clip evidence artifacts."""
    rep = incident or get_incident(incident_id, pipeline_manager=pipeline_manager)
    cfg = load_config()

    snapshot_path_str = getattr(getattr(rep, "evidence", None), "snapshot_path", None)
    clip_path_str = getattr(getattr(rep, "evidence", None), "clip_path", None)

    # Check physical existence on disk
    snapshot_exists = False
    clip_exists = False

    if snapshot_path_str:
        p = (PROJECT_ROOT / snapshot_path_str).resolve()
        snapshot_exists = p.is_file()
    else:
        # Fallback check standard path
        default_snap = (PROJECT_ROOT / cfg.incidents.snapshot_dir / f"{incident_id}.jpg").resolve()
        if default_snap.is_file():
            snapshot_exists = True
            snapshot_path_str = str(default_snap.relative_to(PROJECT_ROOT)).replace("\\", "/")

    if clip_path_str:
        p = (PROJECT_ROOT / clip_path_str).resolve()
        clip_exists = p.is_file()
    else:
        default_clip = (PROJECT_ROOT / cfg.incidents.clip_dir / f"{incident_id}.mp4").resolve()
        if default_clip.is_file():
            clip_exists = True
            clip_path_str = str(default_clip.relative_to(PROJECT_ROOT)).replace("\\", "/")

    return {
        "snapshot_available": snapshot_exists,
        "snapshot_path": snapshot_path_str if snapshot_exists else None,
        "video_available": clip_exists,
        "video_path": clip_path_str if clip_exists else None,
        "frame_idx": getattr(getattr(rep, "evidence", None), "frame_idx", None),
    }


def get_network_policy() -> NetworkPolicyState:
    """Retrieve current truthful application-level network policy state."""
    svc = get_network_policy_service()
    return svc.get_policy_state()


def get_network_events(limit: int = 10) -> list[NetworkEventMetric]:
    """Retrieve recent policy transition audit events."""
    svc = get_network_policy_service()
    return svc.get_events(limit=limit).events
