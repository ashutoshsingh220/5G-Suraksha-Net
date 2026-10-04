import csv
import json
import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.dependencies import get_engine
from config import SNAPSHOT_DIR, DISPATCH_LOG_PATH, OUTPUT_DIR
from surveillance_engine import SurveillanceEngine

router = APIRouter(prefix="/api", tags=["Incidents & Dispatches"])


@router.get("/incidents", summary="Query recorded incident log")
def get_incidents(
    limit: int = Query(50, ge=1, le=500, description="Max incident records to return"),
    event_type: Optional[str] = Query(
        None, description="Optional filter e.g. ACCIDENT, FIRE"
    ),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """Returns recent surveillance incident events parsed from the incident CSV log."""
    if engine.incident_logger:
        records = engine.incident_logger.get_recent_incidents(
            limit=limit, event_filter=event_type
        )
        return {"total_returned": len(records), "incidents": records}

    # Direct CSV fallback
    csv_path = os.path.join(OUTPUT_DIR, "incident_log.csv")
    if os.path.exists(csv_path):
        records = []
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (
                        event_type
                        and event_type.upper() not in row.get("Event_Type", "").upper()
                    ):
                        continue
                    records.append(row)
            recent = records[-limit:][::-1]
            return {"total_returned": len(recent), "incidents": recent}
        except Exception as e:
            print(f"[RoutesIncidents] Error reading incident CSV fallback: {e}")

    return {"total_returned": 0, "incidents": []}


@router.get("/dispatches", summary="Query emergency dispatch audit log")
def get_dispatches(
    limit: int = Query(50, ge=1, le=500, description="Max dispatch records to return"),
    incident_type: Optional[str] = Query(
        None, description="Optional filter by incident type"
    ),
    engine: SurveillanceEngine = Depends(get_engine),
):
    """Returns recent emergency alerts and hospital/police routing records from the JSON dispatch log."""
    records = []
    if engine.dispatcher:
        records = engine.dispatcher.get_recent_dispatches(limit=limit)
    elif os.path.exists(DISPATCH_LOG_PATH):
        try:
            with open(DISPATCH_LOG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                loaded = [loaded]
            records = loaded[-limit:][::-1]
        except Exception as e:
            print(f"[RoutesIncidents] Error reading dispatch log fallback: {e}")

    if incident_type:
        records = [
            r
            for r in records
            if incident_type.upper() in r.get("incident", {}).get("type", "").upper()
        ]

    return {"total_returned": len(records), "dispatches": records}


@router.get("/snapshots", summary="List captured incident snapshots")
def list_snapshots(
    limit: int = Query(50, ge=1, le=200, description="Max snapshots to list"),
):
    """Returns a list of saved snapshot images with file details and download URLs."""
    if not os.path.exists(SNAPSHOT_DIR):
        return {"total_snapshots": 0, "snapshots": []}

    files = [
        f
        for f in os.listdir(SNAPSHOT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files.sort(
        key=lambda f: os.path.getmtime(os.path.join(SNAPSHOT_DIR, f)), reverse=True
    )

    result = []
    for f in files[:limit]:
        p = os.path.join(SNAPSHOT_DIR, f)
        stat = os.stat(p)
        result.append(
            {
                "filename": f,
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "url": f"/api/snapshots/{f}",
            }
        )

    return {"total_snapshots": len(files), "returned": len(result), "snapshots": result}


@router.get(
    "/snapshots/{filename}", summary="Download or view a specific snapshot image"
)
def get_snapshot_file(filename: str):
    """Serves a captured incident JPEG or PNG image file."""
    # Sanitize filename against directory traversal
    clean_filename = os.path.basename(filename)
    filepath = os.path.join(SNAPSHOT_DIR, clean_filename)

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404, detail=f"Snapshot '{clean_filename}' not found"
        )

    media_type = (
        "image/png" if clean_filename.lower().endswith(".png") else "image/jpeg"
    )
    return FileResponse(filepath, media_type=media_type, filename=clean_filename)
