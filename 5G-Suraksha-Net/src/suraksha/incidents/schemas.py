"""Incident data contracts — the integration surface for the wider 5G Suraksha-Net system.

Teammates' modules (fire, accident, ambulance, orchestration) consume these
Pydantic models via REST/WebSocket. Bump SCHEMA_VERSION on breaking changes.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from suraksha.location.schemas import Location, LocationSource

SCHEMA_VERSION = "1.0"
SOURCE_MODULE = "crowd_fight"  # identifies this module in the larger system


class IncidentStatus(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    RECORDING_POST_EVENT = "recording_post_event"
    FINALIZED = "finalized"


class IncidentType(str, Enum):
    FIGHT = "fight"
    CROWD_DENSITY_HIGH = "crowd_density_high"
    CROWD_DENSITY_CRITICAL = "crowd_density_critical"
    CROWD_RAPID_GROWTH = "crowd_rapid_growth"
    CROWD_PANIC = "crowd_panic"
    WEAPON = "weapon"
    ARMED_FIGHT = "armed_fight"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, arr) -> "BBox":
        x1, y1, x2, y2 = [float(v) for v in arr]
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


class Evidence(BaseModel):
    snapshot_path: str | None = None
    clip_path: str | None = None
    frame_idx: int | None = None


class IncidentReport(BaseModel):
    """One verified incident. Serialized to JSON for the orchestration layer."""
    schema_version: str = SCHEMA_VERSION
    source_module: str = SOURCE_MODULE
    incident_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: IncidentStatus = IncidentStatus.VERIFIED
    camera_id: str
    incident_type: IncidentType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    start_time: datetime
    end_time: datetime | None = None
    zone: str | None = None            # crowd zone name, if applicable
    track_ids: list[int] = Field(default_factory=list)  # ByteTrack IDs involved
    bbox: BBox | None = None           # region of interest in frame pixels
    person_count: int | None = None    # crowd size at incident time
    details: dict = Field(default_factory=dict)
    evidence: Evidence = Field(default_factory=Evidence)
    location: Location | None = None
    finalized_at: datetime | None = None
    source_mode: str = "REAL"

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(timezone.utc)
