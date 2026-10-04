"""Data contracts and schemas for Phase 3C Deterministic Response Planner.

Defines typed schemas for emergency actions, external resources (police and
hospitals), human-approval gates, and complete incident response plans.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from suraksha.incidents.schemas import Evidence, IncidentReport, IncidentType, Severity
from suraksha.location.schemas import Location


class ActionType(str, Enum):
    """Supported emergency action types for Phase 3C."""
    POLICE_SECURITY = "POLICE_SECURITY"
    MEDICAL_ASSISTANCE = "MEDICAL_ASSISTANCE"


class ActionPriority(str, Enum):
    """Priority level for generated response actions."""
    LOW = "low"
    MEDIUM = "medium"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseStatus(str, Enum):
    """Deterministic status of an emergency action or overall response plan."""
    RECOMMENDED = "RECOMMENDED"
    AWAITING_HUMAN_APPROVAL = "AWAITING_HUMAN_APPROVAL"
    NO_ACTION = "NO_ACTION"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"


class EmergencyResource(BaseModel):
    """Model representing a verified emergency facility (Police or Hospital).

    IMPORTANT: Distances are straight-line geographic distances from the incident location.
    They are NOT road driving distances and NOT estimated travel times (ETAs).
    """
    name: str
    category: str | None = None
    place_type: str | None = None
    distance_km: float | None = None
    distance_m: float | None = None
    formatted_address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    phone_number: str | None = None
    rating: float | None = None
    user_ratings_count: int | None = None
    website: str | None = None
    place_id: str | None = None
    distance_type: str = "straight_line"

    @model_validator(mode="before")
    @classmethod
    def _compute_distance_units(cls, values: Any) -> Any:
        if isinstance(values, dict):
            d_km = values.get("distance_km")
            d_m = values.get("distance_m")
            if d_km is not None and d_m is None:
                try:
                    values["distance_m"] = round(float(d_km) * 1000.0, 1)
                except (ValueError, TypeError):
                    pass
            elif d_m is not None and d_km is None:
                try:
                    values["distance_km"] = round(float(d_m) / 1000.0, 3)
                except (ValueError, TypeError):
                    pass
        return values


class EmergencyResourceDirectory(BaseModel):
    """Container holding pre-discovered, verified emergency facilities."""
    hospitals: list[EmergencyResource] = Field(default_factory=list)
    police_stations: list[EmergencyResource] = Field(default_factory=list)
    location: Any | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmergencyResourceDirectory":
        """Instantiate directory from dictionary matching query_nearby_locations format."""
        if not isinstance(data, dict):
            return cls()

        responders = data.get("emergency_responders", data)
        hospitals_raw = responders.get("hospitals", []) if isinstance(responders, dict) else []
        police_raw = responders.get("police_stations", []) if isinstance(responders, dict) else []

        hospitals: list[EmergencyResource] = []
        for h in hospitals_raw:
            if isinstance(h, dict) and h.get("name"):
                try:
                    hospitals.append(EmergencyResource(**h))
                except Exception:
                    pass

        police: list[EmergencyResource] = []
        for p in police_raw:
            if isinstance(p, dict) and p.get("name"):
                try:
                    police.append(EmergencyResource(**p))
                except Exception:
                    pass

        loc_obj = None
        demo_loc = data.get("demonstration_location")
        if isinstance(demo_loc, dict):
            try:
                from suraksha.location.schemas import Location, LocationSource
                loc_obj = Location(
                    name=demo_loc.get("query") or demo_loc.get("formatted_address") or "Yashobhoomi, Sector 25 Dwarka",
                    latitude=demo_loc.get("latitude"),
                    longitude=demo_loc.get("longitude"),
                    source=LocationSource.GEOCODED_DEMO,
                )
            except Exception:
                pass

        return cls(hospitals=hospitals, police_stations=police, location=loc_obj)

    @classmethod
    def from_file(cls, path: str | Path) -> "EmergencyResourceDirectory":
        """Load directory from JSON file on disk."""
        p = Path(path)
        if not p.is_file():
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            return cls()


class ResponseAction(BaseModel):
    """A single deterministic recommendation within an incident response plan.

    Every emergency response action must be approved by a human operator before dispatch.
    """
    action_type: ActionType
    target: str
    priority: str
    reason: str
    human_approval_required: bool = True
    status: ResponseStatus = ResponseStatus.AWAITING_HUMAN_APPROVAL
    recommended_resource: EmergencyResource | None = None


class IncidentResponse(BaseModel):
    """Structured, machine-readable emergency response plan for a verified incident.

    Preserves original incident information (incident_id, incident_type, severity,
    location, evidence) without recalculating or altering them.
    """
    schema_version: str = "1.0"
    incident_id: str
    incident_type: IncidentType | str
    severity: Severity | str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    location: Location | None = None
    response_actions: list[ResponseAction] = Field(default_factory=list)
    police_resources: list[EmergencyResource] = Field(default_factory=list)
    medical_resources: list[EmergencyResource] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)
    human_approval_required: bool = True
    status: ResponseStatus = ResponseStatus.AWAITING_HUMAN_APPROVAL
    warning: str | None = None
    email_status: str | None = None
    disclaimer: str = (
        "Operational decision support only. Straight-line geographic distance; "
        "not road driving ETA. External emergency response requires operator authorization."
    )
