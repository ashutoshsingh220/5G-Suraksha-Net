"""Data contracts and schemas for Phase 3F IMC Demo & Controlled Scenario Mode.

Defines schemas for deterministic scenarios, simulation timelines, and demo status.
Strictly separates DEMO incidents from production real-world records.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

from suraksha.incidents.schemas import IncidentReport, IncidentType, Severity


class DemoScenarioId(str, Enum):
    """Supported deterministic demonstration scenarios."""
    WEAPON = "WEAPON"
    CROWD_PANIC = "CROWD_PANIC"
    ARMED_FIGHT = "ARMED_FIGHT"
    NORMAL = "NORMAL"


class DemoSimulationStep(BaseModel):
    """Simulation step on the controlled demonstration timeline."""
    step_id: str
    timestamp_label: str  # e.g. "T+00.0s", "T+01.0s"
    title: str
    details: str
    status: str = "COMPLETED"


class DemoScenarioMeta(BaseModel):
    """Metadata describing a selectable demonstration scenario."""
    id: DemoScenarioId
    name: str
    description: str
    incident_type: IncidentType | None = None
    severity: Severity | None = None
    expected_network_policy: str
    expected_network_priority: str
    location_name: str
    video_choices: list[dict[str, str]] = Field(default_factory=list)


class DemoStatusResponse(BaseModel):
    """Real-time operational status of demonstration mode."""
    is_demo_active: bool = False
    active_scenario: DemoScenarioId | str | None = None
    source_mode: str = "REAL"
    incident: IncidentReport | None = None
    simulation_timeline: list[DemoSimulationStep] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    video_file: str | None = None
    video_choice: str | None = None
    disclaimer: str = ""
