"""Network and 5G policy data contracts for 5G Suraksha-Net.

Defines schemas for event-driven application-level network policy,
transmission prioritization, and policy event metrics.
Per safety rules, strictly declares when physical network control is unattached.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class NetworkPolicy(str, Enum):
    """Application-level transmission policy state."""
    NORMAL = "NORMAL"
    EVENT = "EVENT"
    CRITICAL = "CRITICAL"


class ApplicationPriority(str, Enum):
    """Data and streaming transmission priority tier."""
    ROUTINE = "ROUTINE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyEventType(str, Enum):
    """Audit event types for policy transitions and transmission events."""
    POLICY_INITIALIZED = "POLICY_INITIALIZED"
    POLICY_ESCALATED = "POLICY_ESCALATED"
    POLICY_DEESCALATED = "POLICY_DEESCALATED"
    INCIDENT_PRIORITY_ASSIGNED = "INCIDENT_PRIORITY_ASSIGNED"
    EVIDENCE_READY = "EVIDENCE_READY"


class NetworkPolicyState(BaseModel):
    """Truthful real-time status of application-level network policy."""
    policy: NetworkPolicy = NetworkPolicy.NORMAL
    priority: ApplicationPriority = ApplicationPriority.ROUTINE
    active_incident_id: str | None = None
    active_incident_type: str | None = None
    active_severity: str | None = None
    reason: str = "Routine monitoring active — no verified critical or elevated incidents."
    actual_network_control: bool = False
    control_plane_connected: bool = False
    policy_scope: str = "APPLICATION_LAYER"
    slice_allocated: str = "DEFAULT_BE"
    measurement_source: str = "APPLICATION_POLICY"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NetworkEventMetric(BaseModel):
    """Individual policy audit / instrumentation metric."""
    event_id: str = Field(default_factory=lambda: f"pe-{uuid.uuid4().hex[:8]}")
    event_type: PolicyEventType
    policy: NetworkPolicy
    priority: ApplicationPriority
    incident_id: str | None = None
    details: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    actual_network_control: bool = False


class NetworkEventsResponse(BaseModel):
    """Paginated or bounded list of recent policy events."""
    count: int
    events: list[NetworkEventMetric]
