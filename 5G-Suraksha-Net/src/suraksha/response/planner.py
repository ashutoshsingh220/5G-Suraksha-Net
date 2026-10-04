"""Deterministic Response Planner for 5G Suraksha-Net Phase 3C.

Converts an already-detected and already-fused IncidentReport into a structured,
machine-readable IncidentResponse plan using pre-discovered emergency resources.

Purely deterministic:
- Zero HTTP or Google Maps calls
- Zero LLMs or agent frameworks
- Zero model inference
- Zero modification of incident severity
- Strictly requires human approval for all emergency actions
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from suraksha.config import PROJECT_ROOT
from suraksha.incidents.schemas import IncidentReport
from suraksha.logging_utils import get_logger
from suraksha.response.policy import evaluate_response_policy
from suraksha.response.schemas import (
    ActionType,
    EmergencyResource,
    EmergencyResourceDirectory,
    IncidentResponse,
    ResponseAction,
    ResponseStatus,
)

log = get_logger(__name__)

DEFAULT_RESOURCE_PATH = PROJECT_ROOT / "outputs" / "phase3" / "nearby_emergency_locations_yashobhoomi.json"


class ResponsePlanner:
    """Deterministic, policy-driven emergency response planner."""

    def __init__(
        self,
        resource_directory: EmergencyResourceDirectory | None = None,
        resource_file: str | Path | None = None,
    ) -> None:
        if resource_directory is not None:
            self.resource_directory = resource_directory
        else:
            path = Path(resource_file) if resource_file else DEFAULT_RESOURCE_PATH
            if path.is_file():
                self.resource_directory = EmergencyResourceDirectory.from_file(path)
            else:
                self.resource_directory = EmergencyResourceDirectory()

    @staticmethod
    def _sort_resources(resources: list[EmergencyResource]) -> list[EmergencyResource]:
        """Sort resources deterministically by straight-line distance, breaking ties by name.

        Distances are straight-line geographic distances, not driving routes or ETAs.
        """
        return sorted(
            resources,
            key=lambda r: (
                r.distance_km if r.distance_km is not None else float("inf"),
                r.name or "",
            ),
        )

    def plan(
        self,
        incident: IncidentReport,
        resources: EmergencyResourceDirectory | dict[str, Any] | None = None,
    ) -> IncidentResponse:
        """Generate a deterministic IncidentResponse plan for a verified incident.

        Args:
            incident: Verified IncidentReport from the incident manager or fusion engine.
            resources: Optional override for emergency resources (police/hospitals).
                       If omitted, uses the planner's pre-loaded resource directory.

        Returns:
            Structured IncidentResponse machine-readable plan.
        """
        # 1. Resolve emergency resource directory
        if resources is not None:
            if isinstance(resources, EmergencyResourceDirectory):
                active_res = resources
            elif isinstance(resources, dict):
                active_res = EmergencyResourceDirectory.from_dict(resources)
            else:
                log.warning("Unrecognized resources type (%s); using pre-loaded directory", type(resources))
                active_res = self.resource_directory
        else:
            active_res = self.resource_directory

        # 2. Deterministically sort available police and hospital resources
        police_sorted = self._sort_resources(active_res.police_stations)
        medical_sorted = self._sort_resources(active_res.hospitals)

        # 3. Evaluate deterministic policy table
        decision = evaluate_response_policy(incident)

        actions: list[ResponseAction] = []
        warnings: list[str] = []

        # 4. Construct Police/Security Response Action if required
        if decision.police_required:
            primary_police = police_sorted[0] if police_sorted else None
            if primary_police is None:
                warnings.append("Required police/security resource is unavailable in local directory")
                police_status = ResponseStatus.RESOURCE_UNAVAILABLE
            else:
                police_status = ResponseStatus.AWAITING_HUMAN_APPROVAL

            actions.append(
                ResponseAction(
                    action_type=ActionType.POLICE_SECURITY,
                    target="POLICE_SECURITY_DISPATCH",
                    priority=decision.police_priority,
                    reason=decision.police_reason,
                    human_approval_required=True,
                    status=police_status,
                    recommended_resource=primary_police,
                )
            )

        # 5. Construct Medical Assistance Response Action if required
        if decision.medical_required:
            primary_medical = medical_sorted[0] if medical_sorted else None
            if primary_medical is None:
                warnings.append("Required medical/hospital resource is unavailable in local directory")
                medical_status = ResponseStatus.RESOURCE_UNAVAILABLE
            else:
                medical_status = ResponseStatus.AWAITING_HUMAN_APPROVAL

            actions.append(
                ResponseAction(
                    action_type=ActionType.MEDICAL_ASSISTANCE,
                    target="MEDICAL_TRAUMA_ASSISTANCE",
                    priority=decision.medical_priority,
                    reason=decision.medical_reason,
                    human_approval_required=True,
                    status=medical_status,
                    recommended_resource=primary_medical,
                )
            )

        # 6. Determine overall response plan status
        if not decision.is_supported:
            overall_status = ResponseStatus.NO_ACTION
            warnings.append("No response policy configured for this incident state")
        elif not actions:
            overall_status = ResponseStatus.NO_ACTION
            if decision.policy_notes:
                warnings.append(decision.policy_notes)
        elif any(a.status == ResponseStatus.RESOURCE_UNAVAILABLE for a in actions):
            overall_status = ResponseStatus.RESOURCE_UNAVAILABLE
        else:
            overall_status = ResponseStatus.AWAITING_HUMAN_APPROVAL

        warning_str = "; ".join(warnings) if warnings else None

        # 7. Construct IncidentResponse preserving original incident fields
        return IncidentResponse(
            schema_version="1.0",
            incident_id=incident.incident_id,
            incident_type=incident.incident_type,
            severity=incident.severity,
            timestamp=incident.start_time,
            location=incident.location,
            response_actions=actions,
            police_resources=police_sorted,
            medical_resources=medical_sorted,
            evidence=incident.evidence,
            human_approval_required=True,
            status=overall_status,
            warning=warning_str,
        )
