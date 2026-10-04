"""Deterministic response policy table for 5G Suraksha-Net Phase 3C.

Evaluates an IncidentReport according to configured policy rules:
- FIGHT: Police/security REQUIRED. Medical assistance ONLY IF explicit injury/person-down indicator present.
- WEAPON: Police/security REQUIRED. Medical assistance ONLY IF explicit injury indicator present.
- ARMED_FIGHT: Police/security REQUIRED and Medical assistance REQUIRED (CRITICAL escalation).
- CROWD_PANIC: Police/security REQUIRED. Medical assistance ONLY IF explicit injury indicator present.
- CROWD_DENSITY: Routine monitoring only; no automatic emergency actions.
- UNSUPPORTED: Safe no-action plan with structured reason.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from suraksha.incidents.schemas import IncidentReport, IncidentType, Severity
from suraksha.response.schemas import ActionPriority, ActionType


def check_has_injury(incident: IncidentReport) -> tuple[bool, str | None]:
    """Inspect incident details for explicit injury, person-down, or medical-need indicators.

    Does NOT infer or fabricate injuries; relies strictly on existing structured fields.
    """
    details = incident.details if isinstance(incident.details, dict) else {}

    # Check direct boolean or numeric flags in details
    injury_keys = ("injury", "injuries", "person_down", "casualty", "medical_need", "has_injury")
    for key in injury_keys:
        val = details.get(key)
        if val is True:
            return True, f"Details flag '{key}' is True"
        if isinstance(val, (int, float)) and val > 0:
            return True, f"Details flag '{key}' has positive count ({val})"
        if isinstance(val, str) and val.strip().lower() in ("true", "yes", "confirmed", "critical", "active"):
            return True, f"Details flag '{key}' is '{val}'"

    # Check medical details sub-dict if present
    medical_ctx = details.get("medical")
    if isinstance(medical_ctx, dict):
        if medical_ctx.get("required") is True or medical_ctx.get("injuries_reported") is True:
            return True, "Details sub-context 'medical' reports active medical requirement"

    return False, None


class PolicyDecision(BaseModel):
    """Structured result of evaluating the deterministic response policy."""
    police_required: bool = False
    medical_required: bool = False
    police_priority: str = "high"
    medical_priority: str = "moderate"
    police_reason: str = ""
    medical_reason: str = ""
    is_supported: bool = True
    policy_notes: str = ""


def evaluate_response_policy(incident: IncidentReport) -> PolicyDecision:
    """Evaluate deterministic policy table against an IncidentReport.

    Pure deterministic decision logic:
    - Zero network or API calls
    - Zero AI / LLM inference
    - Zero modification of incident severity
    """
    itype = incident.incident_type
    sev = incident.severity
    has_injury, injury_detail = check_has_injury(incident)

    sev_str = sev.value if isinstance(sev, Severity) else str(sev)

    # 1. ARMED_FIGHT (CRITICAL)
    if itype == IncidentType.ARMED_FIGHT:
        return PolicyDecision(
            police_required=True,
            medical_required=True,
            police_priority=sev_str,
            medical_priority=sev_str,
            police_reason="Armed fight combines a confirmed weapon with a physical fight; police/security response is required.",
            medical_reason="Armed fight indicates high risk of critical trauma; medical assistance readiness is required.",
            is_supported=True,
            policy_notes="Armed fight requires immediate joint police and medical dispatch readiness.",
        )

    # 2. FIGHT (MODERATE / HIGH)
    if itype == IncidentType.FIGHT:
        med_req = has_injury
        if med_req:
            med_reason = f"Physical fight with explicit medical/injury evidence ({injury_detail}); medical assistance required."
            med_priority = "high" if sev == Severity.HIGH else "moderate"
        else:
            med_reason = "No explicit injury/person-down indicator is present; medical response was not automatically added."
            med_priority = "low"

        return PolicyDecision(
            police_required=True,
            medical_required=med_req,
            police_priority=sev_str,
            medical_priority=med_priority,
            police_reason="Physical fight detected; requires police/security response.",
            medical_reason=med_reason,
            is_supported=True,
            policy_notes="Physical fight policy: police required; medical requires explicit injury evidence.",
        )

    # 3. WEAPON (HIGH / CRITICAL)
    if itype == IncidentType.WEAPON:
        med_req = has_injury
        if med_req:
            med_reason = f"Confirmed weapon with explicit medical/injury evidence ({injury_detail}); medical assistance required."
            med_priority = "critical" if sev == Severity.CRITICAL else "high"
        else:
            med_reason = "No explicit injury/person-down indicator is present; medical response was not automatically added."
            med_priority = "low"

        return PolicyDecision(
            police_required=True,
            medical_required=med_req,
            police_priority=sev_str,
            medical_priority=med_priority,
            police_reason="Confirmed weapon incident requires police/security response.",
            medical_reason=med_reason,
            is_supported=True,
            policy_notes="Confirmed weapon policy: police required; hospital response added only on explicit medical need.",
        )

    # 4. CROWD_PANIC (HIGH)
    if itype == IncidentType.CROWD_PANIC:
        med_req = has_injury
        if med_req:
            med_reason = f"Crowd panic with explicit injury indicator ({injury_detail}); medical assistance required."
            med_priority = "high"
        else:
            med_reason = "No explicit injury/person-down indicator is present; medical response was not automatically added."
            med_priority = "low"

        return PolicyDecision(
            police_required=True,
            medical_required=med_req,
            police_priority=sev_str,
            medical_priority=med_priority,
            police_reason="Crowd panic detected; requires police/security response for crowd control and evacuation.",
            medical_reason=med_reason,
            is_supported=True,
            policy_notes="Crowd panic policy: police required for dispersal/evacuation; medical conditional on injury.",
        )

    # 5. CROWD_DENSITY (Routine monitoring - HIGH, CRITICAL, RAPID_GROWTH)
    if itype in (
        IncidentType.CROWD_DENSITY_HIGH,
        IncidentType.CROWD_DENSITY_CRITICAL,
        IncidentType.CROWD_RAPID_GROWTH,
    ):
        return PolicyDecision(
            police_required=False,
            medical_required=False,
            police_priority="low",
            medical_priority="low",
            police_reason="",
            medical_reason="",
            is_supported=True,
            policy_notes="Crowd density/growth is under routine monitoring; no emergency intervention policy configured.",
        )

    # 6. Unconfigured / Unknown incident types
    return PolicyDecision(
        police_required=False,
        medical_required=False,
        police_priority="low",
        medical_priority="low",
        police_reason="",
        medical_reason="",
        is_supported=False,
        policy_notes="No response policy configured for this incident state.",
    )
