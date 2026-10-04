"""Event-driven network policy service for 5G Suraksha-Net.

Maps detected incidents and their severity directly into application-level
transmission priority tiers. Emits audit instrumentation events on escalation
and de-escalation. Never fabricates physical 5G slice control or cellular core parameters.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Sequence

from suraksha.incidents.schemas import IncidentReport, IncidentStatus, Severity
from suraksha.logging_utils import get_logger
from suraksha.network.schemas import (
    ApplicationPriority,
    NetworkEventMetric,
    NetworkEventsResponse,
    NetworkPolicy,
    NetworkPolicyState,
    PolicyEventType,
)

log = get_logger(__name__)

SEVERITY_WEIGHTS: dict[Severity, int] = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MODERATE: 2,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
}


class NetworkPolicyService:
    """Computes truthful, event-driven network policy from incident lifecycle."""

    def __init__(self, active_window_seconds: float = 60.0, max_events: int = 200) -> None:
        self.active_window_seconds = active_window_seconds
        self._events: deque[NetworkEventMetric] = deque(maxlen=max_events)
        self._active_incidents: dict[str, IncidentReport] = {}
        self._current_policy: NetworkPolicy = NetworkPolicy.NORMAL
        self._current_priority: ApplicationPriority = ApplicationPriority.ROUTINE
        self._lock = threading.Lock()

        # Record initial state
        init_evt = NetworkEventMetric(
            event_type=PolicyEventType.POLICY_INITIALIZED,
            policy=NetworkPolicy.NORMAL,
            priority=ApplicationPriority.ROUTINE,
            incident_id=None,
            details="Network policy engine initialized in NORMAL standby mode.",
            actual_network_control=False,
        )
        self._events.append(init_evt)

    def reset(self) -> None:
        """Reset service state (primarily for tests)."""
        with self._lock:
            self._events.clear()
            self._active_incidents.clear()
            self._current_policy = NetworkPolicy.NORMAL
            self._current_priority = ApplicationPriority.ROUTINE
            init_evt = NetworkEventMetric(
                event_type=PolicyEventType.POLICY_INITIALIZED,
                policy=NetworkPolicy.NORMAL,
                priority=ApplicationPriority.ROUTINE,
                incident_id=None,
                details="Network policy engine initialized in NORMAL standby mode.",
                actual_network_control=False,
            )
            self._events.append(init_evt)

    def on_incident(self, report: IncidentReport) -> NetworkPolicyState:
        """Process incoming incident report from pipeline bus and update policy."""
        with self._lock:
            self._active_incidents[report.incident_id] = report

            # Determine assigned priority for this specific incident
            assigned_prio = ApplicationPriority.ROUTINE
            sev = getattr(report, "severity", Severity.LOW)
            if sev == Severity.CRITICAL:
                assigned_prio = ApplicationPriority.CRITICAL
            elif sev in (Severity.HIGH, Severity.MODERATE, Severity.MEDIUM):
                assigned_prio = ApplicationPriority.HIGH

            itype_val = getattr(report.incident_type, "value", str(report.incident_type))
            sev_val = getattr(report.severity, "value", str(report.severity))

            # Record priority assignment event
            assign_evt = NetworkEventMetric(
                event_type=PolicyEventType.INCIDENT_PRIORITY_ASSIGNED,
                policy=self._current_policy,
                priority=assigned_prio,
                incident_id=report.incident_id,
                details=f"Assigned transmission priority {assigned_prio.value} for incident {itype_val} ({sev_val}).",
                actual_network_control=False,
            )
            self._events.append(assign_evt)

        # Re-evaluate global policy with updated incidents
        return self.get_policy_state()

    def record_evidence_event(self, incident_id: str, details: str = "Forensic evidence package ready for transmission.") -> None:
        """Record forensic evidence generation event."""
        with self._lock:
            evt = NetworkEventMetric(
                event_type=PolicyEventType.EVIDENCE_READY,
                policy=self._current_policy,
                priority=self._current_priority,
                incident_id=incident_id,
                details=details,
                actual_network_control=False,
            )
            self._events.append(evt)

    def _prune_expired_incidents_locked(self, now: float) -> None:
        """Remove incidents that have aged past the active window or are finalized and old."""
        to_delete = []
        for inc_id, inc in self._active_incidents.items():
            start_dt = inc.start_time
            if isinstance(start_dt, datetime):
                start_ts = start_dt.timestamp()
            else:
                start_ts = now
            age = now - start_ts

            # If incident is past active window, or finalized and past 15 seconds, prune
            if age > self.active_window_seconds or (inc.status == IncidentStatus.FINALIZED and age > 15.0):
                to_delete.append(inc_id)

        for inc_id in to_delete:
            del self._active_incidents[inc_id]

    def get_policy_state(
        self,
        candidate_incidents: Sequence[IncidentReport] | None = None,
        now: float | None = None,
    ) -> NetworkPolicyState:
        """Evaluate active incidents and return current truthful network policy state."""
        now_ts = now if now is not None else time.time()

        with self._lock:
            # Sync candidate incidents if provided
            if candidate_incidents is not None:
                for rep in candidate_incidents:
                    inc_id = getattr(rep, "incident_id", None)
                    if inc_id and inc_id not in self._active_incidents:
                        self._active_incidents[inc_id] = rep

            self._prune_expired_incidents_locked(now_ts)

            # Find active incidents
            active_list = list(self._active_incidents.values())

            # Sort active incidents by severity weight descending
            def _weight(inc: IncidentReport) -> int:
                sev = getattr(inc, "severity", Severity.LOW)
                return SEVERITY_WEIGHTS.get(sev, 1)

            active_list.sort(key=_weight, reverse=True)

            target_policy = NetworkPolicy.NORMAL
            target_priority = ApplicationPriority.ROUTINE
            active_id = None
            active_type = None
            active_sev = None
            reason = "Routine monitoring active — no verified critical or elevated incidents."

            if active_list:
                top_incident = active_list[0]
                sev = getattr(top_incident, "severity", Severity.LOW)
                top_type = getattr(top_incident.incident_type, "value", str(top_incident.incident_type))
                top_sev_str = getattr(sev, "value", str(sev)).upper()

                if sev == Severity.CRITICAL:
                    target_policy = NetworkPolicy.CRITICAL
                    target_priority = ApplicationPriority.CRITICAL
                    active_id = top_incident.incident_id
                    active_type = top_type
                    active_sev = top_sev_str
                    reason = f"Active CRITICAL incident [{top_type}] (#{top_incident.incident_id}) requires maximum transmission priority."
                elif sev in (Severity.HIGH, Severity.MODERATE, Severity.MEDIUM):
                    target_policy = NetworkPolicy.EVENT
                    target_priority = ApplicationPriority.HIGH
                    active_id = top_incident.incident_id
                    active_type = top_type
                    active_sev = top_sev_str
                    reason = f"Active elevated incident [{top_type}] (#{top_incident.incident_id}) assigned high transmission priority."

            # Check for policy transition
            prev_policy = self._current_policy
            if target_policy != prev_policy:
                # Check if escalated or deescalated
                prev_rank = 3 if prev_policy == NetworkPolicy.CRITICAL else (2 if prev_policy == NetworkPolicy.EVENT else 1)
                curr_rank = 3 if target_policy == NetworkPolicy.CRITICAL else (2 if target_policy == NetworkPolicy.EVENT else 1)

                evt_type = PolicyEventType.POLICY_ESCALATED if curr_rank > prev_rank else PolicyEventType.POLICY_DEESCALATED
                transition_details = (
                    f"Policy transitioned from {prev_policy.value} to {target_policy.value} "
                    f"(priority: {target_priority.value}). {reason}"
                )

                transition_evt = NetworkEventMetric(
                    event_type=evt_type,
                    policy=target_policy,
                    priority=target_priority,
                    incident_id=active_id,
                    details=transition_details,
                    actual_network_control=False,
                )
                self._events.append(transition_evt)
                log.info("NETWORK POLICY: %s -> %s [reason: %s]", prev_policy.value, target_policy.value, reason)

            self._current_policy = target_policy
            self._current_priority = target_priority

            return NetworkPolicyState(
                policy=target_policy,
                priority=target_priority,
                active_incident_id=active_id,
                active_incident_type=active_type,
                active_severity=active_sev,
                reason=reason,
                actual_network_control=False,
                control_plane_connected=False,
                policy_scope="APPLICATION_LAYER",
                slice_allocated="DEFAULT_BE",
                measurement_source="APPLICATION_POLICY",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def get_events(self, limit: int = 50) -> NetworkEventsResponse:
        """Return bounded audit trail of network policy events (most recent first)."""
        with self._lock:
            evts = list(self._events)
            evts.reverse()
            sliced = evts[:limit]
            return NetworkEventsResponse(count=len(sliced), events=sliced)
