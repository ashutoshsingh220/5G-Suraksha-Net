"""Agent orchestrator service for 5G Suraksha-Net.

Orchestrates the logical agent modules (Incident, Severity, Location, Evidence,
Response, and Network) to produce a unified operator briefing.
Maintains a per-incident assessment cache to prevent redundant evaluations.
"""
from __future__ import annotations

import threading
from typing import Any

from suraksha.agents import tools
from suraksha.agents.providers import AgentProvider, DeterministicAgentProvider
from suraksha.agents.schemas import AgentAssessment
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


class AgentOrchestrator:
    """Orchestrates structured tool retrieval and agent reasoning."""

    def __init__(self, provider: AgentProvider | None = None) -> None:
        self.provider = provider or DeterministicAgentProvider()
        self._cache: dict[str, AgentAssessment] = {}
        self._lock = threading.Lock()

    def get_assessment(
        self,
        incident_id: str,
        force_refresh: bool = False,
        pipeline_manager: Any = None,
    ) -> AgentAssessment | None:
        """Generate or retrieve cached agent assessment for an incident."""
        with self._lock:
            if not force_refresh and incident_id in self._cache:
                return self._cache[incident_id]

        # 1. Retrieve authoritative incident report via read-only tool
        incident = tools.get_incident(incident_id, pipeline_manager=pipeline_manager)
        if incident is None:
            return None

        # 2. Retrieve response plan
        plan = tools.get_response_plan(incident_id, incident=incident, pipeline_manager=pipeline_manager)

        # 3. Retrieve geospatial location
        loc = tools.get_location(incident_id, incident=incident, pipeline_manager=pipeline_manager)

        # 4. Retrieve nearby emergency directory resources
        resources = tools.get_nearby_resources(incident_id, plan=plan, incident=incident, pipeline_manager=pipeline_manager)

        # 5. Retrieve forensic evidence metadata
        evidence_meta = tools.get_evidence_metadata(incident_id, incident=incident, pipeline_manager=pipeline_manager)

        # 6. Retrieve real-time network policy state
        network_state = tools.get_network_policy()

        # 7. Generate structured assessment via provider
        assessment = self.provider.generate_assessment(
            incident=incident,
            response_plan=plan,
            location=loc,
            resources=resources,
            evidence_meta=evidence_meta,
            network_state=network_state,
        )

        with self._lock:
            self._cache[incident_id] = assessment

        return assessment

    def clear_cache(self, incident_id: str | None = None) -> None:
        """Clear assessment cache for a specific incident or all incidents."""
        with self._lock:
            if incident_id:
                self._cache.pop(incident_id, None)
            else:
                self._cache.clear()
