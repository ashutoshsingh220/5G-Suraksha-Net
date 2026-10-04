"""Agentic Emergency Orchestration Subsystem for 5G Suraksha-Net."""
from __future__ import annotations

from suraksha.agents.orchestrator import AgentOrchestrator
from suraksha.agents.providers import AgentProvider, DeterministicAgentProvider, LLMAgentProvider
from suraksha.agents.schemas import (
    ActionStep,
    ActionStepStatus,
    AgentAssessment,
    AgentProviderType,
    DataSourceTraceability,
    EvidenceAssessment,
    LocationAssessment,
    NetworkAssessment,
    ResponseAssessment,
    SeverityAssessment,
)

__all__ = [
    "ActionStep",
    "ActionStepStatus",
    "AgentAssessment",
    "AgentOrchestrator",
    "AgentProvider",
    "AgentProviderType",
    "DataSourceTraceability",
    "DeterministicAgentProvider",
    "EvidenceAssessment",
    "LLMAgentProvider",
    "LocationAssessment",
    "NetworkAssessment",
    "ResponseAssessment",
    "SeverityAssessment",
    "get_agent_orchestrator",
]

_instance: AgentOrchestrator | None = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Get or initialize global AgentOrchestrator singleton."""
    global _instance
    if _instance is None:
        _instance = AgentOrchestrator()
    return _instance
