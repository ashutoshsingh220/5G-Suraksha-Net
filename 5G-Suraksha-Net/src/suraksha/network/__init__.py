"""Network policy and event-driven transmission intelligence subsystem."""
from __future__ import annotations

from suraksha.network.schemas import (
    ApplicationPriority,
    NetworkEventMetric,
    NetworkEventsResponse,
    NetworkPolicy,
    NetworkPolicyState,
    PolicyEventType,
)
from suraksha.network.service import NetworkPolicyService

__all__ = [
    "ApplicationPriority",
    "NetworkEventMetric",
    "NetworkEventsResponse",
    "NetworkPolicy",
    "NetworkPolicyState",
    "PolicyEventType",
    "NetworkPolicyService",
    "get_network_policy_service",
]

_instance: NetworkPolicyService | None = None


def get_network_policy_service() -> NetworkPolicyService:
    """Get or instantiate the global NetworkPolicyService singleton."""
    global _instance
    if _instance is None:
        _instance = NetworkPolicyService()
    return _instance
