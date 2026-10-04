"""Controlled IMC Demonstration & Scenario Simulation Subsystem."""
from __future__ import annotations

from suraksha.demo.schemas import (
    DemoScenarioId,
    DemoScenarioMeta,
    DemoSimulationStep,
    DemoStatusResponse,
)
from suraksha.demo.service import DemoScenarioService

__all__ = [
    "DemoScenarioId",
    "DemoScenarioMeta",
    "DemoSimulationStep",
    "DemoStatusResponse",
    "DemoScenarioService",
    "get_demo_service",
]

_instance: DemoScenarioService | None = None


def get_demo_service() -> DemoScenarioService:
    """Get or initialize global DemoScenarioService singleton."""
    global _instance
    if _instance is None:
        _instance = DemoScenarioService()
    return _instance
