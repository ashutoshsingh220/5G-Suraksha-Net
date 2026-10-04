"""Tests for event-driven network policy service and REST endpoints."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from suraksha.api.app import app
from suraksha.incidents.schemas import IncidentReport, IncidentStatus, IncidentType, Severity
from suraksha.network import (
    ApplicationPriority,
    NetworkPolicy,
    NetworkPolicyService,
    NetworkPolicyState,
    PolicyEventType,
    get_network_policy_service,
)


@pytest.fixture
def policy_service():
    """Isolated NetworkPolicyService instance."""
    svc = NetworkPolicyService(active_window_seconds=10.0)
    svc.reset()
    return svc


@pytest.fixture
def client():
    """TestClient for FastAPI app."""
    return TestClient(app)


def _make_report(
    incident_id: str,
    incident_type: IncidentType = IncidentType.FIGHT,
    severity: Severity = Severity.MODERATE,
    status: IncidentStatus = IncidentStatus.VERIFIED,
) -> IncidentReport:
    return IncidentReport(
        schema_version="1.0",
        source_module="crowd_fight",
        incident_id=incident_id,
        camera_id="cam_yashobhoomi_01",
        incident_type=incident_type,
        severity=severity,
        confidence=0.92,
        start_time=datetime.now(timezone.utc),
        status=status,
    )


def test_initial_policy_state(policy_service):
    """Test policy starts in NORMAL standby mode without physical network control."""
    state = policy_service.get_policy_state()
    assert state.policy == NetworkPolicy.NORMAL
    assert state.priority == ApplicationPriority.ROUTINE
    assert state.actual_network_control is False
    assert state.control_plane_connected is False
    assert state.policy_scope == "APPLICATION_LAYER"
    assert state.slice_allocated == "DEFAULT_BE"
    assert state.active_incident_id is None

    events = policy_service.get_events()
    assert events.count >= 1
    assert events.events[-1].event_type == PolicyEventType.POLICY_INITIALIZED


def test_escalation_to_event_on_moderate_or_high_incident(policy_service):
    """Test policy escalates to EVENT and HIGH priority on fight or crowd incident."""
    rep = _make_report("inc-fight-01", IncidentType.FIGHT, Severity.MODERATE)
    state = policy_service.on_incident(rep)

    assert state.policy == NetworkPolicy.EVENT
    assert state.priority == ApplicationPriority.HIGH
    assert state.active_incident_id == "inc-fight-01"
    assert state.actual_network_control is False

    events = policy_service.get_events()
    event_types = [e.event_type for e in events.events]
    assert PolicyEventType.POLICY_ESCALATED in event_types
    assert PolicyEventType.INCIDENT_PRIORITY_ASSIGNED in event_types


def test_escalation_to_critical_on_armed_fight(policy_service):
    """Test policy escalates to CRITICAL for critical severity incidents."""
    rep = _make_report("inc-armed-01", IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    state = policy_service.on_incident(rep)

    assert state.policy == NetworkPolicy.CRITICAL
    assert state.priority == ApplicationPriority.CRITICAL
    assert state.active_incident_id == "inc-armed-01"


def test_highest_severity_wins_multiple_incidents(policy_service):
    """If both a MODERATE incident and a CRITICAL incident are active, CRITICAL wins."""
    rep_mod = _make_report("inc-mod-01", IncidentType.FIGHT, Severity.MODERATE)
    rep_crit = _make_report("inc-crit-02", IncidentType.ARMED_FIGHT, Severity.CRITICAL)

    policy_service.on_incident(rep_mod)
    state = policy_service.on_incident(rep_crit)

    assert state.policy == NetworkPolicy.CRITICAL
    assert state.priority == ApplicationPriority.CRITICAL
    assert state.active_incident_id == "inc-crit-02"


def test_deescalation_when_incidents_clear(policy_service):
    """When active incidents expire, policy deescalates back to NORMAL."""
    rep = _make_report("inc-temp-01", IncidentType.WEAPON, Severity.HIGH)
    state1 = policy_service.on_incident(rep)
    assert state1.policy == NetworkPolicy.EVENT

    # Fast forward past active window
    future_time = datetime.now(timezone.utc).timestamp() + 100.0
    state2 = policy_service.get_policy_state(now=future_time)

    assert state2.policy == NetworkPolicy.NORMAL
    assert state2.priority == ApplicationPriority.ROUTINE
    assert state2.active_incident_id is None

    events = policy_service.get_events()
    event_types = [e.event_type for e in events.events]
    assert PolicyEventType.POLICY_DEESCALATED in event_types


def test_record_evidence_ready_event(policy_service):
    """Verify evidence generation event recording."""
    policy_service.record_evidence_event("inc-ev-01", "Snapshot and 10s MP4 clip ready")
    events = policy_service.get_events()
    ev_types = [e.event_type for e in events.events]
    assert PolicyEventType.EVIDENCE_READY in ev_types


def test_api_network_policy_endpoint(client):
    """Test GET /network/policy returns compliant schema and truthful flags."""
    svc = get_network_policy_service()
    svc.reset()

    resp = client.get("/network/policy")
    assert resp.status_code == 200
    data = resp.json()

    assert data["policy"] in ("NORMAL", "EVENT", "CRITICAL")
    assert data["priority"] in ("ROUTINE", "HIGH", "CRITICAL")
    assert data["actual_network_control"] is False
    assert data["control_plane_connected"] is False
    assert data["policy_scope"] == "APPLICATION_LAYER"
    assert "timestamp" in data


def test_api_network_events_endpoint(client):
    """Test GET /network/events returns event list."""
    svc = get_network_policy_service()
    svc.reset()
    rep = _make_report("inc-api-01", IncidentType.FIGHT, Severity.HIGH)
    svc.on_incident(rep)

    resp = client.get("/network/events?limit=10")
    assert resp.status_code == 200
    data = resp.json()

    assert "count" in data
    assert "events" in data
    assert data["count"] >= 1
    assert any(e["incident_id"] == "inc-api-01" for e in data["events"])
