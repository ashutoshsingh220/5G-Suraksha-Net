"""Tests for Part 12: Live IMC Demonstration & Controlled Scenario Mode.

Verifies:
1. DemoScenarioService lifecycle: WEAPON, CROWD_PANIC, ARMED_FIGHT, NORMAL.
2. source_mode == 'DEMO' on all generated incidents and responses.
3. Storage isolation in outputs/demo/incidents/ (zero contamination of outputs/incidents/).
4. Network policy escalation and de-escalation on reset.
5. Strict email notification suppression for demo incidents.
6. REST API endpoints (/demo/status, /demo/scenarios, /demo/scenario, /demo/reset).
7. Downstream endpoints (/incidents/{id}, /incidents/{id}/response, /incidents/{id}/assessment, /incidents/{id}/snapshot).
"""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from suraksha.config import PROJECT_ROOT
from suraksha.demo import get_demo_service, DemoScenarioId
from suraksha.incidents.schemas import IncidentReport, IncidentType, Severity
from suraksha.network import get_network_policy_service
from suraksha.notifications.service import EmailNotificationService
from suraksha.notifications.schemas import EmailNotificationStatus
from suraksha.api.app import app


@pytest.fixture(autouse=True)
def cleanup_demo_state():
    """Ensure clean demo and network state before and after each test."""
    svc = get_demo_service()
    svc.stop_scenario()
    net = get_network_policy_service()
    net.reset()
    yield
    svc.stop_scenario()
    net.reset()


def test_demo_scenarios_metadata():
    """Verify available demonstration scenario definitions."""
    svc = get_demo_service()
    scenarios = svc.get_available_scenarios()
    assert len(scenarios) == 4
    ids = [s.id for s in scenarios]
    assert DemoScenarioId.WEAPON in ids
    assert DemoScenarioId.CROWD_PANIC in ids
    assert DemoScenarioId.ARMED_FIGHT in ids
    assert DemoScenarioId.NORMAL in ids


def test_start_weapon_scenario():
    """Verify start_scenario for WEAPON scenario."""
    svc = get_demo_service()
    res = svc.start_scenario(DemoScenarioId.WEAPON)
    assert res.is_demo_active is True
    assert res.active_scenario == DemoScenarioId.WEAPON
    assert res.source_mode == "DEMO"
    assert res.incident is not None
    assert res.incident.incident_type == IncidentType.WEAPON
    assert res.incident.severity == Severity.HIGH
    assert res.incident.source_mode == "DEMO"
    assert res.incident.incident_id.startswith("demo-weapon-")
    assert len(res.simulation_timeline) >= 5

    # Storage isolation check
    demo_file = PROJECT_ROOT / "outputs" / "demo" / "incidents" / f"{res.incident.incident_id}.json"
    prod_file = PROJECT_ROOT / "outputs" / "incidents" / f"{res.incident.incident_id}.json"
    assert demo_file.is_file(), "Demo incident should be stored in outputs/demo/incidents/"
    assert not prod_file.is_file(), "Demo incident MUST NOT contaminate outputs/incidents/"

    # Verify Network Policy escalated
    net = get_network_policy_service()
    pol = net.get_policy_state()
    assert pol.policy == "EVENT"
    assert pol.priority == "HIGH"


def test_start_armed_fight_scenario_critical():
    """Verify start_scenario for ARMED_FIGHT scenario escalates to CRITICAL."""
    svc = get_demo_service()
    res = svc.start_scenario(DemoScenarioId.ARMED_FIGHT)
    assert res.is_demo_active is True
    assert res.active_scenario == DemoScenarioId.ARMED_FIGHT
    assert res.incident.incident_type == IncidentType.ARMED_FIGHT
    assert res.incident.severity == Severity.CRITICAL
    assert res.incident.source_mode == "DEMO"

    # Verify Network Policy escalated to CRITICAL
    net = get_network_policy_service()
    pol = net.get_policy_state()
    assert pol.policy == "CRITICAL"
    assert pol.priority == "CRITICAL"


def test_demo_reset_and_normal():
    """Verify stop_scenario resets demo state and network policy cleanly."""
    svc = get_demo_service()
    svc.start_scenario(DemoScenarioId.CROWD_PANIC)
    assert svc.get_status().is_demo_active is True

    # Reset
    res = svc.stop_scenario()
    assert res.is_demo_active is False
    assert res.active_scenario is None
    assert res.source_mode == "REAL"
    assert res.incident is None

    # Verify Network Policy restored to NORMAL
    net = get_network_policy_service()
    pol = net.get_policy_state()
    assert pol.policy == "NORMAL"
    assert pol.priority == "ROUTINE"


def test_notifications_suppressed_for_demo_mode():
    """Verify that email notifications are strictly suppressed when source_mode is DEMO."""
    from suraksha.config import EmailConfig
    svc = get_demo_service()
    res = svc.start_scenario(DemoScenarioId.ARMED_FIGHT)
    incident = res.incident
    assert incident.source_mode == "DEMO"

    cfg = EmailConfig(enabled=True)
    notif = EmailNotificationService(config=cfg)
    result = notif.send_incident_email(incident)
    assert result.status == EmailNotificationStatus.NOT_ATTEMPTED
    assert "Suppressed: demonstration mode active" in result.error_message


def test_demo_api_endpoints():
    """Verify the /demo/* REST API endpoints using FastAPI TestClient."""
    client = TestClient(app)

    # 1. GET /demo/scenarios
    r = client.get("/demo/scenarios")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 4

    # 2. GET /demo/status initial
    r = client.get("/demo/status")
    assert r.status_code == 200
    status = r.json()
    assert status["is_demo_active"] is False

    # 3. POST /demo/scenario
    r = client.post("/demo/scenario", json={"scenario_id": "WEAPON"})
    assert r.status_code == 200
    active = r.json()
    assert active["is_demo_active"] is True
    assert active["active_scenario"] == "WEAPON"
    inc_id = active["incident"]["incident_id"]

    # 4. GET /incidents/{inc_id}
    r = client.get(f"/incidents/{inc_id}")
    assert r.status_code == 200
    inc_data = r.json()
    assert inc_data["incident_id"] == inc_id
    assert inc_data["source_mode"] == "DEMO"

    # 5. GET /incidents/{inc_id}/response
    r = client.get(f"/incidents/{inc_id}/response")
    assert r.status_code == 200
    resp_plan = r.json()
    assert resp_plan["incident_id"] == inc_id
    assert resp_plan["human_approval_required"] is True

    # 6. GET /incidents/{inc_id}/assessment
    r = client.get(f"/incidents/{inc_id}/assessment")
    assert r.status_code == 200
    assessment = r.json()
    assert assessment["incident_id"] == inc_id
    assert assessment["human_approval_required"] is True
    assert assessment["severity_assessment"]["severity"] == "HIGH"

    # 7. GET /incidents/{inc_id}/snapshot
    r = client.get(f"/incidents/{inc_id}/snapshot")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"

    # 8. POST /demo/reset
    r = client.post("/demo/reset")
    assert r.status_code == 200
    reset_data = r.json()
    assert reset_data["is_demo_active"] is False
    assert reset_data["active_scenario"] is None
