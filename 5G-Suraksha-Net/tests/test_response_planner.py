"""Tests for Phase 3C: Deterministic Response Planner for 5G Suraksha-Net."""
from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import sys
from fastapi.testclient import TestClient

import suraksha.api.app
app_mod = sys.modules["suraksha.api.app"]
from suraksha.api.app import app
from suraksha.incidents.schemas import BBox, Evidence, IncidentReport, IncidentType, Severity
from suraksha.location.schemas import Location, LocationSource
from suraksha.response import (
    ActionPriority,
    ActionType,
    EmergencyResource,
    EmergencyResourceDirectory,
    IncidentResponse,
    PolicyDecision,
    ResponseAction,
    ResponsePlanner,
    ResponseStatus,
    check_has_injury,
    evaluate_response_policy,
)


@pytest.fixture
def sample_location():
    return Location(
        name="Yashobhoomi, Dwarka Sector 25, New Delhi",
        latitude=28.552553,
        longitude=77.044893,
        source=LocationSource.GEOCODED_DEMO,
    )


@pytest.fixture
def mock_resources():
    return EmergencyResourceDirectory(
        hospitals=[
            EmergencyResource(
                name="Venkateshwar Super Speciality Hospital",
                distance_km=4.06,
                phone_number="011 4855 5555",
                formatted_address="Sector 18A Dwarka",
            ),
            EmergencyResource(
                name="Max Super Speciality Hospital Dwarka",
                distance_km=3.40,
                phone_number="088604 44888",
                formatted_address="Sector 10 Dwarka",
            ),
            EmergencyResource(
                name="Indira Gandhi Hospital",
                distance_km=3.59,
                phone_number="011 2089 5995",
                formatted_address="Sector 9 Dwarka",
            ),
        ],
        police_stations=[
            EmergencyResource(
                name="DCP Office District Dwarka Sec 19",
                distance_km=3.04,
                phone_number="011 2804 2990",
                formatted_address="Sector 19 Dwarka",
            ),
            EmergencyResource(
                name="Police Station Dwarka Sector 23",
                distance_km=1.91,
                phone_number="011 2805 1585",
                formatted_address="Sector 23 Dwarka",
            ),
            EmergencyResource(
                name="Police Station Dwarka South",
                distance_km=3.56,
                phone_number="011 2508 9326",
                formatted_address="Sector 9 Dwarka",
            ),
        ],
    )


def make_incident(
    incident_type: IncidentType,
    severity: Severity = Severity.HIGH,
    details: dict | None = None,
    location: Location | None = None,
) -> IncidentReport:
    return IncidentReport(
        schema_version="1.0",
        source_module="crowd_fight",
        incident_id="test_inc_123",
        camera_id="cam_yashobhoomi_01",
        incident_type=incident_type,
        severity=severity,
        confidence=0.89,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        details=details or {},
        location=location,
    )


# ---------------------------------------------------------------------------
# Policy Tests (1-6)
# ---------------------------------------------------------------------------

def test_policy_fight_requires_police_only_by_default(mock_resources):
    """Test 1: FIGHT requires police/security response, no automatic medical."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.FIGHT, Severity.MODERATE)
    resp = planner.plan(inc)

    assert resp.status == ResponseStatus.AWAITING_HUMAN_APPROVAL
    assert len(resp.response_actions) == 1
    action = resp.response_actions[0]
    assert action.action_type == ActionType.POLICE_SECURITY
    assert action.priority == "moderate"
    assert "Physical fight detected" in action.reason


def test_policy_weapon_requires_police_only_by_default(mock_resources):
    """Test 2: WEAPON requires police/security response, no automatic medical."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.WEAPON, Severity.HIGH)
    resp = planner.plan(inc)

    assert resp.status == ResponseStatus.AWAITING_HUMAN_APPROVAL
    assert len(resp.response_actions) == 1
    action = resp.response_actions[0]
    assert action.action_type == ActionType.POLICE_SECURITY
    assert action.priority == "high"
    assert "Confirmed weapon incident" in action.reason


def test_policy_armed_fight_requires_both_police_and_medical(mock_resources):
    """Test 3: ARMED_FIGHT requires both police and medical assistance (CRITICAL)."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    resp = planner.plan(inc)

    assert resp.status == ResponseStatus.AWAITING_HUMAN_APPROVAL
    assert len(resp.response_actions) == 2
    action_types = {a.action_type for a in resp.response_actions}
    assert action_types == {ActionType.POLICE_SECURITY, ActionType.MEDICAL_ASSISTANCE}

    for action in resp.response_actions:
        assert action.priority == "critical"
        assert action.human_approval_required is True
        assert action.status == ResponseStatus.AWAITING_HUMAN_APPROVAL


def test_policy_crowd_panic_requires_police_only_by_default(mock_resources):
    """Test 4: CROWD_PANIC requires police/security for crowd control/evacuation."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.CROWD_PANIC, Severity.HIGH)
    resp = planner.plan(inc)

    assert resp.status == ResponseStatus.AWAITING_HUMAN_APPROVAL
    assert len(resp.response_actions) == 1
    assert resp.response_actions[0].action_type == ActionType.POLICE_SECURITY
    assert "Crowd panic detected" in resp.response_actions[0].reason


def test_policy_crowd_density_no_emergency_action(mock_resources):
    """Test 5: CROWD_DENSITY is routine monitoring, creates no emergency action."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    for dtype in (
        IncidentType.CROWD_DENSITY_HIGH,
        IncidentType.CROWD_DENSITY_CRITICAL,
        IncidentType.CROWD_RAPID_GROWTH,
    ):
        inc = make_incident(dtype, Severity.MEDIUM)
        resp = planner.plan(inc)
        assert resp.status == ResponseStatus.NO_ACTION
        assert len(resp.response_actions) == 0
        assert "routine monitoring" in resp.warning


def test_policy_unsupported_incident_state_safe_no_action(mock_resources):
    """Test 6: Unsupported or unconfigured incident states create safe no-action plan."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    # Mocking an unknown incident type
    inc = make_incident(IncidentType.FIGHT)
    inc.incident_type = "unknown_incident_type"  # type: ignore

    resp = planner.plan(inc)
    assert resp.status == ResponseStatus.NO_ACTION
    assert len(resp.response_actions) == 0
    assert "No response policy configured" in resp.warning


# ---------------------------------------------------------------------------
# Medical Evidence Tests (7-10)
# ---------------------------------------------------------------------------

def test_fight_without_injury_no_automatic_medical(mock_resources):
    """Test 7: FIGHT without injury indicators does not add medical assistance."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.FIGHT, details={"person_count": 2})
    resp = planner.plan(inc)

    assert not any(a.action_type == ActionType.MEDICAL_ASSISTANCE for a in resp.response_actions)


def test_fight_with_explicit_injury_adds_medical_action(mock_resources):
    """Test 8: FIGHT with explicit injury/person-down indicator triggers medical action."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    # Positive boolean flag in details
    inc = make_incident(
        IncidentType.FIGHT,
        details={"injury": True, "person_down": True},
    )
    resp = planner.plan(inc)

    action_types = {a.action_type for a in resp.response_actions}
    assert ActionType.POLICE_SECURITY in action_types
    assert ActionType.MEDICAL_ASSISTANCE in action_types

    med_action = next(a for a in resp.response_actions if a.action_type == ActionType.MEDICAL_ASSISTANCE)
    assert "explicit medical/injury evidence" in med_action.reason


def test_weapon_without_injury_no_automatic_medical(mock_resources):
    """Test 9: WEAPON without injury evidence does not add medical response."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.WEAPON, details={"weapon_class": "knife"})
    resp = planner.plan(inc)

    assert not any(a.action_type == ActionType.MEDICAL_ASSISTANCE for a in resp.response_actions)


def test_weapon_with_injury_adds_medical_action(mock_resources):
    """Test 10: WEAPON with explicit injury indicator triggers medical action."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.WEAPON, details={"injuries": 1})
    resp = planner.plan(inc)

    action_types = {a.action_type for a in resp.response_actions}
    assert ActionType.MEDICAL_ASSISTANCE in action_types


# ---------------------------------------------------------------------------
# Resource Selection Tests (11-17)
# ---------------------------------------------------------------------------

def test_nearest_police_resource_selected_as_primary(mock_resources):
    """Test 11: The nearest valid police resource by straight-line distance is recommended."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.WEAPON)
    resp = planner.plan(inc)

    police_action = resp.response_actions[0]
    assert police_action.recommended_resource is not None
    # Sector 23 is at 1.91 km (nearest)
    assert police_action.recommended_resource.name == "Police Station Dwarka Sector 23"
    assert police_action.recommended_resource.distance_km == 1.91


def test_nearest_hospital_resource_selected_as_primary(mock_resources):
    """Test 12: The nearest valid hospital resource by straight-line distance is recommended."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    resp = planner.plan(inc)

    med_action = next(a for a in resp.response_actions if a.action_type == ActionType.MEDICAL_ASSISTANCE)
    assert med_action.recommended_resource is not None
    # Max Super Speciality is at 3.40 km (nearest in fixture)
    assert med_action.recommended_resource.name == "Max Super Speciality Hospital Dwarka"
    assert med_action.recommended_resource.distance_km == 3.40


def test_resources_sorted_deterministically_ascending(mock_resources):
    """Test 13: Resources are sorted deterministically ascending by straight-line distance."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    resp = planner.plan(inc)

    police_dists = [r.distance_km for r in resp.police_resources]
    assert police_dists == sorted(police_dists)
    assert police_dists == [1.91, 3.04, 3.56]

    med_dists = [r.distance_km for r in resp.medical_resources]
    assert med_dists == sorted(med_dists)
    assert med_dists == [3.40, 3.59, 4.06]


def test_missing_police_resources_handled_safely():
    """Test 14: Missing police resources do not crash; marked as RESOURCE_UNAVAILABLE."""
    empty_res = EmergencyResourceDirectory(hospitals=[], police_stations=[])
    planner = ResponsePlanner(resource_directory=empty_res)
    inc = make_incident(IncidentType.FIGHT)
    resp = planner.plan(inc)

    assert resp.status == ResponseStatus.RESOURCE_UNAVAILABLE
    assert len(resp.response_actions) == 1
    assert resp.response_actions[0].status == ResponseStatus.RESOURCE_UNAVAILABLE
    assert resp.response_actions[0].recommended_resource is None
    assert "Required police/security resource is unavailable" in resp.warning


def test_missing_hospital_resources_handled_safely():
    """Test 15: Missing hospital resources do not crash; marked as RESOURCE_UNAVAILABLE."""
    only_police = EmergencyResourceDirectory(
        police_stations=[EmergencyResource(name="PS Sector 23", distance_km=1.91)],
        hospitals=[],
    )
    planner = ResponsePlanner(resource_directory=only_police)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    resp = planner.plan(inc)

    assert resp.status == ResponseStatus.RESOURCE_UNAVAILABLE
    med_action = next(a for a in resp.response_actions if a.action_type == ActionType.MEDICAL_ASSISTANCE)
    assert med_action.status == ResponseStatus.RESOURCE_UNAVAILABLE
    assert med_action.recommended_resource is None
    assert "Required medical/hospital resource is unavailable" in resp.warning


def test_malformed_resource_does_not_crash():
    """Test 16: Malformed input resource dictionary does not crash the planner."""
    malformed_dict = {
        "emergency_responders": {
            "hospitals": [{"bad_key": 123}, {"name": "Valid Hospital", "distance_km": "invalid"}],
            "police_stations": [None, "string_instead_of_dict"],
        }
    }
    planner = ResponsePlanner()
    inc = make_incident(IncidentType.FIGHT)
    # Passing malformed dictionary directly
    resp = planner.plan(inc, resources=malformed_dict)
    assert isinstance(resp, IncidentResponse)
    assert resp.incident_id == inc.incident_id


def test_no_fabricated_resource():
    """Test 17: When directory is empty, planner NEVER invents or fabricates a facility."""
    empty_dir = EmergencyResourceDirectory()
    planner = ResponsePlanner(resource_directory=empty_dir)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    resp = planner.plan(inc)

    assert len(resp.police_resources) == 0
    assert len(resp.medical_resources) == 0
    assert resp.response_actions[0].recommended_resource is None
    assert resp.response_actions[1].recommended_resource is None


# ---------------------------------------------------------------------------
# Security & Human Approval Tests (18-20)
# ---------------------------------------------------------------------------

def test_all_emergency_actions_require_human_approval(mock_resources):
    """Test 18: Every generated emergency response action requires human approval."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    resp = planner.plan(inc)

    assert resp.human_approval_required is True
    for action in resp.response_actions:
        assert action.human_approval_required is True
        assert action.status == ResponseStatus.AWAITING_HUMAN_APPROVAL


def test_planner_never_contacts_external_network(mock_resources, monkeypatch):
    """Test 19: Pure in-memory execution; socket creation is blocked and never attempted."""
    def guarded_socket(*args, **kwargs):
        raise RuntimeError("Network socket call attempted during deterministic planning!")

    monkeypatch.setattr(socket, "socket", guarded_socket)
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)

    # Should run with zero network activity
    resp = planner.plan(inc)
    assert resp.incident_id == inc.incident_id


def test_planner_never_exposes_api_key(mock_resources, monkeypatch):
    """Test 20: Response plan serialization never exposes GOOGLE_MAPS_API_KEY."""
    fake_key = "AIzaSy_SUPER_CONFIDENTIAL_KEY_98765"
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", fake_key)

    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)
    resp = planner.plan(inc)

    serialized = resp.model_dump_json()
    assert fake_key not in serialized
    assert "api_key" not in serialized.lower()


# ---------------------------------------------------------------------------
# Integration & API Tests (21-25)
# ---------------------------------------------------------------------------

def test_incident_to_response_preserves_attributes(mock_resources, sample_location):
    """Test 21: Original incident attributes are strictly preserved without re-calculation."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    evidence = Evidence(snapshot_path="snap.jpg", clip_path="clip.mp4", frame_idx=100)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL, location=sample_location)
    inc.evidence = evidence

    resp = planner.plan(inc)
    assert resp.incident_id == inc.incident_id
    assert resp.incident_type == IncidentType.ARMED_FIGHT
    assert resp.severity == Severity.CRITICAL
    assert resp.location == sample_location
    assert resp.evidence.snapshot_path == "snap.jpg"
    assert resp.evidence.clip_path == "clip.mp4"


def test_existing_incident_serialization_compatible(mock_resources):
    """Test 22: IncidentResponse is fully serializable to clean JSON without schema conflicts."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.FIGHT)
    resp = planner.plan(inc)

    data = resp.model_dump(mode="json")
    assert data["incident_id"] == "test_inc_123"
    assert data["status"] == "AWAITING_HUMAN_APPROVAL"
    assert isinstance(data["response_actions"], list)
    assert isinstance(data["police_resources"], list)


def test_api_response_plan_endpoint(mock_resources, sample_location):
    """Test 23: GET /incidents/{incident_id}/response returns 200 OK with response plan."""
    client = TestClient(app)
    mock_pipeline = MagicMock()
    mock_pipeline.state.running = True

    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL, location=sample_location)
    mock_pipeline.manager.recent = [inc]
    mock_pipeline.planner = ResponsePlanner(resource_directory=mock_resources)
    mock_pipeline.manager.get_response_plan.return_value = mock_pipeline.planner.plan(inc)

    app_mod._pipeline = mock_pipeline

    res = client.get("/incidents/test_inc_123/response")
    assert res.status_code == 200
    plan_json = res.json()
    assert plan_json["incident_id"] == "test_inc_123"
    assert plan_json["incident_type"] == "armed_fight"
    assert plan_json["severity"] == "critical"
    assert len(plan_json["response_actions"]) == 2
    assert plan_json["human_approval_required"] is True


def test_api_response_plan_endpoint_404():
    """Test 24: GET /incidents/{incident_id}/response returns 404 for unknown incident."""
    client = TestClient(app)
    mock_pipeline = MagicMock()
    mock_pipeline.state.running = True
    mock_pipeline.manager.recent = []
    mock_pipeline.manager.get_response_plan.return_value = None

    app_mod._pipeline = mock_pipeline

    res = client.get("/incidents/nonexistent_id/response")
    assert res.status_code == 404
    assert res.json()["error"] == "response plan not found"


# ---------------------------------------------------------------------------
# Strict Determinism & Performance Tests (26-28)
# ---------------------------------------------------------------------------

def test_strict_determinism_across_100_runs(mock_resources, sample_location):
    """Test 26: 100 consecutive runs on the exact same incident yield 100% identical outputs."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL, location=sample_location)

    first_dump = planner.plan(inc).model_dump_json()

    for _ in range(100):
        subsequent_dump = planner.plan(inc).model_dump_json()
        assert subsequent_dump == first_dump, "Response plan output had non-deterministic variance!"


def test_lightweight_execution_speed(mock_resources):
    """Test 27: Response planning executes in sub-millisecond time (< 5ms)."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)

    t0 = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        _ = planner.plan(inc)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    # Must be sub-millisecond, far below any video frame threshold (e.g. 33ms)
    assert avg_ms < 2.0, f"Response planning took {avg_ms:.3f}ms, expected < 2.0ms"


def test_distance_units_synchronized():
    """Test 28: EmergencyResource automatically synchronizes distance_km and distance_m."""
    res1 = EmergencyResource(name="Station A", distance_km=1.91)
    assert res1.distance_m == 1910.0
    assert res1.distance_type == "straight_line"

    res2 = EmergencyResource(name="Station B", distance_m=3400.0)
    assert res2.distance_km == 3.4


def test_default_yashobhoomi_directory_loading(sample_location):
    """Test 29: Planner loads Yashobhoomi emergency resources by default when available."""
    planner = ResponsePlanner()
    assert len(planner.resource_directory.police_stations) > 0
    assert len(planner.resource_directory.hospitals) > 0

    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL, location=sample_location)
    resp = planner.plan(inc)
    assert resp.status == ResponseStatus.AWAITING_HUMAN_APPROVAL
    # Yashobhoomi verified nearest is Sector 23 (1.91 km)
    assert resp.police_resources[0].name == "Police Station Dwarka Sector 23"
    assert resp.police_resources[0].distance_km == 1.91
    # Yashobhoomi verified nearest hospital is Max Super Speciality (3.40 km)
    assert resp.medical_resources[0].name == "Max Super Speciality Hospital Dwarka"
    assert resp.medical_resources[0].distance_km == 3.40


def test_incident_manager_integration(mock_resources):
    """Test 30: IncidentManager records and retrieves response plan on incident emission."""
    from suraksha.config import VerifyConfig
    from suraksha.incidents.evidence import EvidenceWriter
    from suraksha.incidents.manager import EventBus, IncidentManager

    bus = EventBus()
    evidence = MagicMock()
    planner = ResponsePlanner(resource_directory=mock_resources)
    mgr = IncidentManager(
        camera_id="cam_test_01",
        verify_cfg=VerifyConfig(),
        evidence=evidence,
        bus=bus,
        response_planner=planner,
    )

    inc = make_incident(IncidentType.WEAPON, Severity.HIGH)
    mgr._emit(inc)

    plan = mgr.get_response_plan(inc.incident_id)
    assert plan is not None
    assert plan.incident_id == inc.incident_id
    assert plan.status == ResponseStatus.AWAITING_HUMAN_APPROVAL
    assert plan.response_actions[0].action_type == ActionType.POLICE_SECURITY


def test_websocket_incident_broadcast_compatible():
    """Test 31: Existing WebSocket client broadcast continues functioning with no schema break."""
    import asyncio
    from suraksha.api.app import _broadcast, _ws_clients

    client = TestClient(app)
    report = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL)

    with client.websocket_connect("/ws/incidents") as ws:
        assert len(_ws_clients) == 1
        payload = report.model_dump_json()
        for active_ws in list(_ws_clients):
            asyncio.run(active_ws.send_text(payload))

        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["incident_id"] == "test_inc_123"
        assert data["incident_type"] == "armed_fight"
        assert data["severity"] == "critical"
