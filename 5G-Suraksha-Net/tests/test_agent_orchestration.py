"""Tests for Agentic Emergency Orchestration and assessment endpoints."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

import sys
from suraksha.api.app import app
app_mod = sys.modules['suraksha.api.app']
from suraksha.agents import (
    ActionStepStatus,
    AgentAssessment,
    AgentOrchestrator,
    DeterministicAgentProvider,
    get_agent_orchestrator,
)
from suraksha.config import PROJECT_ROOT, load_config
from suraksha.incidents.schemas import BBox, Evidence, IncidentReport, IncidentStatus, IncidentType, Severity
from suraksha.location.schemas import Location, LocationSource
from suraksha.network import get_network_policy_service
from suraksha.response.schemas import ActionType, IncidentResponse, ResponseAction, ResponseStatus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def orchestrator():
    orch = AgentOrchestrator(provider=DeterministicAgentProvider())
    orch.clear_cache()
    return orch


def _create_mock_incident(
    incident_id: str,
    incident_type: IncidentType = IncidentType.WEAPON,
    severity: Severity = Severity.HIGH,
    confidence: float = 0.91,
    with_location: bool = True,
    with_evidence: bool = True,
) -> IncidentReport:
    loc = None
    if with_location:
        loc = Location(
            name="Yashobhoomi, Dwarka Sector 25, New Delhi",
            latitude=28.5528,
            longitude=77.0601,
            source=LocationSource.CAMERA,
        )

    ev = Evidence()
    if with_evidence:
        ev = Evidence(
            snapshot_path=f"outputs/snapshots/{incident_id}.jpg",
            clip_path=f"outputs/clips/{incident_id}.mp4",
        )

    return IncidentReport(
        schema_version="1.0",
        source_module="crowd_fight",
        incident_id=incident_id,
        camera_id="cam_yashobhoomi_01",
        incident_type=incident_type,
        severity=severity,
        confidence=confidence,
        start_time=datetime.now(timezone.utc),
        location=loc,
        evidence=ev,
        status=IncidentStatus.VERIFIED,
    )


def test_assessment_weapon_incident(orchestrator):
    """Test assessment for verified weapon incident preserves deterministic severity and recommends police."""
    inc = _create_mock_incident("inc-wpn-001", IncidentType.WEAPON, Severity.HIGH, 0.94)
    mgr = MagicMock()
    mgr.recent = [inc]

    assessment = orchestrator.get_assessment("inc-wpn-001", pipeline_manager=mgr)
    assert assessment is not None
    assert assessment.incident_id == "inc-wpn-001"
    # Deterministic severity preserved
    assert assessment.severity_assessment.severity == "HIGH"
    assert assessment.severity_assessment.is_deterministic is True
    # Human approval required
    assert assessment.human_approval_required is True
    # Traceability
    assert assessment.data_sources.incident_report is True
    assert assessment.data_sources.network_policy is True
    # Recommended sequence present
    assert len(assessment.recommended_sequence) >= 5
    assert assessment.recommended_sequence[0].title == "VERIFY INCIDENT"


def test_assessment_critical_armed_fight(orchestrator):
    """Test assessment for CRITICAL armed fight maintains CRITICAL severity and elevated priority."""
    inc = _create_mock_incident("inc-armed-002", IncidentType.ARMED_FIGHT, Severity.CRITICAL, 0.96)
    mgr = MagicMock()
    mgr.recent = [inc]

    # Escalate network policy
    net_svc = get_network_policy_service()
    net_svc.on_incident(inc)

    assessment = orchestrator.get_assessment("inc-armed-002", pipeline_manager=mgr)
    assert assessment is not None
    assert assessment.severity_assessment.severity == "CRITICAL"
    assert "CRITICAL" in assessment.situation_summary
    assert assessment.network_assessment.policy == "CRITICAL"
    assert assessment.network_assessment.actual_network_control is False


def test_assessment_fight_medical_conditional(orchestrator):
    """Test that medical assistance is marked conditional when casualty is unconfirmed."""
    inc = _create_mock_incident("inc-fight-003", IncidentType.FIGHT, Severity.MODERATE, 0.85)
    mgr = MagicMock()
    mgr.recent = [inc]

    assessment = orchestrator.get_assessment("inc-fight-003", pipeline_manager=mgr)
    assert assessment is not None
    assert assessment.response_assessment.medical_conditional is True
    assert any("Medical injury status not confirmed" in gap for gap in assessment.missing_information)


def test_missing_location_and_evidence_handling(orchestrator):
    """Test that missing location and evidence are truthfully flagged in missing_information."""
    inc = _create_mock_incident(
        "inc-sparse-004",
        IncidentType.CROWD_PANIC,
        Severity.MODERATE,
        confidence=0.75,
        with_location=False,
        with_evidence=False,
    )
    mgr = MagicMock()
    mgr.recent = [inc]

    assessment = orchestrator.get_assessment("inc-sparse-004", pipeline_manager=mgr)
    assert assessment is not None
    assert assessment.location_assessment.gps_fix_available is False
    assert assessment.evidence_assessment.snapshot_available is False
    assert assessment.evidence_assessment.video_available is False
    assert any("GPS" in gap for gap in assessment.missing_information)
    assert any("snapshot" in gap.lower() for gap in assessment.missing_information)


def test_persisted_disk_incident_assessment(orchestrator, tmp_path):
    """Test retrieving assessment for persisted disk incident without in-memory manager."""
    cfg = load_config()
    report_dir = (PROJECT_ROOT / cfg.incidents.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    test_file = report_dir / "inc-disk-005.json"

    inc = _create_mock_incident("inc-disk-005", IncidentType.FIGHT, Severity.HIGH)
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(inc.model_dump_json())

        assessment = orchestrator.get_assessment("inc-disk-005", pipeline_manager=None)
        assert assessment is not None
        assert assessment.incident_id == "inc-disk-005"
        assert assessment.severity_assessment.severity == "HIGH"
    finally:
        if test_file.is_file():
            test_file.unlink()


def test_api_assessment_endpoint_success(client):
    """Test GET /incidents/{incident_id}/assessment returns 200 and schema."""
    inc = _create_mock_incident("inc-api-006", IncidentType.WEAPON, Severity.HIGH)
    mock_pipeline = MagicMock()
    mock_pipeline.manager.recent = [inc]
    app_mod._pipeline = mock_pipeline

    resp = client.get("/incidents/inc-api-006/assessment")
    assert resp.status_code == 200
    data = resp.json()

    assert data["incident_id"] == "inc-api-006"
    assert data["severity_assessment"]["severity"] == "HIGH"
    assert data["human_approval_required"] is True
    assert "situation_summary" in data
    assert "recommended_sequence" in data
    assert "data_sources" in data


def test_api_assessment_endpoint_not_found(client):
    """Test GET /incidents/{id}/assessment returns 404 for non-existent incident."""
    mock_pipeline = MagicMock()
    mock_pipeline.manager.recent = []
    app_mod._pipeline = mock_pipeline

    resp = client.get("/incidents/non_existent_inc_99999/assessment")
    assert resp.status_code == 404
