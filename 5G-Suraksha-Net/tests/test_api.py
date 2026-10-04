"""Automated integration tests for FastAPI REST and WebSocket endpoints."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import suraksha.api.app
app_mod = sys.modules['suraksha.api.app']
from suraksha.api.app import app, _on_incident, _ws_clients, _broadcast
from suraksha.incidents.schemas import IncidentReport, BBox, Evidence, IncidentType, Severity


@pytest.fixture
def mock_pipeline():
    """Mock CrowdFightPipeline state for hermetic API testing."""
    mock = MagicMock()
    mock.state.running = True
    mock.state.device = "cuda"
    mock.device_info.gpu_name = "NVIDIA GeForce RTX 4050"
    mock.device_info.cuda_available = True
    mock.state.frames_processed = 120
    mock.state.effective_fps = 29.5
    mock.state.latest_crowd = {
        "timestamp": 100.0,
        "person_count": 5,
        "zones": [{"name": "zone_1", "person_count": 3, "density": 0.42, "level": "medium"}],
        "growth_per_min": 2.1,
        "growth_alert": False,
        "movement": None,
    }

    sample_report = IncidentReport(
        schema_version="1.0",
        source_module="crowd_fight",
        incident_id="test_inc_001",
        camera_id="cam_default",
        incident_type=IncidentType.FIGHT,
        severity=Severity.HIGH,
        confidence=0.88,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        track_ids=[1, 2],
        bbox=BBox(x1=100.0, y1=150.0, x2=250.0, y2=400.0),
        evidence=Evidence(
            snapshot_path="outputs/snapshots/56dc59e0f761.jpg",
            clip_path="outputs/clips/56dc59e0f761.mp4",
        ),
    )
    mock.manager.recent = [sample_report]
    return mock


@pytest.fixture
def client():
    """FastAPI TestClient without starting background RTSP capture loop."""
    from suraksha.config import load_config
    load_config.cache_clear()
    return TestClient(app)


def test_api_health_endpoint(client, mock_pipeline):
    """Test GET /health returns 200 OK with expected pipeline metrics."""
    app_mod._pipeline = mock_pipeline
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["device"] == "cuda"
    assert data["frames_processed"] == 120
    assert data["effective_fps"] == 29.5


def test_api_crowd_status_endpoint(client, mock_pipeline):
    """Test GET /crowd/status returns latest crowd analytics snapshot."""
    app_mod._pipeline = mock_pipeline
    resp = client.get("/crowd/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["person_count"] == 5
    assert len(data["zones"]) == 1
    assert data["zones"][0]["name"] == "zone_1"


def test_api_incidents_list_and_filter(client, mock_pipeline):
    """Test GET /incidents returns incidents and filters by type."""
    app_mod._pipeline = mock_pipeline
    resp = client.get("/incidents?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["incidents"][0]["incident_id"] == "test_inc_001"

    # Filter matching
    resp_fight = client.get("/incidents?incident_type=fight")
    assert resp_fight.status_code == 200
    assert resp_fight.json()["count"] == 1

    # Filter non-matching
    resp_crowd = client.get("/incidents?incident_type=crowd_panic")
    assert resp_crowd.status_code == 200
    assert resp_crowd.json()["count"] == 0


def test_api_incident_detail(client, mock_pipeline):
    """Test GET /incidents/{id} returns single report or 404."""
    app_mod._pipeline = mock_pipeline
    resp = client.get("/incidents/test_inc_001")
    assert resp.status_code == 200
    assert resp.json()["incident_id"] == "test_inc_001"

    resp_404 = client.get("/incidents/nonexistent_id")
    assert resp_404.status_code == 404


def test_api_evidence_endpoints(client):
    """Test GET /incidents/{id}/snapshot and /clip serve real media or 404, and reject traversal."""
    # Existing verified incident on disk
    resp_snap = client.get("/incidents/56dc59e0f761/snapshot")
    assert resp_snap.status_code == 200
    assert resp_snap.headers["content-type"] == "image/jpeg"
    assert len(resp_snap.content) > 1000

    resp_clip = client.get("/incidents/56dc59e0f761/clip")
    assert resp_clip.status_code == 200
    assert resp_clip.headers["content-type"] == "video/mp4"
    assert len(resp_clip.content) > 10000

    # Test alias routes
    assert client.get("/evidence/snapshot/56dc59e0f761").status_code == 200
    assert client.get("/evidence/clip/56dc59e0f761").status_code == 200

    # Nonexistent IDs
    assert client.get("/incidents/unknown_id/snapshot").status_code == 404
    assert client.get("/incidents/unknown_id/clip").status_code == 404
    assert client.get("/evidence/snapshot/unknown_id").status_code == 404
    assert client.get("/evidence/clip/unknown_id").status_code == 404

    # Path traversal rejection (must return 400 or 404, never 200 or 500)
    assert client.get("/incidents/..%2F..%2Fconfig/snapshot").status_code in (400, 404)
    assert client.get("/incidents/bad*id/clip").status_code in (400, 404)
    assert client.get("/evidence/snapshot/../test").status_code in (400, 404)


def test_websocket_incident_broadcast(client):
    """Test /ws/incidents connects, receives incident push, and handles disconnection."""
    report = IncidentReport(
        schema_version="1.0",
        source_module="crowd_fight",
        incident_id="ws_test_999",
        camera_id="cam_default",
        incident_type=IncidentType.FIGHT,
        severity=Severity.HIGH,
        confidence=0.92,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        track_ids=[3, 7],
    )

    with client.websocket_connect("/ws/incidents") as ws:
        assert len(_ws_clients) == 1

        # Broadcast incident payload
        payload = report.model_dump_json()
        for active_ws in list(_ws_clients):
            asyncio.run(active_ws.send_text(payload))

        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["incident_id"] == "ws_test_999"
        assert data["confidence"] == 0.92
        assert data["track_ids"] == [3, 7]

    # After exiting context, client is removed
    assert len(_ws_clients) == 0


def test_api_video_status_endpoint(client, mock_pipeline):
    """Test GET /video/status returns video streaming pipeline status."""
    app_mod._pipeline = mock_pipeline
    resp = client.get("/video/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "effective_fps" in data


def test_api_system_status_endpoint(client, mock_pipeline):
    """Test GET /system/status returns truthful system, AI, GPU, and telemetry metrics."""
    import time
    mock_pipeline.started_at = time.time() - 42.0
    mock_pipeline.capture.kind = "webcam"
    mock_pipeline.capture.source = "0"
    mock_pipeline.capture.fps = 30.0
    mock_pipeline.capture.width = 1280
    mock_pipeline.capture.height = 720
    mock_pipeline.state.last_frame_time = time.time() - 0.1
    mock_pipeline.state.metrics.total_ms = 14.5
    mock_pipeline.state.metrics.detect_track_ms = 8.2
    mock_pipeline.state.metrics.crowd_ms = 1.1
    mock_pipeline.state.metrics.fight_ms = 2.0
    mock_pipeline.state.metrics.weapon_ms = 3.2

    app_mod._pipeline = mock_pipeline
    app_mod._latest_frame_jpeg = b"\xff\xd8\xff\xe0fakejpeg"
    try:
        resp = client.get("/system/status")
        assert resp.status_code == 200
        data = resp.json()
    finally:
        app_mod._latest_frame_jpeg = None

    assert data["status"] in ("ONLINE", "DEGRADED", "OFFLINE")
    assert data["uptime_seconds"] >= 40.0
    assert data["video_source"]["source_kind"] == "webcam"
    assert data["video_source"]["resolution"] == "1280x720"
    assert data["video_source"]["status"] == "LIVE"
    assert data["ai_inference"]["device"] == "cuda"
    assert data["ai_inference"]["effective_fps"] == 29.5
    assert data["ai_inference"]["inference_latency_ms"] == 14.5
    assert "gpu" in data
    assert data["gpu"]["utilization_percent"] is None  # Truthfully null without continuous profiling
    assert "drone_telemetry" in data
    assert "incidents_count" in data


def test_api_system_status_when_pipeline_none(client):
    """Test GET /system/status gracefully reports OFFLINE when pipeline is not initialized."""
    old_pipeline = app_mod._pipeline
    try:
        app_mod._pipeline = None
        resp = client.get("/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "OFFLINE"
        assert data["video_source"]["status"] == "OFFLINE"
        assert data["ai_inference"]["status"] == "OFFLINE"
    finally:
        app_mod._pipeline = old_pipeline


def test_api_incidents_persisted_disk_and_response(client, mock_pipeline, tmp_path):
    """Test GET /incidents and GET /incidents/{id}/response can query persisted disk files."""
    from suraksha.config import PROJECT_ROOT, load_config
    cfg = load_config()
    report_dir = PROJECT_ROOT / cfg.incidents.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    test_id = "disk_inc_test99"
    test_file = report_dir / f"{test_id}.json"
    sample = {
        "schema_version": "1.0",
        "source_module": "crowd_fight",
        "incident_id": test_id,
        "camera_id": "cam_dwarka_test",
        "incident_type": "weapon",
        "severity": "critical",
        "confidence": 0.94,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "track_ids": [5],
        "details": {"weapon_class": "knife"},
    }
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(sample, f)

    try:
        app_mod._pipeline = mock_pipeline
        # Query detail from disk
        resp = client.get(f"/incidents/{test_id}")
        assert resp.status_code == 200
        assert resp.json()["incident_id"] == test_id

        # Query response plan for disk incident
        resp_plan = client.get(f"/incidents/{test_id}/response")
        assert resp_plan.status_code == 200
        plan_data = resp_plan.json()
        assert "incident_id" in plan_data
        assert plan_data["incident_id"] == test_id
        assert "response_actions" in plan_data
        assert plan_data["human_approval_required"] is True
    finally:
        if test_file.exists():
            test_file.unlink()



