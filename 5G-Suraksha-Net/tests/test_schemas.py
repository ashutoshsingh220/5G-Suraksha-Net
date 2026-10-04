import json
from datetime import datetime, timezone

from suraksha.incidents.schemas import (
    SCHEMA_VERSION,
    SOURCE_MODULE,
    BBox,
    Evidence,
    IncidentReport,
    IncidentType,
    Severity,
)


def test_report_serialization_roundtrip():
    r = IncidentReport(
        camera_id="cam_default",
        incident_type=IncidentType.FIGHT,
        severity=Severity.HIGH,
        confidence=0.87,
        start_time=datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc),
        track_ids=[3, 7],
        bbox=BBox(x1=100.0, y1=200.0, x2=300.0, y2=500.0),
        details={"windows_scored": 4},
        evidence=Evidence(snapshot_path="outputs/snapshots/abc.jpg", clip_path="outputs/clips/abc.mp4"),
    )
    raw = r.model_dump_json()
    back = IncidentReport.model_validate_json(raw)
    assert back.incident_type == IncidentType.FIGHT
    assert back.track_ids == [3, 7]
    assert back.schema_version == SCHEMA_VERSION
    assert back.source_module == SOURCE_MODULE


def test_json_is_integration_friendly():
    """Teammates' modules parse plain JSON — verify keys are stable strings."""
    r = IncidentReport(
        camera_id="cam1",
        incident_type=IncidentType.CROWD_DENSITY_CRITICAL,
        severity=Severity.CRITICAL,
        confidence=0.95,
        start_time=IncidentReport.utc_now(),
        zone="market_square",
        person_count=87,
    )
    data = json.loads(r.model_dump_json())
    assert data["incident_type"] == "crowd_density_critical"
    assert data["severity"] == "critical"
    assert data["zone"] == "market_square"
    assert set(data) >= {
        "schema_version", "source_module", "incident_id", "camera_id",
        "incident_type", "severity", "confidence", "start_time", "evidence",
    }
