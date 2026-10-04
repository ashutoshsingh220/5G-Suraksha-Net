"""Tests for Phase 3A: Location Foundation with Env-Based Demo Location."""
from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from suraksha.config import AppConfig, LocationConfig, PROJECT_ROOT, VerifyConfig, load_config
from suraksha.incidents.evidence import EvidenceWriter
from suraksha.incidents.manager import EventBus, IncidentManager
from suraksha.incidents.schemas import BBox, IncidentReport, IncidentType, Severity
from suraksha.location import (
    DemoLocationProvider,
    GPSLocationProvider,
    Location,
    LocationProvider,
    LocationSource,
    UnknownLocationProvider,
    create_location_provider,
    validate_google_maps_api_key,
)


# ---------------------------------------------------------------------------
# Test 1: DEMO_LOCATION_QUERY loads from environment
# ---------------------------------------------------------------------------
def test_demo_location_query_loads_from_env(monkeypatch):
    """Verify DEMO_LOCATION_QUERY is loaded from environment and accessible in config."""
    import suraksha.config as c

    c.load_config.cache_clear()
    expected_location = "Yashobhoomi, Dwarka Sector 25, New Delhi"
    monkeypatch.setenv("DEMO_LOCATION_QUERY", expected_location)

    cfg = c.load_config()
    assert cfg.location.query_from_env is True
    assert cfg.location.get_effective_query() == expected_location

    provider = DemoLocationProvider()
    assert provider.query == expected_location
    assert provider.demo_location_query == expected_location
    c.load_config.cache_clear()


# ---------------------------------------------------------------------------
# Test 2: Missing DEMO_LOCATION_QUERY handled safely
# ---------------------------------------------------------------------------
def test_missing_demo_location_query_handled_safely(monkeypatch):
    """Verify that an empty or missing query does not cause errors or crashes."""
    monkeypatch.delenv("DEMO_LOCATION_QUERY", raising=False)
    monkeypatch.delenv("SURAKSHA_DEMO_LOCATION_QUERY", raising=False)

    provider = DemoLocationProvider(query=None)
    assert provider.query == ""

    loc = provider.get_location(camera_id="cam_test_01")
    assert isinstance(loc, Location)
    assert loc.name == "Unknown Location"
    assert loc.source == LocationSource.UNKNOWN
    assert loc.latitude is None
    assert loc.longitude is None
    assert loc.has_coordinates is False


# ---------------------------------------------------------------------------
# Test 3: GOOGLE_MAPS_API_KEY is never exposed in logs or representations
# ---------------------------------------------------------------------------
def test_google_maps_api_key_never_exposed_in_logs(monkeypatch, caplog):
    """Verify secret Google Maps API key is redacted from repr, dump, and logs."""
    fake_secret = "AIzaSy_SUPER_SECRET_KEY_NEVER_PRINT_12345"
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", fake_secret)

    loc_cfg = LocationConfig(mode="demo", google_maps_api_key=fake_secret)

    # 1. Repr must mask the key
    repr_str = repr(loc_cfg)
    assert fake_secret not in repr_str
    assert "***REDACTED***" in repr_str

    # 2. Str must mask the key
    str_val = str(loc_cfg)
    assert fake_secret not in str_val

    # 3. Model dump must mask the key
    dump_dict = loc_cfg.model_dump()
    assert fake_secret not in str(dump_dict)
    assert dump_dict.get("google_maps_api_key") == "***REDACTED***"

    # 4. Logger output must not reveal the key
    with caplog.at_level(logging.DEBUG):
        logger = logging.getLogger("suraksha.test_security")
        logger.info("Config state: %s", repr_str)
    assert fake_secret not in caplog.text

    # 5. Validation function retrieves the secret value safely without logging
    valid_key = validate_google_maps_api_key()
    assert valid_key == fake_secret
    assert fake_secret not in caplog.text

    # 6. If key is missing, validate_google_maps_api_key raises clean error
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.delenv("SURAKSHA_GOOGLE_MAPS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY is not configured"):
        validate_google_maps_api_key(api_key=None)


# ---------------------------------------------------------------------------
# Test 4: .env is ignored by Git
# ---------------------------------------------------------------------------
def test_env_is_ignored_by_git():
    """Verify that .gitignore properly excludes .env and environment secrets."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore file must exist"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
    assert ".env" in lines, ".env must be explicitly listed in .gitignore"
    assert any(".env.*" in line for line in lines), ".env.* wildcard should be ignored"
    assert "!.env.example" in lines, "!.env.example must be preserved for repository documentation"


# ---------------------------------------------------------------------------
# Test 5: DemoLocationProvider exposes the configured query
# ---------------------------------------------------------------------------
def test_demo_location_provider_exposes_configured_query():
    """Verify DemoLocationProvider exposes the exact query string and non-fabricated coords."""
    demo_query = "Yashobhoomi, Dwarka Sector 25, New Delhi"
    provider = DemoLocationProvider(query=demo_query)

    assert provider.query == demo_query
    assert provider.demo_location_query == demo_query

    loc = provider.get_location(camera_id="cam_imc_01")
    assert loc.name == demo_query
    assert loc.camera_id == "cam_imc_01"
    assert loc.source == LocationSource.GEOCODED_DEMO
    # Strictly non-fabricated coordinates in Phase 3A
    assert loc.latitude is None
    assert loc.longitude is None
    assert loc.has_coordinates is False


# ---------------------------------------------------------------------------
# Test 6: No API request occurs in Phase 3A
# ---------------------------------------------------------------------------
def test_no_api_request_occurs_in_phase_3a(monkeypatch):
    """Ensure zero network sockets or HTTP requests are made during Phase 3A location usage."""
    network_called = False

    def guard_socket_connect(*args, **kwargs):
        nonlocal network_called
        network_called = True
        raise RuntimeError("Network access detected! Phase 3A must not make network calls.")

    # Guard socket creation and connection
    monkeypatch.setattr(socket.socket, "connect", guard_socket_connect)

    demo_query = "Yashobhoomi, Dwarka Sector 25, New Delhi"
    provider = DemoLocationProvider(query=demo_query)
    loc = provider.get_location(camera_id="cam_main")

    assert loc.name == demo_query
    assert not network_called, "DemoLocationProvider made an unauthorized network call in Phase 3A"


# ---------------------------------------------------------------------------
# Test 7: Incident can contain location
# ---------------------------------------------------------------------------
def test_incident_can_contain_location():
    """Verify IncidentReport carries typed Location model and roundtrips through JSON."""
    loc = Location(
        name="Yashobhoomi, Dwarka Sector 25, New Delhi",
        latitude=None,
        longitude=None,
        source=LocationSource.GEOCODED_DEMO,
        camera_id="cam_gate_01",
    )
    report = IncidentReport(
        camera_id="cam_gate_01",
        incident_type=IncidentType.WEAPON,
        severity=Severity.HIGH,
        confidence=0.88,
        start_time=datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc),
        location=loc,
        details={"weapon_class": "knife"},
    )

    json_str = report.model_dump_json()
    parsed = json.loads(json_str)

    assert "location" in parsed
    assert parsed["location"]["name"] == "Yashobhoomi, Dwarka Sector 25, New Delhi"
    assert parsed["location"]["source"] == "geocoded_demo"
    assert parsed["location"]["latitude"] is None
    assert parsed["location"]["longitude"] is None
    assert parsed["location"]["camera_id"] == "cam_gate_01"

    # Validate deserialization back to IncidentReport
    recovered = IncidentReport.model_validate_json(json_str)
    assert recovered.location is not None
    assert recovered.location.name == "Yashobhoomi, Dwarka Sector 25, New Delhi"
    assert recovered.location.source == LocationSource.GEOCODED_DEMO
    assert recovered.location.has_coordinates is False


# ---------------------------------------------------------------------------
# Test 8: Incident without location remains valid
# ---------------------------------------------------------------------------
def test_incident_without_location_remains_valid():
    """Verify legacy incident creation without location operates identically (backward compatibility)."""
    report = IncidentReport(
        camera_id="cam_gate_02",
        incident_type=IncidentType.FIGHT,
        severity=Severity.MODERATE,
        confidence=0.75,
        start_time=datetime.now(timezone.utc),
    )
    assert report.location is None

    json_str = report.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["location"] is None

    recovered = IncidentReport.model_validate_json(json_str)
    assert recovered.location is None
    assert recovered.incident_type == IncidentType.FIGHT


# ---------------------------------------------------------------------------
# Test 9: Malformed configuration handled safely
# ---------------------------------------------------------------------------
def test_malformed_configuration_handled_safely():
    """Verify fallback behavior when location config contains unrecognized or empty values."""
    # 1. Unrecognized mode -> UnknownLocationProvider
    cfg_invalid = LocationConfig(mode="satellite_telemetry_invalid")
    provider_fallback = create_location_provider(cfg_invalid)
    assert isinstance(provider_fallback, UnknownLocationProvider)
    loc = provider_fallback.get_location()
    assert loc.source == LocationSource.UNKNOWN

    # 2. None config -> UnknownLocationProvider
    provider_none = create_location_provider(None)
    assert isinstance(provider_none, UnknownLocationProvider)

    # 3. GPS stub provider -> GPSLocationProvider
    cfg_gps = LocationConfig(mode="gps")
    provider_gps = create_location_provider(cfg_gps)
    assert isinstance(provider_gps, GPSLocationProvider)
    loc_gps = provider_gps.get_location(camera_id="cam_drone")
    assert loc_gps.source == LocationSource.GPS
    assert loc_gps.latitude is None


# ---------------------------------------------------------------------------
# Test 10: Location provider integration with IncidentManager
# ---------------------------------------------------------------------------
def test_location_provider_integration_with_incident_manager(tmp_path):
    """Verify IncidentManager automatically attaches location to generated incidents."""
    demo_query = "Yashobhoomi, Dwarka Sector 25, New Delhi"
    provider = DemoLocationProvider(query=demo_query)

    bus = EventBus()
    evidence = MagicMock(spec=EvidenceWriter)
    evidence.cfg = MagicMock()
    evidence.cfg.report_dir = str(tmp_path)
    verify_cfg = VerifyConfig()

    manager = IncidentManager(
        camera_id="cam_main_hall",
        verify_cfg=verify_cfg,
        evidence=evidence,
        bus=bus,
        location_provider=provider,
    )

    emitted: list[IncidentReport] = []
    bus.subscribe(lambda r: emitted.append(r))

    from suraksha.detection.weapon import WeaponTelemetry
    telemetry = WeaponTelemetry(
        weapon_class="pistol",
        confidence=0.85,
        bbox=[100.0, 100.0, 200.0, 200.0],
        frame_timestamp=100.0,
        persistence_count=3,
        confirmation_state="WEAPON_CONFIRMED",
        track_id=1,
    )

    report = manager.report_weapon(telemetry, timestamp=100.0)
    assert report is not None
    assert report.location is not None
    assert report.location.name == demo_query
    assert report.location.source == LocationSource.GEOCODED_DEMO
    assert report.location.latitude is None
    assert report.location.longitude is None
    assert report.location.camera_id == "cam_main_hall"
    assert len(emitted) == 1
    assert emitted[0].location.name == demo_query
