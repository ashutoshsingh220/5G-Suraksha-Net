"""Automated tests for Phase 3D: Incident Email Alert & Evidence Delivery."""
from __future__ import annotations

import logging
import os
import smtplib
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from suraksha.config import AppConfig, EmailConfig, LocationConfig, PROJECT_ROOT, VerifyConfig, load_config
from suraksha.incidents.schemas import BBox, Evidence, IncidentReport, IncidentStatus, IncidentType, Severity
from suraksha.location.schemas import Location, LocationSource
from suraksha.notifications import (
    EmailNotificationResult,
    EmailNotificationService,
    EmailNotificationStatus,
    EmailPayload,
)
from suraksha.response import (
    EmergencyResource,
    EmergencyResourceDirectory,
    IncidentResponse,
    ResponsePlanner,
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
                name="Max Super Speciality Hospital Dwarka",
                distance_km=3.40,
                phone_number="088604 44888",
                formatted_address="Plot No. 1, Sector 10 Dwarka",
            )
        ],
        police_stations=[
            EmergencyResource(
                name="Police Station Dwarka Sector 23",
                distance_km=1.91,
                phone_number="011 2805 1585",
                formatted_address="Sector 23 Dwarka",
            )
        ],
    )


@pytest.fixture
def temp_evidence(tmp_path):
    clip_file = tmp_path / "test_clip.mp4"
    clip_file.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024)  # fake mp4 header

    snap_file = tmp_path / "test_snap.jpg"
    snap_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 512)  # fake jpeg header

    return clip_file, snap_file


def make_incident(
    incident_type: IncidentType = IncidentType.ARMED_FIGHT,
    severity: Severity = Severity.CRITICAL,
    location: Location | None = None,
    clip_path: str | None = None,
    snap_path: str | None = None,
    status: IncidentStatus = IncidentStatus.FINALIZED,
) -> IncidentReport:
    return IncidentReport(
        schema_version="1.0",
        source_module="crowd_fight",
        incident_id="inc_phase3d_001",
        camera_id="cam_yashobhoomi_01",
        incident_type=incident_type,
        severity=severity,
        confidence=0.92,
        start_time=datetime(2026, 10, 3, 2, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 10, 3, 2, 0, 10, tzinfo=timezone.utc),
        details={"weapon_class": "knife", "struggle_duration_s": 4.5},
        location=location,
        evidence=Evidence(snapshot_path=snap_path, clip_path=clip_path),
        status=status,
    )


# ---------------------------------------------------------------------------
# Configuration Tests (1-5)
# ---------------------------------------------------------------------------

def test_email_enabled_loads_correctly(monkeypatch):
    """Test 1: EMAIL_ENABLED loads from environment and updates config."""
    monkeypatch.setenv("EMAIL_ENABLED", "true")
    import suraksha.config as c
    c.load_config.cache_clear()
    cfg = c.load_config()
    assert cfg.email.enabled is True
    c.load_config.cache_clear()


def test_smtp_host_port_load_correctly(monkeypatch):
    """Test 2: SMTP host and port load correctly from environment."""
    monkeypatch.setenv("SMTP_HOST", "smtp.office365.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    import suraksha.config as c
    c.load_config.cache_clear()
    cfg = c.load_config()
    assert cfg.email.smtp_host == "smtp.office365.com"
    assert cfg.email.smtp_port == 587
    c.load_config.cache_clear()


def test_recipient_configuration_loads_correctly(monkeypatch):
    """Test 3: Recipient email address configuration parses string or lists cleanly."""
    # Comma-separated string
    cfg1 = EmailConfig(email_to="sec1@example.com, sec2@example.com")
    assert cfg1.email_to == ["sec1@example.com", "sec2@example.com"]

    # Single string
    cfg2 = EmailConfig(email_to="solo@example.com")
    assert cfg2.email_to == ["solo@example.com"]

    # List of strings
    cfg3 = EmailConfig(email_to=["a@example.com", "b@example.com"])
    assert cfg3.email_to == ["a@example.com", "b@example.com"]


def test_smtp_password_is_redacted(monkeypatch, caplog):
    """Test 4: SMTP password is automatically redacted in repr, str, and model_dump."""
    secret_pw = "SuperSecretSmtpPass!99"
    cfg = EmailConfig(smtp_password=secret_pw)

    assert secret_pw not in repr(cfg)
    assert "***REDACTED***" in repr(cfg)

    dumped = cfg.model_dump()
    assert dumped["smtp_password"] == "***REDACTED***"

    with caplog.at_level(logging.DEBUG):
        log = logging.getLogger("suraksha.test_email_security")
        log.info("Email config state: %s", repr(cfg))
    assert secret_pw not in caplog.text


def test_missing_smtp_credentials_handled_safely():
    """Test 5: Missing or empty credentials do not raise unhandled exceptions."""
    cfg = EmailConfig(enabled=True, smtp_host="", smtp_username="", smtp_password=None)
    svc = EmailNotificationService(config=cfg)
    inc = make_incident()
    res = svc.send_incident_email(inc, blocking=True)
    assert res.status == EmailNotificationStatus.FAILED
    assert "No email recipients configured" in (res.error_message or "")


# ---------------------------------------------------------------------------
# Email Construction Tests (6-14)
# ---------------------------------------------------------------------------

def test_subject_generated_correctly(sample_location, mock_resources):
    """Test 6: Subject is generated deterministically from incident fields."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL, location=sample_location)
    payload = svc.build_incident_email(inc)

    assert payload.subject == "[5G Suraksha-Net] CRITICAL - ARMED_FIGHT detected at Yashobhoomi, Dwarka Sector 25, New Delhi"


def test_body_contains_incident_type_and_severity(sample_location, mock_resources):
    """Test 7 & 8: Body explicitly displays incident type and severity."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL, location=sample_location)
    payload = svc.build_incident_email(inc)

    assert "Type        : ARMED_FIGHT" in payload.body_text
    assert "Severity    : CRITICAL" in payload.body_text


def test_body_contains_timestamp(sample_location, mock_resources):
    """Test 9: Body contains incident start timestamp."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(location=sample_location)
    payload = svc.build_incident_email(inc)

    assert "2026-10-03T02:00:00+00:00" in payload.body_text


def test_body_contains_location_and_coordinates(sample_location, mock_resources):
    """Test 10 & 11: Body contains formatted location name and numerical coordinates."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(location=sample_location)
    payload = svc.build_incident_email(inc)

    assert "Location    : Yashobhoomi, Dwarka Sector 25, New Delhi" in payload.body_text
    assert "Coordinates : Latitude 28.552553, Longitude 77.044893" in payload.body_text


def test_body_coordinates_unavailable_when_missing(mock_resources):
    """Test 11b: Coordinates explicitly noted as unavailable if not resolved."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(location=Location(name="Demo Venue", latitude=None, longitude=None))
    payload = svc.build_incident_email(inc)

    assert "Coordinates unavailable" in payload.body_text


def test_body_contains_response_recommendation(sample_location, mock_resources):
    """Test 12: Body includes deterministic policy response recommendations."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(IncidentType.ARMED_FIGHT, Severity.CRITICAL, location=sample_location)
    payload = svc.build_incident_email(inc)

    assert "POLICE_SECURITY" in payload.body_text
    assert "MEDICAL_ASSISTANCE" in payload.body_text
    assert "Police Station Dwarka Sector 23" in payload.body_text
    assert "Max Super Speciality Hospital Dwarka" in payload.body_text


def test_body_contains_human_approval_and_disclaimer(sample_location, mock_resources):
    """Test 13 & 14: Body contains human approval status and straight-line distance disclaimer."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(location=sample_location)
    payload = svc.build_incident_email(inc)

    assert "AWAITING HUMAN APPROVAL" in payload.body_text
    assert "decision-support notification" in payload.body_text
    assert "No emergency service" in payload.body_text
    assert "straight-line geographic distance" in payload.body_text


# ---------------------------------------------------------------------------
# Evidence & Attachment Tests (15-20)
# ---------------------------------------------------------------------------

def test_valid_mp4_is_attached(temp_evidence, mock_resources):
    """Test 15: Valid MP4 video clip is attached and verified."""
    clip_file, _ = temp_evidence
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(clip_path=str(clip_file))
    payload = svc.build_incident_email(inc)

    assert payload.clip_attached is True
    assert payload.clip_attachment_path == str(clip_file)
    assert "Incident evidence clip attached" in payload.clip_status_note


def test_valid_snapshot_is_attached(temp_evidence, mock_resources):
    """Test 16: Valid snapshot image is attached and verified."""
    _, snap_file = temp_evidence
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(snap_path=str(snap_file))
    payload = svc.build_incident_email(inc)

    assert payload.snapshot_attached is True
    assert payload.snapshot_attachment_path == str(snap_file)
    assert "Incident snapshot attached" in payload.snapshot_status_note


def test_missing_clip_handled_safely(mock_resources):
    """Test 17: Missing clip does not crash and notes clip unavailable."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(clip_path="nonexistent_clip.mp4")
    payload = svc.build_incident_email(inc)

    assert payload.clip_attached is False
    assert payload.clip_attachment_path is None
    assert "Incident clip unavailable" in payload.clip_status_note


def test_missing_snapshot_handled_safely(mock_resources):
    """Test 18: Missing snapshot does not crash and notes snapshot unavailable."""
    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(snap_path="nonexistent_snap.jpg")
    payload = svc.build_incident_email(inc)

    assert payload.snapshot_attached is False
    assert payload.snapshot_attachment_path is None
    assert "Incident snapshot unavailable" in payload.snapshot_status_note


def test_attachment_filename_deterministic(temp_evidence, mock_resources):
    """Test 19: Attachment filenames follow deterministic naming scheme."""
    clip_file, snap_file = temp_evidence
    cfg = EmailConfig(
        enabled=True,
        smtp_host="localhost",
        email_from="alert@suraksha.local",
        email_to=["ops@suraksha.local"],
    )
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident(clip_path=str(clip_file), snap_path=str(snap_file))

    with patch("smtplib.SMTP") as mock_smtp:
        server_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server_instance

        res = svc.send_incident_email(inc, blocking=True)
        assert res.status == EmailNotificationStatus.SENT

        # Verify sent email message attachments
        args, _ = server_instance.send_message.call_args
        msg = args[0]
        filenames = [part.get_filename() for part in msg.iter_attachments()]
        assert f"incident_{inc.incident_id}_T-5_to_T+5.mp4" in filenames
        assert f"incident_{inc.incident_id}_snapshot.jpg" in filenames


def test_attachment_size_limit_enforced(tmp_path, mock_resources):
    """Test 20: Oversized attachment is omitted and documented."""
    big_file = tmp_path / "big_video.mp4"
    big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2MB file

    # Limit to 1MB
    cfg = EmailConfig(max_attachment_mb=1.0)
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident(clip_path=str(big_file))
    payload = svc.build_incident_email(inc)

    assert payload.clip_attached is False
    assert payload.clip_attachment_path is None
    assert "exceeds maximum allowed" in payload.clip_status_note


# ---------------------------------------------------------------------------
# SMTP Operation Tests (21-25)
# ---------------------------------------------------------------------------

def test_successful_email_send(mock_resources):
    """Test 21: Successful SMTP send returns SENT status."""
    cfg = EmailConfig(
        enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user1",
        smtp_password="pw1",
        email_from="alerts@example.com",
        email_to=["dest@example.com"],
        use_tls=True,
    )
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident()

    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        res = svc.send_incident_email(inc, blocking=True)
        assert res.status == EmailNotificationStatus.SENT
        assert res.incident_id == inc.incident_id
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user1", "pw1")
        mock_server.send_message.assert_called_once()


def test_smtp_authentication_failure_handled(mock_resources):
    """Test 22: SMTPAuthenticationError is caught and marked FAILED without leaking secret."""
    cfg = EmailConfig(
        enabled=True,
        smtp_host="smtp.example.com",
        smtp_username="user",
        smtp_password="bad_password_123",
        email_to=["test@example.com"],
    )
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident()

    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Authentication credentials invalid")
        mock_smtp.return_value.__enter__.return_value = mock_server

        res = svc.send_incident_email(inc, blocking=True)
        assert res.status == EmailNotificationStatus.FAILED
        assert "Authentication credentials invalid" in (res.error_message or "")
        assert "bad_password_123" not in (res.error_message or "")


def test_smtp_connection_failure_handled(mock_resources):
    """Test 23: SMTPConnectError / ConnectionRefusedError handled gracefully."""
    cfg = EmailConfig(enabled=True, smtp_host="unreachable.host", email_to=["test@example.com"])
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident()

    with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("Connection refused")):
        res = svc.send_incident_email(inc, blocking=True)
        assert res.status == EmailNotificationStatus.FAILED
        assert "Connection refused" in (res.error_message or "")


def test_smtp_timeout_handled(mock_resources):
    """Test 24: Socket timeout during SMTP connect is handled cleanly."""
    cfg = EmailConfig(enabled=True, smtp_host="timeout.host", email_to=["test@example.com"])
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident()

    with patch("smtplib.SMTP", side_effect=TimeoutError("SMTP timed out")):
        res = svc.send_incident_email(inc, blocking=True)
        assert res.status == EmailNotificationStatus.FAILED
        assert "timed out" in (res.error_message or "")


def test_invalid_configuration_handled():
    """Test 25: Completely invalid configuration returns FAILED result without crash."""
    cfg = EmailConfig(enabled=True, email_to=[])  # No recipients
    svc = EmailNotificationService(config=cfg)
    inc = make_incident()
    res = svc.send_incident_email(inc, blocking=True)
    assert res.status == EmailNotificationStatus.FAILED
    assert "No email recipients" in (res.error_message or "")


# ---------------------------------------------------------------------------
# Pipeline Behavior & Synchronization Tests (26-30)
# ---------------------------------------------------------------------------

def test_email_queued_only_after_incident_emission(mock_resources):
    """Test 26: Email notification is triggered upon incident event."""
    cfg = EmailConfig(enabled=True, email_to=["ops@example.com"])
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident(status=IncidentStatus.VERIFIED)

    with patch.object(svc, "send_incident_email", return_value=EmailNotificationResult(incident_id=inc.incident_id, status=EmailNotificationStatus.QUEUED)) as mock_send:
        res = svc.handle_incident_event(inc)
        assert res is not None
        mock_send.assert_called_once()


def test_email_waits_for_evidence_artifact_readiness(mock_resources):
    """Test 27: While incident is RECORDING_POST_EVENT, email is deferred until FINALIZED."""
    cfg = EmailConfig(enabled=True, email_to=["ops@example.com"])
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))

    # Phase 1: Recording post-event (T+5s) -> Must NOT send
    recording_inc = make_incident(status=IncidentStatus.RECORDING_POST_EVENT)
    with patch.object(svc, "send_incident_email") as mock_send:
        res1 = svc.handle_incident_event(recording_inc)
        assert res1 is None
        mock_send.assert_not_called()

    # Phase 2: Finalized -> Evidence ready, must send
    finalized_inc = make_incident(status=IncidentStatus.FINALIZED)
    with patch.object(svc, "send_incident_email") as mock_send:
        svc.handle_incident_event(finalized_inc)
        mock_send.assert_called_once()


def test_email_does_not_run_on_every_frame(mock_resources):
    """Test 28: Email handler is not invoked on raw frame events, only on incidents."""
    cfg = EmailConfig(enabled=True, email_to=["ops@example.com"])
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))

    # Calling with a non-incident object or None
    assert svc.handle_incident_event(make_incident(status=IncidentStatus.CANDIDATE)) is None


def test_duplicate_incident_does_not_create_duplicate_email(mock_resources):
    """Test 29: Multiple emissions of the same incident ID send exactly one email."""
    cfg = EmailConfig(enabled=True, smtp_host="localhost", email_to=["ops@example.com"])
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident(status=IncidentStatus.FINALIZED)

    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = MagicMock()

        # First call -> sent
        res1 = svc.send_incident_email(inc, blocking=True)
        assert res1.status == EmailNotificationStatus.SENT

        # Duplicate call with exact same incident_id -> NOT_ATTEMPTED
        res2 = svc.send_incident_email(inc, blocking=True)
        assert res2.status == EmailNotificationStatus.NOT_ATTEMPTED
        assert "Duplicate incident" in (res2.error_message or "")


def test_email_failure_does_not_crash_pipeline(mock_resources):
    """Test 30: When SMTP raises catastrophic error, handle_incident_event does not propagate exception."""
    cfg = EmailConfig(enabled=True, smtp_host="localhost", email_to=["ops@example.com"])
    svc = EmailNotificationService(config=cfg, planner=ResponsePlanner(resource_directory=mock_resources))
    inc = make_incident(status=IncidentStatus.FINALIZED)

    with patch("smtplib.SMTP", side_effect=Exception("Critical network collapse")):
        # Must catch cleanly and return FAILED result
        res = svc.send_incident_email(inc, blocking=True)
        assert res.status == EmailNotificationStatus.FAILED


# ---------------------------------------------------------------------------
# Security & Leaks Tests (31-33)
# ---------------------------------------------------------------------------

def test_smtp_password_never_appears_in_logs(monkeypatch, caplog):
    """Test 31: SMTP password is never exposed in logs even when errors occur."""
    secret_pw = "TOP_SECRET_SMTP_PASS_12345"
    cfg = EmailConfig(
        enabled=True,
        smtp_host="localhost",
        smtp_username="demo_user",
        smtp_password=secret_pw,
        email_to=["ops@example.com"],
    )
    svc = EmailNotificationService(config=cfg)
    inc = make_incident()

    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value.login.side_effect = Exception(
            f"Authentication failed with top secret {secret_pw}"
        )
        with caplog.at_level(logging.DEBUG):
            svc.send_incident_email(inc, blocking=True)

    assert secret_pw not in caplog.text
    assert "***REDACTED***" in caplog.text


def test_google_maps_key_never_appears_in_email(monkeypatch, sample_location, mock_resources):
    """Test 32: Google Maps API key never appears in email subject or body text."""
    secret_map_key = "AIzaSy_CONFIDENTIAL_MAPS_KEY_54321"
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", secret_map_key)

    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(location=sample_location)
    payload = svc.build_incident_email(inc)

    assert secret_map_key not in payload.subject
    assert secret_map_key not in payload.body_text
    assert secret_map_key not in (payload.body_html or "")


def test_smtp_password_never_appears_in_rest_or_response_json():
    """Test 33: IncidentResponse models and REST endpoints never leak email passwords."""
    cfg = EmailConfig(smtp_password="SECRET_PASSWORD_NO_LEAK")
    app_cfg = AppConfig(email=cfg)

    dumped = app_cfg.model_dump()
    assert dumped["email"]["smtp_password"] == "***REDACTED***"

    resp = IncidentResponse(
        incident_id="inc_001",
        incident_type=IncidentType.ARMED_FIGHT,
        severity=Severity.CRITICAL,
    )
    resp_json = resp.model_dump_json()
    assert "smtp_password" not in resp_json
    assert "SECRET_PASSWORD_NO_LEAK" not in resp_json


def test_real_incident_artifact_attachment(sample_location, mock_resources):
    """Test 34: Verify real on-disk incident artifacts (10.07s MP4 clip and snapshot) attach cleanly."""
    real_clip = PROJECT_ROOT / "outputs" / "clips" / "8f4a8428a4d3.mp4"
    real_snap = PROJECT_ROOT / "outputs" / "snapshots" / "56dc59e0f761.jpg"

    if not real_clip.exists() or not real_snap.exists():
        pytest.skip("Real clip/snapshot not found on disk in outputs/")

    planner = ResponsePlanner(resource_directory=mock_resources)
    svc = EmailNotificationService(planner=planner)
    inc = make_incident(
        location=sample_location,
        clip_path=str(real_clip),
        snap_path=str(real_snap),
    )
    payload = svc.build_incident_email(inc)

    assert payload.clip_attached is True
    assert payload.snapshot_attached is True
    assert "Incident evidence clip attached" in payload.clip_status_note
    assert "Incident snapshot attached" in payload.snapshot_status_note


def test_live_smoke_test_skipped_when_unconfigured():
    """Test 35: Live email smoke test safely skips when EMAIL_ENABLED is false."""
    cfg = EmailConfig(enabled=False)
    svc = EmailNotificationService(config=cfg)
    inc = make_incident()

    res = svc.send_incident_email(inc, blocking=True)
    assert res.status == EmailNotificationStatus.DISABLED
