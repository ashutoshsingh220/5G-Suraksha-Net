"""Hermetic unit tests for the drone telemetry module."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from suraksha.api.app import app
from suraksha.telemetry.bridge import MavlinkBridge, ARDUCOPTER_MODES, GPS_FIX_MAP
from suraksha.telemetry.models import DroneTelemetry
from suraksha.telemetry.service import TelemetryService, get_telemetry_service, reset_telemetry_service


class MockMavMessage:
    """Mock MAVLink message with attributes."""

    def __init__(self, msg_type: str, **kwargs):
        self._msg_type = msg_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_type(self) -> str:
        return self._msg_type


@pytest.fixture(autouse=True)
def cleanup_service():
    yield
    reset_telemetry_service()


def test_initial_offline_telemetry():
    """Verify initial state without messages reports OFFLINE."""
    bridge = MavlinkBridge(timeout_s=3.0)
    snap = bridge.get_snapshot()

    assert isinstance(snap, DroneTelemetry)
    assert snap.connected is False
    assert snap.status == "OFFLINE"
    assert snap.telemetry_age_s is None
    assert snap.armed is False
    assert snap.flight_mode == "UNKNOWN"


def test_heartbeat_processing():
    """Verify HEARTBEAT decodes armed flag and copter flight mode."""
    bridge = MavlinkBridge(timeout_s=3.0)

    # 1. Disarmed in STABILIZE (custom_mode=0, base_mode=0)
    hb_disarmed = MockMavMessage("HEARTBEAT", type=2, base_mode=0, custom_mode=0)
    bridge.process_message(hb_disarmed)
    snap = bridge.get_snapshot()

    assert snap.connected is True
    assert snap.status == "ONLINE"
    assert snap.armed is False
    assert snap.flight_mode == "STABILIZE"
    assert snap.system_type == "QUADROTOR"

    # 2. Armed in LOITER (custom_mode=5, base_mode=128)
    hb_armed = MockMavMessage("HEARTBEAT", type=2, base_mode=128, custom_mode=5)
    bridge.process_message(hb_armed)
    snap = bridge.get_snapshot()

    assert snap.armed is True
    assert snap.flight_mode == "LOITER"


def test_global_position_processing():
    """Verify GLOBAL_POSITION_INT decodes coordinates, altitudes, and heading."""
    bridge = MavlinkBridge()
    msg = MockMavMessage(
        "GLOBAL_POSITION_INT",
        lat=286139000,       # 28.6139° N
        lon=772090000,       # 77.2090° E
        alt=250500,          # 250.5 m AMSL
        relative_alt=45200,  # 45.2 m AGL
        vz=-150,             # climb rate +1.5 m/s
        hdg=18050,           # 180.5 deg
    )
    bridge.process_message(msg)
    snap = bridge.get_snapshot()

    assert snap.latitude == pytest.approx(28.6139, abs=1e-5)
    assert snap.longitude == pytest.approx(77.2090, abs=1e-5)
    assert snap.altitude_amsl_m == 250.5
    assert snap.altitude_relative_m == 45.2
    assert snap.climb_rate_m_s == 1.5
    assert snap.heading_deg == 180.5


def test_vfr_hud_processing():
    """Verify VFR_HUD decodes speed, heading, and climb."""
    bridge = MavlinkBridge()
    msg = MockMavMessage(
        "VFR_HUD",
        groundspeed=8.45,
        airspeed=8.80,
        heading=270,
        climb=0.5,
        alt=50.0,
    )
    bridge.process_message(msg)
    snap = bridge.get_snapshot()

    assert snap.groundspeed_m_s == pytest.approx(8.4, abs=0.1)
    assert snap.airspeed_m_s == pytest.approx(8.8, abs=0.1)
    assert snap.heading_deg == 270.0
    assert snap.climb_rate_m_s == 0.5


def test_gps_raw_int_processing():
    """Verify GPS_RAW_INT decodes fix status, satellites, and hdop."""
    bridge = MavlinkBridge()
    msg = MockMavMessage(
        "GPS_RAW_INT",
        fix_type=3,           # 3D Fix
        satellites_visible=14,
        eph=120,              # HDOP 1.20
        lat=286139000,
        lon=772090000,
    )
    bridge.process_message(msg)
    snap = bridge.get_snapshot()

    assert snap.gps_fix_type == "3D Fix"
    assert snap.satellites_visible == 14
    assert snap.hdop == 1.20


def test_sys_status_battery_processing():
    """Verify SYS_STATUS decodes battery percentage, voltage, and current."""
    bridge = MavlinkBridge()
    msg = MockMavMessage(
        "SYS_STATUS",
        battery_remaining=87,
        voltage_battery=16200,   # 16.20 V
        current_battery=1250,    # 12.50 A
    )
    bridge.process_message(msg)
    snap = bridge.get_snapshot()

    assert snap.battery_percent == 87
    assert snap.battery_voltage_v == 16.20
    assert snap.battery_current_a == 12.50


def test_attitude_processing():
    """Verify ATTITUDE converts Euler radians to degrees."""
    import math

    bridge = MavlinkBridge()
    msg = MockMavMessage(
        "ATTITUDE",
        roll=math.radians(10.5),
        pitch=math.radians(-5.2),
        yaw=math.radians(90.0),
    )
    bridge.process_message(msg)
    snap = bridge.get_snapshot()

    assert snap.roll_deg == pytest.approx(10.5, abs=0.1)
    assert snap.pitch_deg == pytest.approx(-5.2, abs=0.1)
    assert snap.yaw_deg == pytest.approx(90.0, abs=0.1)


def test_stale_telemetry_transition():
    """Verify that timeout triggers STALE status."""
    bridge = MavlinkBridge(timeout_s=0.2)

    hb = MockMavMessage("HEARTBEAT", type=2, base_mode=128, custom_mode=5)
    bridge.process_message(hb)

    # Immediately after heartbeat, status should be ONLINE
    snap1 = bridge.get_snapshot()
    assert snap1.status == "ONLINE"
    assert snap1.connected is True

    # Sleep longer than timeout_s (0.2s)
    time.sleep(0.3)
    snap2 = bridge.get_snapshot()
    assert snap2.status == "STALE"
    assert snap2.connected is False
    assert snap2.telemetry_age_s is not None
    assert snap2.telemetry_age_s >= 0.2


def test_telemetry_rest_endpoint():
    """Verify GET /drone/telemetry returns valid snapshot via FastAPI TestClient."""
    client = TestClient(app)
    response = client.get("/drone/telemetry")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "connected" in data
    assert "altitude_relative_m" in data
    assert "battery_percent" in data
    assert "gps_fix_type" in data
