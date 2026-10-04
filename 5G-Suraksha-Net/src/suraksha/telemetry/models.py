"""Pydantic data models for drone and flight controller telemetry."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DroneTelemetry(BaseModel):
    """Normalized snapshot of real-time flight controller telemetry."""

    # Connection & Freshness
    connected: bool = Field(default=False, description="True if valid heartbeat received recently")
    status: str = Field(default="OFFLINE", description="ONLINE | STALE | OFFLINE")
    telemetry_age_s: float | None = Field(default=None, description="Seconds elapsed since last received heartbeat")
    last_heartbeat_timestamp: float | None = Field(default=None, description="Unix timestamp of last heartbeat")

    # Identity
    drone_id: str = Field(default="SURAKSHA-DRONE-01", description="Vehicle identifier")
    platform: str = Field(default="Edge AI (Raspberry Pi + ArduPilot)", description="Platform / Hardware description")
    system_type: str = Field(default="QUADROTOR", description="MAVLink vehicle system type")

    # Flight State & Kinematics
    armed: bool = Field(default=False, description="Vehicle armed status")
    flight_mode: str = Field(default="UNKNOWN", description="Flight controller mode (LOITER, AUTO, RTL, STABILIZE, etc.)")
    altitude_relative_m: float = Field(default=0.0, description="Altitude relative to home/takeoff point (m)")
    altitude_amsl_m: float = Field(default=0.0, description="Altitude above mean sea level (m)")
    groundspeed_m_s: float = Field(default=0.0, description="Ground speed in m/s")
    airspeed_m_s: float = Field(default=0.0, description="Air speed in m/s")
    climb_rate_m_s: float = Field(default=0.0, description="Vertical climb/descent speed in m/s")
    heading_deg: float = Field(default=0.0, description="Compass heading 0-360 degrees")

    # Position
    latitude: float | None = Field(default=None, description="Latitude in decimal degrees")
    longitude: float | None = Field(default=None, description="Longitude in decimal degrees")

    # Power
    battery_percent: int = Field(default=0, description="Remaining battery percentage (0-100)")
    battery_voltage_v: float = Field(default=0.0, description="Main battery voltage (V)")
    battery_current_a: float = Field(default=0.0, description="Battery current draw (A)")

    # Navigation & GPS Quality
    gps_fix_type: str = Field(default="No GPS", description="GPS Fix type (No GPS, 2D Fix, 3D Fix, DGPS, RTK)")
    satellites_visible: int = Field(default=0, description="Number of satellites currently locked")
    hdop: float = Field(default=0.0, description="Horizontal dilution of precision")

    # Attitude (Euler angles)
    roll_deg: float = Field(default=0.0, description="Roll angle in degrees")
    pitch_deg: float = Field(default=0.0, description="Pitch angle in degrees")
    yaw_deg: float = Field(default=0.0, description="Yaw angle in degrees")

    # Companion / Edge Subsystem indicators
    edge_compute: str = Field(default="Active", description="Raspberry Pi edge compute module status")
    camera_status: str = Field(default="Online", description="Camera payload status")
