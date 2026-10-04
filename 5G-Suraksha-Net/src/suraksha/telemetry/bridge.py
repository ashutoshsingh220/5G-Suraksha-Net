"""MAVLink Bridge for real-time telemetry from Mission Planner / ArduPilot.

SAFETY CRITICAL:
This bridge is strictly READ-ONLY. It never transmits flight commands,
never sends heartbeat packets, never arms/disarms, and never sets vehicle modes.
It passively listens to forwarded MAVLink datagrams (e.g. from Mission Planner UDP mirror).
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any

from suraksha.logging_utils import get_logger
from suraksha.telemetry.models import DroneTelemetry

log = get_logger(__name__)

ARDUCOPTER_MODES: dict[int, str] = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    8: "POSITION",
    9: "LAND",
    10: "OF_LOITER",
    11: "DRIFT",
    13: "SPORT",
    14: "FLIP",
    15: "AUTOTUNE",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
    22: "FLOWHOLD",
    23: "FOLLOW",
    24: "ZIGZAG",
    25: "SYSTEMID",
    26: "AUTOROTATE",
    27: "AUTO_RTL",
    28: "TURTLE",
}

GPS_FIX_MAP: dict[int, str] = {
    0: "No GPS",
    1: "No Fix",
    2: "2D Fix",
    3: "3D Fix",
    4: "DGPS",
    5: "RTK Float",
    6: "RTK Fixed",
}


def detect_flight_controller_port() -> tuple[str, int] | None:
    """Scan local serial COM ports for connected ArduPilot / Pixhawk USB devices."""
    try:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            if "ardupilot" in desc or "pixhawk" in desc or "px4" in desc or "1209:5741" in hwid:
                return p.device, 115200
    except Exception:
        pass
    return None


class MavlinkBridge:
    """Passively receives and parses MAVLink telemetry streams."""

    def __init__(
        self,
        connection_string: str = "udpin:127.0.0.1:14550",
        baud_rate: int = 57600,
        timeout_s: float = 3.0,
    ) -> None:
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._mav = None
        self._active_conn_info = connection_string
        self._last_stream_req_time: float = 0.0

        # Internal state store
        self._last_heartbeat_time: float = 0.0
        self._state: dict[str, Any] = {
            "drone_id": "SURAKSHA-DRONE-01",
            "platform": "Edge AI (Raspberry Pi + ArduPilot)",
            "system_type": "QUADROTOR",
            "armed": False,
            "flight_mode": "UNKNOWN",
            "altitude_relative_m": 0.0,
            "altitude_amsl_m": 0.0,
            "groundspeed_m_s": 0.0,
            "airspeed_m_s": 0.0,
            "climb_rate_m_s": 0.0,
            "heading_deg": 0.0,
            "latitude": None,
            "longitude": None,
            "battery_percent": 0,
            "battery_voltage_v": 0.0,
            "battery_current_a": 0.0,
            "gps_fix_type": "No GPS",
            "satellites_visible": 0,
            "hdop": 0.0,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
            "edge_compute": "Active",
            "camera_status": "Online",
        }

    def start(self) -> None:
        """Start the background receiver thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("MavlinkBridge is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="mavlink_bridge_worker",
            daemon=True,
        )
        self._thread.start()
        log.info("MavlinkBridge started on %s", self.connection_string)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background worker thread."""
        self._stop_event.set()
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        log.info("MavlinkBridge stopped")

    def _worker_loop(self) -> None:
        """Continuously receive MAVLink packets without ever sending commands."""
        try:
            from pymavlink import mavutil
        except ImportError:
            log.error("pymavlink is not installed. MavlinkBridge cannot run.")
            return

        while not self._stop_event.is_set():
            if self._mav is None:
                conn_target = self.connection_string
                baud = self.baud_rate

                # If connection_string is "auto", "udpin:127.0.0.1:14550", or explicit COM port,
                # attempt direct USB Pixhawk connection first
                if conn_target.lower() == "auto" or conn_target == "udpin:127.0.0.1:14550" or conn_target.upper().startswith("COM"):
                    detected = detect_flight_controller_port()
                    if detected and (conn_target.lower() == "auto" or conn_target == "udpin:127.0.0.1:14550" or conn_target.upper() == detected[0].upper()):
                        port_dev, port_baud = detected
                        try:
                            log.info("Detected Pixhawk USB on %s. Connecting directly...", port_dev)
                            self._mav = mavutil.mavlink_connection(
                                port_dev,
                                baud=port_baud,
                                autoreconnect=True,
                            )
                            self._active_conn_info = f"{port_dev} @ {port_baud} (Direct USB)"
                        except Exception as e:
                            log.warning("Could not open %s directly (%s). Falling back to UDP mirror...", port_dev, e)
                            self._mav = None

                # Fall back to UDP mirror (e.g. from Mission Planner) if direct serial port isn't available
                if self._mav is None:
                    target_to_use = "udpin:127.0.0.1:14550" if conn_target.lower() == "auto" else conn_target
                    try:
                        log.info("Connecting MAVLink receiver to %s", target_to_use)
                        self._mav = mavutil.mavlink_connection(
                            target_to_use,
                            baud=baud,
                            autoreconnect=True,
                        )
                        self._active_conn_info = target_to_use
                    except Exception as e:
                        log.warning("Failed to bind MAVLink connection (%s): %s. Retrying in 2s...", target_to_use, e)
                        time.sleep(2.0)
                        continue

            try:
                # Read next message with small timeout to allow graceful stop check
                msg = self._mav.recv_match(blocking=True, timeout=0.25)
                if msg is None:
                    continue

                self.process_message(msg)
            except Exception as e:
                if not self._stop_event.is_set():
                    log.debug("MAVLink receive exception: %s", e)
                    time.sleep(0.1)

    def process_message(self, msg: Any) -> None:
        """Decode a MAVLink message and update internal state in a thread-safe manner."""
        from pymavlink import mavutil

        msg_type = msg.get_type() if hasattr(msg, "get_type") else ""
        now = time.time()

        with self._lock:
            if msg_type == "HEARTBEAT":
                self._last_heartbeat_time = now
                base_mode = getattr(msg, "base_mode", 0)
                # MAV_MODE_FLAG_SAFETY_ARMED = 128
                self._state["armed"] = bool(base_mode & 128)

                # Decode flight mode
                custom_mode = getattr(msg, "custom_mode", 0)
                mode_str = ARDUCOPTER_MODES.get(custom_mode, f"MODE_{custom_mode}")
                self._state["flight_mode"] = mode_str

                # Vehicle type
                v_type = getattr(msg, "type", 2)
                if v_type in (1, 2, 3, 4, 13, 14, 15):
                    self._state["system_type"] = "QUADROTOR"
                elif v_type == 1:
                    self._state["system_type"] = "FIXED_WING"

                # Request data stream periodically to maintain telemetry flow on direct serial
                if now - self._last_stream_req_time > 4.0:
                    self._last_stream_req_time = now
                    if self._mav is not None and getattr(self._mav, "target_system", 0):
                        try:
                            self._mav.mav.request_data_stream_send(
                                self._mav.target_system,
                                self._mav.target_component,
                                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                                4,
                                1,
                            )
                        except Exception:
                            pass

            elif msg_type == "GLOBAL_POSITION_INT":
                lat = getattr(msg, "lat", 0)
                lon = getattr(msg, "lon", 0)
                if lat != 0 or lon != 0:
                    self._state["latitude"] = round(lat / 1e7, 7)
                    self._state["longitude"] = round(lon / 1e7, 7)

                alt_amsl = getattr(msg, "alt", 0)
                rel_alt = getattr(msg, "relative_alt", 0)
                self._state["altitude_amsl_m"] = round(alt_amsl / 1000.0, 1)
                self._state["altitude_relative_m"] = round(rel_alt / 1000.0, 1)

                vz = getattr(msg, "vz", 0)
                self._state["climb_rate_m_s"] = round(-vz / 100.0, 1)

                hdg = getattr(msg, "hdg", 65535)
                if hdg != 65535:
                    self._state["heading_deg"] = round(hdg / 100.0, 1)

            elif msg_type == "VFR_HUD":
                self._state["heading_deg"] = float(getattr(msg, "heading", self._state["heading_deg"]))
                self._state["groundspeed_m_s"] = round(float(getattr(msg, "groundspeed", 0.0)), 1)
                self._state["airspeed_m_s"] = round(float(getattr(msg, "airspeed", 0.0)), 1)
                self._state["climb_rate_m_s"] = round(float(getattr(msg, "climb", 0.0)), 1)
                alt = getattr(msg, "alt", 0.0)
                if self._state["altitude_amsl_m"] == 0.0:
                    self._state["altitude_amsl_m"] = round(float(alt), 1)

            elif msg_type == "GPS_RAW_INT":
                fix_type = getattr(msg, "fix_type", 0)
                self._state["gps_fix_type"] = GPS_FIX_MAP.get(fix_type, f"Fix {fix_type}")
                self._state["satellites_visible"] = int(getattr(msg, "satellites_visible", 0))
                eph = getattr(msg, "eph", 65535)
                if eph < 65535:
                    self._state["hdop"] = round(eph / 100.0, 2)

                lat = getattr(msg, "lat", 0)
                lon = getattr(msg, "lon", 0)
                if (lat != 0 or lon != 0) and self._state["latitude"] is None:
                    self._state["latitude"] = round(lat / 1e7, 7)
                    self._state["longitude"] = round(lon / 1e7, 7)

            elif msg_type == "SYS_STATUS":
                rem = getattr(msg, "battery_remaining", -1)
                if rem >= 0:
                    self._state["battery_percent"] = max(0, min(100, int(rem)))

                volt = getattr(msg, "voltage_battery", -1)
                if volt > 0:
                    self._state["battery_voltage_v"] = round(volt / 1000.0, 2)

                curr = getattr(msg, "current_battery", -1)
                if curr >= 0:
                    self._state["battery_current_a"] = round(curr / 100.0, 2)

            elif msg_type == "BATTERY_STATUS":
                rem = getattr(msg, "battery_remaining", -1)
                if rem >= 0:
                    self._state["battery_percent"] = max(0, min(100, int(rem)))

                curr = getattr(msg, "current_battery", -1)
                if curr >= 0:
                    self._state["battery_current_a"] = round(curr / 100.0, 2)

            elif msg_type == "ATTITUDE":
                roll = getattr(msg, "roll", 0.0)
                pitch = getattr(msg, "pitch", 0.0)
                yaw = getattr(msg, "yaw", 0.0)
                self._state["roll_deg"] = round(math.degrees(roll), 1)
                self._state["pitch_deg"] = round(math.degrees(pitch), 1)
                self._state["yaw_deg"] = round((math.degrees(yaw) + 360) % 360, 1)

    def get_snapshot(self) -> DroneTelemetry:
        """Return a fresh DroneTelemetry model evaluated against freshness timeout."""
        now = time.time()
        with self._lock:
            last_hb = self._last_heartbeat_time
            state_copy = dict(self._state)

        if last_hb == 0.0:
            connected = False
            status = "OFFLINE"
            age = None
        else:
            age = round(now - last_hb, 1)
            if age <= self.timeout_s:
                connected = True
                status = "ONLINE"
            else:
                connected = False
                status = "STALE"

        return DroneTelemetry(
            connected=connected,
            status=status,
            telemetry_age_s=age,
            last_heartbeat_timestamp=last_hb if last_hb > 0.0 else None,
            **state_copy,
        )
