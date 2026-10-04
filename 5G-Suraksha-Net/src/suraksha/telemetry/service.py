"""Telemetry Service managing lifecycle, caching, and WebSocket subscriptions."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket
from suraksha.logging_utils import get_logger
from suraksha.telemetry.bridge import MavlinkBridge
from suraksha.telemetry.models import DroneTelemetry

log = get_logger(__name__)

_instance: TelemetryService | None = None


class TelemetryService:
    """High-level service coordinating the MAVLink bridge and real-time frontend delivery."""

    def __init__(
        self,
        connection_string: str = "udpin:127.0.0.1:14550",
        baud_rate: int = 57600,
        timeout_s: float = 3.0,
        broadcast_rate_hz: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.broadcast_rate_hz = max(1.0, min(20.0, broadcast_rate_hz))
        self.bridge = MavlinkBridge(
            connection_string=connection_string,
            baud_rate=baud_rate,
            timeout_s=timeout_s,
        )
        self._ws_clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        """Start the telemetry background bridge and subscriber worker."""
        if not self.enabled:
            log.info("TelemetryService is disabled by configuration.")
            return

        self._running = True
        self.bridge.start()
        log.info("TelemetryService started (rate: %.1f Hz)", self.broadcast_rate_hz)

    def stop(self) -> None:
        """Stop bridge and cleanup clients."""
        self._running = False
        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
            self._broadcast_task = None

        self.bridge.stop()
        self._ws_clients.clear()
        log.info("TelemetryService stopped")

    def get_telemetry(self) -> DroneTelemetry:
        """Get latest telemetry snapshot."""
        if not self.enabled:
            return DroneTelemetry(
                connected=False,
                status="OFFLINE",
                telemetry_age_s=None,
            )
        return self.bridge.get_snapshot()

    def register_client(self, ws: WebSocket) -> None:
        """Register a WebSocket client for real-time telemetry push."""
        self._ws_clients.add(ws)

    def unregister_client(self, ws: WebSocket) -> None:
        """Unregister a WebSocket client."""
        self._ws_clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._ws_clients)

    async def broadcast_loop(self) -> None:
        """Async task broadcasting normalized snapshots to connected clients at fixed frequency."""
        interval = 1.0 / self.broadcast_rate_hz
        log.info("Starting telemetry WebSocket broadcaster (interval=%.3fs)", interval)
        try:
            while self._running:
                if self._ws_clients:
                    snapshot = self.get_telemetry()
                    payload = snapshot.model_dump_json()

                    dead = []
                    for ws in list(self._ws_clients):
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self._ws_clients.discard(ws)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("Error in telemetry broadcast loop: %s", e)


def get_telemetry_service(cfg: Any = None) -> TelemetryService:
    """Return the global TelemetryService singleton."""
    global _instance
    if _instance is None:
        if cfg is not None and hasattr(cfg, "telemetry"):
            t_cfg = cfg.telemetry
            _instance = TelemetryService(
                connection_string=t_cfg.connection_string,
                baud_rate=getattr(t_cfg, "baud_rate", 57600),
                timeout_s=t_cfg.timeout_s,
                broadcast_rate_hz=t_cfg.broadcast_rate_hz,
                enabled=t_cfg.enabled,
            )
        else:
            _instance = TelemetryService()
    return _instance


def reset_telemetry_service() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _instance
    if _instance is not None:
        _instance.stop()
        _instance = None
