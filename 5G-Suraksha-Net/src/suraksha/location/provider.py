"""LocationProvider interface and implementations for 5G Suraksha-Net."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from suraksha.location.schemas import Location, LocationSource
from suraksha.logging_utils import get_logger

if TYPE_CHECKING:
    from suraksha.config import LocationConfig

log = get_logger(__name__)


def validate_google_maps_api_key(api_key: str | None = None) -> str:
    """Validate that Google Maps API key exists when Google Maps services are invoked.

    NOTE: In Phase 3A, Google Maps calls are deferred. This function is only
    called when actual Google Maps operations are invoked (in Phase 3B).
    """
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("SURAKSHA_GOOGLE_MAPS_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "GOOGLE_MAPS_API_KEY is not configured. "
            "Please set GOOGLE_MAPS_API_KEY in your .env file or environment."
        )
    return key.strip()


class LocationProvider(ABC):
    """Abstract base class for all location resolution providers."""

    @abstractmethod
    def get_location(
        self,
        camera_id: str | None = None,
        timestamp: float | None = None,
    ) -> Location:
        """Retrieve location for the given camera/incident context."""
        pass


class DemoLocationProvider(LocationProvider):
    """Demonstration location provider backed by Google Maps Geocoding API."""

    def __init__(self, query: str | None = None, api_key: str | None = None) -> None:
        env_query = os.environ.get("DEMO_LOCATION_QUERY") or os.environ.get("SURAKSHA_DEMO_LOCATION_QUERY")
        raw_query = query if query is not None else env_query
        self.query: str = (raw_query or "Yashobhoomi, Dwarka Sector 25, New Delhi").strip()
        self.demo_location_query: str = self.query
        self.api_key: str = (
            api_key
            or os.environ.get("GOOGLE_MAPS_API_KEY")
            or os.environ.get("SURAKSHA_GOOGLE_MAPS_API_KEY")
            or ""
        ).strip()
        self._cached_location: Location | None = None

    def _resolve_geocoding(self) -> Location:
        if self._cached_location is not None:
            return self._cached_location

        if self.api_key and self.query:
            try:
                import json
                import urllib.parse
                import urllib.request
                url = f"https://maps.googleapis.com/maps/api/geocode/json?address={urllib.parse.quote(self.query)}&key={self.api_key}"
                req = urllib.request.Request(url, headers={"User-Agent": "5G-Suraksha-Net/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "OK" and data.get("results"):
                        res = data["results"][0]
                        fmt_addr = res.get("formatted_address") or self.query
                        geom = res.get("geometry", {}).get("location", {})
                        lat = geom.get("lat")
                        lng = geom.get("lng")
                        if lat is not None and lng is not None:
                            self._cached_location = Location(
                                name=f"{self.query} ({fmt_addr})",
                                latitude=float(lat),
                                longitude=float(lng),
                                source=LocationSource.GEOCODED_DEMO,
                            )
                            log.info("GOOGLE_MAPS_GEOCODE_OK query='%s' lat=%.6f lng=%.6f", self.query, lat, lng)
                            return self._cached_location
            except Exception as e:
                log.warning("Google Maps geocoding query failed: %s", e)

        # Verified fallback coordinates for Yashobhoomi, Sector 25 Dwarka
        self._cached_location = Location(
            name=f"{self.query} (Sector 25 Dwarka, New Delhi, 110077, India)",
            latitude=28.552553,
            longitude=77.044893,
            source=LocationSource.GEOCODED_DEMO,
        )
        return self._cached_location

    def get_location(
        self,
        camera_id: str | None = None,
        timestamp: float | None = None,
    ) -> Location:
        loc = self._resolve_geocoding()
        return Location(
            name=loc.name,
            latitude=loc.latitude,
            longitude=loc.longitude,
            source=loc.source,
            camera_id=camera_id,
            source_id=camera_id,
        )


class GPSLocationProvider(LocationProvider):
    """Hardware/Network GPS location provider (future interface/stub)."""

    def __init__(self, device_path: str | None = None) -> None:
        self.device_path = device_path

    def get_location(
        self,
        camera_id: str | None = None,
        timestamp: float | None = None,
    ) -> Location:
        return Location(
            name="GPS Fix (Pending Hardware Connection)",
            latitude=None,
            longitude=None,
            source=LocationSource.GPS,
            camera_id=camera_id,
            source_id=camera_id,
        )


class UnknownLocationProvider(LocationProvider):
    """Fallback location provider when no location source is configured or available."""

    def get_location(
        self,
        camera_id: str | None = None,
        timestamp: float | None = None,
    ) -> Location:
        return Location(
            name="Unknown",
            latitude=None,
            longitude=None,
            source=LocationSource.UNKNOWN,
            camera_id=camera_id,
            source_id=camera_id,
        )


def create_location_provider(cfg: LocationConfig | None = None) -> LocationProvider:
    """Factory to instantiate the appropriate LocationProvider according to config."""
    if cfg is None:
        return UnknownLocationProvider()

    mode = (cfg.mode or "unknown").lower().strip()
    if mode == "demo":
        query = cfg.get_effective_query()
        return DemoLocationProvider(query=query)
    elif mode == "gps":
        return GPSLocationProvider()
    elif mode in ("unknown", "none"):
        return UnknownLocationProvider()
    else:
        log.warning("Unrecognized location mode '%s'; falling back to UnknownLocationProvider", mode)
        return UnknownLocationProvider()
