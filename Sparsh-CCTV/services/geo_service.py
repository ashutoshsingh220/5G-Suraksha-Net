import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv

from config import (
    DEFAULT_CAMERA_LOCATION,
    DEFAULT_CAMERA_LATITUDE,
    DEFAULT_CAMERA_LONGITUDE,
)

# Load environment variables
load_dotenv()

# Known static coordinate fallbacks for common surveillance deployment nodes
STATIC_COORDINATE_FALLBACKS = {
    "H23V+F8X, Sector 25 Dwarka, Dwarka, Delhi, 110077": (
        DEFAULT_CAMERA_LATITUDE,
        DEFAULT_CAMERA_LONGITUDE,
        "H23V+F8X Sector 25 Dwarka, Dwarka, Delhi, India",
    ),
    "Sector 25 Dwarka, Dwarka, Delhi, 110077": (
        DEFAULT_CAMERA_LATITUDE,
        DEFAULT_CAMERA_LONGITUDE,
        "Sector 25 Dwarka, Dwarka, Delhi, India",
    ),
}


@dataclass
class GeoLocation:
    """Represents a geographic location with coordinates and formatted address."""

    latitude: float
    longitude: float
    formatted_address: str
    source: str  # "googlemaps_api" or "static_fallback"

    @property
    def maps_url(self) -> str:
        return f"https://www.google.com/maps/search/?api=1&query={self.latitude:.7f},{self.longitude:.7f}"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two points in kilometers using Haversine formula.
    """
    radius = 6371.0  # Earth's radius in kilometers

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius * c


class GeoService:
    """Provides geocoding, coordinate resolution, and distance calculations for cameras."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MAPS_API_KEY")
        self._gmaps_client = None

        if self.api_key:
            try:
                import googlemaps

                self._gmaps_client = googlemaps.Client(key=self.api_key)
            except Exception as e:
                print(
                    f"[GeoService] Warning: Failed to initialize Google Maps client: {e}"
                )

    def resolve_location(self, location_query: str) -> GeoLocation:
        """
        Resolves an address or Plus Code into a GeoLocation.
        Prioritizes Google Maps Geocoding API if available, falling back to static lookup.
        """
        normalized_query = location_query.strip()

        # 1. Attempt Google Maps Geocoding if client is active
        if self._gmaps_client:
            try:
                results = self._gmaps_client.geocode(normalized_query)
                if results and len(results) > 0:
                    first = results[0]
                    loc = first.get("geometry", {}).get("location", {})
                    lat = loc.get("lat")
                    lon = loc.get("lng")
                    addr = first.get("formatted_address", normalized_query)
                    if lat is not None and lon is not None:
                        return GeoLocation(
                            latitude=float(lat),
                            longitude=float(lon),
                            formatted_address=addr,
                            source="googlemaps_api",
                        )
            except Exception as e:
                print(
                    f"[GeoService] Warning: Geocoding API query failed: {e}. Using fallback."
                )

        # 2. Check Static Fallback Dictionary
        for key, (flat, flon, faddr) in STATIC_COORDINATE_FALLBACKS.items():
            if (
                key.lower() in normalized_query.lower()
                or normalized_query.lower() in key.lower()
            ):
                return GeoLocation(
                    latitude=flat,
                    longitude=flon,
                    formatted_address=faddr,
                    source="static_fallback",
                )

        # 3. Default fallback to Sector 25 Dwarka reference coordinates
        return GeoLocation(
            latitude=DEFAULT_CAMERA_LATITUDE,
            longitude=DEFAULT_CAMERA_LONGITUDE,
            formatted_address=normalized_query or DEFAULT_CAMERA_LOCATION,
            source="static_fallback",
        )
