"""Location data contracts and schemas for 5G Suraksha-Net."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LocationSource(str, Enum):
    """Source mechanism providing location data."""
    GEOCODED_DEMO = "geocoded_demo"
    GPS = "gps"
    CAMERA = "camera"
    UNKNOWN = "unknown"


class Location(BaseModel):
    """Typed location model associated with cameras, streams, or incidents."""
    name: str
    latitude: float | None = None
    longitude: float | None = None
    source: LocationSource = LocationSource.UNKNOWN
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accuracy: float | None = None
    camera_id: str | None = None
    source_id: str | None = None

    @property
    def has_coordinates(self) -> bool:
        """True if non-null, valid numerical coordinates are present."""
        return self.latitude is not None and self.longitude is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize location model to clean dictionary."""
        return {
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source": self.source.value if isinstance(self.source, LocationSource) else str(self.source),
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "accuracy": self.accuracy,
            "camera_id": self.camera_id,
            "source_id": self.source_id,
            "has_coordinates": self.has_coordinates,
        }
