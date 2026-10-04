import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from config import (
    DEFAULT_CAMERA_LATITUDE,
    DEFAULT_CAMERA_LONGITUDE,
    EMERGENCY_SEARCH_RADIUS_KM,
    EMERGENCY_CACHE_PATH,
    MAX_DISPATCH_FACILITIES_PER_TYPE,
)
from services.geo_service import haversine_distance

load_dotenv()

# Curated offline reference facilities within 3km of Sector 25 Dwarka (28.5537° N, 77.0434° E)
CURATED_OFFLINE_FACILITIES = [
    # Fire Stations
    {
        "name": "FIRE STATION DWARKA SECTOR 25",
        "facility_type": "fire_station",
        "address": "Sector 25 Dwarka, Dwarka, Delhi, 110077, India",
        "latitude": 28.5517775,
        "longitude": 77.0376441,
        "phone": "101 / +91-11-28080101",
    },
    # Police Stations / Check Posts
    {
        "name": "Dwarka Sector 21 Police Check Post",
        "facility_type": "police",
        "address": "Service Rd, Sector 21, Dwarka, New Delhi, Delhi, 110077, India",
        "latitude": 28.5512175,
        "longitude": 77.056413,
        "phone": "112 / +91-11-28052100",
    },
    {
        "name": "Police Station Dwarka Sector 23",
        "facility_type": "police",
        "address": "H375+RH4, Sector 23, Dwarka, New Delhi, Delhi, 110077, India",
        "latitude": 28.5645101,
        "longitude": 77.0589427,
        "phone": "112 / +91-11-28080112",
    },
    {
        "name": "Traffic Inspector Dwarka Office",
        "facility_type": "police",
        "address": "H375+V6M, Sector 23, Dwarka, New Delhi, Delhi, 110077, India",
        "latitude": 28.5647137,
        "longitude": 77.0580418,
        "phone": "1095 / +91-11-28080100",
    },
    # Hospitals & Trauma Centers
    {
        "name": "Maple Care Hospital",
        "facility_type": "hospital",
        "address": "Plot 10, opposite Ranjit Vihar II, Sector 23, Dwarka, New Delhi, 110077, India",
        "latitude": 28.5605619,
        "longitude": 77.0511742,
        "phone": "102 / +91-11-40004000",
    },
    {
        "name": "Ambe Hospital",
        "facility_type": "hospital",
        "address": "47, Block A, Sector 23, Dwarka, New Delhi, Delhi, 110077, India",
        "latitude": 28.562368,
        "longitude": 77.0493619,
        "phone": "+91-11-28081234",
    },
    {
        "name": "HELPLINE MULTISPECIALITY HOSPITAL",
        "facility_type": "hospital",
        "address": "Village-Pochanpur, Block A, Sector 23, Dwarka, New Delhi, 110077, India",
        "latitude": 28.5626713,
        "longitude": 77.0492646,
        "phone": "+91-11-28085678",
    },
    {
        "name": "Ayushman Hospital & Health Services",
        "facility_type": "hospital",
        "address": "41, Sector 21, Dwarka, Delhi, 110077, India",
        "latitude": 28.5665566,
        "longitude": 77.06235,
        "phone": "+91-11-45678900",
    },
    {
        "name": "Amerix Super Speciality Hospital",
        "facility_type": "hospital",
        "address": "opposite Welcomhotel, Sector 19, Dwarka, Delhi, 110075, India",
        "latitude": 28.5772725,
        "longitude": 77.0554256,
        "phone": "+91-11-47000000",
    },
]


@dataclass
class EmergencyFacility:
    """Represents an emergency responder facility."""

    name: str
    facility_type: str  # "hospital", "police", "fire_station"
    address: str
    latitude: float
    longitude: float
    distance_km: float
    phone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "facility_type": self.facility_type,
            "address": self.address,
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "distance_km": round(self.distance_km, 2),
            "distance_meters": int(round(self.distance_km * 1000)),
            "phone": self.phone or "Emergency 112",
        }


class EmergencyRegistry:
    """
    Manages and queries emergency facilities (hospitals, fire stations, police stations)
    within a specified radius of a surveillance camera.
    """

    def __init__(
        self,
        camera_lat: float = DEFAULT_CAMERA_LATITUDE,
        camera_lon: float = DEFAULT_CAMERA_LONGITUDE,
        max_radius_km: float = EMERGENCY_SEARCH_RADIUS_KM,
        cache_path: str = EMERGENCY_CACHE_PATH,
        api_key: Optional[str] = None,
    ):
        self.camera_lat = camera_lat
        self.camera_lon = camera_lon
        self.max_radius_km = max_radius_km
        self.cache_path = cache_path
        self.api_key = api_key or os.getenv("MAPS_API_KEY")
        self.facilities: List[EmergencyFacility] = []

        self._initialize_registry()

    def _initialize_registry(self):
        """Loads facilities from cache, or fetches live from Google Places, or uses offline fallback."""
        # 1. Try loading from cache if available
        if self._load_cache():
            return

        # 2. Try fetching live from Google Places API (New)
        if self.api_key:
            try:
                live_facilities = self._fetch_from_google_places()
                if live_facilities:
                    self.facilities = live_facilities
                    self._save_cache()
                    return
            except Exception as e:
                print(
                    f"[EmergencyRegistry] Warning: Live Places fetch failed: {e}. Using fallback."
                )

        # 3. Fallback to curated offline database
        self._load_curated_fallback()
        self._save_cache()

    def _load_cache(self) -> bool:
        """Loads cached facilities from disk and recomputes distance to current camera."""
        if not os.path.exists(self.cache_path):
            return False

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded = []
            for item in data.get("facilities", []):
                dist = haversine_distance(
                    self.camera_lat,
                    self.camera_lon,
                    item["latitude"],
                    item["longitude"],
                )
                if dist <= self.max_radius_km:
                    loaded.append(
                        EmergencyFacility(
                            name=item["name"],
                            facility_type=item["facility_type"],
                            address=item["address"],
                            latitude=item["latitude"],
                            longitude=item["longitude"],
                            distance_km=dist,
                            phone=item.get("phone"),
                        )
                    )

            if loaded:
                self.facilities = sorted(loaded, key=lambda f: f.distance_km)
                print(
                    f"[EmergencyRegistry] Loaded {len(self.facilities)} facilities within {self.max_radius_km}km from cache."
                )
                return True
        except Exception as e:
            print(f"[EmergencyRegistry] Error reading cache file: {e}")
        return False

    def _save_cache(self):
        """Persists current facilities to disk cache."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            data = {
                "reference_lat": self.camera_lat,
                "reference_lon": self.camera_lon,
                "max_radius_km": self.max_radius_km,
                "facilities": [f.to_dict() for f in self.facilities],
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(
                f"[EmergencyRegistry] Cached {len(self.facilities)} facilities to {self.cache_path}"
            )
        except Exception as e:
            print(f"[EmergencyRegistry] Error saving cache: {e}")

    def _load_curated_fallback(self):
        """Loads curated offline facilities within max_radius_km."""
        loaded = []
        for item in CURATED_OFFLINE_FACILITIES:
            dist = haversine_distance(
                self.camera_lat, self.camera_lon, item["latitude"], item["longitude"]
            )
            if dist <= self.max_radius_km:
                loaded.append(
                    EmergencyFacility(
                        name=item["name"],
                        facility_type=item["facility_type"],
                        address=item["address"],
                        latitude=item["latitude"],
                        longitude=item["longitude"],
                        distance_km=dist,
                        phone=item.get("phone"),
                    )
                )

        self.facilities = sorted(loaded, key=lambda f: f.distance_km)
        print(
            f"[EmergencyRegistry] Loaded {len(self.facilities)} curated offline facilities within {self.max_radius_km}km."
        )

    def _fetch_from_google_places(self) -> List[EmergencyFacility]:
        """Fetches hospitals, fire stations, and police stations using Google Places API (New)."""
        url = "https://places.googleapis.com/v1/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.primaryType",
        }

        discovered: List[EmergencyFacility] = []
        types_to_query = [
            ("hospital", "hospital"),
            ("police", "police"),
            ("fire_station", "fire_station"),
        ]

        radius_meters = float(self.max_radius_km * 1000.0)

        for place_type, facility_cat in types_to_query:
            body = {
                "includedTypes": [place_type],
                "maxResultCount": 10,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": self.camera_lat,
                            "longitude": self.camera_lon,
                        },
                        "radius": radius_meters,
                    }
                },
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    for place in resp_data.get("places", []):
                        name = place.get("displayName", {}).get(
                            "text", "Unknown Facility"
                        )
                        address = place.get("formattedAddress", "Dwarka, New Delhi")
                        loc = place.get("location", {})
                        lat = loc.get("latitude")
                        lon = loc.get("longitude")

                        if lat is not None and lon is not None:
                            dist = haversine_distance(
                                self.camera_lat, self.camera_lon, lat, lon
                            )
                            # Strict radius check
                            if dist <= self.max_radius_km:
                                # Determine phone fallback based on category
                                phone = (
                                    "101"
                                    if facility_cat == "fire_station"
                                    else (
                                        "102" if facility_cat == "hospital" else "112"
                                    )
                                )
                                discovered.append(
                                    EmergencyFacility(
                                        name=name,
                                        facility_type=facility_cat,
                                        address=address,
                                        latitude=lat,
                                        longitude=lon,
                                        distance_km=dist,
                                        phone=phone,
                                    )
                                )
            except Exception as e:
                print(f"[EmergencyRegistry] Query for {place_type} failed: {e}")

        # Remove duplicate names and sort by proximity
        seen = set()
        unique = []
        for f in sorted(discovered, key=lambda x: x.distance_km):
            if f.name not in seen:
                seen.add(f.name)
                unique.append(f)

        return unique

    def get_facilities_by_type(
        self, facility_type: str, limit: int = 3
    ) -> List[EmergencyFacility]:
        """Returns the nearest facilities matching the specified type within the search radius."""
        matches = [f for f in self.facilities if f.facility_type == facility_type]
        return sorted(matches, key=lambda f: f.distance_km)[:limit]

    def get_nearest_hospitals(self, limit: int = 2) -> List[EmergencyFacility]:
        return self.get_facilities_by_type("hospital", limit=limit)

    def get_nearest_police(self, limit: int = 2) -> List[EmergencyFacility]:
        return self.get_facilities_by_type("police", limit=limit)

    def get_nearest_fire_stations(self, limit: int = 2) -> List[EmergencyFacility]:
        return self.get_facilities_by_type("fire_station", limit=limit)

    def get_facilities_for_incident(
        self, incident_type: str, limit_per_type: int = MAX_DISPATCH_FACILITIES_PER_TYPE
    ) -> Dict[str, List[EmergencyFacility]]:
        """
        Routes and returns appropriate emergency services based on incident type:
        - Accidents: Hospitals & Police Stations
        - Fire / Smoke: Fire Stations, Hospitals, Police Stations
        """
        upper_type = incident_type.upper()
        results = {}

        if "FIRE" in upper_type or "SMOKE" in upper_type:
            results["fire_stations"] = self.get_nearest_fire_stations(
                limit=limit_per_type
            )
            results["hospitals"] = self.get_nearest_hospitals(limit=limit_per_type)
            results["police_stations"] = self.get_nearest_police(limit=limit_per_type)
        else:
            # Accident / Crash event
            results["hospitals"] = self.get_nearest_hospitals(limit=limit_per_type)
            results["police_stations"] = self.get_nearest_police(limit=limit_per_type)

        return results
