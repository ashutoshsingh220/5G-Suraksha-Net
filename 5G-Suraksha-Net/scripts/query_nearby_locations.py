"""Query Google Maps Geocoding & Places API (New) to discover emergency facilities near Yashobhoomi."""
import json
import math
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
query = os.environ.get("DEMO_LOCATION_QUERY", "Yashobhoomi, Dwarka Sector 25, New Delhi").strip()

if not api_key:
    print("Error: GOOGLE_MAPS_API_KEY not found in .env")
    sys.exit(1)


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return 6371.0 * c


print("=" * 75)
print(f"1. GEOCODING DEMO LOCATION: '{query}'")
print("=" * 75)

geo_url = "https://maps.googleapis.com/maps/api/geocode/json"
geo_resp = requests.get(geo_url, params={"address": query, "key": api_key}).json()

if geo_resp.get("status") != "OK" or not geo_resp.get("results"):
    print(f"Geocoding failed: {geo_resp.get('status')} - {geo_resp.get('error_message')}")
    sys.exit(1)

result = geo_resp["results"][0]
formatted_address = result.get("formatted_address")
loc = result["geometry"]["location"]
center_lat = loc["lat"]
center_lng = loc["lng"]
place_id = result.get("place_id")

print(f"Formatted Address : {formatted_address}")
print(f"Coordinates       : Latitude {center_lat:.6f}, Longitude {center_lng:.6f}")
print(f"Place ID          : {place_id}")
print(f"Location Type     : {result['geometry'].get('location_type')}")
print()


def search_nearby_places(category_name, place_type, radius_meters=6000.0, max_results=10):
    print("=" * 75)
    print(f"SEARCHING NEARBY: {category_name.upper()} (type={place_type}, radius={radius_meters/1000.0:.1f}km)")
    print("=" * 75)

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,places.location,"
            "places.rating,places.userRatingCount,places.types,places.nationalPhoneNumber,"
            "places.websiteUri"
        ),
    }
    body = {
        "includedTypes": [place_type],
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": center_lat,
                    "longitude": center_lng,
                },
                "radius": radius_meters,
            }
        },
    }

    resp = requests.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return []

    data = resp.json()
    places_raw = data.get("places", [])
    print(f"Discovered: {len(places_raw)} facilities\n")

    items = []
    for p in places_raw:
        p_loc = p.get("location", {})
        p_lat = p_loc.get("latitude")
        p_lng = p_loc.get("longitude")
        dist_km = haversine_km(center_lat, center_lng, p_lat, p_lng) if (p_lat and p_lng) else None

        name = p.get("displayName", {}).get("text", "Unknown Facility")
        addr = p.get("formattedAddress", "N/A")
        phone = p.get("nationalPhoneNumber", "N/A")
        rating = p.get("rating")
        rating_count = p.get("userRatingCount")
        website = p.get("websiteUri")

        item = {
            "name": name,
            "category": category_name,
            "place_type": place_type,
            "distance_km": round(dist_km, 2) if dist_km else None,
            "formatted_address": addr,
            "latitude": p_lat,
            "longitude": p_lng,
            "phone_number": phone,
            "rating": rating,
            "user_ratings_count": rating_count,
            "website": website,
            "place_id": p.get("id"),
        }
        items.append(item)

    # Sort strictly by proximity to Yashobhoomi
    items.sort(key=lambda x: (x["distance_km"] if x["distance_km"] is not None else 999))

    for idx, pl in enumerate(items, 1):
        rating_str = f"{pl['rating']}/5.0 ({pl['user_ratings_count']} reviews)" if pl['rating'] else "No rating"
        print(f"  [{idx}] {pl['name']}")
        print(f"      Distance : {pl['distance_km']} km (straight-line from Yashobhoomi)")
        print(f"      Address  : {pl['formatted_address']}")
        print(f"      Phone    : {pl['phone_number']}")
        print(f"      Rating   : {rating_str}")
        print(f"      Coords   : ({pl['latitude']:.6f}, {pl['longitude']:.6f})")
        if pl['website']:
            print(f"      Website  : {pl['website']}")
        print()

    return items


# 1. Hospitals & Super-Speciality / Trauma Centers
hospitals = search_nearby_places("Hospitals & Emergency Trauma Centers", "hospital", radius_meters=6000.0)

# 2. Police Stations & Law Enforcement
police_stations = search_nearby_places("Police Stations & Law Enforcement", "police", radius_meters=6000.0)

# 3. Fire Stations & Rescue
fire_stations = search_nearby_places("Fire Stations & Rescue", "fire_station", radius_meters=7000.0)

# Compile complete payload
summary_payload = {
    "demonstration_location": {
        "query": query,
        "formatted_address": formatted_address,
        "latitude": center_lat,
        "longitude": center_lng,
        "place_id": place_id,
    },
    "emergency_responders": {
        "hospitals": hospitals,
        "police_stations": police_stations,
        "fire_stations": fire_stations,
    },
}

out_json = PROJECT_ROOT / "outputs" / "phase3" / "nearby_emergency_locations_yashobhoomi.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

print("=" * 75)
print(f"SAVED STRUCTURED EMERGENCY DIRECTORY TO: {out_json}")
print("=" * 75)
