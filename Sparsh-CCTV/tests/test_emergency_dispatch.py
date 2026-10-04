import json
import os
import sys
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    DEFAULT_CAMERA_LOCATION,
    DEFAULT_CAMERA_LATITUDE,
    DEFAULT_CAMERA_LONGITUDE,
    EMERGENCY_SEARCH_RADIUS_KM,
    DISPATCH_LOG_PATH,
)
from services.geo_service import GeoService, haversine_distance
from services.emergency_registry import EmergencyRegistry, CURATED_OFFLINE_FACILITIES
from services.alert_dispatcher import EmergencyAlertDispatcher
from incident_logger import IncidentLogger
from detectors.accident_detector import VehicleDetection
from detectors.fire_detector import FireResult


class TestEmergencyDispatch(unittest.TestCase):
    def setUp(self):
        self.camera_query = DEFAULT_CAMERA_LOCATION
        self.geo_service = GeoService()
        self.registry = EmergencyRegistry(
            camera_lat=DEFAULT_CAMERA_LATITUDE,
            camera_lon=DEFAULT_CAMERA_LONGITUDE,
            max_radius_km=EMERGENCY_SEARCH_RADIUS_KM,
        )

    def test_01_geoservice_coordinate_resolution(self):
        """Verify camera location resolves to valid Dwarka Sector 25 coordinates."""
        loc = self.geo_service.resolve_location(self.camera_query)
        self.assertAlmostEqual(loc.latitude, DEFAULT_CAMERA_LATITUDE, places=3)
        self.assertAlmostEqual(loc.longitude, DEFAULT_CAMERA_LONGITUDE, places=3)
        self.assertTrue(loc.maps_url.startswith("https://www.google.com/maps/search/"))
        self.assertIn(str(round(DEFAULT_CAMERA_LATITUDE, 2)), loc.maps_url)

    def test_02_haversine_distance_calculation(self):
        """Verify Haversine formula calculation for identical and offset points."""
        zero_dist = haversine_distance(28.5537, 77.0434, 28.5537, 77.0434)
        self.assertAlmostEqual(zero_dist, 0.0, places=4)

        # Distance to Dwarka Sector 25 Fire Station (28.5518, 77.0376) ~0.60 km
        dist_to_fire = haversine_distance(
            28.5537375, 77.0433594, 28.5517775, 77.0376441
        )
        self.assertGreater(dist_to_fire, 0.4)
        self.assertLess(dist_to_fire, 0.8)

    def test_03_emergency_registry_strict_3km_radius(self):
        """Verify that all discovered facilities are strictly within the 3.0 km search boundary."""
        self.assertGreater(
            len(self.registry.facilities), 0, "Registry should not be empty"
        )

        for fac in self.registry.facilities:
            self.assertLessEqual(
                fac.distance_km,
                EMERGENCY_SEARCH_RADIUS_KM,
                f"Facility '{fac.name}' distance {fac.distance_km}km exceeds {EMERGENCY_SEARCH_RADIUS_KM}km limit",
            )

    def test_04_emergency_registry_facility_categories(self):
        """Verify hospitals, fire stations, and police stations are all present in registry."""
        hospitals = self.registry.get_nearest_hospitals(limit=5)
        police = self.registry.get_nearest_police(limit=5)
        fire = self.registry.get_nearest_fire_stations(limit=5)

        self.assertGreater(len(hospitals), 0, "Should have discovered hospitals")
        self.assertGreater(len(police), 0, "Should have discovered police stations")
        self.assertGreater(len(fire), 0, "Should have discovered fire stations")

        # Proximity ordering check
        for fac_list in [hospitals, police, fire]:
            distances = [f.distance_km for f in fac_list]
            self.assertEqual(
                distances,
                sorted(distances),
                "Facilities must be sorted by ascending proximity",
            )

    def test_05_incident_routing_matrix(self):
        """Verify appropriate routing matrix for accidents vs fires."""
        # 1. Accident Event
        accident_routed = self.registry.get_facilities_for_incident("ACCIDENT_LVL_3")
        self.assertIn("hospitals", accident_routed)
        self.assertIn("police_stations", accident_routed)
        self.assertNotIn("fire_stations", accident_routed)

        # 2. Fire Event
        fire_routed = self.registry.get_facilities_for_incident("FIRE_ALERT")
        self.assertIn("fire_stations", fire_routed)
        self.assertIn("hospitals", fire_routed)
        self.assertIn("police_stations", fire_routed)

        # 3. Smoke Event
        smoke_routed = self.registry.get_facilities_for_incident("SMOKE_ALERT")
        self.assertIn("fire_stations", smoke_routed)

    def test_06_emergency_alert_dispatcher_payload(self):
        """Verify full emergency alert dispatcher payload creation and notification channels."""
        test_log_path = os.path.join(
            os.path.dirname(DISPATCH_LOG_PATH), "test_emergency_dispatch.json"
        )
        if os.path.exists(test_log_path):
            os.remove(test_log_path)

        dispatcher = EmergencyAlertDispatcher(
            camera_id="TEST-CAM-01",
            camera_location=self.camera_query,
            dispatch_log_path=test_log_path,
            enable_console=False,
        )

        result = dispatcher.dispatch(
            incident_type="ACCIDENT_LVL_3",
            severity_label="Severe Accident",
            confidence=0.89,
            snapshot_path="output/snapshots/test_accident.jpg",
            vehicles_count=2,
        )

        self.assertTrue(result["dispatch_id"].startswith("DISPATCH-"))
        self.assertEqual(result["camera"]["camera_id"], "TEST-CAM-01")
        self.assertEqual(result["incident"]["type"], "ACCIDENT_LVL_3")
        self.assertEqual(result["incident"]["vehicles_involved"], 2)
        self.assertIn("hospitals", result["dispatched_services"])
        self.assertIn("police_stations", result["dispatched_services"])
        self.assertTrue(result["notification_status"]["json_logged"])

        # Check JSON persistence
        self.assertTrue(os.path.exists(test_log_path))
        with open(test_log_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["dispatch_id"], result["dispatch_id"])

        # Clean up
        if os.path.exists(test_log_path):
            os.remove(test_log_path)

    def test_07_incident_logger_integration(self):
        """Verify IncidentLogger end-to-end integration with EmergencyAlertDispatcher."""
        test_csv_path = os.path.join(
            os.path.dirname(DISPATCH_LOG_PATH), "test_incident_log.csv"
        )
        test_json_path = os.path.join(
            os.path.dirname(DISPATCH_LOG_PATH), "test_dispatch_log.json"
        )

        for p in [test_csv_path, test_json_path]:
            if os.path.exists(p):
                os.remove(p)

        dispatcher = EmergencyAlertDispatcher(
            camera_id="TEST-CAM-02",
            camera_location=self.camera_query,
            dispatch_log_path=test_json_path,
            enable_console=False,
        )

        logger = IncidentLogger(
            log_file=test_csv_path, dispatcher=dispatcher, enable_dispatch=True
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_det = VehicleDetection(
            bbox=(100, 100, 200, 200),
            confidence=0.91,
            class_id=3,
            raw_name="accident_severe",
            display_name="Severe Accident",
            short_name="SEVERE",
            color=(0, 0, 255),
            is_accident=True,
            severity_level=3,
        )

        # Trigger check_and_log with severity 3
        snapshot = logger.check_and_log(
            frame=dummy_frame,
            highest_severity=3,
            highest_det=dummy_det,
            fire_result=None,
            detections=[dummy_det],
        )

        self.assertIsNotNone(snapshot)
        self.assertTrue(os.path.exists(snapshot))
        self.assertTrue(os.path.exists(test_csv_path))
        self.assertTrue(os.path.exists(test_json_path))

        # Verify CSV row contains Dispatch ID
        with open(test_csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), 2)  # Header + 1 record
        header = lines[0].split(",")
        self.assertIn("Dispatch_ID", header)
        record = lines[1].split(",")
        self.assertTrue(record[-1].startswith("DISPATCH-"))

        # Clean up
        for p in [test_csv_path, test_json_path, snapshot]:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    unittest.main()
