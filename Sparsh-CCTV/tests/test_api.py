import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import app
from config import (
    get_settings,
    update_settings,
    reset_settings_to_defaults,
    DEFAULT_VIDEO_SOURCE,
)
from surveillance_engine import get_surveillance_engine


class TestFastAPIEndpoints(unittest.TestCase):
    """Integration test suite verifying all Sparsh CCTV FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        reset_settings_to_defaults()
        engine = get_surveillance_engine(auto_init_detectors=False)
        if engine.is_running:
            engine.stop()

    def test_01_health_and_info_endpoints(self):
        """Verify /api/system/health and /api/system/info."""
        res_health = self.client.get("/api/system/health")
        self.assertEqual(res_health.status_code, 200)
        health_data = res_health.json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertIn("uptime_seconds", health_data)
        self.assertIn("camera_id", health_data)

        res_info = self.client.get("/api/system/info")
        self.assertEqual(res_info.status_code, 200)
        info_data = res_info.json()
        self.assertIn("platform", info_data)
        self.assertIn("hardware", info_data)
        self.assertIn("model_weights", info_data)

    def test_02_dashboard_html_endpoint(self):
        """Verify web dashboard at / and /dashboard returns 200 HTML with unified feed selector and hydration elements."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("text/html", res_root.headers["content-type"])
        self.assertIn("Sparsh CCTV", res_root.text)
        self.assertIn("feedSourceSelect", res_root.text)
        self.assertIn("feedTestVideosOptgroup", res_root.text)
        self.assertIn("btnQuickRTSP", res_root.text)
        self.assertIn("initDashboard", res_root.text)
        self.assertIn("handleStreamError", res_root.text)
        self.assertIn("settingsStatusBadge", res_root.text)

        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        self.assertIn("Sparsh CCTV", res_dash.text)
        self.assertIn("feedSourceSelect", res_dash.text)

    def test_03_get_settings_endpoint(self):
        """Verify GET /api/settings returns complete configuration dictionary."""
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("camera", data)
        self.assertIn("detectors", data)
        self.assertIn("alerts", data)
        self.assertIn("email", data)
        self.assertIn("twilio", data)

        # Check mask_secrets option
        res_masked = self.client.get("/api/settings?mask_secrets=true")
        self.assertEqual(res_masked.status_code, 200)

    def test_04_patch_settings_endpoint(self):
        """Verify dynamic runtime update via PATCH /api/settings."""
        test_update = {
            "camera_id": "SPARSH-TEST-CAM-99",
            "crash_conf": 0.82,
            "fire_conf": 0.85,
            "alert_cooldown_sec": 8.0,
            "emergency_search_radius_km": 4.5,
            "default_recipient_email": "test-admin@sparsh.org",
        }
        res = self.client.patch("/api/settings", json=test_update)
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "SUCCESS")

        # Verify persisted dynamically in active settings
        curr = get_settings()
        self.assertEqual(curr.camera_id, "SPARSH-TEST-CAM-99")
        self.assertEqual(curr.crash_conf, 0.82)
        self.assertEqual(curr.fire_conf, 0.85)
        self.assertEqual(curr.alert_cooldown_sec, 8.0)
        self.assertEqual(curr.emergency_search_radius_km, 4.5)
        self.assertEqual(curr.default_recipient_email, "test-admin@sparsh.org")

        # Test custom RTSP URL set and clear
        res_rtsp = self.client.patch(
            "/api/settings", json={"custom_rtsp_url": "rtsp://camera.local/live"}
        )
        self.assertEqual(res_rtsp.status_code, 200)
        self.assertEqual(get_settings().custom_rtsp_url, "rtsp://camera.local/live")

        res_reset_rtsp = self.client.patch(
            "/api/settings", json={"custom_rtsp_url": None}
        )
        self.assertEqual(res_reset_rtsp.status_code, 200)
        self.assertIsNone(get_settings().custom_rtsp_url)

    def test_05_granular_settings_sub_routes(self):
        """Verify individual sub-resource settings endpoints."""
        # Camera
        res_cam = self.client.get("/api/settings/camera")
        self.assertEqual(res_cam.status_code, 200)
        self.assertIn("camera_id", res_cam.json())

        patch_cam = self.client.patch("/api/settings/camera", json={"camera_fps": 20.0})
        self.assertEqual(patch_cam.status_code, 200)
        self.assertEqual(get_settings().camera_fps, 20.0)

        # Detectors
        res_det = self.client.get("/api/settings/detectors")
        self.assertEqual(res_det.status_code, 200)
        self.assertIn("crash_conf", res_det.json())

        patch_det = self.client.patch(
            "/api/settings/detectors", json={"crash_conf": 0.65}
        )
        self.assertEqual(patch_det.status_code, 200)
        self.assertEqual(get_settings().crash_conf, 0.65)

        # Alerts
        res_alt = self.client.get("/api/settings/alerts")
        self.assertEqual(res_alt.status_code, 200)
        self.assertIn("alert_cooldown_sec", res_alt.json())

        patch_alt = self.client.patch(
            "/api/settings/alerts", json={"alert_cooldown_sec": 6.0}
        )
        self.assertEqual(patch_alt.status_code, 200)
        self.assertEqual(get_settings().alert_cooldown_sec, 6.0)

        # Email & Department dictionary merging
        res_mail = self.client.get("/api/settings/email")
        self.assertEqual(res_mail.status_code, 200)
        self.assertIn("smtp_host", res_mail.json())

        patch_mail = self.client.patch(
            "/api/settings/email",
            json={
                "smtp_port": 465,
                "department_emails": {"police": "test-police@sparsh.org"},
            },
        )
        self.assertEqual(patch_mail.status_code, 200)
        self.assertEqual(get_settings().smtp_port, 465)
        self.assertEqual(
            get_settings().department_emails["police"], "test-police@sparsh.org"
        )
        self.assertIn("hospital", get_settings().department_emails)

        # Twilio
        res_tw = self.client.get("/api/settings/twilio")
        self.assertEqual(res_tw.status_code, 200)
        self.assertIn("emergency_dispatch_phone", res_tw.json())

        patch_tw = self.client.patch(
            "/api/settings/twilio", json={"emergency_dispatch_phone": "+919876543210"}
        )
        self.assertEqual(patch_tw.status_code, 200)
        self.assertEqual(get_settings().emergency_dispatch_phone, "+919876543210")

    def test_06_surveillance_status_and_detections(self):
        """Verify /api/surveillance/status and /api/surveillance/detections."""
        res_status = self.client.get("/api/surveillance/status")
        self.assertEqual(res_status.status_code, 200)
        s_data = res_status.json()
        self.assertIn("is_running", s_data)
        self.assertIn("current_status", s_data)
        self.assertIn("fps", s_data)

        res_dets = self.client.get("/api/surveillance/detections")
        self.assertEqual(res_dets.status_code, 200)
        d_data = res_dets.json()
        self.assertIn("vehicles", d_data)
        self.assertIn("fire_hazard", d_data)

    def test_07_surveillance_frame_endpoint(self):
        """Verify /api/surveillance/frame returns JPEG image bytes."""
        res_frame = self.client.get("/api/surveillance/frame")
        self.assertEqual(res_frame.status_code, 200)
        self.assertEqual(res_frame.headers["content-type"], "image/jpeg")
        self.assertGreater(len(res_frame.content), 1000)

    def test_08_manual_snapshot_and_trigger_alert(self):
        """Verify snapshot creation and manual alert triggering endpoints."""
        # Manual Snapshot
        res_snap = self.client.post(
            "/api/surveillance/snapshot", json={"tag": "api_test"}
        )
        self.assertEqual(res_snap.status_code, 200)
        snap_data = res_snap.json()
        self.assertEqual(snap_data["status"], "SUCCESS")
        self.assertIn("filename", snap_data)
        self.assertTrue(os.path.exists(snap_data["filepath"]))

        # Check retrieval of that snapshot via /api/snapshots/{filename}
        res_get_snap = self.client.get(f"/api/snapshots/{snap_data['filename']}")
        self.assertEqual(res_get_snap.status_code, 200)
        self.assertEqual(res_get_snap.headers["content-type"], "image/jpeg")

        # Cleanup snapshot file
        if os.path.exists(snap_data["filepath"]):
            os.remove(snap_data["filepath"])

        # Trigger Manual Alert
        res_alert = self.client.post(
            "/api/surveillance/trigger-alert",
            json={
                "incident_type": "ACCIDENT_LVL_3",
                "severity_label": "Severe Accident",
                "confidence": 0.94,
                "dispatch_emergency": False,
            },
        )
        self.assertEqual(res_alert.status_code, 200)
        alert_data = res_alert.json()
        self.assertEqual(alert_data["status"], "SUCCESS")

        # Cleanup manual alert snapshot
        if "snapshot" in alert_data and os.path.exists(
            alert_data["snapshot"].get("filepath", "")
        ):
            os.remove(alert_data["snapshot"]["filepath"])

    def test_09_incidents_and_dispatches_query(self):
        """Verify querying /api/incidents and /api/dispatches."""
        res_inc = self.client.get("/api/incidents?limit=10")
        self.assertEqual(res_inc.status_code, 200)
        inc_data = res_inc.json()
        self.assertIn("incidents", inc_data)

        res_disp = self.client.get("/api/dispatches?limit=10")
        self.assertEqual(res_disp.status_code, 200)
        disp_data = res_disp.json()
        self.assertIn("dispatches", disp_data)

        res_snaps = self.client.get("/api/snapshots?limit=10")
        self.assertEqual(res_snaps.status_code, 200)
        snaps_data = res_snaps.json()
        self.assertIn("snapshots", snaps_data)

    def test_10_emergency_endpoints(self):
        """Verify /api/emergency/location, /facilities, and /resolve-location."""
        res_loc = self.client.get("/api/emergency/location")
        self.assertEqual(res_loc.status_code, 200)
        loc_data = res_loc.json()
        self.assertIn("latitude", loc_data)
        self.assertIn("longitude", loc_data)
        self.assertIn("maps_url", loc_data)

        res_fac = self.client.get("/api/emergency/facilities?limit=3")
        self.assertEqual(res_fac.status_code, 200)
        fac_data = res_fac.json()
        self.assertIn("hospitals", fac_data)
        self.assertIn("police_stations", fac_data)
        self.assertIn("fire_stations", fac_data)

        res_geocode = self.client.post(
            "/api/emergency/resolve-location",
            json={"location_query": "Sector 25 Dwarka, Dwarka, Delhi, 110077"},
        )
        self.assertEqual(res_geocode.status_code, 200)
        geo_data = res_geocode.json()
        self.assertAlmostEqual(geo_data["latitude"], 28.5537, places=2)

        # Verify test SMS endpoint with overrides
        res_sms = self.client.post(
            "/api/emergency/test-sms",
            json={"to_phone": "+919876543210", "message": "Integration Test SMS"},
        )
        self.assertEqual(res_sms.status_code, 200)
        self.assertEqual(res_sms.json()["status"], "SUCCESS")

    def test_11_surveillance_lifecycle_endpoints(self):
        """Verify starting, pausing, resuming, and stopping stream via API."""
        # Start
        res_start = self.client.post(
            "/api/surveillance/start", json={"save_recording": False}
        )
        self.assertEqual(res_start.status_code, 200)
        self.assertTrue(res_start.json()["details"]["is_running"])

        # Pause
        res_pause = self.client.post("/api/surveillance/pause")
        self.assertEqual(res_pause.status_code, 200)
        self.assertTrue(res_pause.json()["details"]["is_paused"])

        # Resume
        res_resume = self.client.post("/api/surveillance/resume")
        self.assertEqual(res_resume.status_code, 200)
        self.assertFalse(res_resume.json()["details"]["is_paused"])

        # Stop
        res_stop = self.client.post("/api/surveillance/stop")
        self.assertEqual(res_stop.status_code, 200)
        self.assertFalse(res_stop.json()["details"]["is_running"])

    def test_12_available_sources_endpoint(self):
        """Verify GET /api/surveillance/available-sources returns RTSP and test video files."""
        res = self.client.get("/api/surveillance/available-sources")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("current_source", data)
        self.assertIn("current_source_type", data)
        self.assertIn("is_rtsp", data)
        self.assertIn("rtsp", data)
        self.assertIn("test_videos", data)

        self.assertIn("url", data["rtsp"])
        self.assertIn("camera_ip", data["rtsp"])

        test_videos = data["test_videos"]
        self.assertIsInstance(test_videos, list)
        self.assertGreaterEqual(len(test_videos), 5)

        filenames = [v["filename"] for v in test_videos]
        self.assertIn("crash.mp4", filenames)
        self.assertIn("bikeacc.mp4", filenames)
        self.assertIn("fire.mp4", filenames)
        self.assertIn("blast.mp4", filenames)
        self.assertIn("truck_acc.mp4", filenames)

        for v in test_videos:
            self.assertIn("filename", v)
            self.assertIn("path", v)
            self.assertIn("size_mb", v)
            self.assertGreater(v["size_mb"], 0)

    def test_13_switch_source_test_video(self):
        """Verify POST /api/surveillance/switch-source switches to test video."""
        res = self.client.post(
            "/api/surveillance/switch-source", json={"source": "crash.mp4"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["source_type"], "test_video")
        self.assertEqual(data["source_label"], "crash.mp4")
        self.assertTrue(data["details"]["is_running"])

        # Check dynamic settings updated
        settings = get_settings()
        self.assertTrue(settings.video_source.endswith("crash.mp4"))

        # Clean up
        self.client.post("/api/surveillance/stop")

    @patch("surveillance_engine.StreamCapture")
    def test_14_switch_source_rtsp(self, mock_stream):
        """Verify POST /api/surveillance/switch-source switches to RTSP stream."""
        instance = MagicMock()
        instance.width = 1280
        instance.height = 720
        instance.fps = 15.0
        instance.read.return_value = (False, None)
        mock_stream.return_value = instance

        res = self.client.post(
            "/api/surveillance/switch-source", json={"source": "rtsp"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["source_type"], "rtsp")
        self.assertEqual(data["source_label"], "Sparsh CCTV (Live RTSP)")

        # Check dynamic settings updated
        settings = get_settings()
        self.assertEqual(settings.video_source, "rtsp")

        # Clean up
        self.client.post("/api/surveillance/stop")

    def test_15_switch_source_invalid(self):
        """Verify invalid or non-existent source returns 400 Bad Request."""
        res_empty = self.client.post(
            "/api/surveillance/switch-source", json={"source": "   "}
        )
        self.assertEqual(res_empty.status_code, 400)

        res_missing = self.client.post(
            "/api/surveillance/switch-source",
            json={"source": "completely_nonexistent_file_999.mp4"},
        )
        self.assertEqual(res_missing.status_code, 400)
        self.assertIn("not found", res_missing.json()["detail"])

    def test_16_default_source_is_rtsp(self):
        """Verify default video source in system settings is 'rtsp'."""
        reset_settings_to_defaults()
        settings = get_settings()
        self.assertEqual(settings.video_source, "rtsp")
        self.assertEqual(DEFAULT_VIDEO_SOURCE, "rtsp")

    @patch("surveillance_engine.StreamCapture")
    def test_17_no_stream_state_when_rtsp_offline(self, mock_stream):
        """Verify graceful NO_STREAM state when RTSP device is offline."""
        mock_stream.side_effect = RuntimeError(
            "Could not open video stream: Connection timed out"
        )

        res = self.client.post(
            "/api/surveillance/switch-source", json={"source": "rtsp"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "WARNING")
        self.assertFalse(data["stream_connected"])
        self.assertIn("NO_STREAM", data["stream_status"])

        # Check status endpoint
        res_status = self.client.get("/api/surveillance/status")
        self.assertEqual(res_status.status_code, 200)
        s_data = res_status.json()
        self.assertFalse(s_data["stream_connected"])
        self.assertEqual(s_data["current_status"], "NO_STREAM")
        self.assertEqual(s_data["highest_severity_label"], "No Feed")
        self.assertEqual(s_data["fire_status_label"], "No Feed")
        self.assertEqual(s_data["fps"], 0.0)

        # Check frame endpoint returns the NO_STREAM canvas
        res_frame = self.client.get("/api/surveillance/frame")
        self.assertEqual(res_frame.status_code, 200)
        self.assertEqual(res_frame.headers["content-type"], "image/jpeg")
        self.assertGreater(len(res_frame.content), 1000)

        # Clean up
        self.client.post("/api/surveillance/stop")

    @patch("surveillance_engine.StreamCapture")
    def test_18_no_incident_logging_when_offline(self, mock_stream):
        """Verify zero detections or incidents are logged while in NO_STREAM mode."""
        mock_stream.side_effect = RuntimeError("Camera offline")

        self.client.post("/api/surveillance/switch-source", json={"source": "rtsp"})
        engine = get_surveillance_engine()
        self.assertTrue(engine.is_running)
        self.assertFalse(engine.stream_connected)
        status = engine.get_status()
        self.assertEqual(status["monitored_vehicles"], 0)
        self.assertEqual(len(engine.latest_detections), 0)

        # Query incidents endpoint - verify no new incidents generated by offline stream
        res_inc = self.client.get("/api/incidents?limit=5")
        self.assertEqual(res_inc.status_code, 200)

        # Clean up
        self.client.post("/api/surveillance/stop")

    def test_19_trigger_accident_alert_endpoint(self):
        """Verify POST /api/surveillance/trigger-accident-alert routes to police & hospital."""
        res = self.client.post(
            "/api/surveillance/trigger-accident-alert",
            json={
                "severity_label": "Severe Accident (Unit Test)",
                "confidence": 0.96,
                "dispatch_emergency": False,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["incident_type"], "ACCIDENT_LVL_3")
        self.assertIn("police", data["departments_notified"])
        self.assertIn("hospital", data["departments_notified"])
        self.assertIn("traffic_control", data["departments_notified"])
        self.assertTrue(data["snapshot"]["filename"].startswith("manual_accident_"))

        # Clean up snapshot
        snap_path = data["snapshot"].get("filepath")
        if snap_path and os.path.exists(snap_path):
            os.remove(snap_path)

    def test_20_trigger_fire_alert_endpoint(self):
        """Verify POST /api/surveillance/trigger-fire-alert routes to fire dept & hospital."""
        res = self.client.post(
            "/api/surveillance/trigger-fire-alert",
            json={
                "severity_label": "Active Fire Hazard (Unit Test)",
                "confidence": 0.99,
                "dispatch_emergency": False,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["incident_type"], "FIRE_ALERT")
        self.assertIn("fire", data["departments_notified"])
        self.assertIn("hospital", data["departments_notified"])
        self.assertIn("police", data["departments_notified"])
        self.assertTrue(data["snapshot"]["filename"].startswith("manual_fire_"))

        # Clean up snapshot
        snap_path = data["snapshot"].get("filepath")
        if snap_path and os.path.exists(snap_path):
            os.remove(snap_path)


if __name__ == "__main__":
    unittest.main()


