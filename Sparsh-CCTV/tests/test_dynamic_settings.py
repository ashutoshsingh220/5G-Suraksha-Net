import os
import sys
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    get_settings,
    update_settings,
    register_settings_listener,
    unregister_settings_listener,
    SystemSettings,
)
import config
from surveillance_engine import get_surveillance_engine


class TestDynamicSettings(unittest.TestCase):
    """
    Validates Requirement 2: Dynamic runtime reconfiguration of all system settings
    (Camera, RTSP, Location, Thresholds, Cooldown, Search Radius, Email, Twilio)
    without requiring a process restart.
    """

    def setUp(self):
        self.engine = get_surveillance_engine(auto_init_detectors=True)

    def test_01_listener_notification_mechanism(self):
        """Verify settings listeners receive updates synchronously."""
        received_updates = []

        def listener(settings, updates):
            received_updates.append((settings, updates))

        register_settings_listener(listener)
        try:
            update_settings({"camera_id": "NOTIFY-TEST-CAM"})
            self.assertEqual(len(received_updates), 1)
            self.assertEqual(received_updates[0][1]["camera_id"], "NOTIFY-TEST-CAM")
            self.assertEqual(received_updates[0][0].camera_id, "NOTIFY-TEST-CAM")
        finally:
            unregister_settings_listener(listener)

    def test_02_dynamic_detector_thresholds_synchronization(self):
        """Verify updating crash and fire thresholds modifies active detectors immediately."""
        new_crash_conf = 0.44
        new_fire_conf = 0.82

        update_settings({"crash_conf": new_crash_conf, "fire_conf": new_fire_conf})

        # Check SystemSettings
        settings = get_settings()
        self.assertEqual(settings.crash_conf, new_crash_conf)
        self.assertEqual(settings.fire_conf, new_fire_conf)

        # Check config module-level globals
        self.assertEqual(config.DEFAULT_CRASH_CONF, new_crash_conf)
        self.assertEqual(config.DEFAULT_FIRE_CONF, new_fire_conf)

        # Check active detectors in running surveillance engine
        if self.engine.accident_detector:
            self.assertEqual(self.engine.accident_detector.conf_thres, new_crash_conf)
        if self.engine.fire_detector:
            self.assertEqual(self.engine.fire_detector.conf_thres, new_fire_conf)

    def test_03_dynamic_alert_cooldown_and_dispatch_toggle(self):
        """Verify updating cooldown and dispatch toggle dynamically updates incident logger."""
        new_cooldown = 15.0

        update_settings({"alert_cooldown_sec": new_cooldown, "enable_dispatch": False})

        # Check config globals
        self.assertEqual(config.ALERT_SNAPSHOT_COOLDOWN_SEC, new_cooldown)

        # Check incident logger on engine
        if self.engine.incident_logger:
            self.assertEqual(self.engine.incident_logger.cooldown, new_cooldown)
            self.assertFalse(self.engine.incident_logger.enable_dispatch)

        # Restore dispatch
        update_settings({"enable_dispatch": True, "alert_cooldown_sec": 5.0})
        if self.engine.incident_logger:
            self.assertTrue(self.engine.incident_logger.enable_dispatch)
            self.assertEqual(self.engine.incident_logger.cooldown, 5.0)

    def test_04_dynamic_camera_location_and_search_radius(self):
        """Verify changing camera location re-geocodes and re-initializes emergency facilities."""
        test_location = "Sector 21 Dwarka, Dwarka, Delhi, 110077"
        test_radius = 2.5

        update_settings(
            {
                "camera_location": test_location,
                "emergency_search_radius_km": test_radius,
            }
        )

        settings = get_settings()
        self.assertEqual(settings.camera_location, test_location)
        self.assertEqual(settings.emergency_search_radius_km, test_radius)
        self.assertEqual(config.DEFAULT_CAMERA_LOCATION, test_location)
        self.assertEqual(config.EMERGENCY_SEARCH_RADIUS_KM, test_radius)

        if self.engine.dispatcher:
            self.assertEqual(self.engine.dispatcher.camera_location_str, test_location)
            self.assertEqual(self.engine.dispatcher.registry.max_radius_km, test_radius)
            # Facilities must all be within 2.5 km
            for fac in self.engine.dispatcher.registry.facilities:
                self.assertLessEqual(fac.distance_km, test_radius)

    def test_05_dynamic_email_settings_and_recipients(self):
        """Verify dynamic updates to SMTP credentials and department email recipients."""
        new_email = "emergency-dispatch-center@sparsh.org"
        new_port = 465

        update_settings(
            {
                "default_recipient_email": new_email,
                "smtp_port": new_port,
                "smtp_enabled": True,
            }
        )

        settings = get_settings()
        self.assertEqual(settings.default_recipient_email, new_email)
        self.assertEqual(settings.smtp_port, new_port)
        self.assertEqual(config.DEFAULT_RECIPIENT_EMAIL, new_email)
        self.assertEqual(config.SMTP_PORT, new_port)

        if self.engine.email_service:
            self.assertEqual(self.engine.email_service.port, new_port)
            self.assertTrue(self.engine.email_service.enabled)

    def test_06_dynamic_twilio_settings(self):
        """Verify dynamic updates to Twilio credentials and dispatch phone number."""
        test_sid = "AC_DYNAMIC_TEST_SID"
        test_auth = "AUTH_DYNAMIC_TEST_TOKEN"
        test_from = "+15550001111"
        test_to = "+919999999999"

        update_settings(
            {
                "twilio_account_sid": test_sid,
                "twilio_auth_token": test_auth,
                "twilio_phone_number": test_from,
                "emergency_dispatch_phone": test_to,
            }
        )

        settings = get_settings()
        self.assertEqual(settings.twilio_account_sid, test_sid)
        self.assertEqual(settings.twilio_auth_token, test_auth)
        self.assertEqual(settings.twilio_phone_number, test_from)
        self.assertEqual(settings.emergency_dispatch_phone, test_to)

        if self.engine.dispatcher:
            self.assertEqual(self.engine.dispatcher.twilio_sid, test_sid)
            self.assertEqual(self.engine.dispatcher.twilio_auth, test_auth)
            self.assertEqual(self.engine.dispatcher.twilio_from, test_from)
            self.assertEqual(self.engine.dispatcher.dispatch_to, test_to)

    def test_07_dynamic_rtsp_url_computation(self):
        """Verify computed RTSP URL adapts when IP, port, channel, or credentials change."""
        update_settings(
            {
                "rtsp_camera_ip": "10.0.0.50",
                "rtsp_port": 8554,
                "rtsp_channel": 2,
                "rtsp_stream": 1,
                "rtsp_username": "operator",
                "rtsp_password": "securepassword",
            }
        )

        settings = get_settings()
        expected_url = "rtsp://operator:securepassword@10.0.0.50:8554/avstream/channel=2/stream=1.sdp"
        self.assertEqual(settings.rtsp_url, expected_url)
        self.assertEqual(config.SPARSH_RTSP_URL, expected_url)

        # Custom RTSP URL override
        custom_url = "rtsp://custom-cctv.local:554/live"
        update_settings({"custom_rtsp_url": custom_url})
        self.assertEqual(get_settings().rtsp_url, custom_url)
        self.assertEqual(config.SPARSH_RTSP_URL, custom_url)

        # Reset custom
        update_settings({"custom_rtsp_url": None})
        self.assertIsNone(get_settings().custom_rtsp_url)
        self.assertEqual(get_settings().rtsp_url, expected_url)
        self.assertEqual(config.SPARSH_RTSP_URL, expected_url)

    @classmethod
    def tearDownClass(cls):
        from config import reset_settings_to_defaults

        reset_settings_to_defaults()


if __name__ == "__main__":
    unittest.main()
