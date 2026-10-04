import json
import os
import sys
import time
import unittest
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_LOCATION,
    DEFAULT_RECIPIENT_EMAIL,
    SMTP_HOST,
    SMTP_PORT,
)
from services.email_service import EmailService
from services.alert_dispatcher import EmergencyAlertDispatcher


class TestEmailService(unittest.TestCase):
    def setUp(self):
        self.service = EmailService(
            enabled=True,
            department_emails={
                "police": "police@test.org",
                "hospital": "hospital@test.org",
                "fire": "fire@test.org",
                "traffic_control": "traffic@test.org",
            },
        )

        self.sample_payload = {
            "dispatch_id": "DISPATCH-TEST-001",
            "timestamp": "2026-10-03 12:00:00",
            "camera": {
                "camera_id": "TEST-CAM-01",
                "location_string": DEFAULT_CAMERA_LOCATION,
                "coordinates": {"latitude": 28.5537375, "longitude": 77.0433594},
                "maps_url": "https://www.google.com/maps/search/?api=1&query=28.5537375,77.0433594",
            },
            "incident": {
                "type": "ACCIDENT_LVL_3",
                "severity_label": "Severe Accident",
                "confidence": 0.92,
                "vehicles_involved": 2,
                "snapshot_path": "N/A",
            },
            "dispatched_services": {
                "hospitals": [
                    {
                        "name": "Maple Care Hospital",
                        "distance_km": 1.08,
                        "phone": "102",
                        "address": "Sector 23",
                    }
                ],
                "police_stations": [
                    {
                        "name": "Sector 21 Police Post",
                        "distance_km": 1.31,
                        "phone": "112",
                        "address": "Sector 21",
                    }
                ],
            },
        }

    def test_01_department_routing_logic(self):
        """Verify incident-to-department routing rules."""
        # Accident routing
        accident_depts = self.service.get_departments_for_incident("ACCIDENT_LVL_3")
        self.assertIn("police", accident_depts)
        self.assertIn("hospital", accident_depts)
        self.assertIn("traffic_control", accident_depts)
        self.assertNotIn("fire", accident_depts)

        # Fire routing
        fire_depts = self.service.get_departments_for_incident("FIRE_ALERT")
        self.assertIn("fire", fire_depts)
        self.assertIn("hospital", fire_depts)
        self.assertIn("police", fire_depts)

        # Smoke routing
        smoke_depts = self.service.get_departments_for_incident("SMOKE_ALERT")
        self.assertIn("fire", smoke_depts)
        self.assertIn("police", smoke_depts)

    def test_02_recipients_resolution(self):
        """Verify recipient email address mapping and deduplication."""
        recipients = self.service.get_recipients_for_incident("ACCIDENT_LVL_3")
        self.assertEqual(len(recipients), 3)
        self.assertIn("police@test.org", recipients)
        self.assertIn("hospital@test.org", recipients)
        self.assertIn("traffic@test.org", recipients)

        # Default fallback test
        default_svc = EmailService()
        def_recips = default_svc.get_recipients_for_incident("FIRE_ALERT")
        self.assertIn(DEFAULT_RECIPIENT_EMAIL, def_recips)

    def test_03_mime_email_construction(self):
        """Verify MIME structure, subject, HTML body, and maps link."""
        recipients = ["dept1@test.org", "dept2@test.org"]
        departments = ["police", "hospital"]

        msg = self.service.build_email_message(
            payload=self.sample_payload, recipients=recipients, departments=departments
        )

        self.assertIn("Severe Accident", msg["Subject"])
        self.assertIn("TEST-CAM-01", msg["Subject"])
        self.assertEqual(msg["To"], "dept1@test.org, dept2@test.org")

        # Walk parts and inspect text and HTML
        parts = [
            p.get_payload(decode=True).decode("utf-8")
            for p in msg.walk()
            if p.get_content_type() in ("text/plain", "text/html")
        ]
        self.assertGreaterEqual(len(parts), 2)

        plain_text, html_body = parts[0], parts[1]
        self.assertIn("ACCIDENT_LVL_3", plain_text)
        self.assertIn("https://www.google.com/maps/search/", plain_text)
        self.assertIn("Maple Care Hospital", html_body)
        self.assertIn("1.08 km", html_body)
        self.assertIn("POLICE, HOSPITAL", html_body)

    def test_04_snapshot_attachment(self):
        """Verify inline snapshot JPEG attachment embedding."""
        temp_img_path = os.path.join(
            os.path.dirname(__file__), "temp_test_snapshot.jpg"
        )
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(temp_img_path, dummy_img)

        try:
            msg = self.service.build_email_message(
                payload=self.sample_payload,
                recipients=["test@example.com"],
                departments=["police"],
                snapshot_path=temp_img_path,
            )

            # Check for image/jpeg part
            img_parts = [p for p in msg.walk() if p.get_content_type() == "image/jpeg"]
            self.assertEqual(len(img_parts), 1)
            self.assertEqual(img_parts[0]["Content-ID"], "<incident_snapshot_img>")
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    def test_05_disabled_mode(self):
        """Verify that when disabled, send_incident_alert returns DISABLED without connecting."""
        disabled_service = EmailService(enabled=False)
        res = disabled_service.send_incident_alert(self.sample_payload)
        self.assertEqual(res["status"], "DISABLED")

    def test_06_async_execution_non_blocking(self):
        """Verify that send_incident_alert_async returns immediately without blocking."""
        # Intentionally mock/simulated to test thread timing
        test_service = EmailService(enabled=False)

        start = time.time()
        future = test_service.send_incident_alert_async(self.sample_payload)
        elapsed = time.time() - start

        # Call must return quickly without waiting for email delivery
        self.assertLess(elapsed, 0.10, f"Async call took too long: {elapsed:.4f}s")
        result = future.result(timeout=2.0)
        self.assertEqual(result["status"], "DISABLED")

    def test_07_alert_dispatcher_integration(self):
        """Verify EmergencyAlertDispatcher includes email dispatch in notification_status."""
        test_email_svc = EmailService(enabled=False)
        dispatcher = EmergencyAlertDispatcher(
            camera_id="CAM-EMAIL-TEST",
            camera_location=DEFAULT_CAMERA_LOCATION,
            email_service=test_email_svc,
            enable_email=True,
            enable_console=False,
        )

        res = dispatcher.dispatch(
            incident_type="ACCIDENT_LVL_3",
            severity_label="Severe Accident",
            confidence=0.88,
        )

        self.assertIn("email", res["notification_status"])
        email_status = res["notification_status"]["email"]
        self.assertEqual(email_status["status"], "DISPATCHED_ASYNC")
        self.assertIn("police", email_status["departments"])


if __name__ == "__main__":
    unittest.main()
