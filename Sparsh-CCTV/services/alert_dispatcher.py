import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from config import (
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_LOCATION,
    DEFAULT_CAMERA_LATITUDE,
    DEFAULT_CAMERA_LONGITUDE,
    DISPATCH_LOG_PATH,
    OUTPUT_DIR,
    EMAIL_ALERTS_ENABLED,
    MAX_DISPATCH_FACILITIES_PER_TYPE,
)
from services.geo_service import GeoService, GeoLocation
from services.emergency_registry import EmergencyRegistry, EmergencyFacility
from services.email_service import EmailService

load_dotenv()


class EmergencyAlertDispatcher:
    """
    Coordinates multi-channel emergency alert dispatching (Console HUD,
    JSON audit log, Twilio SMS, and Webhooks) when accidents or fires are detected.
    """

    def __init__(
        self,
        camera_id: str = DEFAULT_CAMERA_ID,
        camera_location: str = DEFAULT_CAMERA_LOCATION,
        geo_service: Optional[GeoService] = None,
        registry: Optional[EmergencyRegistry] = None,
        dispatch_log_path: str = DISPATCH_LOG_PATH,
        enable_console: bool = True,
        webhook_url: Optional[str] = None,
        email_service: Optional[EmailService] = None,
        enable_email: bool = EMAIL_ALERTS_ENABLED,
        max_dispatch_facilities_per_type: int = MAX_DISPATCH_FACILITIES_PER_TYPE,
    ):
        self.camera_id = camera_id
        self.camera_location_str = camera_location
        self.dispatch_log_path = dispatch_log_path
        self.enable_console = enable_console
        self.webhook_url = webhook_url
        self.enable_email = enable_email
        self.max_dispatch_facilities_per_type = max_dispatch_facilities_per_type

        # Initialize Geolocation Service
        self.geo_service = geo_service or GeoService()
        self.location: GeoLocation = self.geo_service.resolve_location(
            self.camera_location_str
        )

        # Initialize Emergency Registry (within 3km of resolved coordinates)
        self.registry = registry or EmergencyRegistry(
            camera_lat=self.location.latitude, camera_lon=self.location.longitude
        )

        # Initialize Email Service for Department Notifications
        self.email_service = email_service or (
            EmailService(enabled=enable_email) if self.enable_email else None
        )

        # Twilio Configuration
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = os.getenv("TWILIO_PHONE_NUMBER")
        self.dispatch_to = os.getenv("EMERGENCY_DISPATCH_PHONE")

        self.last_dispatch_time = 0.0

    def dispatch(
        self,
        incident_type: str,
        severity_label: str = "High",
        confidence: float = 1.0,
        snapshot_path: Optional[str] = None,
        vehicles_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Builds incident payload, routes to nearest emergency facilities (hospitals,
        fire stations, police stations within 3km), and broadcasts across channels.
        """
        now_dt = datetime.now()
        dispatch_id = f"DISPATCH-{now_dt.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"

        # 1. Route to emergency facilities according to incident classification
        routed_facilities = self.registry.get_facilities_for_incident(
            incident_type, limit_per_type=self.max_dispatch_facilities_per_type
        )

        # 2. Construct dispatch payload
        payload = {
            "dispatch_id": dispatch_id,
            "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "camera": {
                "camera_id": self.camera_id,
                "location_string": self.camera_location_str,
                "formatted_address": self.location.formatted_address,
                "coordinates": {
                    "latitude": round(self.location.latitude, 7),
                    "longitude": round(self.location.longitude, 7),
                },
                "maps_url": self.location.maps_url,
            },
            "incident": {
                "type": incident_type,
                "severity_label": severity_label,
                "confidence": round(float(confidence), 3),
                "vehicles_involved": vehicles_count,
                "snapshot_path": snapshot_path or "N/A",
            },
            "dispatched_services": {
                cat: [f.to_dict() for f in fac_list]
                for cat, fac_list in routed_facilities.items()
            },
            "notification_status": {},
        }

        # 3. Attempt Twilio SMS notification
        sms_status = self._send_twilio_alert(payload, routed_facilities)
        payload["notification_status"]["sms"] = sms_status

        # 4. Dispatch automated email alert to designated departments (asynchronously)
        if self.enable_email and self.email_service:
            email_depts = self.email_service.get_departments_for_incident(incident_type)
            email_recips = self.email_service.get_recipients_for_incident(incident_type)
            # Async non-blocking submission to worker thread
            self.email_service.send_incident_alert_async(
                payload=payload, snapshot_path=snapshot_path
            )
            payload["notification_status"]["email"] = {
                "status": "DISPATCHED_ASYNC",
                "departments": email_depts,
                "recipients": email_recips,
            }
        else:
            payload["notification_status"]["email"] = {
                "status": "DISABLED",
                "message": "Email alerts disabled in configuration",
            }

        # 5. Print high-visibility Console Banner
        if self.enable_console:
            self._render_console_banner(payload, routed_facilities)
            payload["notification_status"]["console_broadcast"] = True

        # 6. Record to persistent JSON dispatch log
        self._record_json_dispatch(payload)
        payload["notification_status"]["json_logged"] = True

        self.last_dispatch_time = time.time()
        return payload

    def _render_console_banner(
        self, payload: Dict[str, Any], routed: Dict[str, List[EmergencyFacility]]
    ):
        """Displays formatted ASCII emergency dispatch banner in console."""
        inc = payload["incident"]
        cam = payload["camera"]

        print("\n" + "=" * 76)
        print(" [EMERGENCY ALERT DISPATCH] ".center(76, "!"))
        print("=" * 76)
        print(
            f" Incident Event   : {inc['type']} ({inc['severity_label']}) - Conf: {inc['confidence']:.2f}"
        )
        print(f" Camera ID        : {cam['camera_id']}")
        print(f" Camera Location  : {cam['location_string']}")
        print(
            f" GPS Coordinates  : {cam['coordinates']['latitude']}, {cam['coordinates']['longitude']}"
        )
        print(f" Google Maps Link : {cam['maps_url']}")
        if inc["snapshot_path"] != "N/A":
            print(f" Snapshot Image   : {inc['snapshot_path']}")
        print("-" * 76)
        print(" NOTIFIED EMERGENCY SERVICES (Within 3km Radius):")

        for cat, facs in routed.items():
            cat_label = cat.replace("_", " ").upper()
            if facs:
                for f in facs:
                    print(
                        f"   * [{cat_label}] {f.name} - {f.distance_km:.2f} km away (Phone: {f.phone})"
                    )
            else:
                print(f"   * [{cat_label}] None found within 3.0 km")

        # Display Email Dispatch Status
        email_status = payload.get("notification_status", {}).get("email", {})
        if email_status.get("status") == "DISPATCHED_ASYNC":
            dept_names = ", ".join(
                d.upper() for d in email_status.get("departments", [])
            )
            recip_names = ", ".join(email_status.get("recipients", []))
            print("-" * 76)
            print(f" EMAILS DISPATCHED: [{dept_names}] -> {recip_names}")

        print("=" * 76 + "\n")

    def _send_twilio_alert(
        self,
        payload: Dict[str, Any],
        routed: Dict[str, List[EmergencyFacility]],
        to_override: Optional[str] = None,
        message_override: Optional[str] = None,
    ) -> str:
        """Sends SMS via Twilio if configured, or logs simulated transmission."""
        inc = payload.get("incident", {})
        cam = payload.get("camera", {})

        # Summarize top responder facilities for SMS body
        responder_snippets = []
        for cat, facs in routed.items():
            if facs:
                top_fac = facs[0]
                responder_snippets.append(
                    f"{top_fac.name} ({top_fac.distance_km:.1f}km)"
                )
        responder_summary = ", ".join(responder_snippets) or "Local Units Alerted"

        sms_body = message_override or (
            f"URGENT ALERT: {inc.get('type', 'INCIDENT')} ({inc.get('severity_label', 'Alert')}) detected at {cam.get('location_string', 'Camera Location')}. "
            f"Pinging nearest: {responder_summary}. Map: {cam.get('maps_url', '')}"
        )

        target_to = to_override or self.dispatch_to

        if self.twilio_sid and self.twilio_auth and self.twilio_from and target_to:
            try:
                from twilio.rest import Client

                client = Client(self.twilio_sid, self.twilio_auth)
                message = client.messages.create(
                    body=sms_body, from_=self.twilio_from, to=target_to
                )
                print(
                    f"[Twilio SMS] Emergency SMS successfully sent to {target_to} (SID: {message.sid})"
                )
                return f"SENT (Twilio SID: {message.sid})"
            except Exception as e:
                print(f"[Twilio SMS] Failed to send SMS: {e}")
                return f"FAILED ({str(e)})"
        else:
            # Simulated mode
            print(
                f"[Twilio SMS Simulation] Ready. Formatted SMS: '{sms_body[:80]}...' (Target: {target_to or 'Simulated'})"
            )
            return "SIMULATED (Provide TWILIO_PHONE_NUMBER and EMERGENCY_DISPATCH_PHONE in .env for live SMS)"

    def _record_json_dispatch(self, payload: Dict[str, Any]):
        """Appends dispatch record to the emergency dispatch JSON log file."""
        try:
            os.makedirs(os.path.dirname(self.dispatch_log_path), exist_ok=True)
            records = []
            if os.path.exists(self.dispatch_log_path):
                try:
                    with open(self.dispatch_log_path, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    if not isinstance(records, list):
                        records = [records]
                except Exception:
                    records = []

            records.append(payload)
            with open(self.dispatch_log_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            print(f"[EmergencyAlertDispatcher] Error recording dispatch JSON: {e}")

    def update_location(
        self,
        new_location: str,
        new_radius_km: Optional[float] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ):
        """Dynamically re-resolves coordinates and updates emergency facilities registry."""
        self.camera_location_str = new_location
        if latitude is not None and longitude is not None:
            self.location = GeoLocation(
                latitude=float(latitude),
                longitude=float(longitude),
                formatted_address=new_location,
                source="manual_coordinates",
            )
        else:
            self.location = self.geo_service.resolve_location(new_location)

        search_radius = (
            new_radius_km
            if new_radius_km is not None
            else (self.registry.max_radius_km if self.registry else 3.0)
        )
        self.registry = EmergencyRegistry(
            camera_lat=self.location.latitude,
            camera_lon=self.location.longitude,
            max_radius_km=search_radius,
        )

    def update_search_radius(self, new_radius_km: float):
        """Dynamically updates the emergency services search radius."""
        if self.registry:
            self.registry.max_radius_km = new_radius_km
            self.registry._initialize_registry()

    def update_twilio(
        self,
        sid: Optional[str] = None,
        auth: Optional[str] = None,
        phone_from: Optional[str] = None,
        dispatch_to: Optional[str] = None,
    ):
        """Dynamically updates Twilio SMS credentials and target numbers."""
        if sid is not None:
            self.twilio_sid = sid
        if auth is not None:
            self.twilio_auth = auth
        if phone_from is not None:
            self.twilio_from = phone_from
        if dispatch_to is not None:
            self.dispatch_to = dispatch_to

    def update_email_service(
        self,
        email_service: Optional[EmailService] = None,
        enable_email: Optional[bool] = None,
    ):
        """Dynamically updates associated EmailService reference and alert enablement."""
        if email_service is not None:
            self.email_service = email_service
        if enable_email is not None:
            self.enable_email = enable_email

    def get_recent_dispatches(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Reads recent emergency dispatches from the JSON dispatch log file."""
        if not os.path.exists(self.dispatch_log_path):
            return []
        try:
            with open(self.dispatch_log_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = [records]
            return records[-limit:][::-1]  # Return newest first
        except Exception as e:
            print(f"[EmergencyAlertDispatcher] Error reading dispatch log: {e}")
            return []
