import os
import smtplib
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any, Callable

from config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_SENDER_EMAIL,
    SMTP_APP_PASSWORD,
    SMTP_TIMEOUT_SECONDS,
    EMAIL_ALERTS_ENABLED,
    DEPARTMENT_EMAILS,
    INCIDENT_DEPARTMENT_ROUTING,
    ATTACH_SNAPSHOT_IN_EMAIL,
    DEFAULT_RECIPIENT_EMAIL,
)


class EmailService:
    """
    Manages automated emergency email alerts to designated municipal and
    emergency departments (Police, Hospital, Fire Department) with rich HTML formatting,
    inline snapshot image embedding, and non-blocking asynchronous dispatch.
    """

    def __init__(
        self,
        host: str = SMTP_HOST,
        port: int = SMTP_PORT,
        use_tls: bool = SMTP_USE_TLS,
        sender_email: str = SMTP_SENDER_EMAIL,
        password: str = SMTP_APP_PASSWORD,
        timeout: int = SMTP_TIMEOUT_SECONDS,
        enabled: bool = EMAIL_ALERTS_ENABLED,
        department_emails: Optional[Dict[str, str]] = None,
        incident_routing: Optional[Dict[str, List[str]]] = None,
        attach_snapshot: bool = ATTACH_SNAPSHOT_IN_EMAIL,
    ):
        self.host = host
        self.port = port
        self.use_tls = use_tls
        self.sender_email = sender_email
        self.password = (password or "").replace(" ", "")
        self.timeout = timeout
        self.enabled = enabled
        self.department_emails = department_emails or DEPARTMENT_EMAILS
        self.incident_routing = incident_routing or INCIDENT_DEPARTMENT_ROUTING
        self.attach_snapshot = attach_snapshot

        # Background thread pool executor for non-blocking 15 FPS surveillance loop
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="EmergencyEmailWorker"
        )

    def get_departments_for_incident(self, incident_type: str) -> List[str]:
        """Resolves target departments according to the incident category."""
        upper = incident_type.upper()
        if "FIRE" in upper:
            return self.incident_routing.get("fire", ["fire", "hospital", "police"])
        elif "SMOKE" in upper:
            return self.incident_routing.get("smoke", ["fire", "police"])
        else:
            # Accident / Crash event
            return self.incident_routing.get(
                "accident", ["police", "hospital", "traffic_control"]
            )

    def get_recipients_for_incident(self, incident_type: str) -> List[str]:
        """Returns unique list of recipient email addresses for an incident."""
        departments = self.get_departments_for_incident(incident_type)
        recipients = []
        for dept in departments:
            email = self.department_emails.get(dept, DEFAULT_RECIPIENT_EMAIL)
            if email and email not in recipients:
                recipients.append(email)
        return recipients or [DEFAULT_RECIPIENT_EMAIL]

    def build_email_message(
        self,
        payload: Dict[str, Any],
        recipients: List[str],
        departments: List[str],
        snapshot_path: Optional[str] = None,
    ) -> MIMEMultipart:
        """Constructs a multipart MIME email with HTML body, plain text, and embedded snapshot."""
        inc = payload.get("incident", {})
        cam = payload.get("camera", {})
        dispatched_services = payload.get("dispatched_services", {})
        event_type = inc.get("type", "INCIDENT")
        severity_label = inc.get("severity_label", "Alert")
        confidence = inc.get("confidence", 1.0)
        camera_id = cam.get("camera_id", "SPARSH-CAM")
        location_str = cam.get("location_string", "Unknown Location")
        coords = cam.get("coordinates", {})
        maps_url = cam.get("maps_url", "#")
        timestamp = payload.get(
            "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Determine theme color based on severity
        is_fire = "FIRE" in event_type.upper() or "SMOKE" in event_type.upper()
        theme_color = (
            "#d32f2f"
            if (
                is_fire
                or "SEVERE" in severity_label.upper()
                or "TOTALED" in severity_label.upper()
            )
            else "#f57c00"
        )
        header_title = (
            "CRITICAL EMERGENCY ALERT"
            if theme_color == "#d32f2f"
            else "URGENT INCIDENT ALERT"
        )

        subject = f"[{header_title}] {severity_label} Detected by {camera_id} at {location_str}"

        # Root message
        msg_root = MIMEMultipart("related")
        msg_root["Subject"] = subject
        msg_root["From"] = f"Sparsh CCTV Emergency Center <{self.sender_email}>"
        msg_root["To"] = ", ".join(recipients)
        msg_root["X-Priority"] = "1"  # High priority flag

        # Alternative container (Plain text + HTML)
        msg_alt = MIMEMultipart("alternative")
        msg_root.attach(msg_alt)

        # 1. Plain Text Version
        dept_str = ", ".join(d.upper() for d in departments)
        plain_text = f"""*** {header_title} ***
Timestamp       : {timestamp}
Event Type      : {event_type} ({severity_label})
Confidence      : {confidence * 100:.1f}%
Camera ID       : {camera_id}
Location        : {location_str}
Coordinates     : {coords.get("latitude")}, {coords.get("longitude")}
Google Maps     : {maps_url}
Notified Depts  : {dept_str}

NOTIFIED EMERGENCY SERVICES WITHIN 3KM:
"""
        for cat, facs in dispatched_services.items():
            plain_text += f"\n[{cat.replace('_', ' ').upper()}]:\n"
            for f in facs:
                plain_text += f" - {f.get('name')} | {f.get('distance_km')} km away | Phone: {f.get('phone')}\n"

        plain_text += (
            "\nAutomated alert sent by Sparsh CCTV AI Agentic Surveillance System."
        )
        msg_alt.attach(MIMEText(plain_text, "plain", "utf-8"))

        # 2. HTML Version with Embedded CSS
        facilities_html = ""
        for cat, facs in dispatched_services.items():
            cat_name = cat.replace("_", " ").title()
            facilities_html += f"""
            <h4 style="margin: 12px 0 6px 0; color: #37474f; font-size: 14px; text-transform: uppercase;">{cat_name}</h4>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 12px; font-size: 13px;">
                <thead>
                    <tr style="background-color: #f1f3f4; text-align: left;">
                        <th style="padding: 6px 8px; border: 1px solid #e0e0e0;">Facility Name</th>
                        <th style="padding: 6px 8px; border: 1px solid #e0e0e0; width: 90px;">Distance</th>
                        <th style="padding: 6px 8px; border: 1px solid #e0e0e0; width: 110px;">Contact</th>
                    </tr>
                </thead>
                <tbody>
            """
            for f in facs:
                facilities_html += f"""
                    <tr>
                        <td style="padding: 6px 8px; border: 1px solid #e0e0e0;"><strong>{f.get("name")}</strong><br><span style="font-size: 11px; color: #616161;">{f.get("address")}</span></td>
                        <td style="padding: 6px 8px; border: 1px solid #e0e0e0; color: #d32f2f; font-weight: bold;">{f.get("distance_km")} km</td>
                        <td style="padding: 6px 8px; border: 1px solid #e0e0e0;">{f.get("phone")}</td>
                    </tr>
                """
            facilities_html += "</tbody></table>"

        snapshot_cid = "incident_snapshot_img"
        snapshot_html_block = ""
        if snapshot_path and os.path.exists(snapshot_path):
            snapshot_html_block = f"""
            <div style="margin: 16px 0; text-align: center;">
                <p style="font-weight: bold; margin-bottom: 6px; color: #37474f;">Captured Incident Snapshot:</p>
                <img src="cid:{snapshot_cid}" alt="Incident Snapshot" style="max-width: 100%; height: auto; border: 2px solid {theme_color}; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);" />
            </div>
            """

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.5; color: #212121; background-color: #f8f9fa; margin: 0; padding: 20px;">
            <div style="max-width: 650px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <!-- Header Banner -->
                <div style="background-color: {theme_color}; color: #ffffff; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 22px; letter-spacing: 1px;">🚨 {header_title}</h1>
                    <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.95;">Sparsh CCTV Agentic Surveillance & Rapid Emergency Response</p>
                </div>
                
                <!-- Main Body Content -->
                <div style="padding: 24px;">
                    <!-- Alert Details Table -->
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; width: 140px; color: #424242;">Incident Event:</td>
                            <td style="padding: 8px 0;"><span style="background-color: {theme_color}; color: #ffffff; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 13px;">{event_type} - {severity_label}</span> (Conf: {confidence * 100:.1f}%)</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #424242;">Camera Source:</td>
                            <td style="padding: 8px 0;"><strong>{camera_id}</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #424242;">Camera Location:</td>
                            <td style="padding: 8px 0;">{location_str}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #424242;">GPS Coordinates:</td>
                            <td style="padding: 8px 0;">{coords.get("latitude")}, {coords.get("longitude")}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #424242;">Detected Time:</td>
                            <td style="padding: 8px 0;">{timestamp}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #424242;">Routed Departments:</td>
                            <td style="padding: 8px 0;"><strong style="color: #0d47a1;">{dept_str}</strong></td>
                        </tr>
                    </table>

                    <!-- Google Maps Button -->
                    <div style="text-align: center; margin: 20px 0;">
                        <a href="{maps_url}" target="_blank" style="background-color: #1a73e8; color: #ffffff; text-decoration: none; padding: 12px 24px; border-radius: 6px; font-weight: bold; display: inline-block; font-size: 14px; box-shadow: 0 2px 4px rgba(26,115,232,0.3);">
                            📍 Open Exact Location in Google Maps
                        </a>
                    </div>

                    {snapshot_html_block}

                    <!-- 3km Emergency Responders List -->
                    <div style="margin-top: 24px; border-top: 1px solid #eeeeee; padding-top: 16px;">
                        <h3 style="margin: 0 0 12px 0; color: #263238; font-size: 16px;">Nearby Emergency Facilities (Within 3km Radius):</h3>
                        {facilities_html}
                    </div>
                </div>

                <!-- Footer -->
                <div style="background-color: #f1f3f4; padding: 14px 20px; font-size: 11px; color: #757575; text-align: center; border-top: 1px solid #e0e0e0;">
                    This is an automated dispatch from the Sparsh CCTV Real-Time Video Analytics Pipeline.<br>
                    Camera Reference: {camera_id} | Location: {location_str}
                </div>
            </div>
        </body>
        </html>
        """
        msg_alt.attach(MIMEText(html_body, "html", "utf-8"))

        # 3. Attach Inline Snapshot Image
        if snapshot_path and os.path.exists(snapshot_path) and self.attach_snapshot:
            try:
                with open(snapshot_path, "rb") as img_f:
                    img_data = img_f.read()
                mime_img = MIMEImage(img_data)
                mime_img.add_header("Content-ID", f"<{snapshot_cid}>")
                mime_img.add_header(
                    "Content-Disposition",
                    "inline",
                    filename=os.path.basename(snapshot_path),
                )
                msg_root.attach(mime_img)
            except Exception as e:
                print(f"[EmailService] Warning: Could not attach snapshot image: {e}")

        return msg_root

    def send_incident_alert(
        self,
        payload: Dict[str, Any],
        snapshot_path: Optional[str] = None,
        override_recipients: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronously sends emergency alert email to designated departments.
        Returns a delivery status dictionary.
        """
        incident_type = payload.get("incident", {}).get("type", "INCIDENT")
        departments = self.get_departments_for_incident(incident_type)
        recipients = override_recipients or self.get_recipients_for_incident(
            incident_type
        )

        if not self.enabled:
            return {
                "status": "DISABLED",
                "recipients": recipients,
                "departments": departments,
                "message": "Email alerts disabled in configuration",
            }

        if not self.sender_email or not self.password:
            print(
                "[EmailService] Warning: Missing SMTP credentials. Running in simulated mode."
            )
            return {
                "status": "SIMULATED",
                "recipients": recipients,
                "departments": departments,
                "message": "SMTP credentials missing in .env",
            }

        snapshot = snapshot_path or payload.get("incident", {}).get("snapshot_path")
        msg = self.build_email_message(payload, recipients, departments, snapshot)

        try:
            server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
            if self.use_tls:
                server.starttls()
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, recipients, msg.as_string())
            server.quit()

            dept_summary = ", ".join(departments)
            recip_summary = ", ".join(recipients)
            print(
                f"[EmailService] Emergency alert email sent to [{dept_summary}] -> {recip_summary}"
            )
            return {
                "status": "SENT",
                "recipients": recipients,
                "departments": departments,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            err_msg = str(e)
            print(f"[EmailService] Failed to send emergency email: {err_msg}")
            return {
                "status": "FAILED",
                "recipients": recipients,
                "departments": departments,
                "error": err_msg,
            }

    def send_incident_alert_async(
        self,
        payload: Dict[str, Any],
        snapshot_path: Optional[str] = None,
        override_recipients: Optional[List[str]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Future:
        """
        Asynchronously submits email dispatch to background thread pool.
        Guarantees 0ms blockage on real-time surveillance processing loop.
        """

        def _task():
            res = self.send_incident_alert(payload, snapshot_path, override_recipients)
            if callback:
                try:
                    callback(res)
                except Exception as cb_err:
                    print(f"[EmailService] Callback error: {cb_err}")
            return res

        return self._executor.submit(_task)

    def update_config(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_tls: Optional[bool] = None,
        sender_email: Optional[str] = None,
        password: Optional[str] = None,
        timeout: Optional[int] = None,
        enabled: Optional[bool] = None,
        department_emails: Optional[Dict[str, str]] = None,
        incident_routing: Optional[Dict[str, List[str]]] = None,
        attach_snapshot: Optional[bool] = None,
    ):
        """Dynamically updates SMTP connection and recipient settings at runtime."""
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        if use_tls is not None:
            self.use_tls = use_tls
        if sender_email is not None:
            self.sender_email = sender_email
        if password is not None:
            self.password = (password or "").replace(" ", "")
        if timeout is not None:
            self.timeout = timeout
        if enabled is not None:
            self.enabled = enabled
        if department_emails is not None:
            self.department_emails = dict(department_emails)
        if incident_routing is not None:
            self.incident_routing = dict(incident_routing)
        if attach_snapshot is not None:
            self.attach_snapshot = attach_snapshot
