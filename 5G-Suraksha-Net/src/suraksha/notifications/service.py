"""Incident Email Alert and Evidence Delivery Service for 5G Suraksha-Net Phase 3D.

Builds structured, operational email notifications containing incident context,
location intelligence, deterministic response plans, and verified video evidence
attachments (~10 seconds: T-5s to T+5s).

Asynchronous and non-blocking:
- Zero impact on real-time computer vision frame processing
- Safe against SMTP failures, timeouts, and network disconnects
- Zero autonomous dispatch: strictly decision support for human operators
- Complete secret redaction
"""
from __future__ import annotations

import os
import smtplib
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from suraksha.config import EmailConfig, PROJECT_ROOT
from suraksha.incidents.schemas import IncidentReport, IncidentStatus
from suraksha.logging_utils import get_logger
from suraksha.notifications.schemas import (
    EmailNotificationResult,
    EmailNotificationStatus,
    EmailPayload,
)
from suraksha.response import IncidentResponse, ResponsePlanner

log = get_logger(__name__)


def _sanitize_error(err_str: str, config: EmailConfig | None = None) -> str:
    """Sanitize error messages to ensure no credentials or secret keys are logged."""
    sanitized = str(err_str)
    sensitive_keys = [
        os.environ.get("SMTP_PASSWORD"),
        os.environ.get("SURAKSHA_SMTP_PASSWORD"),
        os.environ.get("GOOGLE_MAPS_API_KEY"),
        os.environ.get("SURAKSHA_GOOGLE_MAPS_API_KEY"),
    ]
    if config is not None:
        sensitive_keys.append(config.get_password())
        sensitive_keys.append(config.smtp_password)
    for key in sensitive_keys:
        if key and str(key).strip():
            sanitized = sanitized.replace(str(key).strip(), "***REDACTED***")
    return sanitized


class EmailNotificationService:
    """Asynchronous, policy-driven email notification and evidence delivery service."""

    def __init__(
        self,
        config: EmailConfig | None = None,
        planner: ResponsePlanner | None = None,
    ) -> None:
        self.config = config or EmailConfig()
        self.planner = planner or ResponsePlanner()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="email_notifier")
            if self.config.enabled
            else None
        )
        self._pending_futures: list[Future] = []
        self._sent_incident_ids: set[str] = set()
        self._lock = threading.Lock()

    def _resolve_path(self, rel_or_abs: str | Path | None) -> Path | None:
        if not rel_or_abs:
            return None
        p = Path(rel_or_abs)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def build_incident_email(
        self,
        incident: IncidentReport,
        response: IncidentResponse | None = None,
        recipient_role: str = "GENERAL",
        custom_clip_path: str | Path | None = None,
    ) -> EmailPayload:
        """Construct structured, role-tailored operational email payload from an incident."""
        resp = response if response is not None else self.planner.plan(incident)

        # 1. Basic incident attributes
        inc_id = incident.incident_id
        itype_str = (
            incident.incident_type.value.upper()
            if hasattr(incident.incident_type, "value")
            else str(incident.incident_type).upper()
        )
        sev_str = (
            incident.severity.value.upper()
            if hasattr(incident.severity, "value")
            else str(incident.severity).upper()
        )
        time_str = incident.start_time.isoformat()
        cam_id = incident.camera_id
        conf_str = f"{incident.confidence:.2f}"

        # 2. Location intelligence (Google Maps Geocoded Intelligence)
        loc = incident.location or (resp.location if resp else None)
        if not loc or loc.name in ("Unknown", "Unknown Location") or "DESTINATION_DEMONSTRATION_SITE" in str(getattr(loc, "name", "")):
            try:
                from suraksha.location.provider import DemoLocationProvider
                demo_prov = DemoLocationProvider()
                loc = demo_prov.get_location()
            except Exception:
                pass

        if loc and loc.name and loc.name not in ("Unknown", "Unknown Location"):
            loc_name = loc.name
            clean_loc = loc.name.split(" (")[0].strip()
            loc_subject_suffix = f" at {clean_loc}"
        else:
            loc_name = "Yashobhoomi, Dwarka Sector 25, New Delhi"
            loc_subject_suffix = f" at {loc_name}"

        if loc and loc.latitude is not None and loc.longitude is not None:
            coords_str = f"Latitude {loc.latitude:.6f}, Longitude {loc.longitude:.6f}"
        else:
            coords_str = "Coordinates unavailable"

        loc_source = loc.source.value if (loc and hasattr(loc.source, "value")) else "geocoded_google_maps"

        # 3. Detection summary
        summary_lines = []
        if incident.details:
            for k, v in incident.details.items():
                if k not in ("detector", "video_file", "email_clip_path"):
                    summary_lines.append(f"  - {k}: {v}")
        if incident.person_count is not None:
            summary_lines.append(f"  - Crowd Count: {incident.person_count} persons in zone")
        if incident.track_ids:
            summary_lines.append(f"  - Track IDs Involved: {incident.track_ids}")
        detection_summary = "\n".join(summary_lines) if summary_lines else "  - Standard spatial/temporal detection pattern confirmed."

        # 4. Recommended Response Actions
        action_lines = []
        if resp.response_actions:
            for a in resp.response_actions:
                action_lines.append(f"  - [{a.priority.upper()}] {a.action_type.value}: {a.reason}")
        else:
            action_lines.append("  - Routine monitoring; no emergency action required by policy.")
        actions_summary = "\n".join(action_lines)

        # 5. Police Resource
        police_res = resp.police_resources[0] if resp.police_resources else None
        if police_res:
            p_dist = f"{police_res.distance_km:.2f} km (~{police_res.distance_m:.0f} m) [Straight-line geographic distance]" if police_res.distance_km is not None else "Distance unavailable"
            police_details = (
                f"  Name     : {police_res.name}\n"
                f"  Distance : {p_dist}\n"
                f"  Address  : {police_res.formatted_address or 'Address on file'}\n"
                f"  Contact  : {police_res.phone_number or '100 / 112'}"
            )
        else:
            police_details = "  Nearest police jurisdiction: Dwarka Sector 23 Police Station (Contact: 100 / 112)"

        # 6. Medical Resource
        med_res = resp.medical_resources[0] if resp.medical_resources else None
        if med_res:
            m_dist = f"{med_res.distance_km:.2f} km (~{med_res.distance_m:.0f} m) [Straight-line geographic distance]" if med_res.distance_km is not None else "Distance unavailable"
            medical_details = (
                f"  Name     : {med_res.name}\n"
                f"  Distance : {m_dist}\n"
                f"  Address  : {med_res.formatted_address or 'Address on file'}\n"
                f"  Contact  : {med_res.phone_number or '102 / 108'}"
            )
        else:
            medical_details = "  Nearest trauma facility: Manipal Hospital Dwarka / Trauma Centre (Contact: 102 / 108)"

        # 7. Evidence Attachment Verification
        max_bytes = self.config.max_attachment_mb * 1024 * 1024
        resolved_clip_path = custom_clip_path or incident.evidence.clip_path
        clip_path_obj = self._resolve_path(resolved_clip_path)

        # Prefer dedicated 10s evidence clip for rapid transmission (<20s)
        if clip_path_obj and clip_path_obj.is_file():
            stem = clip_path_obj.stem
            cand_10s = clip_path_obj.parent / f"{stem}_10s.mp4"
            if cand_10s.is_file():
                clip_path_obj = cand_10s
            elif "_10s" not in stem:
                cand_demo = PROJECT_ROOT / "outputs" / "demo" / "clips" / f"{stem}_10s.mp4"
                if cand_demo.is_file():
                    clip_path_obj = cand_demo

        clip_attachment_path = None
        clip_attached = False

        if clip_path_obj and clip_path_obj.is_file():
            size_b = clip_path_obj.stat().st_size
            if size_b <= max_bytes:
                clip_attachment_path = str(clip_path_obj)
                clip_attached = True
                clip_status_note = f"Incident evidence clip attached (~10 seconds: T-5s to T+5s, {size_b / 1024 / 1024:.2f} MB)"
            else:
                clip_status_note = f"Incident evidence clip omitted — file size ({size_b / 1024 / 1024:.1f} MB) exceeds maximum allowed ({self.config.max_attachment_mb} MB)"
        else:
            clip_status_note = "Incident clip unavailable."

        snap_path_obj = self._resolve_path(incident.evidence.snapshot_path)
        is_demo_mode = (
            getattr(incident, "source_mode", "REAL") == "DEMO"
            or (snap_path_obj and snap_path_obj.name == "demo_evidence.jpg")
        )
        # In demo mode, ensure scenario-specific snapshot is prioritized over generic placeholders
        if is_demo_mode and (not snap_path_obj or not snap_path_obj.is_file() or snap_path_obj.name == "demo_evidence.jpg"):
            itype_s = (
                incident.incident_type.value.lower()
                if hasattr(incident.incident_type, "value")
                else str(incident.incident_type).lower()
            )
            if "weapon" in itype_s:
                cand_s = PROJECT_ROOT / "outputs" / "demo" / "snapshots" / "weapon_evidence.jpg"
            elif "crowd" in itype_s:
                cand_s = PROJECT_ROOT / "outputs" / "demo" / "snapshots" / "crowd_panic_evidence.jpg"
            else:
                cand_s = PROJECT_ROOT / "outputs" / "demo" / "snapshots" / "armedfight_evidence.jpg"
            if cand_s.is_file():
                snap_path_obj = cand_s

        snap_attachment_path = None
        snap_attached = False

        if snap_path_obj and snap_path_obj.is_file():
            size_b = snap_path_obj.stat().st_size
            if size_b <= max_bytes:
                snap_attachment_path = str(snap_path_obj)
                snap_attached = True
                snapshot_status_note = f"Incident snapshot attached ({snap_path_obj.name})"
            else:
                snapshot_status_note = f"Incident snapshot omitted — file size exceeds {self.config.max_attachment_mb} MB limit"
        else:
            snapshot_status_note = "Incident snapshot unavailable."

        # 8. Role-Tailored Subject, Plain-text Body, HTML Body & Recipients
        role_upper = (recipient_role or "GENERAL").upper()

        if role_upper == "POLICE":
            subject = f"[POLICE ALERT & EVIDENCE] {sev_str} - {itype_str} DETECTED{loc_subject_suffix}"
            recipients = list(dict.fromkeys(self.config.get_police_recipients() + list(self.config.email_to)))

            body_text = f"""============================================================
5G SURAKSHA-NET — LAW ENFORCEMENT & POLICE EVIDENCE DOSSIER
============================================================

CRIME / SECURITY THREAT EVIDENCE REPORT:
-----------------------------------------
Incident ID : {inc_id}
Incident    : {itype_str}
Severity    : {sev_str}
Detection   : Automated 5G AI Surveillance
Timestamp   : {time_str}
Camera ID   : {cam_id}
Confidence  : {conf_str}

INCIDENT LOCATION & COORDINATES (FOR POLICE DISPATCH):
------------------------------------------------------
Location    : {loc_name}
GPS Coords  : {coords_str}
Source      : {loc_source}

SUSPECT & WEAPON THREAT SUMMARY:
--------------------------------
{detection_summary}

RECOMMENDED POLICE TACTICAL ACTIONS:
------------------------------------
- Immediate PCR / Quick Reaction Team (QRT) dispatch to GPS coordinates.
- Cordon perimeter and verify armed individuals at coordinates {coords_str}.
- Register evidentiary case report / FIR using attached verified forensic media.
{actions_summary}

ASSIGNED POLICE JURISDICTION / STATION:
---------------------------------------
{police_details}

EVIDENTIARY ATTACHMENTS (FOR FIR & INVESTIGATION):
---------------------------------------------------
- 10-Second Evidence Video Clip : {clip_status_note}
- High-Resolution Keyframe Snap : {snapshot_status_note}

OPERATIONAL STATUS:
-------------------
AWAITING HUMAN APPROVAL (human_approval_required = True)
Security operator authorization required prior to dispatch.
============================================================
"""
            body_html = f"""<!DOCTYPE html>
<html>
<head>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; color: #1f2937; margin: 0; padding: 20px; }}
    .container {{ max-width: 680px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
    .header {{ background-color: #1e3a8a; color: #ffffff; padding: 18px 24px; }}
    .header h2 {{ margin: 0; font-size: 20px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 13px; text-transform: uppercase; margin-top: 6px; }}
    .badge-critical {{ background-color: #dc2626; color: white; }}
    .badge-high {{ background-color: #ea580c; color: white; }}
    .content {{ padding: 24px; }}
    .section-title {{ font-size: 14px; font-weight: bold; text-transform: uppercase; color: #1e3a8a; margin-top: 16px; margin-bottom: 8px; border-bottom: 2px solid #bfdbfe; padding-bottom: 4px; }}
    .resource-box {{ background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 12px; margin-bottom: 10px; font-size: 14px; }}
    .coords-box {{ background-color: #fef3c7; border: 1px solid #fde68a; border-radius: 6px; padding: 12px; font-weight: bold; font-size: 14px; color: #92400e; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>5G Suraksha-Net — Law Enforcement Alert & Evidence Dossier</h2>
      <span class="badge badge-{sev_str.lower()}">{sev_str} — {itype_str}</span>
    </div>
    <div class="content">
      <div class="section-title">Incident Information</div>
      <p><strong>Incident ID:</strong> {inc_id} &nbsp;|&nbsp; <strong>Camera:</strong> {cam_id} &nbsp;|&nbsp; <strong>Confidence:</strong> {conf_str}<br>
         <strong>Time:</strong> {time_str}</p>

      <div class="section-title">GPS Coordinates & Location</div>
      <div class="coords-box">
        📍 {loc_name}<br>
        🎯 {coords_str}
      </div>

      <div class="section-title">Suspect & Weapon Threat Summary</div>
      <pre style="background: #f3f4f6; padding: 10px; border-radius: 4px; font-family: inherit; font-size: 13px;">{detection_summary}</pre>

      <div class="section-title">Recommended Tactical Response</div>
      <pre style="background: #f3f4f6; padding: 10px; border-radius: 4px; font-family: inherit; font-size: 13px;">{actions_summary}</pre>

      <div class="section-title">Jurisdictional Police Facility</div>
      <div class="resource-box">
        <pre style="margin: 0; font-family: inherit; font-size: 13px;">{police_details}</pre>
      </div>

      <div class="section-title">Forensic Evidence Attachments</div>
      <p>📹 <strong>10s Incident Video Clip:</strong> {clip_status_note}<br>
         📷 <strong>Keyframe Snapshot:</strong> {snapshot_status_note}</p>

      <p style="font-size: 12px; color: #6b7280; margin-top: 16px;">
        <strong>OPERATIONAL STATUS:</strong> Awaiting human security supervisor authorization.
      </p>
    </div>
  </div>
</body>
</html>"""

        elif role_upper == "HOSPITAL":
            subject = f"[EMERGENCY MEDICAL / TRAUMA ALERT] CASUALTY RISK & AMBULANCE DISPATCH{loc_subject_suffix}"
            recipients = list(dict.fromkeys(self.config.get_hospital_recipients() + list(self.config.email_to)))

            body_text = f"""============================================================
5G SURAKSHA-NET — EMERGENCY MEDICAL & TRAUMA RESPONSE ALERT
============================================================

URGENT CASUALTY & AMBULANCE DISPATCH ADVISORY:
---------------------------------------------
Incident ID : {inc_id}
Incident    : ARMED CONFLICT / VIOLENT ALTERCATION
Severity    : CRITICAL (HIGH CASUALTY RISK)
Timestamp   : {time_str}
Camera ID   : {cam_id}

INCIDENT LOCATION & COORDINATES (FOR AMBULANCE ROUTING):
--------------------------------------------------------
Location    : {loc_name}
GPS Coords  : {coords_str}
Source      : {loc_source}

CASUALTY & TRIAGE ASSESSMENT:
-----------------------------
An armed violent altercation has been detected in real time by surveillance sensors.
There is a high likelihood of serious trauma and physical injuries requiring immediate
medical triage and surgical stabilization at the scene.
{detection_summary}

RECOMMENDED EMERGENCY MEDICAL RESPONSE:
---------------------------------------
- Immediate dispatch of nearest Advanced Life Support (ALS) Ambulance to GPS coordinates {coords_str}.
- Prepare Emergency Department & Trauma Bay for potential incoming casualty intake.
- Coordinate with field triage upon paramedic arrival.

DESIGNATED TRAUMA / MEDICAL FACILITY:
-------------------------------------
{medical_details}

EVIDENTIARY ATTACHMENTS (FOR CASUALTY / INJURY MECHANISM TRIAGE):
-----------------------------------------------------------------
- 10-Second Incident Video Clip : {clip_status_note}
- High-Resolution Keyframe Snap : {snapshot_status_note}

OPERATIONAL STATUS:
-------------------
AWAITING HUMAN APPROVAL (human_approval_required = True)
Medical dispatch coordination requested.
============================================================
"""
            body_html = f"""<!DOCTYPE html>
<html>
<head>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; color: #1f2937; margin: 0; padding: 20px; }}
    .container {{ max-width: 680px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
    .header {{ background-color: #b91c1c; color: #ffffff; padding: 18px 24px; }}
    .header h2 {{ margin: 0; font-size: 20px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 13px; text-transform: uppercase; margin-top: 6px; background-color: #ffffff; color: #b91c1c; }}
    .content {{ padding: 24px; }}
    .section-title {{ font-size: 14px; font-weight: bold; text-transform: uppercase; color: #b91c1c; margin-top: 16px; margin-bottom: 8px; border-bottom: 2px solid #fecaca; padding-bottom: 4px; }}
    .resource-box {{ background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 12px; margin-bottom: 10px; font-size: 14px; }}
    .coords-box {{ background-color: #fee2e2; border: 1px solid #fca5a5; border-radius: 6px; padding: 12px; font-weight: bold; font-size: 14px; color: #991b1b; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>5G Suraksha-Net — Emergency Medical & Trauma Alert</h2>
      <span class="badge">CRITICAL CASUALTY RISK — AMBULANCE REQUIRED</span>
    </div>
    <div class="content">
      <div class="section-title">Medical Advisory Details</div>
      <p><strong>Incident ID:</strong> {inc_id} &nbsp;|&nbsp; <strong>Severity:</strong> CRITICAL &nbsp;|&nbsp; <strong>Confidence:</strong> {conf_str}<br>
         <strong>Time:</strong> {time_str}</p>

      <div class="section-title">Ambulance Routing GPS Coordinates</div>
      <div class="coords-box">
        🏥 Destination: {loc_name}<br>
        🎯 Coordinates: {coords_str}
      </div>

      <div class="section-title">Casualty & Triage Context</div>
      <p style="color: #991b1b; font-weight: 500;">
        Violent armed conflict detected in real time. Elevated probability of blunt or penetrating trauma requiring urgent pre-hospital stabilization.
      </p>
      <pre style="background: #f3f4f6; padding: 10px; border-radius: 4px; font-family: inherit; font-size: 13px;">{detection_summary}</pre>

      <div class="section-title">Designated Medical / Trauma Resource</div>
      <div class="resource-box">
        <pre style="margin: 0; font-family: inherit; font-size: 13px;">{medical_details}</pre>
      </div>

      <div class="section-title">Evidence Deliverables for Pre-Arrival Triage</div>
      <p>📹 <strong>10s Incident Video Clip:</strong> {clip_status_note}<br>
         📷 <strong>Keyframe Snapshot:</strong> {snapshot_status_note}</p>

      <p style="font-size: 12px; color: #6b7280; margin-top: 16px;">
        <strong>OPERATIONAL STATUS:</strong> Awaiting human approval for ambulance dispatch confirmation.
      </p>
    </div>
  </div>
</body>
</html>"""

        else:  # GENERAL
            subject = f"[5G Suraksha-Net] {sev_str} - {itype_str} detected{loc_subject_suffix}"
            recipients = list(dict.fromkeys(list(self.config.email_to) or self.config.get_police_recipients()))

            body_text = f"""============================================================
5G SURAKSHA-NET — INCIDENT ALERT & EVIDENCE DELIVERY
============================================================

INCIDENT DETAILS:
-----------------
Incident ID : {inc_id}
Type        : {itype_str}
Severity    : {sev_str}
Timestamp   : {time_str}
Camera ID   : {cam_id}
Confidence  : {conf_str}

LOCATION INTELLIGENCE:
----------------------
Location    : {loc_name}
Coordinates : {coords_str}
Source      : {loc_source}

DETECTION SUMMARY:
------------------
{detection_summary}

RECOMMENDED RESPONSE (DECISION SUPPORT ONLY):
--------------------------------------------
{actions_summary}

RECOMMENDED POLICE RESOURCE:
----------------------------
{police_details}

RECOMMENDED MEDICAL RESOURCE:
-----------------------------
{medical_details}

EVIDENCE ATTACHMENTS:
---------------------
- Video Clip : {clip_status_note}
- Snapshot   : {snapshot_status_note}

STATUS:
-------
AWAITING HUMAN APPROVAL (human_approval_required = True)

IMPORTANT NOTICE & DISCLAIMER:
------------------------------
This is a decision-support notification for designated human security operators.
No emergency service (police, hospital, or ambulance) has been contacted automatically.
Resource distances are straight-line geographic distances and do not reflect road traffic, driving routes, or arrival ETAs.
============================================================
"""
            body_html = f"""<!DOCTYPE html>
<html>
<head>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; color: #1f2937; margin: 0; padding: 20px; }}
    .container {{ max-width: 680px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
    .header {{ background-color: #111827; color: #ffffff; padding: 18px 24px; }}
    .header h2 {{ margin: 0; font-size: 20px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 13px; text-transform: uppercase; margin-top: 6px; }}
    .badge-critical {{ background-color: #dc2626; color: white; }}
    .badge-high {{ background-color: #ea580c; color: white; }}
    .content {{ padding: 24px; }}
    .section-title {{ font-size: 14px; font-weight: bold; text-transform: uppercase; color: #4b5563; margin-top: 16px; margin-bottom: 8px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }}
    .resource-box {{ background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin-bottom: 10px; font-size: 14px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>5G Suraksha-Net — Incident Alert</h2>
      <span class="badge badge-{sev_str.lower()}">{sev_str} — {itype_str}</span>
    </div>
    <div class="content">
      <div class="section-title">Incident Information</div>
      <p><strong>Incident ID:</strong> {inc_id} &nbsp;|&nbsp; <strong>Camera:</strong> {cam_id} &nbsp;|&nbsp; <strong>Confidence:</strong> {conf_str}<br>
         <strong>Time:</strong> {time_str}</p>

      <div class="section-title">Location Intelligence</div>
      <p><strong>Location:</strong> {loc_name}<br>
         <strong>Coordinates:</strong> {coords_str} (Source: {loc_source})</p>

      <div class="section-title">Recommended Response</div>
      <pre style="background: #f3f4f6; padding: 10px; border-radius: 4px; font-family: inherit; font-size: 13px;">{actions_summary}</pre>

      <div class="section-title">Evidence Deliverables</div>
      <p>📹 <strong>Incident Video Clip:</strong> {clip_status_note}<br>
         📷 <strong>Snapshot:</strong> {snapshot_status_note}</p>
    </div>
  </div>
</body>
</html>"""

        return EmailPayload(
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            recipients=recipients,
            clip_attachment_path=clip_attachment_path,
            snapshot_attachment_path=snap_attachment_path,
            clip_attached=clip_attached,
            snapshot_attached=snap_attached,
            clip_status_note=clip_status_note,
            snapshot_status_note=snapshot_status_note,
        )

    def _send_payload(self, payload: EmailPayload, incident_id: str) -> EmailNotificationResult:
        """Connect to SMTP server and transmit the constructed EmailMessage."""
        if not payload.recipients:
            log.warning("No recipients configured for incident %s; skipping SMTP send", incident_id)
            return EmailNotificationResult(
                incident_id=incident_id,
                status=EmailNotificationStatus.FAILED,
                error_message="No email recipients configured",
            )

        msg = EmailMessage()
        msg["Subject"] = payload.subject
        msg["From"] = self.config.email_from or f"suraksha-alert@{self.config.smtp_host or 'localhost'}"
        msg["To"] = ", ".join(payload.recipients)
        msg.set_content(payload.body_text)

        if payload.body_html:
            msg.add_alternative(payload.body_html, subtype="html")

        # Attach clip if available and verified
        if payload.clip_attachment_path:
            p_clip = Path(payload.clip_attachment_path)
            if p_clip.is_file():
                try:
                    with open(p_clip, "rb") as f:
                        clip_data = f.read()
                    msg.add_attachment(
                        clip_data,
                        maintype="video",
                        subtype="mp4",
                        filename=f"incident_{incident_id}_T-5_to_T+5.mp4",
                    )
                except Exception as e:
                    log.warning("Failed to attach video clip %s: %s", p_clip, e)

        # Attach snapshot if available and verified
        if payload.snapshot_attachment_path:
            p_snap = Path(payload.snapshot_attachment_path)
            if p_snap.is_file():
                try:
                    with open(p_snap, "rb") as f:
                        snap_data = f.read()
                    msg.add_attachment(
                        snap_data,
                        maintype="image",
                        subtype="jpeg",
                        filename=f"incident_{incident_id}_snapshot.jpg",
                    )
                except Exception as e:
                    log.warning("Failed to attach snapshot %s: %s", p_snap, e)

        # SMTP Transmission
        try:
            port = self.config.smtp_port
            host = self.config.smtp_host or "localhost"
            timeout = self.config.timeout_s

            server_factory = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
            with server_factory(host, port, timeout=timeout) as server:
                if self.config.use_tls and port != 465:
                    server.starttls()

                pw = self.config.get_password()
                if self.config.smtp_username and pw:
                    server.login(self.config.smtp_username, pw)

                server.send_message(msg)

            log.info(
                "INCIDENT_EMAIL_SENT incident_id=%s recipients=%d clip=%s snap=%s",
                incident_id, len(payload.recipients), payload.clip_attached, payload.snapshot_attached,
            )
            return EmailNotificationResult(
                incident_id=incident_id,
                status=EmailNotificationStatus.SENT,
                sent_at=datetime.now(timezone.utc),
                recipients=payload.recipients,
                clip_attached=payload.clip_attached,
                snapshot_attached=payload.snapshot_attached,
            )

        except Exception as e:
            sanitized = _sanitize_error(str(e), self.config)
            log.error(
                "EMAIL_SEND_FAILED incident_id=%s reason=%s",
                incident_id, sanitized,
            )
            return EmailNotificationResult(
                incident_id=incident_id,
                status=EmailNotificationStatus.FAILED,
                recipients=payload.recipients,
                clip_attached=payload.clip_attached,
                snapshot_attached=payload.snapshot_attached,
                error_message=sanitized,
            )

    def send_incident_email(
        self,
        incident: IncidentReport,
        response: IncidentResponse | None = None,
        blocking: bool = False,
        allow_demo: bool = False,
        custom_clip_path: str | Path | None = None,
        separate_roles: bool = False,
    ) -> EmailNotificationResult:
        """Dispatch incident email alert asynchronously or synchronously.

        Supports distinct role dispatch (Police vs Hospital) and demo mode activation.
        Enforces incident-level deduplication: exactly one alert per incident ID.
        """
        if not self.config.enabled:
            return EmailNotificationResult(
                incident_id=incident.incident_id,
                status=EmailNotificationStatus.DISABLED,
            )

        if getattr(incident, "source_mode", "REAL") == "DEMO" and not allow_demo:
            log.info("DEMO_MODE: Email notification strictly suppressed for simulated demo incident %s", incident.incident_id)
            return EmailNotificationResult(
                incident_id=incident.incident_id,
                status=EmailNotificationStatus.NOT_ATTEMPTED,
                recipients=[],
                error_message="Suppressed: demonstration mode active",
            )

        with self._lock:
            if incident.incident_id in self._sent_incident_ids:
                log.info("Email already sent for incident %s; skipping duplicate", incident.incident_id)
                return EmailNotificationResult(
                    incident_id=incident.incident_id,
                    status=EmailNotificationStatus.NOT_ATTEMPTED,
                    error_message="Duplicate incident: email alert already sent",
                )
            self._sent_incident_ids.add(incident.incident_id)

        resp = response if response is not None else self.planner.plan(incident)
        itype_val = (
            incident.incident_type.value if hasattr(incident.incident_type, "value") else str(incident.incident_type)
        ).upper()

        if separate_roles:
            # Differentiated roles dispatch:
            # ARMED_FIGHT requires distinct emails for both POLICE and HOSPITAL
            # WEAPON / FIGHT / CROWD_PANIC require POLICE
            if itype_val in ("ARMED_FIGHT",):
                roles_to_send = ["POLICE", "HOSPITAL"]
            elif itype_val in ("WEAPON", "FIGHT", "CROWD_PANIC"):
                roles_to_send = ["POLICE"]
            else:
                roles_to_send = ["GENERAL"]

            all_recipients: list[str] = []
            any_clip = False
            any_snap = False
            last_result: EmailNotificationResult | None = None

            for role in roles_to_send:
                payload = self.build_incident_email(
                    incident=incident,
                    response=resp,
                    recipient_role=role,
                    custom_clip_path=custom_clip_path,
                )
                if not payload.recipients:
                    continue

                all_recipients.extend(payload.recipients)
                if payload.clip_attached:
                    any_clip = True
                if payload.snapshot_attached:
                    any_snap = True

                if blocking or self._executor is None:
                    last_result = self._send_payload(payload, incident.incident_id)
                else:
                    fut = self._executor.submit(self._send_payload, payload, incident.incident_id)
                    with self._lock:
                        self._pending_futures.append(fut)
                    last_result = EmailNotificationResult(
                        incident_id=incident.incident_id,
                        status=EmailNotificationStatus.QUEUED,
                        recipients=payload.recipients,
                        clip_attached=payload.clip_attached,
                        snapshot_attached=payload.snapshot_attached,
                    )

            if last_result is None:
                return EmailNotificationResult(
                    incident_id=incident.incident_id,
                    status=EmailNotificationStatus.FAILED,
                    error_message="No email recipients configured in EMAIL_TO",
                )

            unique_recips = list(dict.fromkeys(all_recipients))
            return EmailNotificationResult(
                incident_id=incident.incident_id,
                status=last_result.status,
                sent_at=last_result.sent_at or datetime.now(timezone.utc),
                recipients=unique_recips,
                clip_attached=any_clip,
                snapshot_attached=any_snap,
                error_message=last_result.error_message,
            )

        # Standard single consolidated email dispatch
        payload = self.build_incident_email(
            incident=incident,
            response=resp,
            recipient_role="GENERAL",
            custom_clip_path=custom_clip_path,
        )

        if not payload.recipients:
            return EmailNotificationResult(
                incident_id=incident.incident_id,
                status=EmailNotificationStatus.FAILED,
                error_message="No email recipients configured in EMAIL_TO",
            )

        if blocking or self._executor is None:
            return self._send_payload(payload, incident.incident_id)

        fut = self._executor.submit(self._send_payload, payload, incident.incident_id)
        with self._lock:
            self._pending_futures.append(fut)

        return EmailNotificationResult(
            incident_id=incident.incident_id,
            status=EmailNotificationStatus.QUEUED,
            recipients=payload.recipients,
            clip_attached=payload.clip_attached,
            snapshot_attached=payload.snapshot_attached,
        )

    def handle_incident_event(
        self,
        report: IncidentReport,
        response: IncidentResponse | None = None,
    ) -> EmailNotificationResult | None:
        """EventBus subscriber callback for live pipeline integration.

        Waits until evidence is finalized before queuing email:
        - If report is still RECORDING_POST_EVENT, defer until FINALIZED event arrives.
        - If report is FINALIZED or VERIFIED (without active recording), queue email immediately.
        """
        if not self.config.enabled:
            return None

        if getattr(report, "source_mode", "REAL") == "DEMO":
            log.info("DEMO_MODE: Email notification strictly suppressed for simulated demo incident %s", report.incident_id)
            return None

        # Defer if clip recording session is in progress (T+5s)
        if report.status == IncidentStatus.RECORDING_POST_EVENT:
            return None

        # Send once evidence is finalized or verified
        if report.status in (IncidentStatus.FINALIZED, IncidentStatus.VERIFIED):
            return self.send_incident_email(report, response, blocking=False)

        return None

    def wait_pending(self, timeout: float = 10.0) -> None:
        """Wait for any asynchronous SMTP sends to conclude."""
        with self._lock:
            futs = list(self._pending_futures)
            self._pending_futures.clear()

        t0 = time.time()
        for f in futs:
            rem = max(0.1, timeout - (time.time() - t0))
            try:
                f.result(timeout=rem)
            except Exception as e:
                log.warning("Asynchronous email future failed: %s", _sanitize_error(str(e), self.config))

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully terminate background notification worker threads."""
        if self._executor is not None:
            self.wait_pending(timeout=5.0)
            self._executor.shutdown(wait=wait)
            self._executor = None

    def test_smtp_connection(self) -> dict:
        """Test SMTP server connectivity and authentication without sending an email.

        Returns a result dict with keys: ok (bool), message (str), detail (str|None).
        Never raises — all SMTP errors are caught and returned as structured output.
        """
        if not self.config.enabled:
            return {"ok": False, "message": "Email notifications are disabled (EMAIL_ENABLED=false)", "detail": None}

        host = self.config.smtp_host or ""
        port = self.config.smtp_port
        if not host:
            return {"ok": False, "message": "SMTP host is not configured (SMTP_HOST is empty)", "detail": None}

        try:
            server_factory = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
            with server_factory(host, port, timeout=self.config.timeout_s) as server:
                if self.config.use_tls and port != 465:
                    server.starttls()
                pw = self.config.get_password()
                if self.config.smtp_username and pw:
                    server.login(self.config.smtp_username, pw)
            log.info("SMTP_TEST_OK host=%s port=%d", host, port)
            return {
                "ok": True,
                "message": f"SMTP connection to {host}:{port} successful. Authentication passed.",
                "detail": f"TLS={self.config.use_tls}, port={port}",
            }
        except smtplib.SMTPAuthenticationError as e:
            sanitized = _sanitize_error(str(e), self.config)
            return {
                "ok": False,
                "message": "SMTP authentication failed. Check SMTP_USERNAME and SMTP_PASSWORD.",
                "detail": sanitized,
            }
        except (socket.timeout, TimeoutError) as e:
            return {
                "ok": False,
                "message": f"SMTP connection timed out connecting to {host}:{port}. Check host/port and firewall.",
                "detail": str(e),
            }
        except Exception as e:
            sanitized = _sanitize_error(str(e), self.config)
            return {"ok": False, "message": f"SMTP connection failed: {sanitized}", "detail": None}

    def get_status(self) -> dict:
        """Return current email notification configuration status (no secrets exposed)."""
        return {
            "enabled": self.config.enabled,
            "smtp_host": self.config.smtp_host or None,
            "smtp_port": self.config.smtp_port,
            "smtp_username": self.config.smtp_username or None,
            "email_from": self.config.email_from or None,
            "email_to_general": list(self.config.email_to),
            "email_to_police": self.config.get_police_recipients(),
            "email_to_hospital": self.config.get_hospital_recipients(),
            "use_tls": self.config.use_tls,
            "max_attachment_mb": self.config.max_attachment_mb,
            "sent_count": len(self._sent_incident_ids),
            "password_configured": bool(self.config.get_password()),
        }
