import csv
import os
import time
from datetime import datetime
from typing import List, Optional
import cv2
import numpy as np

from config import SNAPSHOT_DIR, OUTPUT_DIR, ALERT_SNAPSHOT_COOLDOWN_SEC
from detectors.accident_detector import VehicleDetection
from detectors.fire_detector import FireResult
from services.alert_dispatcher import EmergencyAlertDispatcher


class IncidentLogger:
    """Manages snapshot logging and emergency dispatch recording for surveillance incidents."""

    def __init__(
        self,
        log_file: Optional[str] = None,
        dispatcher: Optional[EmergencyAlertDispatcher] = None,
        enable_dispatch: bool = True,
    ):
        self.log_file = log_file or os.path.join(OUTPUT_DIR, "incident_log.csv")
        self.last_accident_alert_time = 0.0
        self.last_fire_alert_time = 0.0
        self.cooldown = ALERT_SNAPSHOT_COOLDOWN_SEC

        # Emergency dispatch integration
        self.enable_dispatch = enable_dispatch
        self.dispatcher = dispatcher
        if self.enable_dispatch and self.dispatcher is None:
            self.dispatcher = EmergencyAlertDispatcher()

        self._init_csv()

    def _init_csv(self):
        """Initializes the CSV incident log file if not already present."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Timestamp",
                        "Event_Type",
                        "Max_Accident_Severity",
                        "Fire_Probability",
                        "Vehicles_Count",
                        "Snapshot_Path",
                        "Dispatch_ID",
                    ]
                )

    def save_snapshot(self, frame: np.ndarray, tag: str = "manual") -> str:
        """Saves a timestamped JPEG image snapshot to disk."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{tag}_{ts_str}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def check_and_log(
        self,
        frame: np.ndarray,
        highest_severity: int,
        highest_det: Optional[VehicleDetection],
        fire_result: Optional[FireResult],
        detections: List[VehicleDetection],
    ) -> Optional[str]:
        """
        Evaluates current frame state. If an incident or fire is detected and
        cooldown period has elapsed, saves snapshot and logs event.
        """
        now = time.time()
        triggered_events = []

        # Check Fire & Smoke Alert
        if fire_result:
            if fire_result.has_fire:
                if now - self.last_fire_alert_time > self.cooldown:
                    self.last_fire_alert_time = now
                    triggered_events.append("FIRE_ALERT")
            elif fire_result.has_smoke:
                if now - self.last_fire_alert_time > self.cooldown:
                    self.last_fire_alert_time = now
                    triggered_events.append("SMOKE_ALERT")

        # Check Severe Accident (Severity >= 2: Moderate, Severe, Totaled)
        if highest_severity >= 2:
            if now - self.last_accident_alert_time > self.cooldown:
                self.last_accident_alert_time = now
                triggered_events.append(f"ACCIDENT_LVL_{highest_severity}")

        if not triggered_events:
            return None

        event_str = "_".join(triggered_events)
        snapshot_path = self.save_snapshot(frame, tag=event_str.lower())

        # Determine confidence and severity label
        confidence = 1.0
        severity_label = event_str
        if highest_det:
            confidence = getattr(
                highest_det, "confidence", getattr(highest_det, "conf", 1.0)
            )
            severity_label = highest_det.raw_name
        elif fire_result:
            confidence = fire_result.fire_probability
            severity_label = (
                "Fire Detection" if fire_result.has_fire else "Smoke Detection"
            )

        dispatch_id = "N/A"
        # Trigger emergency dispatch if enabled
        if self.enable_dispatch and self.dispatcher:
            try:
                dispatch_res = self.dispatcher.dispatch(
                    incident_type=event_str,
                    severity_label=severity_label,
                    confidence=confidence,
                    snapshot_path=snapshot_path,
                    vehicles_count=len(detections),
                )
                dispatch_id = dispatch_res.get("dispatch_id", "DISPATCHED")
            except Exception as e:
                print(f"[IncidentLogger] Error dispatching emergency alert: {e}")

        # Write to CSV log
        try:
            with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        event_str,
                        highest_det.raw_name if highest_det else "None",
                        f"{fire_result.fire_probability:.3f}" if fire_result else "0.0",
                        len(detections),
                        snapshot_path,
                        dispatch_id,
                    ]
                )
            print(
                f"[ALERT LOGGED] Event: {event_str} -> Snapshot: {snapshot_path} [Dispatch: {dispatch_id}]"
            )
        except Exception as e:
            print(f"[IncidentLogger] Error saving log: {e}")

        return snapshot_path

    def set_cooldown(self, cooldown: float):
        """Dynamically updates alert cooldown seconds."""
        self.cooldown = max(0.1, float(cooldown))

    def set_enable_dispatch(self, enable: bool):
        """Dynamically toggles emergency dispatch."""
        self.enable_dispatch = bool(enable)

    def get_recent_incidents(
        self, limit: int = 50, event_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Reads recent incidents from the CSV log file."""
        if not os.path.exists(self.log_file):
            return []
        records = []
        try:
            with open(self.log_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (
                        event_filter
                        and event_filter.upper()
                        not in row.get("Event_Type", "").upper()
                    ):
                        continue
                    records.append(row)
            return records[-limit:][::-1]  # Return newest first
        except Exception as e:
            print(f"[IncidentLogger] Error reading log file: {e}")
            return []
