import time
from datetime import datetime
from typing import List, Optional, Tuple
import cv2
import numpy as np

from detectors.accident_detector import VehicleDetection
from detectors.fire_detector import FireResult
from config import SEVERITY_CONFIG


class CCTVVisualizer:
    """Surveillance Camera HUD, Bounding Box, and Incident Visualizer."""

    def __init__(self, camera_name: str = "SPARSH CCTV 01"):
        self.camera_name = camera_name
        self.show_hud = True
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.fps_smoothing = 0.9

    def update_fps(self):
        """Calculates smoothed real-time rendering FPS."""
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = self.fps * self.fps_smoothing + current_fps * (
                1.0 - self.fps_smoothing
            )
        self.frame_count += 1

    @staticmethod
    def _draw_card(
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        bg_color: Tuple[int, int, int] = (15, 15, 18),
        border_color: Tuple[int, int, int] = (60, 60, 65),
        alpha: float = 0.75,
    ):
        """Draws a semi-transparent HUD card with an optional border."""
        sub = canvas[y : y + h, x : x + w]
        if sub.shape[0] != h or sub.shape[1] != w:
            return
        rect = np.full_like(sub, bg_color, dtype=np.uint8)
        cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)
        canvas[y : y + h, x : x + w] = sub
        if border_color:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)

    @staticmethod
    def _draw_box_with_badge(
        canvas: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int],
        thickness: int = 2,
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """Draws a bounding box with an attached solid text badge."""
        h, w = canvas.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_th = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_th)

        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        badge_x2 = min(w - 1, x1 + tw + 10)

        cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, y1 - 4),
            font,
            font_scale,
            text_color,
            font_th,
            cv2.LINE_AA,
        )

    def draw(
        self,
        frame: np.ndarray,
        detections: List[VehicleDetection],
        fire_result: Optional[FireResult] = None,
        source_label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draws vehicle and fire/smoke detections, HUD telemetry, and alert banners.
        """
        self.update_fps()
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Draw vehicle detections
        severity_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        highest_severity = 0
        has_accident = False

        for det in detections:
            sev = det.severity_level
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if det.is_accident:
                has_accident = True
                if sev > highest_severity:
                    highest_severity = sev

            label_text = f"{det.short_name} {det.confidence:.2f}"
            text_color = (0, 0, 0) if sev in [0, 1] else (255, 255, 255)
            thickness = 3 if det.is_accident else 2
            self._draw_box_with_badge(
                canvas=canvas,
                bbox=det.bbox,
                label=label_text,
                color=det.color,
                thickness=thickness,
                text_color=text_color,
            )

        # 2. Draw fire and smoke detections
        if fire_result and fire_result.detections:
            for f_det in fire_result.detections:
                label_text = f"{f_det.label} {f_det.confidence:.2f}"
                text_color = (
                    (255, 255, 255) if f_det.label.lower() == "fire" else (0, 0, 0)
                )
                self._draw_box_with_badge(
                    canvas=canvas,
                    bbox=f_det.bbox,
                    label=label_text,
                    color=f_det.color,
                    thickness=3,
                    text_color=text_color,
                )

        if not self.show_hud:
            return canvas

        # 3. Surveillance Header Bar
        header_h = 50
        self._draw_card(
            canvas,
            0,
            0,
            w,
            header_h,
            bg_color=(20, 20, 24),
            border_color=None,
            alpha=0.85,
        )

        cam_text = (
            f"{self.camera_name}"
            if not source_label
            else f"{self.camera_name} | {source_label}"
        )
        cv2.putText(
            canvas,
            cam_text,
            (20, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        telem_text = f"FPS: {self.fps:4.1f} | {timestamp_str}"
        (tw, _), _ = cv2.getTextSize(telem_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            canvas,
            telem_text,
            (w - tw - 20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # 4. Traffic & Accident Status Card (Top Left)
        card_w, card_h = 310, 95
        card_x, card_y = 20, header_h + 12
        self._draw_card(canvas, card_x, card_y, card_w, card_h)

        v_text = f"Monitored Vehicles: {len(detections)}"
        cv2.putText(
            canvas,
            v_text,
            (card_x + 15, card_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        if not has_accident:
            status_text = "STATUS: ALL CLEAR (NORMAL)"
            status_color = (0, 230, 80)
        else:
            cfg = SEVERITY_CONFIG[highest_severity]
            status_text = f"INCIDENT: {cfg['display_name'].upper()}"
            status_color = cfg["color"]

        cv2.putText(
            canvas,
            status_text,
            (card_x + 15, card_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color,
            2,
            cv2.LINE_AA,
        )

        breakdown_text = f"Mnr:{severity_counts[1]} Mod:{severity_counts[2]} Sev:{severity_counts[3]} Tot:{severity_counts[4]}"
        cv2.putText(
            canvas,
            breakdown_text,
            (card_x + 15, card_y + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # 5. Fire & Hazard Status Card (Top Right)
        if fire_result is not None:
            f_card_w, f_card_h = 240, 52
            f_card_x = w - f_card_w - 20
            f_card_y = header_h + 12

            if fire_result.has_fire:
                f_bg = (0, 0, 180)
                f_border = (0, 0, 255)
                f_label = f"FIRE DETECTED ({fire_result.fire_confidence * 100:.1f}%)"
            elif fire_result.has_smoke:
                f_bg = (50, 50, 60)
                f_border = (180, 180, 180)
                f_label = f"SMOKE DETECTED ({fire_result.smoke_confidence * 100:.1f}%)"
            else:
                f_bg = (15, 15, 18)
                f_border = (60, 60, 65)
                f_label = "FIRE/SMOKE: ALL CLEAR"

            self._draw_card(
                canvas,
                f_card_x,
                f_card_y,
                f_card_w,
                f_card_h,
                bg_color=f_bg,
                border_color=f_border,
            )
            cv2.putText(
                canvas,
                f_label,
                (f_card_x + 12, f_card_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2 if (fire_result.has_fire or fire_result.has_smoke) else 1,
                cv2.LINE_AA,
            )

        # 6. Flashing Emergency Alerts
        pulse = (self.frame_count // 6) % 2 == 0

        # Fire / Smoke Emergency Banner
        if fire_result is not None and (fire_result.has_fire or fire_result.has_smoke):
            hazard_type = "FIRE" if fire_result.has_fire else "SMOKE"
            conf = (
                fire_result.fire_confidence
                if fire_result.has_fire
                else fire_result.smoke_confidence
            )
            banner_color = (0, 0, 220) if fire_result.has_fire else (80, 80, 90)

            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), banner_color, 5)
            if pulse:
                alert_text = (
                    f"*** {hazard_type} HAZARD DETECTED ({conf * 100:.1f}%) ***"
                )
                (atw, ath), _ = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2
                )
                bx = (w - atw) // 2
                by = header_h + 90
                cv2.rectangle(
                    canvas,
                    (bx - 18, by - ath - 10),
                    (bx + atw + 18, by + 10),
                    banner_color,
                    -1,
                )
                cv2.putText(
                    canvas,
                    alert_text,
                    (bx, by),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # Severe Accident Banner
        # if (
        #     highest_severity >= 2
        #     and pulse
        #     and not (fire_result and fire_result.has_fire)
        # ):
        #     cfg = SEVERITY_CONFIG[highest_severity]
        #     sev_banner_text = f"*** WARNING: {cfg['display_name'].upper()} DETECTED ***"
        #     (stw, sth), _ = cv2.getTextSize(
        #         sev_banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2
        #     )
        #     bx = (w - stw) // 2
        #     by = h - 45
        #     cv2.rectangle(
        #         canvas,
        #         (bx - 15, by - sth - 8),
        #         (bx + stw + 15, by + 8),
        #         cfg["color"],
        #         -1,
        #     )
        #     cv2.putText(
        #         canvas,
        #         sev_banner_text,
        #         (bx, by),
        #         cv2.FONT_HERSHEY_DUPLEX,
        #         0.72,
        #         (255, 255, 255),
        #         2,
        #         cv2.LINE_AA,
        #     )

        # 7. Bottom Controls Quick Help
        help_text = "[SPACE]: Pause | [S]: Snapshot | [H]: Toggle HUD | [Q]: Exit"
        cv2.putText(
            canvas,
            help_text,
            (20, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        return canvas
