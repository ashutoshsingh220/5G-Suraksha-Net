"""High-visibility CCTV frame annotator for 5G Suraksha-Net.

Renders YOLO person bounding boxes, ByteTrack IDs, active candidate interactions,
verified fight alerts, and a compact diagnostic HUD overlay.
"""
from __future__ import annotations

import cv2
import numpy as np

from suraksha.crowd.analyzer import CrowdSnapshot
from suraksha.detection.tracker import TrackedPerson
from suraksha.detection.weapon import WeaponEvent
from suraksha.fight.candidate import FightCandidate
from suraksha.fight.recognizer import VerifiedFight


class FrameAnnotator:
    """Renders detection bounding boxes, tracking IDs, fight states, and HUD overlay."""

    COLOR_NORMAL = (0, 220, 0)      # Crisp green (BGR)
    COLOR_CANDIDATE = (0, 215, 255)  # Amber / yellow (BGR)
    COLOR_FIGHT = (0, 0, 255)        # Bold red (BGR)
    COLOR_WEAPON_KNIFE = (0, 165, 255)  # Amber / orange (BGR)
    COLOR_WEAPON_GUN = (0, 0, 255)      # Bold red (BGR)
    COLOR_HUD_BG = (20, 20, 20)      # Dark translucent background
    COLOR_HUD_TEXT = (240, 240, 240) # Bright white
    COLOR_HUD_ACCENT = (0, 215, 255) # Amber accent

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        thickness: int = 2,
    ):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def _classify_track_statuses(
        self,
        tracks: list[TrackedPerson],
        candidates: list[FightCandidate],
        verified_fights: list[VerifiedFight],
    ) -> dict[int, str]:
        """Classify each tracked person into NORMAL, CANDIDATE, or FIGHT status."""
        fight_tids: set[int] = set()
        for vf in verified_fights:
            fight_tids.update(vf.track_ids)

        candidate_tids: set[int] = set()
        for cand in candidates:
            candidate_tids.update(cand.track_ids)

        statuses: dict[int, str] = {}
        for t in tracks:
            tid = t.track_id
            if tid in fight_tids:
                statuses[tid] = "FIGHT"
            elif tid in candidate_tids:
                statuses[tid] = "CANDIDATE"
            else:
                statuses[tid] = "NORMAL"
        return statuses

    def _draw_badge(
        self,
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int] = (255, 255, 255),
        scale: float = 0.45,
        thickness: int = 1,
    ) -> None:
        """Draw a text label with a filled background badge."""
        x, y = pos
        (w, h), baseline = cv2.getTextSize(text, self.font, scale, thickness)
        # Background box above coordinates if possible, else below
        h_total = h + baseline + 4
        y_top = max(0, y - h_total)
        y_bot = y_top + h_total
        x_right = min(img.shape[1], x + w + 6)
        cv2.rectangle(img, (x, y_top), (x_right, y_bot), bg_color, -1)
        cv2.putText(
            img,
            text,
            (x + 3, y_bot - baseline - 2),
            self.font,
            scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_hud(
        self,
        img: np.ndarray,
        crowd: CrowdSnapshot | None,
        tracks: list[TrackedPerson],
        candidates: list[FightCandidate],
        metrics: dict,
        effective_fps: float | None,
        max_fight_score: float | None,
        score_threshold: float | None,
        incident_state: str | None,
        frame_idx: int | None,
        weapon_event: WeaponEvent | None = None,
    ) -> None:
        """Draw a compact diagnostic HUD overlay in the top-left corner."""
        h, w = img.shape[:2]

        # Format values or fallback to "N/A"
        fps_val = None
        if effective_fps is not None and effective_fps > 0:
            fps_val = f"{effective_fps:.1f}"
        elif metrics.get("source_fps"):
            fps_val = f"{metrics['source_fps']:.1f}"
        fps_str = fps_val if fps_val else "N/A"

        res_str = metrics.get("resolution") or f"{w}x{h}"
        frame_str = str(frame_idx) if frame_idx is not None else "N/A"

        p_count = crowd.person_count if crowd is not None else len(tracks)
        tracks_count = len(tracks)
        cand_count = len(candidates)

        score_str = f"{max_fight_score:.2f}" if max_fight_score is not None else "N/A"
        thr_str = f"{score_threshold:.2f}" if score_threshold is not None else "N/A"
        state_str = incident_state or "NORMAL"

        det_ms = f"{metrics['detect_track_ms']:.1f}" if metrics.get("detect_track_ms") is not None else "N/A"
        weapon_ms = f"{metrics['weapon_ms']:.1f}" if metrics.get("weapon_ms") is not None else "0.0"
        tot_ms = f"{metrics['total_ms']:.1f}" if metrics.get("total_ms") is not None else "N/A"

        weapon_summary = "NONE"
        if weapon_event is not None:
            if weapon_event.confirmed_count > 0:
                conf_classes = [t.weapon_class.upper() for t in weapon_event.telemetry if t.confirmation_state == "WEAPON_CONFIRMED"]
                weapon_summary = f"CONFIRMED ({', '.join(conf_classes)})"
            elif weapon_event.candidate_count > 0:
                cand_classes = [t.weapon_class.upper() for t in weapon_event.telemetry if t.confirmation_state == "WEAPON_CANDIDATE"]
                weapon_summary = f"CANDIDATE ({', '.join(cand_classes)})"

        lines = [
            f"FPS: {fps_str} | Frame #{frame_str} | Res: {res_str}",
            f"Persons: {p_count} | Active Tracks: {tracks_count} | Candidates: {cand_count}",
            f"Fight: {score_str} (Thr: {thr_str}) | Weapon: {weapon_summary}",
            f"State: {state_str}",
            f"Latency: {det_ms}ms det+trk | {weapon_ms}ms weapon | {tot_ms}ms total",
        ]

        # Calculate HUD box dimensions
        hud_w = min(w - 20, 520)
        line_h = 20
        hud_h = len(lines) * line_h + 16
        hud_x1, hud_y1 = 10, 10
        hud_x2, hud_y2 = hud_x1 + hud_w, hud_y1 + hud_h

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), self.COLOR_HUD_BG, -1)
        cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (70, 70, 70), 1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Render text lines
        y_text = hud_y1 + 18
        for i, line in enumerate(lines):
            color = self.COLOR_HUD_ACCENT if (i in (2, 3) and state_str != "NORMAL") else self.COLOR_HUD_TEXT
            cv2.putText(
                img,
                line,
                (hud_x1 + 10, y_text),
                self.font,
                0.46,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
            y_text += line_h

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[TrackedPerson],
        candidates: list[FightCandidate],
        verified_fights: list[VerifiedFight],
        crowd: CrowdSnapshot | None = None,
        metrics: dict | None = None,
        effective_fps: float | None = None,
        max_fight_score: float | None = None,
        score_threshold: float | None = None,
        incident_state: str | None = None,
        frame_idx: int | None = None,
        weapon_event: WeaponEvent | None = None,
    ) -> np.ndarray:
        """Produce an annotated copy of the input frame. Never mutates input array."""
        out = frame.copy()
        metrics = metrics or {}
        statuses = self._classify_track_statuses(tracks, candidates, verified_fights)
        by_id = {t.track_id: t for t in tracks}

        # 1. Draw candidate pair interactions (connecting line & candidate ROI)
        for cand in candidates:
            tid1, tid2 = cand.track_ids
            if tid1 in by_id and tid2 in by_id:
                c1 = tuple(by_id[tid1].center.astype(int))
                c2 = tuple(by_id[tid2].center.astype(int))
                cv2.line(out, c1, c2, self.COLOR_CANDIDATE, 2, lineType=cv2.LINE_AA)

            # Candidate ROI bounding box
            rx1, ry1, rx2, ry2 = [int(v) for v in cand.roi_xyxy]
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), self.COLOR_CANDIDATE, 1, lineType=cv2.LINE_AA)
            self._draw_badge(
                out,
                f"CANDIDATE ({tid1}, {tid2})",
                (rx1, ry1),
                self.COLOR_CANDIDATE,
                text_color=(0, 0, 0),
                scale=0.42,
            )

        # 2. Draw person tracks & bounding boxes
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
            status = statuses.get(t.track_id, "NORMAL")

            if status == "FIGHT":
                color = self.COLOR_FIGHT
                label = f"ID: {t.track_id} [FIGHT!]"
                thick = 3
            elif status == "CANDIDATE":
                color = self.COLOR_CANDIDATE
                label = f"ID: {t.track_id} [CANDIDATE]"
                thick = 2
            else:
                color = self.COLOR_NORMAL
                label = f"ID: {t.track_id} ({t.confidence:.2f})"
                thick = 2

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick, lineType=cv2.LINE_AA)
            self._draw_badge(
                out,
                label,
                (x1, y1),
                color,
                text_color=(255, 255, 255) if status != "CANDIDATE" else (0, 0, 0),
            )

        # 3. Draw prominent verified fight alert overlays
        for vf in verified_fights:
            vx1, vy1, vx2, vy2 = [int(v) for v in vf.roi_xyxy]
            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), self.COLOR_FIGHT, 3, lineType=cv2.LINE_AA)
            alert_text = f"! FIGHT DETECTED ! IDs: {vf.track_ids} (Score: {vf.confidence:.2f})"
            self._draw_badge(
                out,
                alert_text,
                (vx1, vy1),
                self.COLOR_FIGHT,
                text_color=(255, 255, 255),
                scale=0.55,
                thickness=2,
            )

        # 4. Draw weapon bounding boxes & badges
        if weapon_event is not None and weapon_event.telemetry:
            for wt in weapon_event.telemetry:
                wx1, wy1, wx2, wy2 = [int(v) for v in wt.bbox]
                is_confirmed = (wt.confirmation_state == "WEAPON_CONFIRMED")
                color = self.COLOR_WEAPON_GUN if wt.weapon_class in ("pistol", "long_gun") else self.COLOR_WEAPON_KNIFE
                thick = 3 if is_confirmed else 2
                cv2.rectangle(out, (wx1, wy1), (wx2, wy2), color, thick, lineType=cv2.LINE_AA)
                state_prefix = "CONFIRMED" if is_confirmed else "CANDIDATE"
                badge_text = f"[{state_prefix}] {wt.weapon_class.upper()} ({wt.confidence:.2f})"
                self._draw_badge(
                    out,
                    badge_text,
                    (wx1, wy1),
                    color,
                    text_color=(255, 255, 255),
                    scale=0.48,
                    thickness=1 if not is_confirmed else 2,
                )

        # 5. Draw diagnostic HUD overlay
        self._draw_hud(
            out,
            crowd=crowd,
            tracks=tracks,
            candidates=candidates,
            metrics=metrics,
            effective_fps=effective_fps,
            max_fight_score=max_fight_score,
            score_threshold=score_threshold,
            incident_state=incident_state,
            frame_idx=frame_idx,
            weapon_event=weapon_event,
        )

        return out
