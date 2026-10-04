"""Multimodal incident fusion layer for 5G Suraksha-Net.

Deterministically fuses WeaponEvent, VerifiedFight, and TrackedPerson telemetry into:
1. ARMED_FIGHT (CRITICAL) - verified physical fight with spatially/temporally correlated confirmed weapon.
2. WEAPON (HIGH) - confirmed weapon alone without correlated physical fight.
3. FIGHT (MODERATE) - verified physical fight alone without correlated weapon.

Strictly protects bystanders holding weapons from false armed-fight association.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from suraksha.config import FusionConfig
from suraksha.detection.tracker import TrackedPerson
from suraksha.detection.weapon import WeaponEvent, WeaponState, WeaponTelemetry, compute_iou
from suraksha.fight.recognizer import VerifiedFight
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class CorrelatedArmedFight:
    """A verified physical fight correlated with a confirmed weapon."""
    fight: VerifiedFight
    weapon_telemetry: WeaponTelemetry
    carrier_track_id: int | None
    correlation_confidence: float
    roi_xyxy: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "weapon_class": self.weapon_telemetry.weapon_class,
            "weapon_track_id": self.weapon_telemetry.track_id,
            "weapon_confidence": round(self.weapon_telemetry.confidence, 4),
            "carrier_track_id": self.carrier_track_id,
            "fight_confidence": round(self.fight.confidence, 4),
            "fight_track_ids": list(self.fight.track_ids),
            "correlation_confidence": round(self.correlation_confidence, 4),
            "roi_xyxy": [round(float(v), 2) for v in self.roi_xyxy],
        }


@dataclass
class FusionDecision:
    """Deterministic output of multimodal incident fusion for one frame."""
    timestamp: float
    armed_fights: list[CorrelatedArmedFight] = field(default_factory=list)
    weapons_alone: list[WeaponTelemetry] = field(default_factory=list)
    unarmed_fights: list[VerifiedFight] = field(default_factory=list)

    @property
    def has_armed_fight(self) -> bool:
        return len(self.armed_fights) > 0

    @property
    def has_weapon(self) -> bool:
        return len(self.weapons_alone) > 0 or len(self.armed_fights) > 0


class IncidentFusionEngine:
    """Associates confirmed weapons with tracked persons and verified fights."""

    def __init__(self, cfg: FusionConfig | None = None) -> None:
        self.cfg = cfg or FusionConfig()

    def associate_weapon_carrier(
        self,
        weapon_bbox: np.ndarray | list[float],
        tracks: list[TrackedPerson],
    ) -> int | None:
        """Associate a weapon bounding box with the most likely carrier TrackedPerson."""
        if not tracks:
            return None

        w_box = np.asarray(weapon_bbox, dtype=np.float32)
        w_cx = (w_box[0] + w_box[2]) / 2.0
        w_cy = (w_box[1] + w_box[3]) / 2.0

        best_tid: int | None = None
        best_score = -1.0

        for tp in tracks:
            px1, py1, px2, py2 = tp.bbox_xyxy
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)

            # 1. Direct containment: centroid within person bounding box
            is_inside = (px1 <= w_cx <= px2) and (py1 <= w_cy <= py2)

            # 2. IoU overlap
            iou = compute_iou(w_box, tp.bbox_xyxy)

            # 3. Center distance normalized by body width
            p_cx = (px1 + px2) / 2.0
            p_cy = (py1 + py2) / 2.0
            dist = np.hypot(w_cx - p_cx, w_cy - p_cy)
            norm_dist = dist / pw

            score = 0.0
            if is_inside:
                # Bonus if in upper 85% of body (hand / torso / shoulder envelope)
                upper_envelope = py1 + 0.85 * ph
                score = 2.0 if w_cy <= upper_envelope else 1.5
            elif iou >= 0.10:
                score = 1.0 + iou
            elif norm_dist <= 1.25:
                score = max(0.0, 1.0 - (norm_dist / 1.25))

            if score > best_score and score >= 0.5:
                best_score = score
                best_tid = tp.track_id

        return best_tid

    def is_weapon_correlated_with_fight(
        self,
        weapon: WeaponTelemetry,
        carrier_id: int | None,
        fight: VerifiedFight,
        tracks: list[TrackedPerson],
    ) -> bool:
        """Determine whether a confirmed weapon is correlated with a verified fight.
        
        Strict rules:
        - Must be within temporal correlation window.
        - Must either:
          a) Have carrier track ID belonging to fight.track_ids, OR
          b) Have weapon centroid or bbox inside/proximate to fight ROI.
        - Bystanders outside fight ROI with different track IDs are strictly NOT correlated.
        """
        # 1. Temporal correlation window check
        time_diff = abs(weapon.frame_timestamp - fight.end_time)
        if time_diff > self.cfg.correlation_window_s:
            return False

        # 2. Direct carrier match: carrier is one of the active combatants
        if carrier_id is not None and carrier_id in fight.track_ids:
            return True

        # 3. Spatial overlap / proximity with fight ROI
        w_box = np.asarray(weapon.bbox, dtype=np.float32)
        w_cx = (w_box[0] + w_box[2]) / 2.0
        w_cy = (w_box[1] + w_box[3]) / 2.0

        fx1, fy1, fx2, fy2 = fight.roi_xyxy

        # Calculate average body width of fight participants for adaptive tolerance
        combatant_widths = [
            (t.bbox_xyxy[2] - t.bbox_xyxy[0])
            for t in tracks
            if t.track_id in fight.track_ids
        ]
        avg_w = float(np.mean(combatant_widths)) if combatant_widths else 50.0
        margin = self.cfg.spatial_proximity_threshold * avg_w

        # Expanded fight envelope
        efx1, efy1 = fx1 - margin, fy1 - margin
        efx2, efy2 = fx2 + margin, fy2 + margin

        # Check if weapon centroid is inside expanded fight envelope
        if efx1 <= w_cx <= efx2 and efy1 <= w_cy <= efy2:
            return True

        # Check direct IoU with fight ROI
        if compute_iou(w_box, fight.roi_xyxy) >= 0.05:
            return True

        return False

    def evaluate(
        self,
        weapon_event: WeaponEvent | None,
        verified_fights: list[VerifiedFight],
        tracks: list[TrackedPerson],
        timestamp: float,
    ) -> FusionDecision:
        """Execute multimodal fusion for the current frame."""
        decision = FusionDecision(timestamp=timestamp)

        if not self.cfg.enabled:
            # Pass-through if fusion layer is disabled
            decision.unarmed_fights = list(verified_fights)
            if weapon_event:
                decision.weapons_alone = [
                    t for t in weapon_event.telemetry
                    if t.confirmation_state == WeaponState.WEAPON_CONFIRMED.value
                ]
            return decision

        # 1. Filter confirmed weapons
        confirmed_weapons: list[WeaponTelemetry] = []
        if weapon_event is not None and weapon_event.telemetry:
            confirmed_weapons = [
                t for t in weapon_event.telemetry
                if t.confirmation_state == WeaponState.WEAPON_CONFIRMED.value
            ]

        # 2. If no confirmed weapons, all verified fights are unarmed fights
        if not confirmed_weapons:
            decision.unarmed_fights = list(verified_fights)
            return decision

        # 3. Associate carrier track IDs for each confirmed weapon
        weapon_carriers: dict[int, int | None] = {}
        for wt in confirmed_weapons:
            weapon_carriers[wt.track_id] = self.associate_weapon_carrier(wt.bbox, tracks)

        # 4. Correlate weapons with verified fights
        matched_fight_indices: set[int] = set()
        matched_weapon_tids: set[int] = set()

        for f_idx, vf in enumerate(verified_fights):
            for wt in confirmed_weapons:
                if wt.track_id in matched_weapon_tids:
                    continue

                carrier_id = weapon_carriers.get(wt.track_id)
                if self.is_weapon_correlated_with_fight(wt, carrier_id, vf, tracks):
                    matched_fight_indices.add(f_idx)
                    matched_weapon_tids.add(wt.track_id)

                    # Compute fused bounding box encompassing fight ROI and weapon bbox
                    w_box = np.asarray(wt.bbox, dtype=np.float32)
                    rx1 = min(float(vf.roi_xyxy[0]), float(w_box[0]))
                    ry1 = min(float(vf.roi_xyxy[1]), float(w_box[1]))
                    rx2 = max(float(vf.roi_xyxy[2]), float(w_box[2]))
                    ry2 = max(float(vf.roi_xyxy[3]), float(w_box[3]))
                    fused_roi = np.array([rx1, ry1, rx2, ry2], dtype=np.float32)

                    fused_conf = max(float(vf.confidence), float(wt.confidence))

                    decision.armed_fights.append(
                        CorrelatedArmedFight(
                            fight=vf,
                            weapon_telemetry=wt,
                            carrier_track_id=carrier_id,
                            correlation_confidence=fused_conf,
                            roi_xyxy=fused_roi,
                        )
                    )
                    break

        # 5. Populate remaining unmatched fights as unarmed fights
        for f_idx, vf in enumerate(verified_fights):
            if f_idx not in matched_fight_indices:
                decision.unarmed_fights.append(vf)

        # 6. Populate remaining unmatched confirmed weapons as weapons alone
        for wt in confirmed_weapons:
            if wt.track_id not in matched_weapon_tids:
                decision.weapons_alone.append(wt)

        return decision
