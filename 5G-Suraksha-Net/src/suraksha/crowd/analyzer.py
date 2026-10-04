"""Crowd monitoring: person count, zone density, growth rate, movement anomaly."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from suraksha.config import CrowdConfig, PROJECT_ROOT
from suraksha.detection.tracker import TrackedPerson
from suraksha.logging_utils import get_logger

log = get_logger(__name__)

# Density levels
DENSITY_LOW = "low"
DENSITY_HIGH = "high"
DENSITY_CRITICAL = "critical"


@dataclass
class ZoneRegion:
    name: str
    polygon_norm: np.ndarray  # (N, 2) normalized coords


@dataclass
class ZoneStat:
    name: str
    person_count: int
    density: float          # fraction of zone area covered by person bboxes
    level: str              # low | high | critical


@dataclass
class MovementInfo:
    mean_flow_magnitude: float = 0.0
    direction_entropy: float = 0.0     # 0..1, high = chaotic movement
    fast_track_ratio: float = 0.0      # fraction of tracks moving above panic speed
    panic: bool = False


@dataclass
class CrowdSnapshot:
    timestamp: float
    person_count: int
    zones: list[ZoneStat] = field(default_factory=list)
    growth_per_min: float = 0.0
    growth_alert: bool = False
    movement: MovementInfo = field(default_factory=MovementInfo)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "person_count": self.person_count,
            "zones": [
                {"name": z.name, "person_count": z.person_count,
                 "density": round(z.density, 4), "level": z.level}
                for z in self.zones
            ],
            "growth_per_min": round(self.growth_per_min, 2),
            "growth_alert": self.growth_alert,
            "movement": {
                "mean_flow_magnitude": round(self.movement.mean_flow_magnitude, 3),
                "direction_entropy": round(self.movement.direction_entropy, 3),
                "fast_track_ratio": round(self.movement.fast_track_ratio, 3),
                "panic": self.movement.panic,
            },
        }


def load_zones(zones_file: str, camera_id: str) -> list[ZoneRegion]:
    path = Path(zones_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        log.warning("Zones file not found: %s — density zones disabled", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    zones: list[ZoneRegion] = []
    for entry in data.get("zones", []):
        if entry.get("camera_id") != camera_id:
            continue
        for region in entry.get("regions", []):
            zones.append(ZoneRegion(
                name=region["name"],
                polygon_norm=np.array(region["polygon"], dtype=np.float32),
            ))
    log.info("Loaded %d zone(s) for camera '%s'", len(zones), camera_id)
    return zones


class _TrackHistory:
    """Per-track kinematics: speed in box-widths/sec."""

    def __init__(self, maxlen: int = 30):
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, tracks: list[TrackedPerson], now: float) -> None:
        seen = set()
        for t in tracks:
            seen.add(t.track_id)
            self.hist[t.track_id].append((now, t.center.copy(), max(t.width, 1e-3)))
        for tid in list(self.hist):
            if tid not in seen:
                del self.hist[tid]

    def speed(self, track_id: int) -> float:
        """Mean speed over recent history, normalized by box width (scale-invariant)."""
        h = self.hist.get(track_id)
        if not h or len(h) < 3:
            return 0.0
        pts = np.array([p[1] for p in h])
        times = np.array([p[0] for p in h])
        widths = np.array([p[2] for p in h])
        dt = np.diff(times)
        valid = dt > 1e-4
        if not valid.any():
            return 0.0
        dist = np.linalg.norm(np.diff(pts, axis=0)[valid], axis=1)
        return float(np.mean(dist / dt[valid] / widths[1:][valid]))


class CrowdAnalyzer:
    """Produces a CrowdSnapshot per processed frame."""

    def __init__(self, cfg: CrowdConfig, camera_id: str, frame_shape: tuple[int, int] | None = None):
        self.cfg = cfg
        self.zones = load_zones(cfg.zones_file, camera_id)
        self.history = _TrackHistory()
        self._count_window: deque[tuple[float, int]] = deque()
        self._prev_gray: np.ndarray | None = None
        self._frame_shape = frame_shape  # (h, w); zone masks built lazily
        self._zone_masks: dict[str, np.ndarray] = {}
        self._mask_scale = 0.25  # masks at quarter resolution for speed

    # ---------- zone masks ----------
    def _ensure_masks(self, shape: tuple[int, int]) -> None:
        h, w = shape[:2]
        self._frame_shape = (h, w)
        mh, mw = int(h * self._mask_scale), int(w * self._mask_scale)
        for z in self.zones:
            mask = np.zeros((mh, mw), dtype=np.uint8)
            poly = (z.polygon_norm * np.array([mw, mh], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
            self._zone_masks[z.name] = mask

    # ---------- optical flow ----------
    def _flow_stats(self, frame: np.ndarray) -> tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0, 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        self._prev_gray = gray
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(mag.mean())
        # Direction entropy over 8 bins, weighted by magnitude
        bins = ((ang / (2 * np.pi) * 8).astype(np.uint8) % 8)
        hist = np.bincount(bins.ravel(), weights=mag.ravel().astype(np.float64), minlength=8)
        p = hist / max(hist.sum(), 1e-9)
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum() / 3.0)  # normalized 0..1 (log2(8)=3)
        return mean_mag, entropy

    # ---------- growth ----------
    def _growth(self, now: float, count: int) -> tuple[float, bool]:
        self._count_window.append((now, count))
        window = self.cfg.growth_window_s
        while self._count_window and now - self._count_window[0][0] > window:
            self._count_window.popleft()
        if len(self._count_window) < 5:
            return 0.0, False
        t = np.array([p[0] for p in self._count_window])
        c = np.array([p[1] for p in self._count_window], dtype=np.float64)
        span = t[-1] - t[0]
        if span < 2.0:
            return 0.0, False
        slope_per_min = float(np.polyfit(t - t[0], c, 1)[0] * 60.0)
        return slope_per_min, slope_per_min >= self.cfg.growth_alert_per_min

    # ---------- main ----------
    def update(self, frame: np.ndarray, tracks: list[TrackedPerson], now: float | None = None) -> CrowdSnapshot:
        now = now if now is not None else time.time()
        h, w = frame.shape[:2]
        if self.zones and (not self._zone_masks or getattr(self, "_frame_shape", None) != (h, w)):
            self._ensure_masks((h, w))

        self.history.update(tracks, now)

        # Zone density
        zone_stats: list[ZoneStat] = []
        if self._zone_masks:
            mh, mw = int(h * self._mask_scale), int(w * self._mask_scale)
            person_mask = np.zeros((mh, mw), dtype=np.uint8)
            for t in tracks:
                x1, y1, x2, y2 = (t.bbox_xyxy * self._mask_scale).astype(np.int32)
                cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)
            for z in self.zones:
                zmask = self._zone_masks.get(z.name)
                if zmask is None or zmask.shape != person_mask.shape:
                    continue
                z_area = max(int((zmask > 0).sum()), 1)
                overlap = int(((zmask > 0) & (person_mask > 0)).sum())
                density = overlap / z_area
                in_zone = 0
                for t in tracks:
                    px, py = t.center * self._mask_scale
                    if 0 <= int(px) < mw and 0 <= int(py) < mh and zmask[int(py), int(px)] > 0:
                        in_zone += 1
                if density >= self.cfg.density_critical_threshold:
                    level = DENSITY_CRITICAL
                elif density >= self.cfg.density_high_threshold:
                    level = DENSITY_HIGH
                else:
                    level = DENSITY_LOW
                zone_stats.append(ZoneStat(z.name, in_zone, density, level))

        # Growth
        growth, growth_alert = self._growth(now, len(tracks))

        # Movement / panic
        mean_mag, entropy = self._flow_stats(frame)
        speeds = np.array([self.history.speed(t.track_id) for t in tracks]) if tracks else np.array([])
        fast_ratio = float((speeds > self.cfg.movement.panic_speed_threshold).mean()) if speeds.size else 0.0
        panic = bool(
            speeds.size >= self.cfg.movement.panic_min_tracks
            and fast_ratio >= 0.5
            and entropy >= 0.6
        )
        movement = MovementInfo(mean_mag, entropy, fast_ratio, panic)

        return CrowdSnapshot(
            timestamp=now,
            person_count=len(tracks),
            zones=zone_stats,
            growth_per_min=growth,
            growth_alert=growth_alert,
            movement=movement,
        )
