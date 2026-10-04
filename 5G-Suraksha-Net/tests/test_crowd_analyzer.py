import numpy as np

from suraksha.crowd.analyzer import CrowdAnalyzer, DENSITY_LOW
from suraksha.config import CrowdConfig
from suraksha.detection.tracker import TrackedPerson


def _track(tid, x1, y1, x2, y2):
    return TrackedPerson(
        track_id=tid,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=0.9,
    )


def test_person_count_and_zones():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cfg = CrowdConfig()  # zones file exists with cam_default regions
    analyzer = CrowdAnalyzer(cfg, "cam_default", (720, 1280))

    tracks = [_track(1, 100, 400, 160, 600), _track(2, 300, 420, 360, 620)]
    snap = analyzer.update(frame, tracks, now=1000.0)

    assert snap.person_count == 2
    assert len(snap.zones) == 2
    market = next(z for z in snap.zones if z.name == "market_square")
    assert market.person_count == 2
    assert market.density >= 0.0
    assert market.level in (DENSITY_LOW, "high", "critical")


def test_growth_alert():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cfg = CrowdConfig(growth_window_s=10, growth_alert_per_min=5.0)
    analyzer = CrowdAnalyzer(cfg, "cam_none", (720, 1280))

    t = 100.0
    # simulate crowd growing 1 person/sec for 8 seconds
    for n in range(1, 9):
        tracks = [_track(i, 50 * i, 300, 50 * i + 40, 500) for i in range(n)]
        snap = analyzer.update(frame, tracks, now=t)
        t += 1.0

    assert snap.growth_per_min > 30.0  # ~60 persons/min
    assert snap.growth_alert is True


def test_no_tracks_is_quiet():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    analyzer = CrowdAnalyzer(CrowdConfig(), "cam_none", (360, 640))
    snap = analyzer.update(frame, [], now=1.0)
    assert snap.person_count == 0
    assert snap.movement.panic is False
