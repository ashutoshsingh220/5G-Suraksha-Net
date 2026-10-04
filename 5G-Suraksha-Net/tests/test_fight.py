import numpy as np

from suraksha.config import CandidateConfig, FightConfig
from suraksha.detection.tracker import TrackedPerson
from suraksha.fight.candidate import FightCandidateDetector
from suraksha.fight.recognizer import FightRecognizer
from suraksha.fight.temporal_classifier import HeuristicTemporalScorer


def _track(tid, cx, cy, w=60, h=160):
    return TrackedPerson(
        track_id=tid,
        bbox_xyxy=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        confidence=0.9,
    )


def test_candidate_emitted_on_close_agitated_pair():
    cfg = CandidateConfig(proximity_min_frames=3, motion_energy_threshold=0.0)
    det = FightCandidateDetector(cfg, (720, 1280))

    rng = np.random.default_rng(0)
    candidates = []
    for i in range(10):
        # two persons overlapping, jittering (simulated struggle)
        frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        jitter = (i % 3) * 8
        tracks = [_track(1, 400 + jitter, 400), _track(2, 430 - jitter, 400)]
        candidates = det.update(frame, tracks, now=100.0 + i * 0.1)

    assert candidates, "expected at least one fight candidate"
    assert candidates[0].track_ids == (1, 2)
    assert candidates[0].frames_engaged >= 3


def test_far_apart_tracks_no_candidate():
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(10):
        tracks = [_track(1, 100, 400), _track(2, 1000, 400)]
        out = det.update(frame, tracks, now=i * 0.1)
    assert out == []


def test_heuristic_scorer_range():
    scorer = HeuristicTemporalScorer()
    window = np.random.default_rng(1).random((32, 8)).astype(np.float32)
    s = scorer.score(window)
    assert 0.0 <= s <= 1.0
    assert scorer.score(window[:2]) == 0.0  # too short


def test_recognizer_requires_temporal_persistence():
    """A single frame must never verify a fight — temporal behavior only."""
    cfg = FightConfig()
    cfg.candidate.proximity_min_frames = 2
    cfg.candidate.motion_energy_threshold = 0.0
    rec = FightRecognizer(cfg, (720, 1280))

    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    tracks = [_track(1, 400, 400), _track(2, 440, 400)]
    verified = rec.update(frame, tracks, now=0.0)
    assert verified == [], "single-frame verification must be impossible"


# ---- PROMPT 5: normalized-motion regression tests ----
#
# feature[2] changed from raw optical-flow px/frame (flow/10) to normalized
# motion in BODY-WIDTHS/SEC: motion = flow_px_per_frame * fps / body_width_px.
# These tests pin the invariances that fix relied on. If someone reverts to the
# raw unit, or drops the fps/scale factor, they fail.


def test_normalized_motion_is_resolution_invariant():
    """Same physical motion at 2x resolution -> same normalized feature[2].

    Doubling resolution doubles both the flow displacement (px) and the body
    width (px); the ratio, and therefore feature[2], must not change.
    """
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)

    lo_a, lo_b = _track(1, 400, 400, w=60), _track(2, 440, 400, w=60)
    hi_a, hi_b = _track(1, 800, 800, w=120), _track(2, 880, 800, w=120)

    flow_lo, flow_hi = 3.0, 6.0  # px/frame doubles with resolution
    f_lo = det._pair_features(lo_a, lo_b, 0.1, flow_lo)[2]
    f_hi = det._pair_features(hi_a, hi_b, 0.1, flow_hi)[2]
    assert abs(f_lo - f_hi) < 1e-4, (f_lo, f_hi)


def test_normalized_motion_is_fps_invariant():
    """Same physical motion sampled at 2x fps -> same normalized feature[2].

    Per-frame displacement halves when fps doubles (same speed, shorter
    interval); multiplying by fps must cancel it exactly.
    """
    cfg = CandidateConfig()
    a, b = _track(1, 400, 400, w=60), _track(2, 440, 400, w=60)

    det30 = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    det60 = FightCandidateDetector(cfg, (720, 1280), fps=60.0)
    f30 = det30._pair_features(a, b, 0.1, 4.0)[2]   # 4 px/frame @30fps
    f60 = det60._pair_features(a, b, 0.1, 2.0)[2]   # 2 px/frame @60fps (same speed)
    assert abs(f30 - f60) < 1e-4, (f30, f60)


def test_normalized_motion_units_are_body_widths_per_second():
    """Pin the exact formula: motion = flow * fps / mean_body_width."""
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=25.0)
    a, b = _track(1, 400, 400, w=50), _track(2, 460, 400, w=70)
    scale = (50.0 + 70.0) / 2.0
    expected = 3.0 * 25.0 / scale
    assert abs(det._pair_features(a, b, 0.1, 3.0)[2] - expected) < 1e-4


def test_normalized_motion_is_finite_and_shape_preserved():
    """Zero/degenerate widths must not produce nan/inf; FEATURE_DIM stays 8."""
    cfg = CandidateConfig()
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    # degenerate zero-width boxes -> scale guard (1e-3) must keep it finite
    a = TrackedPerson(track_id=1, bbox_xyxy=np.array([10, 10, 10, 10], np.float32), confidence=0.5)
    b = TrackedPerson(track_id=2, bbox_xyxy=np.array([12, 12, 12, 12], np.float32), confidence=0.5)
    feat = det._pair_features(a, b, 0.0, 5.0)
    assert feat.shape == (FightCandidateDetector.FEATURE_DIM,)
    assert np.isfinite(feat).all(), feat


def test_update_fps_override_changes_normalized_motion():
    """Passing fps to update() must rescale the gate/feature (per-clip fps)."""
    cfg = CandidateConfig(proximity_min_frames=1, motion_energy_threshold=0.0)
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    tracks = [_track(1, 400, 400), _track(2, 430, 400)]
    det.update(frame, tracks, now=0.0)          # prime prev_gray
    det.update(frame, tracks, now=0.1, fps=15.0)
    assert det.fps == 15.0


def test_gate_uses_normalized_motion_threshold():
    """A high normalized-motion threshold suppresses a low-motion pair."""
    cfg = CandidateConfig(proximity_min_frames=3, motion_energy_threshold=1e6)
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    rng = np.random.default_rng(4)
    out = []
    for i in range(8):
        frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        tracks = [_track(1, 400, 400), _track(2, 430, 400)]
        out = det.update(frame, tracks, now=i * 0.1)
    assert out == [], "unreachable motion threshold must emit no candidate"


def test_candidate_emitted_with_proximity_distance_when_iou_zero():
    """Non-overlapping bboxes (IoU=0) within proximity_distance must emit candidate."""
    # w=60, cx1=400, cx2=475 -> gap of 15px (x2_a=430, x1_b=445), IoU = 0.0
    # Center distance = 75px / 60px = 1.25 body-widths <= 1.5
    cfg_with_dist = CandidateConfig(proximity_min_frames=3, motion_energy_threshold=0.0, proximity_distance=1.5)
    det_dist = FightCandidateDetector(cfg_with_dist, (720, 1280), fps=30.0)

    cfg_no_dist = CandidateConfig(proximity_min_frames=3, motion_energy_threshold=0.0, proximity_distance=0.0)
    det_no_dist = FightCandidateDetector(cfg_no_dist, (720, 1280), fps=30.0)

    rng = np.random.default_rng(5)
    out_dist, out_no_dist = [], []
    for i in range(6):
        frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        tracks = [_track(1, 400, 400, w=60), _track(2, 475, 400, w=60)]
        out_dist = det_dist.update(frame, tracks, now=i * 0.1)
        out_no_dist = det_no_dist.update(frame, tracks, now=i * 0.1)

    assert out_dist != [], "proximity_distance=1.5 should emit candidate for 1.25 bw pair"
    assert out_no_dist == [], "proximity_distance=0.0 (strict IoU) must not emit candidate when IoU=0"


def test_candidate_suppressed_when_distance_exceeds_proximity_distance():
    """Non-overlapping bboxes beyond proximity_distance must not emit candidate."""
    # cx1=400, cx2=550 -> dist = 150px / 60px = 2.5 body-widths > 1.5
    cfg = CandidateConfig(proximity_min_frames=3, motion_energy_threshold=0.0, proximity_distance=1.5)
    det = FightCandidateDetector(cfg, (720, 1280), fps=30.0)
    rng = np.random.default_rng(6)
    out = []
    for i in range(6):
        frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        tracks = [_track(1, 400, 400, w=60), _track(2, 550, 400, w=60)]
        out = det.update(frame, tracks, now=i * 0.1)
    assert out == [], "pair beyond proximity_distance must not emit candidate"


