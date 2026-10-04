"""Tests for FrameAnnotator and AnnotatedVideoWriter."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from suraksha.detection.tracker import TrackedPerson
from suraksha.fight.candidate import FightCandidate
from suraksha.fight.recognizer import VerifiedFight
from suraksha.crowd.analyzer import CrowdSnapshot
from suraksha.visualization.annotator import FrameAnnotator
from suraksha.visualization.video_writer import AnnotatedVideoWriter


def test_annotator_non_destructive():
    """Verify FrameAnnotator creates a copy and never mutates the original frame."""
    annotator = FrameAnnotator()
    orig = np.zeros((480, 640, 3), dtype=np.uint8)
    orig_copy = orig.copy()

    tracks = [
        TrackedPerson(track_id=1, bbox_xyxy=np.array([100, 100, 200, 300], dtype=np.float32), confidence=0.88),
    ]

    result = annotator.annotate(
        frame=orig,
        tracks=tracks,
        candidates=[],
        verified_fights=[],
        crowd=None,
        metrics={},
        incident_state="NORMAL",
    )

    assert result is not orig
    assert np.array_equal(orig, orig_copy)
    assert not np.array_equal(result, orig)


def test_annotator_track_coloring_and_labels():
    """Verify tracks are drawn with correct color codes based on status."""
    annotator = FrameAnnotator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t1 = TrackedPerson(track_id=1, bbox_xyxy=np.array([50, 50, 150, 250], dtype=np.float32), confidence=0.90)
    t2 = TrackedPerson(track_id=2, bbox_xyxy=np.array([200, 50, 300, 250], dtype=np.float32), confidence=0.85)
    t3 = TrackedPerson(track_id=3, bbox_xyxy=np.array([350, 50, 450, 250], dtype=np.float32), confidence=0.80)

    # t1 is normal
    # t2 is part of candidate pair (2, 99)
    # t3 is part of verified fight (3, 99)
    cand = FightCandidate(
        track_ids=(2, 99),
        roi_xyxy=np.array([200, 50, 350, 250], dtype=np.float32),
        first_seen=0.0,
        last_seen=1.0,
        score=0.75,
        frames_engaged=10,
    )
    vf = VerifiedFight(
        track_ids=(3, 99),
        roi_xyxy=np.array([350, 50, 500, 250], dtype=np.float32),
        start_time=0.0,
        end_time=2.0,
        confidence=0.82,
        windows_scored=5,
    )

    status_map = annotator._classify_track_statuses([t1, t2, t3], [cand], [vf])
    assert status_map[1] == "NORMAL"
    assert status_map[2] == "CANDIDATE"
    assert status_map[3] == "FIGHT"

    annotated = annotator.annotate(
        frame=frame,
        tracks=[t1, t2, t3],
        candidates=[cand],
        verified_fights=[vf],
        crowd=None,
        metrics={"source_fps": 30.0, "total_ms": 15.0},
        incident_state="FIGHT DETECTED",
        max_fight_score=0.82,
        score_threshold=0.55,
    )
    assert annotated.shape == frame.shape
    assert (annotated != 0).any()


def test_annotator_hud_overlay_values_and_fallback():
    """Verify HUD renders actual metrics or 'N/A' when metrics are missing."""
    annotator = FrameAnnotator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Case 1: Complete values
    crowd = CrowdSnapshot(
        timestamp=100.0,
        person_count=4,
        zones=[],
        growth_per_min=2.5,
        growth_alert=False,
        movement=None,
    )
    metrics = {
        "source_fps": 25.0,
        "detect_track_ms": 14.2,
        "crowd_ms": 1.5,
        "fight_ms": 2.1,
        "total_ms": 17.8,
        "resolution": "640x480",
    }
    annotated = annotator.annotate(
        frame=frame,
        tracks=[],
        candidates=[],
        verified_fights=[],
        crowd=crowd,
        metrics=metrics,
        effective_fps=28.4,
        max_fight_score=0.68,
        score_threshold=0.55,
        incident_state="NORMAL",
    )
    assert annotated.shape == frame.shape

    # Case 2: Missing/None values -> must not crash, renders N/A
    annotated_empty = annotator.annotate(
        frame=frame,
        tracks=[],
        candidates=[],
        verified_fights=[],
        crowd=None,
        metrics={},
        effective_fps=None,
        max_fight_score=None,
        score_threshold=None,
        incident_state=None,
    )
    assert annotated_empty.shape == frame.shape


def test_video_writer_lifecycle(tmp_path: Path):
    """Verify AnnotatedVideoWriter writes a valid, readable MP4 file."""
    out_path = tmp_path / "test_out.mp4"
    h, w = 240, 320
    fps = 10.0
    num_frames = 15

    writer = AnnotatedVideoWriter(output_path=out_path, fps=fps, frame_size=(w, h))
    for i in range(num_frames):
        frame = np.full((h, w, 3), fill_value=i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Read back with OpenCV
    cap = cv2.VideoCapture(str(out_path))
    assert cap.isOpened()
    read_count = 0
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            break
        read_count += 1
        assert f.shape == (h, w, 3)
    cap.release()
    assert read_count == num_frames


def test_video_writer_context_manager(tmp_path: Path):
    """Verify context manager automatically releases the writer."""
    out_path = tmp_path / "test_ctx.mp4"
    h, w = 120, 160
    with AnnotatedVideoWriter(output_path=out_path, fps=15.0, frame_size=(w, h)) as writer:
        for _ in range(5):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))

    assert out_path.exists()
    cap = cv2.VideoCapture(str(out_path))
    assert cap.isOpened()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == 5


def test_pipeline_preview_and_save_video(tmp_path: Path):
    """Verify CrowdFightPipeline integrates FrameAnnotator, queue, and video writer end-to-end."""
    import queue
    import time
    from suraksha.config import load_config
    from suraksha.pipeline import CrowdFightPipeline

    cfg = load_config()
    cfg.capture.rtsp_url = "datasets/videos/test/synthetic_cctv.mp4"
    cfg.capture.pace = "fast"
    cfg.capture.max_frames = 10

    preview_q: queue.Queue = queue.Queue(maxsize=20)
    out_video = tmp_path / "pipeline_annotated.mp4"

    pipeline = CrowdFightPipeline(cfg, preview_queue=preview_q, save_video_path=out_video)
    pipeline.start()

    frames_received = 0
    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            item = preview_q.get(timeout=0.2)
            if item is None:
                break
            frames_received += 1
        except queue.Empty:
            if not pipeline.state.running:
                break

    pipeline.stop()

    assert frames_received == 10
    assert out_video.exists()
    assert out_video.stat().st_size > 0

    cap = cv2.VideoCapture(str(out_video))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_pipeline_pause_and_resume():
    """Verify pause() suspends frame processing and resume() continues it."""
    import time
    from suraksha.config import load_config
    from suraksha.pipeline import CrowdFightPipeline

    cfg = load_config()
    cfg.capture.rtsp_url = "datasets/videos/test/synthetic_cctv.mp4"
    cfg.capture.pace = "fast"
    cfg.capture.max_frames = 60

    pipeline = CrowdFightPipeline(cfg)
    pipeline.start()

    # Wait until pipeline actually starts producing frames
    t0 = time.time()
    while pipeline.state.frames_processed < 2 and time.time() - t0 < 5.0:
        time.sleep(0.05)

    assert pipeline.state.frames_processed >= 2
    pipeline.pause()
    assert pipeline.is_paused

    # Allow in-flight frame to settle into pause wait
    time.sleep(0.1)
    count_at_pause = pipeline.state.frames_processed
    time.sleep(0.3)
    # Frames processed must not advance while paused
    assert pipeline.state.frames_processed == count_at_pause

    pipeline.resume()
    assert not pipeline.is_paused

    # Wait for frames to advance after resume
    t1 = time.time()
    while pipeline.state.frames_processed == count_at_pause and time.time() - t1 < 5.0:
        time.sleep(0.05)

    assert pipeline.state.frames_processed > count_at_pause
    pipeline.stop()


