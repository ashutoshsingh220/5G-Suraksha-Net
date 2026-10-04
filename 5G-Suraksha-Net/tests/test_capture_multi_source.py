"""Multi-source capture tests: protocol classification, credential sanitization,
uniform FrameEvent contract, and source-independent pipeline ingestion.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from suraksha.capture.stream import FrameEvent, StreamCapture, sanitize_url
from suraksha.config import PROJECT_ROOT, AppConfig, CaptureConfig

FIXTURE = PROJECT_ROOT / "datasets" / "videos" / "test" / "synthetic_cctv.mp4"


def test_sanitize_url_masks_credentials():
    # RTSP with credentials
    raw = "rtsp://admin:secretPass123@192.168.1.100:554/live/stream1"
    sanitized = sanitize_url(raw)
    assert "secretPass123" not in sanitized
    assert "admin:***@192.168.1.100:554" in sanitized
    assert sanitized.startswith("rtsp://")

    # RTSP with username only (no password)
    no_pass = "rtsp://drone@192.168.1.50:8554/drone"
    assert sanitize_url(no_pass) == no_pass

    # Standard RTSP without credentials
    plain = "rtsp://192.168.1.50:8554/drone"
    assert sanitize_url(plain) == plain

    # Local file path
    file_path = "datasets/videos/external/cctv_test_01.mp4"
    assert sanitize_url(file_path) == file_path

    # Webcam index
    assert sanitize_url("0") == "0"


def test_extended_protocol_classification():
    # RTSP / RTSPS
    assert StreamCapture._classify("rtsp://192.168.1.50:8554/drone") == "stream"
    assert StreamCapture._classify("rtsps://camera.lan:322/stream") == "stream"

    # UDP / TCP low-latency transport
    assert StreamCapture._classify("udp://@:8554") == "stream"
    assert StreamCapture._classify("tcp://192.168.1.50:8554") == "stream"

    # HTTP / HTTPS / RTMP
    assert StreamCapture._classify("http://192.168.1.50:8080/video") == "stream"
    assert StreamCapture._classify("https://camera.lan/stream.m3u8") == "stream"
    assert StreamCapture._classify("rtmp://live.server/stream") == "stream"

    # Webcam
    assert StreamCapture._classify("0") == "webcam"
    assert StreamCapture._classify(" 1 ") == "webcam"

    # File
    assert StreamCapture._classify("clip.mp4") == "file"
    assert StreamCapture._classify("C:/videos/test.avi") == "file"


def test_explicit_source_type_override():
    # If source_type is explicitly rtsp, even without scheme it resolves to stream
    cfg_rtsp = CaptureConfig(rtsp_url="192.168.1.50:8554/drone", source_type="rtsp")
    cap = StreamCapture(cfg_rtsp)
    assert cap.source_kind == "stream"
    assert cap.resolved_url.startswith("rtsp://")

    # Explicit webcam
    cfg_cam = CaptureConfig(rtsp_url="0", source_type="webcam")
    cap_cam = StreamCapture(cfg_cam)
    assert cap_cam.source_kind == "webcam"

    # Explicit file
    cfg_file = CaptureConfig(rtsp_url="clip.mp4", source_type="file")
    cap_file = StreamCapture(cfg_file)
    assert cap_file.source_kind == "file"


def test_frame_event_contract_is_uniform():
    # Verify FrameEvent structure is identical for any source
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    event = FrameEvent(
        frame=dummy_frame,
        frame_idx=42,
        timestamp=1000.0,
        source_fps=25.0,
    )
    assert isinstance(event.frame, np.ndarray)
    assert event.frame.shape == (720, 1280, 3)
    assert event.frame_idx == 42
    assert event.timestamp == 1000.0
    assert event.source_fps == 25.0


def test_stream_capture_sets_buffersize():
    """Verify that network streams set CAP_PROP_BUFFERSIZE=1 for low latency."""
    cfg = CaptureConfig(rtsp_url="rtsp://127.0.0.1:8554/test", max_reconnect_attempts=1)
    cap = StreamCapture(cfg)

    with patch("cv2.VideoCapture") as mock_vc_cls:
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc_cls.return_value = mock_vc

        ok = cap._open()
        assert ok is True
        # Verify BUFFERSIZE was set to 1
        mock_vc.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE if hasattr(cv2, "CAP_PROP_BUFFERSIZE") else 38, 1)
    cap.release()


def test_invalid_stream_fails_cleanly_without_unhandled_crash():
    cfg = CaptureConfig(
        rtsp_url="rtsp://non-existent-drone-ip:8554/drone",
        max_reconnect_attempts=2,
        reconnect_delay_s=0.01,
    )
    cap = StreamCapture(cfg)
    assert cap.source_kind == "stream"

    with pytest.raises(ConnectionError) as exc_info:
        next(cap.frames())
    # Verify error message is informative and masks credentials if any
    assert "Cannot open source" in str(exc_info.value)
    cap.release()


def test_downstream_pipeline_does_not_branch_on_source_kind():
    """Assert that CrowdFightPipeline and downstream modules have zero camera/source branching."""
    import inspect
    from suraksha import pipeline
    from suraksha.crowd import analyzer
    from suraksha.detection import tracker
    from suraksha.fight import candidate, recognizer

    # Check source code of update loops for any mention of source_kind, camera_type, 'drone', 'rtsp' branching
    pipeline_src = inspect.getsource(pipeline.CrowdFightPipeline._loop)
    assert "source_kind" not in pipeline_src
    assert "drone" not in pipeline_src

    tracker_src = inspect.getsource(tracker.MultiObjectTracker.update)
    assert "source_kind" not in tracker_src
    assert "camera" not in tracker_src

    crowd_src = inspect.getsource(analyzer.CrowdAnalyzer.update)
    assert "source_kind" not in crowd_src

    fight_src = inspect.getsource(recognizer.FightRecognizer.update)
    assert "source_kind" not in fight_src


def test_mocked_rtsp_stream_ingestion_in_pipeline():
    """Verify that CrowdFightPipeline runs cleanly on frames originating from an RTSP stream."""
    import queue
    from suraksha.pipeline import CrowdFightPipeline

    cfg = AppConfig()
    cfg.capture.rtsp_url = "rtsp://192.168.1.50:8554/drone"
    cfg.capture.pace = "fast"
    cfg.capture.max_frames = 5

    # Mock StreamCapture.frames() to yield 5 synthetic frames as if from RTSP
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    simulated_events = [
        FrameEvent(frame=dummy_frame.copy(), frame_idx=i + 1, timestamp=100.0 + i * 0.04, source_fps=25.0)
        for i in range(5)
    ]

    p_queue = queue.Queue(maxsize=10)
    pipeline_obj = CrowdFightPipeline(cfg, preview_queue=p_queue)

    with patch.object(pipeline_obj.capture, "frames", return_value=iter(simulated_events)):
        # Run loop synchronously
        pipeline_obj._loop()

    assert pipeline_obj.state.frames_processed == 5
    assert pipeline_obj.state.metrics["resolution"] == "640x360"
    assert pipeline_obj.state.metrics["source_fps"] == 25.0
    pipeline_obj.stop()
