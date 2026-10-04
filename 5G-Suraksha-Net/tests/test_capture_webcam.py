"""Tests for laptop webcam source mode and multi-source convergence."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from suraksha.capture.stream import FrameEvent, StreamCapture
from suraksha.config import AppConfig, CaptureConfig, load_config


def test_capture_config_camera_index():
    cfg = CaptureConfig()
    assert hasattr(cfg, "camera_index")
    assert cfg.camera_index == 0

    cfg2 = CaptureConfig(camera_index=2)
    assert cfg2.camera_index == 2


def test_capture_config_env_override(monkeypatch):
    load_config.cache_clear()
    try:
        monkeypatch.setenv("SURAKSHA_CAMERA_INDEX", "3")
        cfg = load_config()
        assert cfg.capture.camera_index == 3
    finally:
        load_config.cache_clear()


def test_webcam_source_resolution():
    # Explicit source_type webcam with camera_index 1
    cfg1 = CaptureConfig(source_type="webcam", camera_index=1)
    cap1 = StreamCapture(cfg1)
    assert cap1.source_kind == "webcam"
    assert cap1.resolved_url == "1"

    # Source type webcam without rtsp_url defaults to camera_index
    cfg2 = CaptureConfig(source_type="webcam", camera_index=0)
    cap2 = StreamCapture(cfg2)
    assert cap2.source_kind == "webcam"
    assert cap2.resolved_url == "0"

    # Numeric rtsp_url classified as webcam
    cfg3 = CaptureConfig(rtsp_url="2")
    cap3 = StreamCapture(cfg3)
    assert cap3.source_kind == "webcam"
    assert cap3.resolved_url == "2"


def test_webcam_mock_frame_event_contract():
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, mock_frame)
    mock_cap.get.return_value = 30.0

    cfg = CaptureConfig(source_type="webcam", camera_index=0, max_frames=2, pace="fast")
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with StreamCapture(cfg) as cap:
            events = list(cap.frames())

    assert len(events) == 2
    for ev in events:
        assert isinstance(ev, FrameEvent)
        assert ev.frame.shape == (480, 640, 3)
        assert ev.source_fps == 30.0
        assert ev.frame_idx > 0
        assert ev.timestamp > 0.0

    # Verify low-latency buffer was requested
    mock_cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)


def test_webcam_invalid_index_fails_fast():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    cfg = CaptureConfig(source_type="webcam", camera_index=99, max_reconnect_attempts=2, reconnect_delay_s=0.01)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        cap = StreamCapture(cfg)
        with pytest.raises(RuntimeError) as exc_info:
            next(cap.frames())
        assert "Cannot open webcam" in str(exc_info.value)
        assert "99" in str(exc_info.value)
        cap.release()


def test_multi_source_convergence_preservation():
    # 1. Local MP4 / File
    cfg_file = CaptureConfig(source_type="file", rtsp_url="datasets/videos/test/synthetic_cctv.mp4")
    cap_file = StreamCapture(cfg_file)
    assert cap_file.source_kind == "file"

    # 2. Raspberry Pi RTSP
    cfg_pi = CaptureConfig(source_type="rtsp", rtsp_url="rtsp://10.254.18.48:8554/drone")
    cap_pi = StreamCapture(cfg_pi)
    assert cap_pi.source_kind == "stream"
    assert cap_pi.resolved_url == "rtsp://10.254.18.48:8554/drone"

    # 3. Generic Sparsh CCTV RTSP
    cfg_sparsh = CaptureConfig(source_type="rtsp", rtsp_url="rtsp://192.168.1.100:554/live")
    cap_sparsh = StreamCapture(cfg_sparsh)
    assert cap_sparsh.source_kind == "stream"

    # 4. Laptop Webcam
    cfg_cam = CaptureConfig(source_type="webcam", camera_index=0)
    cap_cam = StreamCapture(cfg_cam)
    assert cap_cam.source_kind == "webcam"
    assert cap_cam.resolved_url == "0"


def test_cli_webcam_argument_parsing(monkeypatch):
    import subprocess
    import sys

    # Test --help contains --camera-index
    proc = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--camera-index" in proc.stdout
    assert "--source" in proc.stdout
    assert "--display" in proc.stdout
