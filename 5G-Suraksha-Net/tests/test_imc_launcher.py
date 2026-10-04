"""Tests for IMC demo configuration, stream probing, and launcher logic."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from suraksha.config import ImcConfig, load_imc_config
from scripts.probe_stream import probe_stream, check_tcp_port, check_rtsp_ready


def test_imc_config_defaults():
    cfg = ImcConfig()
    assert cfg.pi.host == "10.254.18.48"
    assert cfg.pi.user == "student"
    assert cfg.pi.ssh_port == 22
    assert cfg.pi.mediamtx_path == "/home/student/mediamtx"
    assert cfg.pi.camera_device == "/dev/video0"
    assert cfg.rtsp.port == 8554
    assert cfg.rtsp.path == "drone"
    assert cfg.rtsp.stream_path == "/drone"
    assert cfg.camera.width == 640
    assert cfg.camera.height == 360
    assert cfg.camera.fps == 15
    assert cfg.camera.input_format == "mjpeg"
    assert cfg.camera.encoder == "libx264"
    assert cfg.rtsp_url == "rtsp://10.254.18.48:8554/drone"


def test_imc_config_ffmpeg_command():
    cfg = ImcConfig()
    cmd = cfg.ffmpeg_command
    assert "ffmpeg" in cmd
    assert "-f v4l2" in cmd
    assert "-input_format mjpeg" in cmd
    assert "-video_size 640x360" in cmd
    assert "-framerate 15" in cmd
    assert "-i /dev/video0" in cmd
    assert "-c:v libx264" in cmd
    assert "-preset ultrafast" in cmd
    assert "-tune zerolatency" in cmd
    assert "-g 15" in cmd
    assert "-pix_fmt yuv420p" in cmd
    assert "rtsp://127.0.0.1:8554/drone" in cmd


def test_imc_config_env_overrides(monkeypatch):
    monkeypatch.setenv("SURAKSHA_IMC_PI_HOST", "192.168.1.100")
    monkeypatch.setenv("SURAKSHA_IMC_PI_USER", "operator")
    monkeypatch.setenv("SURAKSHA_IMC_RTSP_PORT", "9554")
    monkeypatch.setenv("SURAKSHA_IMC_RTSP_PATH", "airview")
    monkeypatch.setenv("SURAKSHA_IMC_CAMERA_WIDTH", "1280")
    monkeypatch.setenv("SURAKSHA_IMC_CAMERA_HEIGHT", "720")
    monkeypatch.setenv("SURAKSHA_IMC_CAMERA_FPS", "30")

    cfg = load_imc_config()
    assert cfg.pi.host == "192.168.1.100"
    assert cfg.pi.user == "operator"
    assert cfg.rtsp.port == 9554
    assert cfg.rtsp.path == "airview"
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.camera.fps == 30
    assert cfg.rtsp_url == "rtsp://192.168.1.100:9554/airview"
    assert "1280x720" in cfg.ffmpeg_command
    assert "rtsp://127.0.0.1:9554/airview" in cfg.ffmpeg_command


def test_check_tcp_port_failure():
    # Use a non-routable test address with tiny timeout
    ok, msg = check_tcp_port("127.0.0.1", 59999, timeout_s=0.2)
    assert ok is False
    assert "Connection" in msg


def test_probe_stream_unreachable():
    res = probe_stream("rtsp://127.0.0.1:59999/drone", timeout_s=0.3, num_frames=1)
    assert res["success"] is False
    assert "RTSP server unreachable" in res["error"]
    assert res["frames_read"] == 0


def test_probe_stream_mock_success():
    mock_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, mock_frame)
    mock_cap.get.return_value = 15.0

    with patch("scripts.probe_stream.check_tcp_port", return_value=(True, "Port open")):
        with patch("scripts.probe_stream.check_rtsp_ready", return_value=(True, "Path ready")):
            with patch("cv2.VideoCapture", return_value=mock_cap):
                res = probe_stream("rtsp://mock-pi:8554/drone", timeout_s=2.0, num_frames=3)

    assert res["success"] is True
    assert res["width"] == 640
    assert res["height"] == 360
    assert res["source_fps"] == 15.0
    assert res["frames_read"] == 3
    assert res["error"] is None


def test_check_rtsp_ready_non_rtsp():
    ok, msg = check_rtsp_ready("file://test.mp4")
    assert ok is True
    assert "Non-RTSP" in msg


def test_check_rtsp_ready_200_ok():
    mock_sock = MagicMock()
    mock_sock.__enter__.return_value = mock_sock
    mock_sock.recv.return_value = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\n\r\n"

    with patch("socket.create_connection", return_value=mock_sock):
        ok, msg = check_rtsp_ready("rtsp://10.254.18.48:8554/drone", timeout_s=1.0)
    assert ok is True
    assert "200" in msg or "active" in msg


def test_check_rtsp_ready_404_not_found():
    mock_sock = MagicMock()
    mock_sock.__enter__.return_value = mock_sock
    mock_sock.recv.return_value = b"RTSP/1.0 404 Not Found\r\nCSeq: 1\r\n\r\n"

    with patch("socket.create_connection", return_value=mock_sock):
        ok, msg = check_rtsp_ready("rtsp://10.254.18.48:8554/drone", timeout_s=1.0)
    assert ok is False
    assert "404" in msg


def test_check_rtsp_ready_connection_refused():
    with patch("socket.create_connection", side_effect=ConnectionRefusedError("Connection refused")):
        ok, msg = check_rtsp_ready("rtsp://10.254.18.48:8554/drone", timeout_s=1.0)
    assert ok is False
    assert "Connection refused" in msg or "error" in msg.lower()


def test_probe_stream_cli_help():
    cmd = [sys.executable, str(Path("scripts/probe_stream.py")), "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "--url" in proc.stdout
    assert "--timeout" in proc.stdout
    assert "--frames" in proc.stdout
    assert "--json" in proc.stdout

