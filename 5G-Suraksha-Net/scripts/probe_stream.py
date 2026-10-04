"""Lightweight stream probe utility for 5G Suraksha-Net.

Probes RTSP (or file/webcam) streams to verify reachability, frame decoding,
resolution, and FPS before launching the full AI pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from suraksha.capture.stream import sanitize_url
from suraksha.config import load_imc_config
from suraksha.logging_utils import get_logger

log = get_logger(__name__)


def check_tcp_port(host: str, port: int, timeout_s: float = 2.0) -> tuple[bool, str]:
    """Fast check whether a TCP port is reachable and open."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True, f"Port {port} on {host} is OPEN"
    except socket.timeout:
        return False, f"Connection to {host}:{port} timed out after {timeout_s}s"
    except ConnectionRefusedError:
        return False, f"Connection to {host}:{port} refused (server process not running)"
    except Exception as exc:
        return False, f"Connection to {host}:{port} failed: {exc}"


def check_rtsp_ready(url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    """Check if the RTSP server is accepting requests and the path is available via DESCRIBE."""
    parts = urlsplit(url)
    if parts.scheme not in ("rtsp", "rtsps"):
        return True, "Non-RTSP URL"
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 8554
    path = parts.path or "/drone"
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as s:
            req = f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n\r\n"
            s.sendall(req.encode("utf-8"))
            s.settimeout(timeout_s)
            resp = s.recv(1024).decode("utf-8", errors="ignore")
            if "RTSP/1.0 200" in resp or "RTSP/1.0 401" in resp:
                return True, "RTSP path is active and published"
            elif "RTSP/1.0 404" in resp:
                return False, f"RTSP path '{path}' not yet published (404 Not Found)"
            else:
                first_line = resp.splitlines()[0] if resp else "Empty response"
                return False, f"RTSP server returned: {first_line}"
    except Exception as exc:
        return False, f"RTSP check error: {exc}"


def probe_stream(
    url: str,
    timeout_s: float = 15.0,
    num_frames: int = 3,
    retry_interval_s: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Probe an RTSP stream (or other video source) for decoded frames.

    Returns dict with keys:
        success: bool
        url: str (sanitized)
        width: int
        height: int
        source_fps: float
        frames_read: int
        elapsed_s: float
        error: str | None
    """
    clean_url = sanitize_url(url)
    result = {
        "success": False,
        "url": clean_url,
        "width": 0,
        "height": 0,
        "source_fps": 0.0,
        "frames_read": 0,
        "elapsed_s": 0.0,
        "error": None,
    }

    start_time = time.monotonic()
    deadline = start_time + max(1.0, timeout_s)

    parts = urlsplit(url)
    if parts.scheme in ("rtsp", "rtsps", "tcp", "http"):
        host = parts.hostname or "127.0.0.1"
        port = parts.port or (8554 if "rtsp" in parts.scheme else 80)
        tcp_ok, tcp_msg = check_tcp_port(host, port, timeout_s=min(3.0, timeout_s))
        if not tcp_ok:
            result["elapsed_s"] = round(time.monotonic() - start_time, 2)
            result["error"] = f"RTSP server unreachable: {tcp_msg}"
            return result

    if parts.scheme in ("rtsp", "rtsps"):
        # Bounded poll for RTSP path readiness (DESCRIBE 200 OK)
        path_ready = False
        last_path_err = ""
        while time.monotonic() < deadline:
            ready, msg = check_rtsp_ready(url, timeout_s=min(2.0, max(0.5, deadline - time.monotonic())))
            if ready:
                path_ready = True
                break
            last_path_err = msg
            time.sleep(min(retry_interval_s, max(0.1, deadline - time.monotonic())))

        if not path_ready:
            result["elapsed_s"] = round(time.monotonic() - start_time, 2)
            result["error"] = f"RTSP stream not ready: {last_path_err}"
            return result

    # Configure OpenCV FFMPEG low-latency TCP RTSP transport & fast probe
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;500000|probesize;500000"

    cap = None
    try:
        while time.monotonic() < deadline:
            if cap is not None:
                cap.release()
                cap = None

            if parts.scheme in ("rtsp", "rtsps", "rtmp", "http", "https"):
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            elif url.isdigit():
                cap = cv2.VideoCapture(int(url), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(url, cv2.CAP_ANY)

            if not cap.isOpened():
                time.sleep(retry_interval_s)
                continue

            frames_read = 0
            w, h, fps = 0, 0, 0.0
            # Allow at least 5 seconds or remaining deadline to read the frames
            read_deadline = max(time.monotonic() + 5.0, deadline)

            while time.monotonic() < read_deadline and frames_read < num_frames:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frames_read += 1
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                else:
                    time.sleep(0.05)

            if frames_read >= num_frames:
                result["success"] = True
                result["width"] = int(w)
                result["height"] = int(h)
                result["source_fps"] = round(float(fps), 2)
                result["frames_read"] = frames_read
                result["elapsed_s"] = round(time.monotonic() - start_time, 2)
                return result

            time.sleep(retry_interval_s)

        result["elapsed_s"] = round(time.monotonic() - start_time, 2)
        result["error"] = f"Timeout ({timeout_s}s) reading {num_frames} frames from {clean_url}"
        return result
    finally:
        if cap is not None:
            cap.release()


def main() -> int:
    imc_cfg = load_imc_config()

    parser = argparse.ArgumentParser(description="Probe video stream for decoded frames.")
    parser.add_argument(
        "--url",
        type=str,
        default=imc_cfg.rtsp_url,
        help=f"Stream URL or source path (default: {imc_cfg.rtsp_url})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=imc_cfg.probe.timeout_s,
        help=f"Timeout in seconds (default: {imc_cfg.probe.timeout_s})",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=3,
        help="Number of consecutive frames required for verification (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output (useful in automation)",
    )

    args = parser.parse_args()

    res = probe_stream(
        url=args.url,
        timeout_s=args.timeout,
        num_frames=args.frames,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    elif not args.quiet:
        if res["success"]:
            print(f"[OK] Stream verified: {res['url']}")
            print(f"     Resolution : {res['width']}x{res['height']}")
            print(f"     Source FPS : {res['source_fps']}")
            print(f"     Frames read: {res['frames_read']}")
            print(f"     Elapsed    : {res['elapsed_s']}s")
        else:
            print(f"[FAIL] Stream probe failed: {res['url']}", file=sys.stderr)
            print(f"       Error  : {res['error']}", file=sys.stderr)
            print(f"       Elapsed: {res['elapsed_s']}s", file=sys.stderr)

    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
