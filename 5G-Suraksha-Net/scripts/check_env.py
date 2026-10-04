#!/usr/bin/env python
"""Environment self-test: Python, torch, CUDA/GPU, OpenCV, ultralytics, ffmpeg.

Run:  python scripts/check_env.py
Writes results to logs/environment.log and exits non-zero on hard failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suraksha.logging_utils import setup_logging  # noqa: E402
from suraksha.device import log_environment  # noqa: E402

PASS, WARN, FAIL = "[PASS]", "[WARN]", "[FAIL]"


def main() -> int:
    setup_logging("INFO")
    failures = 0

    print("=" * 62)
    print("5G Suraksha-Net — Crowd & Fight module environment check")
    print("=" * 62)

    # Python version
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 11)
    print(f"{PASS if ok else WARN} Python {v.major}.{v.minor}.{v.micro} (target 3.11)")

    # Device / CUDA (recorded to logs/environment.log)
    info = log_environment("auto")
    d = info.as_dict()
    print(f"{PASS} Torch {d['torch_version']} (CUDA build: {d['torch_cuda_build']})")
    if d["cuda_available"]:
        print(f"{PASS} CUDA available — GPU: {d['gpu_name']} ({d['gpu_memory_mb']} MB)")
        print(f"{PASS} Selected device: {d['device']}")
    else:
        print(f"{WARN} CUDA NOT available — CPU fallback active (driver CUDA: {d['driver_cuda_version']})")
    print(f"{PASS if d['ffmpeg_available'] else WARN} ffmpeg "
          f"{'found' if d['ffmpeg_available'] else 'not on PATH — clips will use OpenCV mp4v codec'}")

    # Core imports
    import_checks = [
        ("cv2", "opencv"), ("numpy", "numpy"), ("pandas", "pandas"),
        ("fastapi", "fastapi"), ("pydantic", "pydantic"), ("yaml", "PyYAML"),
        ("ultralytics", "ultralytics"), ("scipy", "scipy"),
    ]
    for mod, pkg in import_checks:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{PASS} {pkg} {ver}")
        except ImportError as e:
            print(f"{FAIL} {pkg}: {e}")
            failures += 1

    # torch<->numpy bridge (the exact failure that hit the global env)
    try:
        import numpy as np
        import torch

        arr = torch.tensor([1.0, 2.0]).numpy()
        assert isinstance(arr, np.ndarray)
        print(f"{PASS} torch<->numpy bridge (numpy {np.__version__})")
    except Exception as e:
        print(f"{FAIL} torch<->numpy bridge broken: {e}")
        failures += 1

    # CUDA tensor smoke test
    if d["cuda_available"]:
        try:
            import torch

            x = torch.randn(64, 64, device="cuda")
            y = (x @ x).sum().item()
            print(f"{PASS} CUDA tensor matmul smoke test ({y:.2f})")
        except Exception as e:
            print(f"{FAIL} CUDA tensor test: {e}")
            failures += 1

    # Config loads
    try:
        from suraksha.config import load_config

        cfg = load_config()
        print(f"{PASS} config loaded (rtsp={cfg.capture.rtsp_url}, camera={cfg.camera_id})")
    except Exception as e:
        print(f"{FAIL} config load: {e}")
        failures += 1

    # YOLO weights present?
    try:
        from suraksha.config import load_config

        w = Path(load_config().detection.weights)
        if w.exists():
            print(f"{PASS} YOLO weights found: {w}")
        else:
            print(f"{WARN} YOLO weights missing — run: python scripts/download_weights.py")
    except Exception as e:
        print(f"{FAIL} weights check: {e}")
        failures += 1

    print("=" * 62)
    if failures:
        print(f"RESULT: {failures} hard failure(s) — fix before running the pipeline")
        return 1
    print(f"RESULT: OK — device={d['device']}, env recorded to logs/environment.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
