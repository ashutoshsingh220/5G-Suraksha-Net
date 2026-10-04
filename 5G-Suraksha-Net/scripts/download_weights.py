#!/usr/bin/env python
"""Download YOLO11s weights (~19 MB) into models/. Run once before first pipeline start."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suraksha.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    dest = Path(cfg.detection.weights)
    if dest.exists():
        print(f"Already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    # attempt_download_asset fetches yolo11s.pt into CWD
    tmp = Path("yolo11s.pt")
    attempt_download_asset("yolo11s.pt")
    src = tmp if tmp.exists() else Path("yolo11s.pt")
    shutil.move(str(src), str(dest))
    print(f"Downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    # sanity-load
    YOLO(str(dest))
    print("Weights load OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
