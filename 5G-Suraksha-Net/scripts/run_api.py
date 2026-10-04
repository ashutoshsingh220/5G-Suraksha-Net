#!/usr/bin/env python
"""Run the REST + WebSocket API server (starts the pipeline internally).

Usage:
  python scripts/run_api.py            # http://localhost:8100/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    import uvicorn

    from suraksha.config import load_config

    cfg = load_config()
    uvicorn.run("suraksha.api.app:app", host=cfg.api.host, port=cfg.api.port, reload=False)


if __name__ == "__main__":
    main()
