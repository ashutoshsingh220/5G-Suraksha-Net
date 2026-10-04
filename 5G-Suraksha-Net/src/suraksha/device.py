"""Device detection: pick cuda/cpu and record full environment info."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from suraksha.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class DeviceInfo:
    device: str = "cpu"
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    torch_version: str = "unknown"
    torch_cuda_build: str | None = None
    cuda_available: bool = False
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    driver_cuda_version: str | None = None
    ffmpeg_available: bool = False

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "torch_cuda_build": self.torch_cuda_build,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": self.gpu_memory_mb,
            "driver_cuda_version": self.driver_cuda_version,
            "ffmpeg_available": self.ffmpeg_available,
        }


def _nvidia_smi_info() -> tuple[str | None, str | None, int | None]:
    """Return (gpu_name, driver_version, memory_mb) from nvidia-smi, or Nones."""
    if shutil.which("nvidia-smi") is None:
        return None, None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if not out.stdout.strip():
            return None, None, None
        parts = out.stdout.strip().splitlines()[0].split(",")
        return parts[0].strip(), parts[2].strip(), int(float(parts[1]))
    except Exception:
        return None, None, None


def detect_device(preference: str = "auto") -> DeviceInfo:
    """Detect best device honoring preference: auto | cuda | cpu."""
    info = DeviceInfo()
    try:
        import torch

        info.torch_version = torch.__version__
        info.torch_cuda_build = torch.version.cuda
        info.cuda_available = torch.cuda.is_available()
        if info.cuda_available:
            info.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.gpu_memory_mb = props.total_memory // (1024 * 1024)
    except ImportError:
        log.warning("torch not installed — falling back to CPU metadata only")

    gpu_name, driver, mem = _nvidia_smi_info()
    info.gpu_name = info.gpu_name or gpu_name
    info.driver_cuda_version = driver
    if mem:
        info.gpu_memory_mb = mem
    info.ffmpeg_available = shutil.which("ffmpeg") is not None

    if preference == "cuda" and not info.cuda_available:
        log.warning("CUDA requested but unavailable — falling back to CPU")
    info.device = "cuda" if (preference in ("auto", "cuda") and info.cuda_available) else "cpu"

    log.info("Environment: %s", info.as_dict())
    return info


def log_environment(preference: str = "auto") -> DeviceInfo:
    """Detect device and write a persistent env record to logs/environment.log."""
    info = detect_device(preference)
    from suraksha.config import PROJECT_ROOT

    env_log = PROJECT_ROOT / "logs" / "environment.log"
    env_log.parent.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime, timezone

    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **info.as_dict()}
    record["sys_executable"] = sys.executable
    with open(env_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return info
