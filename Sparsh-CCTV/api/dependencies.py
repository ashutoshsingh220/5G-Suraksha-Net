from config import get_settings, SystemSettings
from surveillance_engine import get_surveillance_engine, SurveillanceEngine


def get_engine() -> SurveillanceEngine:
    """Dependency provider for the singleton SurveillanceEngine."""
    return get_surveillance_engine()


def get_sys_settings() -> SystemSettings:
    """Dependency provider for current system settings."""
    return get_settings()
