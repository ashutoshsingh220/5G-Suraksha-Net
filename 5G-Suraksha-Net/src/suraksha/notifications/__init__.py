"""Phase 3D Incident Email Alert & Evidence Delivery module."""
from __future__ import annotations

from suraksha.notifications.schemas import (
    EmailNotificationResult,
    EmailNotificationStatus,
    EmailPayload,
)
from suraksha.notifications.service import EmailNotificationService

__all__ = [
    "EmailNotificationResult",
    "EmailNotificationService",
    "EmailNotificationStatus",
    "EmailPayload",
]
