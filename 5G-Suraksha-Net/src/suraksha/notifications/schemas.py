"""Data contracts and schemas for Phase 3D Incident Email Alert & Evidence Delivery."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EmailNotificationStatus(str, Enum):
    """Deterministic status of an email notification attempt."""
    DISABLED = "DISABLED"
    QUEUED = "QUEUED"
    SENT = "SENT"
    FAILED = "FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class EmailPayload(BaseModel):
    """Structured email contents ready for dispatch."""
    subject: str
    body_text: str
    body_html: str | None = None
    recipients: list[str] = Field(default_factory=list)
    clip_attachment_path: str | None = None
    snapshot_attachment_path: str | None = None
    clip_attached: bool = False
    snapshot_attached: bool = False
    clip_status_note: str = ""
    snapshot_status_note: str = ""


class EmailNotificationResult(BaseModel):
    """Outcome of an email alert dispatch operation."""
    incident_id: str
    status: EmailNotificationStatus
    sent_at: datetime | None = None
    recipients: list[str] = Field(default_factory=list)
    clip_attached: bool = False
    snapshot_attached: bool = False
    error_message: str | None = None
