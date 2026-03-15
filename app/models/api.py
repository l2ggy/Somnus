"""
API request/response models for Somnus.

These are kept separate from the domain models in userstate.py because they
represent the *API surface* — what callers send over HTTP — not the internal
state structure.  Field names and optionality may differ from SharedState.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.userstate import JournalEntry, UserPreferences


class StartSessionRequest(BaseModel):
    """
    Payload for POST /session/start.

    journal_history lets callers seed a new session with past nights so the
    strategist has context to plan from on the very first night.
    """
    user_id: str
    preferences: UserPreferences
    journal_history: List[JournalEntry] = Field(default_factory=list)


class SensorPayload(BaseModel):
    """
    Payload for POST /session/{user_id}/sensor.

    All sensor fields are optional — callers send whatever their device
    provides.  The intake agent clamps values and fills the timestamp if
    missing.  Downstream agents handle None values by falling back to
    conservative defaults.
    """
    timestamp: Optional[str] = None          # ISO 8601; auto-filled if absent
    heart_rate: Optional[float] = None       # bpm
    hrv: Optional[float] = None             # ms
    movement: Optional[float] = None        # 0–1 normalised
    noise_db: Optional[float] = None        # dB
    light_level: Optional[float] = None     # 0–1 normalised lux
    breathing_rate: Optional[float] = None  # breaths per minute
