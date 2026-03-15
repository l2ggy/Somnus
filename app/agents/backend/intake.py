"""
Intake Agent
------------
Responsible for validating and normalising incoming sensor data before it
enters the shared state.  This is the "front door" of the Somnus pipeline —
raw payloads from wearables or mocked sources arrive here, get cleaned, and
are stamped with a timestamp before being written to SharedState.latest_sensor.

In a production system this agent would also:
  - Detect stale or missing sensor fields and fill them with the last known
    good value (carry-forward).
  - Flag anomalous readings (e.g. HR > 200) as unreliable.
  - Trigger re-calibration alerts when sensor quality degrades.
"""

from datetime import datetime, timezone
from typing import Optional

from app.models.userstate import SensorSnapshot, SharedState


def run(state: SharedState, raw: dict) -> SharedState:
    """
    Validate and ingest a raw sensor payload into shared state.

    Args:
        state: Current shared state.
        raw:   Dict of raw sensor values from the device layer.
               Expected keys (all optional): heart_rate, hrv, movement,
               noise_db, light_level, breathing_rate.

    Returns:
        Updated SharedState with latest_sensor populated.
    """
    # Clamp 0-1 fields to valid range; leave physiological fields unclamped
    # so downstream agents can decide what counts as anomalous.
    snapshot = SensorSnapshot(
        timestamp=raw.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        heart_rate=_clamp_optional(raw.get("heart_rate"), 20, 220),
        hrv=_clamp_optional(raw.get("hrv"), 0, 300),
        movement=_clamp_optional(raw.get("movement"), 0.0, 1.0),
        noise_db=raw.get("noise_db"),
        light_level=_clamp_optional(raw.get("light_level"), 0.0, 1.0),
        breathing_rate=_clamp_optional(raw.get("breathing_rate"), 4, 60),
    )

    # Immutable update — return a new state object rather than mutating.
    return state.model_copy(update={"latest_sensor": snapshot})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_optional(value: Optional[float], lo: float, hi: float) -> Optional[float]:
    if value is None:
        return None
    return max(lo, min(hi, value))
