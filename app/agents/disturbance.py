"""
Disturbance Agent
-----------------
First deterministic pass: scan the current sensor reading for environmental
and physiological threats to sleep continuity.

Runs after sleep_state so it can read the freshly-classified phase and apply
phase-specific thresholds (e.g. even modest noise is more disruptive during
deep sleep than during light sleep).

Threat taxonomy:
  noise_spike     — sudden or sustained loud sound
  light_spike     — unexpected increase in ambient light
  high_movement   — significant body movement suggesting partial arousal
  elevated_hr     — heart rate climbing above sleep-typical range

Each tick:
  1. Computes a weighted disturbance score from noise/light/movement/physiology
  2. Selects sleep_state.disturbance_reason using explicit priority order
  3. Maps disturbance score to a wake_risk bump (all risk values clamped to 0–1)

If no threat is detected the disturbance_reason is cleared, allowing the
intervention agent to ramp down on the next tick.

LLM integration plan: the disturbance_reason and wake_risk will be included
in the LLM context so the model can generate a natural-language explanation
of what is disrupting sleep.
"""

from app.models.userstate import SharedState, SleepState


def run(state: SharedState) -> SharedState:
    """
    Detect disturbances and annotate sleep_state accordingly.

    Args:
        state: SharedState with sleep_state and latest_sensor populated.

    Returns:
        Updated SharedState; sleep_state.disturbance_reason and wake_risk
        are updated even when the reason has not changed, so wake_risk can
        accumulate across ticks if a disturbance persists.
    """
    sensor = state.latest_sensor
    if sensor is None:
        return state

    reason, risk_bump = _detect(sensor, state.sleep_state.phase)

    current = state.sleep_state
    new_wake_risk = round(_clamp01(current.wake_risk + risk_bump), 2)

    # Skip update only when nothing has actually changed.
    if reason == current.disturbance_reason and new_wake_risk == current.wake_risk:
        return state

    updated_sleep = current.model_copy(update={
        "disturbance_reason": reason,
        "wake_risk": new_wake_risk,
    })
    return state.model_copy(update={"sleep_state": updated_sleep})


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

# Phase-specific noise thresholds: deep sleep is more sensitive to noise.
_NOISE_THRESHOLD_DB = {"deep": 45.0, "rem": 50.0, "light": 58.0, "awake": 70.0, "unknown": 55.0}

# Component thresholds/ranges for normalizing raw sensor values into 0–1.
_LIGHT_THRESHOLD = 0.35
_LIGHT_RANGE = 0.65
_MOVEMENT_THRESHOLD = 0.60
_MOVEMENT_RANGE = 0.40
_HEART_RATE_THRESHOLD = 85.0
_HEART_RATE_RANGE = 30.0
_BREATHING_RATE_THRESHOLD = 18.0
_BREATHING_RATE_RANGE = 8.0

# Weighted contribution to a single disturbance score.
_DISTURBANCE_WEIGHTS = {
    "noise": 0.28,
    "light": 0.30,
    "movement": 0.27,
    "physiology": 0.15,
}

# Reason selection priority (first active reason wins).
_DISTURBANCE_REASON_PRIORITY = (
    "light_spike",
    "noise_spike",
    "high_movement",
    "elevated_hr",
)

# Score-to-risk mapping; score is first clamped to [0, 1], then scaled.
_MAX_RISK_BUMP_PER_TICK = 0.35


def _detect(sensor, phase: str) -> tuple[str | None, float]:
    """
    Return (disturbance_reason, risk_bump).

    Combines all disturbance channels into a single weighted score, then
    chooses a single disturbance reason using explicit priority.
    """
    noise_threshold = _NOISE_THRESHOLD_DB.get(phase, _NOISE_THRESHOLD_DB["unknown"])
    noise_score = _normalize_above(sensor.noise_db, noise_threshold, 30.0)
    light_score = _normalize_above(sensor.light_level, _LIGHT_THRESHOLD, _LIGHT_RANGE)
    movement_score = _normalize_above(sensor.movement, _MOVEMENT_THRESHOLD, _MOVEMENT_RANGE)

    hr_score = _normalize_above(sensor.heart_rate, _HEART_RATE_THRESHOLD, _HEART_RATE_RANGE)
    br_score = _normalize_above(
        sensor.breathing_rate,
        _BREATHING_RATE_THRESHOLD,
        _BREATHING_RATE_RANGE,
    )
    physiology_score = _clamp01(max(hr_score, br_score))

    disturbance_score = _clamp01(
        (_DISTURBANCE_WEIGHTS["noise"] * noise_score)
        + (_DISTURBANCE_WEIGHTS["light"] * light_score)
        + (_DISTURBANCE_WEIGHTS["movement"] * movement_score)
        + (_DISTURBANCE_WEIGHTS["physiology"] * physiology_score)
    )
    risk_bump = round(_map_disturbance_to_risk_bump(disturbance_score), 2)

    active_reasons = {
        "light_spike": light_score > 0.0,
        "noise_spike": noise_score > 0.0,
        "high_movement": movement_score > 0.0,
        "elevated_hr": hr_score > 0.0,
    }

    for reason_name in _DISTURBANCE_REASON_PRIORITY:
        if active_reasons[reason_name]:
            if reason_name == "light_spike":
                return f"light_spike:{(sensor.light_level or 0.0):.2f}", risk_bump
            if reason_name == "noise_spike":
                return f"noise_spike:{(sensor.noise_db or 0.0):.0f}dB", risk_bump
            if reason_name == "high_movement":
                return f"high_movement:{(sensor.movement or 0.0):.2f}", risk_bump
            return f"elevated_hr:{(sensor.heart_rate or 0.0):.0f}bpm", risk_bump

    return None, 0.0


def _normalize_above(value: float | None, threshold: float, span: float) -> float:
    """Normalize values above a threshold into [0, 1]."""
    if value is None:
        return 0.0
    if span <= 0:
        return 1.0 if value > threshold else 0.0
    return _clamp01((value - threshold) / span)


def _map_disturbance_to_risk_bump(disturbance_score: float) -> float:
    """Map normalized disturbance score into a bounded wake-risk bump."""
    return _clamp01(disturbance_score) * _MAX_RISK_BUMP_PER_TICK


def _clamp01(value: float) -> float:
    """Clamp a numeric value to [0, 1]."""
    return max(0.0, min(1.0, value))
