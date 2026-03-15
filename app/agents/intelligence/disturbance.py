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

Each detected threat:
  1. Sets sleep_state.disturbance_reason (first threat wins per tick)
  2. Adds a proportional wake_risk bump (capped at 1.0)

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
    new_wake_risk = round(min(1.0, current.wake_risk + risk_bump), 2)

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
_NOISE_THRESHOLD = {"deep": 45, "rem": 50, "light": 58, "awake": 70, "unknown": 55}

# How much each disturbance type bumps wake_risk.
_RISK_BUMP = {
    "noise_spike":    0.20,
    "light_spike":    0.25,
    "high_movement":  0.30,
    "elevated_hr":    0.15,
}


def _detect(sensor, phase: str) -> tuple[str | None, float]:
    """
    Return (disturbance_reason, risk_bump).

    Checks threats in priority order: light > noise > movement > hr.
    Only the highest-priority active threat is returned per tick.
    """
    # Light spike — clear and immediate arousal signal
    if sensor.light_level is not None and sensor.light_level > 0.35:
        severity = (sensor.light_level - 0.35) / 0.65  # 0 → 1
        bump = round(_RISK_BUMP["light_spike"] * (0.5 + 0.5 * severity), 2)
        return f"light_spike:{sensor.light_level:.2f}", bump

    # Noise — threshold depends on current sleep phase
    noise_threshold = _NOISE_THRESHOLD.get(phase, 55)
    if sensor.noise_db is not None and sensor.noise_db > noise_threshold:
        severity = min(1.0, (sensor.noise_db - noise_threshold) / 30)
        bump = round(_RISK_BUMP["noise_spike"] * (0.4 + 0.6 * severity), 2)
        return f"noise_spike:{sensor.noise_db:.0f}dB", bump

    # High movement
    if sensor.movement is not None and sensor.movement > 0.60:
        severity = min(1.0, (sensor.movement - 0.60) / 0.40)
        bump = round(_RISK_BUMP["high_movement"] * (0.5 + 0.5 * severity), 2)
        return f"high_movement:{sensor.movement:.2f}", bump

    # Elevated heart rate
    if sensor.heart_rate is not None and sensor.heart_rate > 85:
        severity = min(1.0, (sensor.heart_rate - 85) / 30)
        bump = round(_RISK_BUMP["elevated_hr"] * (0.5 + 0.5 * severity), 2)
        return f"elevated_hr:{sensor.heart_rate:.0f}bpm", bump

    # All clear — no bump, clear the reason
    return None, 0.0
