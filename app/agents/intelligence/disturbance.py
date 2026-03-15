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

Each detected threat contributes to a multi-signal disturbance score. The
highest contributor becomes the human-readable disturbance_reason while all
active contributors are preserved as structured metadata in hypotheses.

When disturbances persist across consecutive ticks, risk ramps faster than
isolated spikes. When disturbances stop, wake_risk decays gradually instead
of dropping abruptly.

LLM integration plan: the disturbance_reason and wake_risk will be included
in the LLM context so the model can generate a natural-language explanation
of what is disrupting sleep.
"""

from app.models.userstate import SharedState


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

    current = state.sleep_state
    previous = _last_disturbance_entry(state)
    reason, contributors, tick_risk = _detect(sensor, current.phase)
    had_previous_disturbance = bool(previous and previous.get("disturbance_active"))
    streak = _compute_streak(previous, bool(contributors))
    risk_delta = _risk_delta(
        tick_risk=tick_risk,
        has_disturbance=bool(contributors),
        streak=streak,
        had_previous_disturbance=had_previous_disturbance,
    )

    if contributors:
        new_wake_risk = round(min(1.0, current.wake_risk + risk_delta), 2)
    else:
        new_wake_risk = round(max(0.0, current.wake_risk - risk_delta), 2)

    disturbance_entry = {
        "source": "disturbance",
        "timestamp": sensor.timestamp,
        "phase": current.phase,
        "primary_reason": reason,
        "contributors": contributors,
        "disturbance_active": bool(contributors),
        "disturbance_streak": streak,
        "tick_risk": round(tick_risk, 3),
        "risk_delta": round(risk_delta, 3),
        "wake_risk_after": new_wake_risk,
    }
    updated_hypotheses = state.hypotheses + [disturbance_entry]

    updated_sleep = current.model_copy(update={
        "disturbance_reason": reason,
        "wake_risk": new_wake_risk,
    })
    return state.model_copy(update={
        "sleep_state": updated_sleep,
        "hypotheses": updated_hypotheses,
    })


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

# Persistence/decay controls.
_PERSISTENCE_STEP = 0.15
_PERSISTENCE_MAX = 0.75
_DECAY_BASE = 0.04
_DECAY_POST_DISTURBANCE = 0.03


def _detect(sensor, phase: str) -> tuple[str | None, list[dict], float]:
    """
    Return (primary_disturbance_reason, contributors, tick_risk).

    Multiple signals can contribute in a single tick.
    """
    contributors: list[dict] = []

    # Light spike — clear and immediate arousal signal.
    if sensor.light_level is not None and sensor.light_level > 0.35:
        severity = (sensor.light_level - 0.35) / 0.65  # 0 → 1
        contributors.append(_contributor(
            kind="light_spike",
            value=sensor.light_level,
            severity=min(1.0, max(0.0, severity)),
            detail=f"{sensor.light_level:.2f}",
        ))

    # Noise — threshold depends on current sleep phase.
    noise_threshold = _NOISE_THRESHOLD.get(phase, 55)
    if sensor.noise_db is not None and sensor.noise_db > noise_threshold:
        severity = min(1.0, (sensor.noise_db - noise_threshold) / 30)
        contributors.append(_contributor(
            kind="noise_spike",
            value=sensor.noise_db,
            severity=severity,
            detail=f"{sensor.noise_db:.0f}dB",
        ))

    # High movement.
    if sensor.movement is not None and sensor.movement > 0.60:
        severity = min(1.0, (sensor.movement - 0.60) / 0.40)
        contributors.append(_contributor(
            kind="high_movement",
            value=sensor.movement,
            severity=severity,
            detail=f"{sensor.movement:.2f}",
        ))

    # Elevated heart rate.
    if sensor.heart_rate is not None and sensor.heart_rate > 85:
        severity = min(1.0, (sensor.heart_rate - 85) / 30)
        contributors.append(_contributor(
            kind="elevated_hr",
            value=sensor.heart_rate,
            severity=severity,
            detail=f"{sensor.heart_rate:.0f}bpm",
        ))

    if not contributors:
        return None, [], 0.0

    primary = max(contributors, key=lambda item: item["risk_bump"])
    total_tick_risk = min(1.0, sum(item["risk_bump"] for item in contributors))
    return f"{primary['kind']}:{primary['detail']}", contributors, total_tick_risk


def _contributor(kind: str, value: float, severity: float, detail: str) -> dict:
    risk_bump = _RISK_BUMP[kind] * (0.35 + 0.65 * severity)
    return {
        "kind": kind,
        "value": round(value, 3),
        "severity": round(severity, 3),
        "risk_bump": round(risk_bump, 3),
        "detail": detail,
    }


def _risk_delta(
    tick_risk: float,
    has_disturbance: bool,
    streak: int,
    had_previous_disturbance: bool,
) -> float:
    if has_disturbance:
        persistence = min(_PERSISTENCE_MAX, max(0.0, (streak - 1) * _PERSISTENCE_STEP))
        return tick_risk * (1.0 + persistence)

    decay = _DECAY_BASE
    if had_previous_disturbance:
        decay += _DECAY_POST_DISTURBANCE
    return decay


def _compute_streak(previous: dict | None, has_disturbance: bool) -> int:
    if not has_disturbance:
        return 0
    if previous and previous.get("disturbance_active"):
        return int(previous.get("disturbance_streak", 0)) + 1
    return 1


def _last_disturbance_entry(state: SharedState) -> dict | None:
    for entry in reversed(state.hypotheses):
        if entry.get("source") == "disturbance":
            return entry
    return None
