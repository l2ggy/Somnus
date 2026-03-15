"""
Sensor Interpreter Agent
------------------------
First deterministic pass: translate raw SensorSnapshot numbers into labelled
signals that downstream agents can reason about without knowing sensor physics.

This is intentionally a pure feature-extraction step — no sleep-phase logic
here, just "what does this number mean?"  That keeps the classifier in
sleep_state.py cleanly separated from the feature engineering.

LLM integration plan: once the LLM layer is added, the full feature summary
produced here will be included in the prompt context so the model understands
the physiological situation in natural-language terms.
"""

from app.models.userstate import SharedState, SensorSnapshot


SENSOR_SOURCE = "sensor_interpreter"
TICK_SOURCE_MARKER = "night_tick.latest_sensor"


def run(state: SharedState) -> SharedState:
    """
    Derive interpretive signals from the latest sensor snapshot and append
    them to state.hypotheses.

    Args:
        state: SharedState with a populated latest_sensor field.

    Returns:
        Updated SharedState with a feature-summary hypothesis appended.
        Returns state unchanged if latest_sensor is None.
    """
    sensor = state.latest_sensor
    if sensor is None:
        return state

    summary = extract_features(sensor)
    signal_hypotheses = _build_signal_hypotheses(summary, sensor)
    tick_timestamp = sensor.timestamp

    updated_hypotheses = state.hypotheses + [
        {
            "source": SENSOR_SOURCE,
            "timestamp": tick_timestamp,
            "tick_source": TICK_SOURCE_MARKER,
            "signal_summary": summary,
            # Keep legacy key for compatibility with existing readers.
            "signals": summary,
            # Canonical per-signal payload for downstream reasoning/tracing.
            "signal_hypotheses": signal_hypotheses,
        }
    ]
    return state.model_copy(update={"hypotheses": updated_hypotheses})


# ---------------------------------------------------------------------------
# Feature extraction — all thresholds are based on published sleep-research
# norms and can be tuned independently per user in a later iteration.
# ---------------------------------------------------------------------------

def extract_features(sensor: SensorSnapshot) -> dict:
    """
    Produce a structured feature summary from a SensorSnapshot.

    Returns a dict with up to six labelled signals.  All fields are always
    present (using "unknown" when the sensor value is missing) so downstream
    agents can rely on a consistent schema.
    """
    return {
        "hr_status":        _hr_status(sensor.heart_rate),
        "hrv_quality":      _hrv_quality(sensor.hrv),
        "movement_level":   _movement_level(sensor.movement),
        "noise_alert":      _noise_alert(sensor.noise_db),
        "light_status":     _light_status(sensor.light_level),
        "breathing_status": _breathing_status(sensor.breathing_rate),
    }


def _build_signal_hypotheses(summary: dict, sensor: SensorSnapshot) -> list[dict]:
    """Build a consistent per-signal hypothesis payload for the current tick."""
    evidence = {
        "heart_rate": sensor.heart_rate,
        "hrv": sensor.hrv,
        "movement": sensor.movement,
        "noise_db": sensor.noise_db,
        "light_level": sensor.light_level,
        "breathing_rate": sensor.breathing_rate,
    }

    signal_specs = (
        ("noise_alert", summary["noise_alert"], _severity_for_noise_alert),
        ("movement_spike", summary["movement_level"], _severity_for_movement),
        ("hr_elevated", summary["hr_status"], _severity_for_hr_status),
        ("hrv_recovery", summary["hrv_quality"], _severity_for_hrv_quality),
        ("light_exposure", summary["light_status"], _severity_for_light_status),
        (
            "breathing_irregularity",
            summary["breathing_status"],
            _severity_for_breathing_status,
        ),
    )

    return [
        {
            "signal": signal_name,
            "severity": severity_fn(label),
            "label": label,
            "evidence": evidence,
            "tick_timestamp": sensor.timestamp,
            "tick_source": TICK_SOURCE_MARKER,
        }
        for signal_name, label, severity_fn in signal_specs
    ]


def _severity_for_hr_status(label: str) -> float:
    return {
        "unknown": 0.0,
        "very_low": 0.65,
        "low_resting": 0.2,
        "normal": 0.1,
        "mildly_elevated": 0.6,
        "elevated": 0.9,
    }.get(label, 0.0)


def _severity_for_hrv_quality(label: str) -> float:
    return {
        "unknown": 0.0,
        "excellent": 0.1,
        "good": 0.2,
        "moderate": 0.45,
        "low": 0.8,
    }.get(label, 0.0)


def _severity_for_movement(label: str) -> float:
    return {
        "unknown": 0.0,
        "still": 0.05,
        "minimal": 0.2,
        "restless": 0.65,
        "active": 0.95,
    }.get(label, 0.0)


def _severity_for_noise_alert(label: str) -> float:
    return {
        "unknown": 0.0,
        "quiet": 0.05,
        "moderate": 0.35,
        "loud": 0.75,
        "very_loud": 1.0,
    }.get(label, 0.0)


def _severity_for_light_status(label: str) -> float:
    return {
        "unknown": 0.0,
        "dark": 0.05,
        "dim": 0.2,
        "moderate": 0.55,
        "bright": 0.9,
    }.get(label, 0.0)


def _severity_for_breathing_status(label: str) -> float:
    return {
        "unknown": 0.0,
        "very_slow": 0.8,
        "normal": 0.1,
        "slightly_elevated": 0.55,
        "elevated": 0.9,
    }.get(label, 0.0)


def _hr_status(hr: float | None) -> str:
    if hr is None:
        return "unknown"
    if hr < 50:
        return "very_low"       # possible bradycardia or sensor artifact
    if hr < 60:
        return "low_resting"    # typical deep-sleep range
    if hr <= 80:
        return "normal"
    if hr <= 90:
        return "mildly_elevated"
    return "elevated"           # likely awake or stressed


def _hrv_quality(hrv: float | None) -> str:
    """
    Higher HRV correlates with parasympathetic (rest/recover) dominance.
    Low HRV suggests sympathetic activation or poor recovery.
    """
    if hrv is None:
        return "unknown"
    if hrv >= 70:
        return "excellent"
    if hrv >= 50:
        return "good"
    if hrv >= 30:
        return "moderate"
    return "low"


def _movement_level(movement: float | None) -> str:
    """movement is a 0–1 normalized score from the wearable accelerometer."""
    if movement is None:
        return "unknown"
    if movement < 0.05:
        return "still"
    if movement < 0.25:
        return "minimal"
    if movement < 0.55:
        return "restless"
    return "active"


def _noise_alert(noise_db: float | None) -> str:
    """
    Below 40 dB is quiet bedroom; above 60 dB is conversational level.
    Returns a severity label rather than a boolean so callers can grade it.
    """
    if noise_db is None:
        return "unknown"
    if noise_db < 40:
        return "quiet"
    if noise_db < 55:
        return "moderate"
    if noise_db < 65:
        return "loud"
    return "very_loud"


def _light_status(light_level: float | None) -> str:
    """light_level is a 0–1 normalized lux reading."""
    if light_level is None:
        return "unknown"
    if light_level < 0.05:
        return "dark"
    if light_level < 0.25:
        return "dim"
    if light_level < 0.6:
        return "moderate"
    return "bright"


def _breathing_status(bpm: float | None) -> str:
    """
    Normal sleeping breathing: 12–16 bpm.
    Elevated breathing may indicate arousal, apnoea recovery, or REM.
    """
    if bpm is None:
        return "unknown"
    if bpm < 10:
        return "very_slow"      # possible apnoea or sensor noise
    if bpm <= 16:
        return "normal"
    if bpm <= 20:
        return "slightly_elevated"
    return "elevated"
