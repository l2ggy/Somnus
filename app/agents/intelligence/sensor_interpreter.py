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

from copy import deepcopy
from functools import lru_cache

from app.models.userstate import SharedState, SensorSnapshot


SENSOR_SOURCE = "sensor_interpreter"
TICK_SOURCE_MARKER = "night_tick.latest_sensor"


# Centralized signal model configuration. Threshold cutoffs and severity
# mappings live here so feature extraction and hypothesis generation stay aligned.
BASE_SIGNAL_PROFILE = {
    "features": {
        "hr_status": {
            "sensor_attr": "heart_rate",
            "unknown_label": "unknown",
            "ranges": [
                {"label": "very_low", "max": 50, "max_inclusive": False},
                {"label": "low_resting", "max": 60, "max_inclusive": False},
                {"label": "normal", "max": 80, "max_inclusive": True},
                {"label": "mildly_elevated", "max": 90, "max_inclusive": True},
                {"label": "elevated", "max": None},
            ],
            "severity_by_label": {
                "unknown": 0.0,
                "very_low": 0.65,
                "low_resting": 0.2,
                "normal": 0.1,
                "mildly_elevated": 0.6,
                "elevated": 0.9,
            },
        },
        "hrv_quality": {
            "sensor_attr": "hrv",
            "unknown_label": "unknown",
            "ranges": [
                {"label": "excellent", "min": 70},
                {"label": "good", "min": 50, "max": 70, "max_inclusive": False},
                {"label": "moderate", "min": 30, "max": 50, "max_inclusive": False},
                {"label": "low", "max": 30, "max_inclusive": False},
            ],
            "severity_by_label": {
                "unknown": 0.0,
                "excellent": 0.1,
                "good": 0.2,
                "moderate": 0.45,
                "low": 0.8,
            },
        },
        "movement_level": {
            "sensor_attr": "movement",
            "unknown_label": "unknown",
            "ranges": [
                {"label": "still", "max": 0.05, "max_inclusive": False},
                {"label": "minimal", "max": 0.25, "max_inclusive": False},
                {"label": "restless", "max": 0.55, "max_inclusive": False},
                {"label": "active", "max": None},
            ],
            "severity_by_label": {
                "unknown": 0.0,
                "still": 0.05,
                "minimal": 0.2,
                "restless": 0.65,
                "active": 0.95,
            },
        },
        "noise_alert": {
            "sensor_attr": "noise_db",
            "unknown_label": "unknown",
            "ranges": [
                {"label": "quiet", "max": 40, "max_inclusive": False},
                {"label": "moderate", "max": 55, "max_inclusive": False},
                {"label": "loud", "max": 65, "max_inclusive": False},
                {"label": "very_loud", "max": None},
            ],
            "severity_by_label": {
                "unknown": 0.0,
                "quiet": 0.05,
                "moderate": 0.35,
                "loud": 0.75,
                "very_loud": 1.0,
            },
        },
        "light_status": {
            "sensor_attr": "light_level",
            "unknown_label": "unknown",
            "ranges": [
                {"label": "dark", "max": 0.05, "max_inclusive": False},
                {"label": "dim", "max": 0.25, "max_inclusive": False},
                {"label": "moderate", "max": 0.6, "max_inclusive": False},
                {"label": "bright", "max": None},
            ],
            "severity_by_label": {
                "unknown": 0.0,
                "dark": 0.05,
                "dim": 0.2,
                "moderate": 0.55,
                "bright": 0.9,
            },
        },
        "breathing_status": {
            "sensor_attr": "breathing_rate",
            "unknown_label": "unknown",
            "ranges": [
                {"label": "very_slow", "max": 10, "max_inclusive": False},
                {"label": "normal", "max": 16, "max_inclusive": True},
                {"label": "slightly_elevated", "max": 20, "max_inclusive": True},
                {"label": "elevated", "max": None},
            ],
            "severity_by_label": {
                "unknown": 0.0,
                "very_slow": 0.8,
                "normal": 0.1,
                "slightly_elevated": 0.55,
                "elevated": 0.9,
            },
        },
    },
    "hypothesis_signals": [
        {"signal": "noise_alert", "feature": "noise_alert"},
        {"signal": "movement_spike", "feature": "movement_level"},
        {"signal": "hr_elevated", "feature": "hr_status"},
        {"signal": "hrv_recovery", "feature": "hrv_quality"},
        {"signal": "light_exposure", "feature": "light_status"},
        {"signal": "breathing_irregularity", "feature": "breathing_status"},
    ],
}

# Hook points for future personalization. Keep defaults equivalent to the
# historic behavior unless specific profile overrides are added.
SIGNAL_PROFILE_OVERRIDES = {
    "default": {},
    "aggressiveness:low": {},
    "aggressiveness:medium": {},
    "aggressiveness:high": {},
}


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

    profile_key = _resolve_profile_key(state.preferences.intervention_aggressiveness)
    summary = extract_features(sensor, profile_key=profile_key)
    signal_hypotheses = _build_signal_hypotheses(summary, sensor, profile_key=profile_key)
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

def extract_features(sensor: SensorSnapshot, profile_key: str = "default") -> dict:
    """
    Produce a structured feature summary from a SensorSnapshot.

    Returns a dict with up to six labelled signals.  All fields are always
    present (using "unknown" when the sensor value is missing) so downstream
    agents can rely on a consistent schema.
    """
    profile = _profile_config(profile_key)
    return {
        feature_key: _label_for_feature(sensor, feature_spec)
        for feature_key, feature_spec in profile["features"].items()
    }


def _build_signal_hypotheses(
    summary: dict,
    sensor: SensorSnapshot,
    profile_key: str = "default",
) -> list[dict]:
    """Build a consistent per-signal hypothesis payload for the current tick."""
    evidence = {
        "heart_rate": sensor.heart_rate,
        "hrv": sensor.hrv,
        "movement": sensor.movement,
        "noise_db": sensor.noise_db,
        "light_level": sensor.light_level,
        "breathing_rate": sensor.breathing_rate,
    }

    profile = _profile_config(profile_key)

    return [
        {
            "signal": signal_spec["signal"],
            "severity": _severity_for_feature_label(
                profile,
                signal_spec["feature"],
                summary[signal_spec["feature"]],
            ),
            "label": label,
            "evidence": evidence,
            "tick_timestamp": sensor.timestamp,
            "tick_source": TICK_SOURCE_MARKER,
        }
        for signal_spec in profile["hypothesis_signals"]
        for label in [summary[signal_spec["feature"]]]
    ]


def _resolve_profile_key(aggressiveness: str | None) -> str:
    if aggressiveness in {"low", "medium", "high"}:
        return f"aggressiveness:{aggressiveness}"
    return "default"


@lru_cache(maxsize=None)
def _profile_config(profile_key: str) -> dict:
    merged = deepcopy(BASE_SIGNAL_PROFILE)
    overrides = SIGNAL_PROFILE_OVERRIDES.get(profile_key, SIGNAL_PROFILE_OVERRIDES["default"])
    _deep_update(merged, overrides)
    return merged


def _deep_update(base: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _label_for_feature(sensor: SensorSnapshot, feature_spec: dict) -> str:
    value = getattr(sensor, feature_spec["sensor_attr"])
    if value is None:
        return feature_spec.get("unknown_label", "unknown")
    return _label_from_ranges(value, feature_spec["ranges"], feature_spec["unknown_label"])


def _label_from_ranges(value: float, ranges: list[dict], fallback_label: str) -> str:
    for range_spec in ranges:
        if _matches_range(value, range_spec):
            return range_spec["label"]
    return fallback_label


def _matches_range(value: float, range_spec: dict) -> bool:
    min_value = range_spec.get("min")
    max_value = range_spec.get("max")
    min_inclusive = range_spec.get("min_inclusive", True)
    max_inclusive = range_spec.get("max_inclusive", False)

    if min_value is not None:
        if min_inclusive and value < min_value:
            return False
        if not min_inclusive and value <= min_value:
            return False

    if max_value is not None:
        if max_inclusive and value > max_value:
            return False
        if not max_inclusive and value >= max_value:
            return False

    return True


def _severity_for_feature_label(profile: dict, feature_key: str, label: str) -> float:
    return profile["features"][feature_key]["severity_by_label"].get(label, 0.0)
