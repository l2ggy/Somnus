"""
Sleep-State ML Model Wrapper
-----------------------------
Wraps a global trained sleep-state classifier with per-user baseline
normalization as the first personalization layer.

Global vs. per-user:
  The underlying model (sleep_model.pkl) is trained on population-level data
  and classifies sleep phase from a normalized feature vector.  Per-user
  normalization adjusts raw physiological values relative to that user's
  personal baselines (resting HR, baseline HRV) before inference, so the
  same model produces more accurate results for users whose physiology sits
  outside the training population's mean.

Future personalization:
  Per-user baseline normalization is intentionally the *first* layer.  Future
  versions may support user-specific fine-tuning (e.g. adapting model weights
  from a user's labelled nights) or a learned per-user feature scaler stored
  in the session database.  The predict() interface is designed to stay stable
  across both approaches.

Model file:
  Expected at models/sleep_model.pkl (relative to the working directory).
  If the file is absent the class falls back to a deterministic mock so the
  rest of the system can run safely without a trained artefact.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Phases the model recognises — must stay in sync with SleepState.phase in
# app/models/userstate.py (excluding "unknown" which is a sensor-gap sentinel).
SLEEP_PHASES: list[str] = ["awake", "light", "deep", "rem"]

_MODEL_PATH = Path("models/sleep_model.pkl")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model():
    """
    Attempt to load the trained classifier from disk using joblib.

    Returns the loaded model object on success, None if the file is missing
    or if joblib is unavailable (graceful degradation for environments that
    don't have scikit-learn installed).
    """
    if not _MODEL_PATH.exists():
        logger.info(
            "sleep_model.pkl not found at %s — using mock classifier",
            _MODEL_PATH.resolve(),
        )
        return None

    try:
        import joblib  # soft dependency — not required for deterministic path
        model = joblib.load(_MODEL_PATH)
        logger.info("Loaded sleep model from %s", _MODEL_PATH.resolve())
        return model
    except Exception as exc:
        logger.warning("Failed to load sleep model (%s) — using mock classifier", exc)
        return None


def _mock_predict_proba(features: dict) -> list[float]:
    """
    Deterministic mock classifier used when no trained model is available.

    Produces a plausible probability distribution from raw features so the
    rest of the system can exercise the full inference path during development
    and testing without requiring a trained artefact.

    Priority logic mirrors the heuristic rules in
    app/agents/intelligence/sleep_state.py so mock output is coherent with
    the existing deterministic pipeline.
    """
    movement  = features.get("movement", 0.0) or 0.0
    hr        = features.get("heart_rate", 65.0) or 65.0
    hrv       = features.get("hrv", 40.0) or 40.0
    noise     = features.get("noise", 0.0) or 0.0

    # Awake signals: high movement or elevated HR
    if movement > 0.55 or hr > 85:
        return [0.75, 0.15, 0.05, 0.05]  # awake, light, deep, rem

    # Deep sleep signals: low HR, good HRV, low movement
    if hr < 60 and hrv >= 55 and movement < 0.10:
        return [0.05, 0.10, 0.75, 0.10]

    # REM proxy: moderate HRV, minimal movement
    if hrv >= 50 and movement < 0.15:
        return [0.05, 0.20, 0.15, 0.60]

    # Default: light sleep, adjusted slightly by noise
    noise_penalty = min(0.15, noise / 500) if noise > 55 else 0.0
    light_prob    = max(0.35, 0.55 - noise_penalty)
    awake_prob    = min(0.45, 0.20 + noise_penalty)
    return [awake_prob, light_prob, 0.15, 0.10]


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

def get_user_baseline(user_id: str) -> dict:  # noqa: ARG001
    """
    Return the baseline physiological profile for a user.

    Currently returns a population-average default for all users.  A future
    version will load stored baselines from the session database, keyed by
    user_id, and update them incrementally from nightly sensor data.

    Args:
        user_id: The user identifier (reserved for future per-user lookup).

    Returns:
        Dict with baseline fields used by normalize_for_user().
    """
    return {
        "resting_hr":    60.0,
        "baseline_hrv":  40.0,
    }


def normalize_for_user(features: dict, user_baseline: dict) -> dict:
    """
    Adjust physiological features relative to a user's personal baseline.

    Only fields that carry meaningful personalization signal are normalized.
    Sensor-physics fields (movement, noise, light) are left unchanged because
    they are already expressed in absolute units (0–1 or dB) that do not vary
    by individual physiology.

    Normalization scheme:
      hr_elevation  = heart_rate - resting_hr
                      (positive → above personal resting rate)
      hrv_ratio     = hrv / baseline_hrv
                      (> 1 → better than baseline recovery;
                       < 1 → worse than baseline)

    Args:
        features:       Raw feature dict (keys: heart_rate, hrv, movement,
                        noise, movement_severity, noise_severity, hr_elevation).
        user_baseline:  Dict from get_user_baseline() with resting_hr and
                        baseline_hrv.

    Returns:
        New dict with all original keys preserved and two normalized fields
        added/overwritten: hr_elevation, hrv_ratio.
    """
    normalized = dict(features)

    resting_hr   = user_baseline.get("resting_hr",   60.0)
    baseline_hrv = user_baseline.get("baseline_hrv", 40.0)

    if "heart_rate" in features and features["heart_rate"] is not None:
        normalized["hr_elevation"] = features["heart_rate"] - resting_hr

    if "hrv" in features and features["hrv"] is not None and baseline_hrv > 0:
        normalized["hrv_ratio"] = features["hrv"] / baseline_hrv

    return normalized


# ---------------------------------------------------------------------------
# SleepModel
# ---------------------------------------------------------------------------

class SleepModel:
    """
    Global sleep-state classifier with per-user baseline normalization.

    A single instance of this class is intended to be shared across all
    requests (module-level singleton pattern).  The trained model weights are
    loaded once at construction time.

    Attributes:
        _model: The loaded sklearn-compatible classifier, or None if the model
                file was absent at startup (triggers mock path).
    """

    def __init__(self) -> None:
        self._model = _load_model()

    @property
    def is_trained(self) -> bool:
        """True if a real trained model was loaded; False if using the mock."""
        return self._model is not None

    def predict(self, features: dict, user_id: str) -> dict:
        """
        Predict sleep phase probabilities for a single feature vector.

        Steps:
          1. Fetch the user's physiological baseline.
          2. Normalize features against that baseline (personalization layer 1).
          3. Run inference: trained model or deterministic mock.
          4. Return a structured result with phase, confidence, and the full
             probability distribution across all four sleep phases.

        Args:
            features: Dict of sensor-derived features.  Recognised keys:
                        heart_rate, hrv, movement, noise,
                        movement_severity, noise_severity, hr_elevation.
                      Unknown keys are passed through unchanged — the model
                      (or mock) ignores what it doesn't need.
            user_id:  Used to look up the per-user physiological baseline.

        Returns:
            {
                "phase":         str,    # winning phase label
                "confidence":    float,  # probability of the winning phase
                "probabilities": {       # full distribution
                    "awake": float,
                    "light": float,
                    "deep":  float,
                    "rem":   float,
                }
            }
        """
        baseline   = get_user_baseline(user_id)
        normalized = normalize_for_user(features, baseline)

        proba = self._run_inference(normalized)

        # Pair each phase with its probability and find the winner.
        phase_probs   = dict(zip(SLEEP_PHASES, proba))
        winning_phase = max(phase_probs, key=phase_probs.__getitem__)
        confidence    = round(phase_probs[winning_phase], 4)

        return {
            "phase":         winning_phase,
            "confidence":    confidence,
            "probabilities": {p: round(v, 4) for p, v in phase_probs.items()},
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_inference(self, features: dict) -> list[float]:
        """
        Run the classifier and return a probability list aligned to
        SLEEP_PHASES order: [awake, light, deep, rem].
        """
        if self._model is None:
            return _mock_predict_proba(features)

        try:
            feature_vector = self._build_feature_vector(features)
            proba = self._model.predict_proba([feature_vector])[0]
            return list(proba)
        except Exception as exc:
            logger.warning(
                "Model inference failed (%s) — falling back to mock", exc
            )
            return _mock_predict_proba(features)

    def _build_feature_vector(self, features: dict) -> list[float]:
        """
        Convert the features dict into an ordered list matching the column
        order the trained model was fitted on.

        The ordering here must stay in sync with the training pipeline that
        produced sleep_model.pkl.  If columns diverge the model will silently
        produce wrong results — a future version should load the column order
        from the pkl artefact itself.
        """
        ordered_keys = [
            "heart_rate",
            "hrv",
            "movement",
            "noise",
            "movement_severity",
            "noise_severity",
            "hr_elevation",
            "hrv_ratio",
        ]
        return [float(features.get(k) or 0.0) for k in ordered_keys]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Instantiated once at import time so the model file is loaded a single time
# per process.  Import this object rather than constructing SleepModel() in
# request handlers.
sleep_model = SleepModel()
