"""
Intervention Agent
------------------
First deterministic pass: decide whether to start, continue, or stop an audio
or pacing intervention based on the current sleep state.

Decision logic:
  1. If wake_risk is low and no disturbance is flagged → clear intervention.
  2. Otherwise select the best compatible type from nightly_plan's ordered list,
     skipping anything in preferences.disliked_audio.
  3. Scale intensity by wake_risk × aggressiveness so lighter risks get softer
     sounds rather than jumping straight to max volume.

Phase-aware selection:
  deep  → prefer low-frequency masking (brown/pink noise); avoid stimulating sounds
  rem   → breathing_pace can help re-anchor rhythm without fully waking
  light → any masking type is acceptable
  awake → wake_ramp if near target wake time; masking otherwise

LLM integration plan: the rationale string will become a structured context
block fed to the LLM, which will produce a more nuanced natural-language
explanation and potentially override the type choice with better reasoning.
"""

from app.models.userstate import SharedState, ActiveIntervention


# Maximum base intensity per aggressiveness level.
_BASE_INTENSITY = {"low": 0.35, "medium": 0.60, "high": 0.85}

# Intervention types that are considered stimulating — avoid during deep sleep.
_STIMULATING = {"white_noise", "wake_ramp"}

# Threshold below which we consider wake_risk low enough to not intervene.
_RISK_THRESHOLD = 0.38


def run(state: SharedState) -> SharedState:
    """
    Select and activate the most appropriate intervention given current state.

    Args:
        state: SharedState with sleep_state, nightly_plan, and preferences set.

    Returns:
        Updated SharedState with active_intervention populated.
    """
    sleep = state.sleep_state
    prefs = state.preferences
    plan  = state.nightly_plan

    risk_is_low    = sleep.wake_risk < _RISK_THRESHOLD
    no_disturbance = sleep.disturbance_reason is None

    if risk_is_low and no_disturbance:
        # Sleep is stable — clear any running intervention.
        if state.active_intervention.type != "none":
            return state.model_copy(update={"active_intervention": ActiveIntervention()})
        return state

    # Build a candidate list: plan's preferred order, minus disliked types.
    disliked = set(prefs.disliked_audio)
    candidates = [t for t in plan.preferred_intervention_order if t not in disliked]

    # Fall back to safe default if user preferences wiped out all options.
    if not candidates:
        candidates = ["brown_noise"]

    chosen_type = _pick_type(candidates, sleep.phase)
    intensity   = _compute_intensity(sleep.wake_risk, prefs.intervention_aggressiveness)

    rationale = (
        f"wake_risk={sleep.wake_risk:.2f}; "
        f"phase={sleep.phase}; "
        f"disturbance={sleep.disturbance_reason or 'none'}; "
        f"aggressiveness={prefs.intervention_aggressiveness}"
    )

    new_intervention = ActiveIntervention(
        type=chosen_type,
        intensity=intensity,
        rationale=rationale,
    )
    return state.model_copy(update={"active_intervention": new_intervention})


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _pick_type(candidates: list[str], phase: str) -> str:
    """
    Choose the most appropriate intervention type for the current sleep phase.

    Args:
        candidates: Ordered list of allowed types (disliked already removed).
        phase:      Current sleep phase from sleep_state.

    Returns:
        The chosen intervention type string.
    """
    # Deep sleep: avoid stimulating sounds; prefer low-frequency masking.
    if phase == "deep":
        for t in candidates:
            if t not in _STIMULATING and t in ("brown_noise", "pink_noise"):
                return t
        # Accept any non-stimulating type.
        for t in candidates:
            if t not in _STIMULATING:
                return t

    # REM: breathing_pace first, then gentle masking.
    if phase == "rem":
        for t in candidates:
            if t == "breathing_pace":
                return t
        for t in candidates:
            if t not in _STIMULATING:
                return t

    # Light / awake / unknown: first non-stimulating candidate is fine.
    for t in candidates:
        if t not in _STIMULATING:
            return t

    # If everything left is stimulating, return the first candidate anyway.
    return candidates[0]


def _compute_intensity(wake_risk: float, aggressiveness: str) -> float:
    """
    Scale intensity: base from aggressiveness × wake_risk, so a low-risk
    event gets a whisper-level intervention rather than full volume.
    """
    base = _BASE_INTENSITY.get(aggressiveness, 0.60)
    # Minimum floor of 0.15 so the intervention is audible even at low risk.
    intensity = max(0.15, base * wake_risk)
    return round(intensity, 2)
