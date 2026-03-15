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

Candidate filtering precedence (lowest → highest priority, each layer overrides):
  1. nightly_plan.preferred_intervention_order  (strategist/GPT guidance)
  2. preferences.disliked_audio                 (user stated preferences)
  3. nightly_plan.blocked_interventions         (outer-loop policy — prior night scores)
  4. _PHASE_BLOCKLIST                           (safety — phase-appropriate constraints)

Outer-loop integration:
  nightly_plan.blocked_interventions is populated by apply_policy_to_state()
  at session start when a UserPolicy exists.  Interventions with a cross-night
  score ≤ -0.50 are placed in this list.  The agent removes them from the
  candidate pool and records the suppression in the rationale so it is visible
  in the tick trace.
"""

import logging

from app.models.userstate import SharedState, ActiveIntervention

logger = logging.getLogger(__name__)


# Maximum base intensity per aggressiveness level.
_BASE_INTENSITY = {"low": 0.35, "medium": 0.60, "high": 0.85}

# Hard cap for sustained interventions by aggressiveness profile.
_CONTINUOUS_INTENSITY_CAP = {"low": 0.40, "medium": 0.65, "high": 0.82}

# Intervention types that are considered stimulating — avoid during deep sleep.
_STIMULATING = {"white_noise", "wake_ramp"}

# Phase hard-block list. Stimulating types are forbidden in deep sleep unless
# explicit wake-transition criteria are met.
_PHASE_BLOCKLIST = {
    "deep": _STIMULATING,
    "rem": {"wake_ramp"},
}

# Threshold below which we consider wake_risk low enough to not intervene.
_RISK_THRESHOLD = 0.38

# Number of ticks to hold a chosen type before allowing a switch.
_TYPE_CHANGE_COOLDOWN_TICKS = 2

# Maximum per-tick intensity movement, to avoid abrupt jumps.
_MAX_INTENSITY_DELTA = 0.12

_META_PREFIX = "meta:"


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

    # Build a candidate list.
    # Precedence (each layer adds exclusions on top of the previous):
    #   1. plan.preferred_intervention_order  — strategist ranking
    #   2. prefs.disliked_audio               — user stated preferences
    #   3. plan.blocked_interventions         — outer-loop policy (prior night scores)
    disliked = set(prefs.disliked_audio)
    policy_blocked = set(getattr(plan, "blocked_interventions", []))

    candidates = [
        t for t in plan.preferred_intervention_order
        if t not in disliked and t not in policy_blocked
    ]

    # Log when policy blocking actually changed what would have been chosen.
    if policy_blocked:
        would_have_been = [t for t in plan.preferred_intervention_order if t not in disliked]
        suppressed = [t for t in would_have_been if t in policy_blocked]
        if suppressed:
            logger.info(
                "intervention: policy blocked %s for user (outer_loop score ≤ -0.50); "
                "remaining candidates: %s",
                suppressed, candidates,
            )

    # Fall back to safe default if all options were excluded.
    if not candidates:
        logger.warning(
            "intervention: all candidates excluded (disliked=%s, policy_blocked=%s); "
            "falling back to brown_noise",
            list(disliked), list(policy_blocked),
        )
        candidates = ["brown_noise"]

    previous = state.active_intervention
    prior_meta = _parse_rationale_meta(previous.rationale)
    safety_notes: list[str] = []

    chosen_type = _pick_type(candidates, sleep.phase)
    chosen_type = _apply_phase_blocklist(
        chosen_type=chosen_type,
        candidates=candidates,
        phase=sleep.phase,
        sleep_state=state.sleep_state,
        safety_notes=safety_notes,
    )

    chosen_type, cooldown_remaining = _apply_type_change_cooldown(
        chosen_type=chosen_type,
        previous_type=previous.type,
        previous_cooldown=prior_meta.get("type_change_cooldown", 0),
        safety_notes=safety_notes,
    )

    target_intensity = _compute_intensity(sleep.wake_risk, prefs.intervention_aggressiveness)
    intensity = _apply_intensity_safety_bounds(
        target_intensity=target_intensity,
        previous=previous,
        aggressiveness=prefs.intervention_aggressiveness,
        safety_notes=safety_notes,
    )
    if chosen_type == "none":
        if intensity != 0.0:
            safety_notes.append(f"type_none_force_zero_intensity:{intensity:.2f}->0.00")
        intensity = 0.0

    rationale_parts = [
        f"wake_risk={sleep.wake_risk:.2f}; "
        f"phase={sleep.phase}; "
        f"disturbance={sleep.disturbance_reason or 'none'}; "
        f"aggressiveness={prefs.intervention_aggressiveness}"
    ]
    if safety_notes:
        rationale_parts.append("safety=" + "|".join(safety_notes))
    # Surface outer-loop policy guidance in the rationale for traceability.
    if policy_blocked:
        rationale_parts.append(f"policy_blocked={','.join(sorted(policy_blocked))}")
    rationale_parts.append(f"{_META_PREFIX}type_change_cooldown={cooldown_remaining}")
    rationale = "; ".join(rationale_parts)

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


def _parse_rationale_meta(rationale: str | None) -> dict[str, int]:
    """Parse machine-friendly metadata tokens from intervention rationale."""
    if not rationale:
        return {}
    parsed: dict[str, int] = {}
    parts = [p.strip() for p in rationale.split(";")]
    for part in parts:
        if not part.startswith(_META_PREFIX):
            continue
        payload = part[len(_META_PREFIX):]
        key, _, value = payload.partition("=")
        if not key or not value:
            continue
        try:
            parsed[key] = int(value)
        except ValueError:
            continue
    return parsed


def _wake_transition_allowed(phase: str, wake_risk: float, disturbance_reason: str | None) -> bool:
    """Allow stimulating types only when a wake transition is clearly intended."""
    if phase == "awake":
        return True
    if wake_risk >= 0.80:
        return True
    if disturbance_reason and disturbance_reason.lower() in {
        "wake_window",
        "target_wake",
        "wake_time_reached",
        "alarm",
    }:
        return True
    return False


def _apply_phase_blocklist(
    chosen_type: str,
    candidates: list[str],
    phase: str,
    sleep_state,
    safety_notes: list[str],
) -> str:
    """Enforce phase-based type blocklist with wake-transition bypass rules."""
    blocked = _PHASE_BLOCKLIST.get(phase, set())
    if chosen_type not in blocked:
        return chosen_type

    if _wake_transition_allowed(phase, sleep_state.wake_risk, sleep_state.disturbance_reason):
        safety_notes.append("phase_block_bypassed:wake_transition")
        return chosen_type

    for t in candidates:
        if t not in blocked:
            safety_notes.append(f"phase_block:{phase}:{chosen_type}->{t}")
            return t

    safety_notes.append(f"phase_block_no_safe_alternative:{phase}:{chosen_type}")
    return "none"


def _apply_type_change_cooldown(
    chosen_type: str,
    previous_type: str,
    previous_cooldown: int,
    safety_notes: list[str],
) -> tuple[str, int]:
    """Prevent tick-to-tick type flip-flopping by holding recent choices."""
    if previous_type == "none":
        if chosen_type == "none":
            return chosen_type, 0
        return chosen_type, _TYPE_CHANGE_COOLDOWN_TICKS

    if chosen_type == previous_type:
        return chosen_type, max(previous_cooldown - 1, 0)

    if previous_cooldown > 0:
        safety_notes.append(
            f"type_change_cooldown_hold:{previous_type}({previous_cooldown})->{chosen_type}"
        )
        return previous_type, previous_cooldown - 1

    return chosen_type, _TYPE_CHANGE_COOLDOWN_TICKS


def _apply_intensity_safety_bounds(
    target_intensity: float,
    previous: ActiveIntervention,
    aggressiveness: str,
    safety_notes: list[str],
) -> float:
    """Apply ramp-rate and continuous-cap safety controls to intensity."""
    intensity = target_intensity

    if previous.type != "none":
        max_up = previous.intensity + _MAX_INTENSITY_DELTA
        max_down = max(0.0, previous.intensity - _MAX_INTENSITY_DELTA)
        bounded = min(max(intensity, max_down), max_up)
        if bounded != intensity:
            safety_notes.append(f"intensity_ramp_limited:{intensity:.2f}->{bounded:.2f}")
        intensity = bounded

    cap = _CONTINUOUS_INTENSITY_CAP.get(aggressiveness, 0.65)
    if intensity > cap:
        safety_notes.append(f"continuous_intensity_cap:{intensity:.2f}->{cap:.2f}")
        intensity = cap

    return round(intensity, 2)
