"""
Strategist Agent
----------------
Builds the NightlyPlan once, before sleep begins.

Two execution paths:

  run(state)       — deterministic, always available, no external calls.
                     Uses keyword matching on goals and direct preference mapping.
                     Fast, predictable, testable.

  run_gpt(state)   — GPT-backed, returns (SharedState, mode_used).
                     Sends the user's profile and recent journal to the LLM and
                     asks for a structured JSON plan.  If the call fails for any
                     reason (network error, bad JSON, validation failure, missing
                     API key) it falls back to the deterministic path and returns
                     mode_used="fallback" so callers know which path was taken.

Why only planning gets the LLM treatment
-----------------------------------------
The real-time tick pipeline (sensor_interpreter → sleep_state → disturbance →
intervention) runs every 30–60 s and must be fast and reliable.  Planning runs
once per night where a 2–5 s LLM latency is acceptable and the richer context
understanding of an LLM genuinely improves output quality.
"""

import logging

from app import llm_client
from app.models.userstate import NightlyPlan, SharedState

logger = logging.getLogger(__name__)

# Valid intervention types the model is allowed to include in its plan.
_VALID_INTERVENTION_TYPES = {
    "brown_noise", "white_noise", "pink_noise",
    "rain", "waves", "breathing_pace", "wake_ramp",
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(state: SharedState) -> SharedState:
    """
    Deterministic planner.  Always succeeds.  Used by the tick pipeline and as
    the fallback when GPT is unavailable.
    """
    plan = _build_plan_deterministic(state)
    return state.model_copy(update={"nightly_plan": plan})


def run_gpt(state: SharedState) -> tuple[SharedState, str]:
    """
    GPT-backed planner with automatic fallback.

    Returns:
        (updated_state, mode_used) where mode_used is "gpt" or "fallback".
    """
    try:
        plan = _build_plan_gpt(state)
        return state.model_copy(update={"nightly_plan": plan}), "gpt"
    except Exception as exc:
        logger.warning("strategist GPT path failed (%s), using fallback", exc)
        return run(state), "fallback"


# ---------------------------------------------------------------------------
# Deterministic planner
# ---------------------------------------------------------------------------

def _build_plan_deterministic(state: SharedState) -> NightlyPlan:
    prefs = state.preferences

    goal = "balanced_sleep"
    if prefs.goals:
        lower = [g.lower() for g in prefs.goals]
        if any("deep" in g for g in lower):
            goal = "deep_sleep_focus"
        elif any("rem" in g or "dream" in g for g in lower):
            goal = "rem_focus"
        elif any("awaken" in g or "interrupt" in g for g in lower):
            goal = "reduce_awakenings"

    intervention_order = list(prefs.preferred_audio) if prefs.preferred_audio else [
        "brown_noise", "breathing_pace", "rain"
    ]

    return NightlyPlan(
        target_bedtime="23:00",
        target_wake_time=prefs.target_wake_time,
        sleep_goal=goal,
        notes=f"Deterministic plan. Goals: {', '.join(prefs.goals) or 'none'}.",
        preferred_intervention_order=intervention_order,
    )


# ---------------------------------------------------------------------------
# GPT planner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a sleep-optimization assistant for an app called Somnus.
Your role is to create a practical nightly sleep plan based on the user's profile and recent sleep history.

Rules:
- Respond with a single JSON object. No prose, no markdown, no extra keys.
- Do not make medical diagnoses or health claims.
- Respect the user's audio preferences — never include disliked types in preferred_intervention_order.
- preferred_intervention_order must contain 2–4 items chosen from this exact set:
  brown_noise, white_noise, pink_noise, rain, waves, breathing_pace, wake_ramp
- sleep_goal must be one of: deep_sleep_focus, rem_focus, balanced_sleep, recovery, reduce_awakenings
- notes should be 2–3 practical sentences — no fluff.

Output schema (JSON only):
{
  "target_bedtime": "HH:MM",
  "target_wake_time": "HH:MM",
  "sleep_goal": "...",
  "notes": "...",
  "preferred_intervention_order": ["...", "..."]
}"""


def _build_plan_gpt(state: SharedState) -> NightlyPlan:
    """Call the LLM and parse the result into a NightlyPlan.  Raises on failure."""
    prefs  = state.preferences
    history = state.journal_history[-5:]  # last 5 nights is enough context

    journal_lines = []
    for e in history:
        journal_lines.append(
            f"  {e.date}: rested={e.rested_score}/10, "
            f"awakenings={e.awakenings_reported}, notes=\"{e.notes}\""
        )
    journal_summary = "\n".join(journal_lines) if journal_lines else "  No journal history yet."

    user_prompt = f"""\
User profile:
  Goals: {', '.join(prefs.goals) or 'none'}
  Preferred audio: {', '.join(prefs.preferred_audio) or 'none'}
  Disliked audio:  {', '.join(prefs.disliked_audio) or 'none'}
  Target wake time: {prefs.target_wake_time}
  Intervention aggressiveness: {prefs.intervention_aggressiveness}

Recent journal ({len(history)} nights):
{journal_summary}

Create tonight's sleep plan."""

    raw = llm_client.chat_json(_SYSTEM_PROMPT, user_prompt)

    # Sanitise intervention_order — reject anything not in the valid set.
    order = raw.get("preferred_intervention_order", [])
    disliked = set(prefs.disliked_audio)
    clean_order = [t for t in order if t in _VALID_INTERVENTION_TYPES and t not in disliked]
    if not clean_order:
        # If the model returned nothing usable, fall back to preferred_audio.
        clean_order = [t for t in prefs.preferred_audio if t in _VALID_INTERVENTION_TYPES]
    if not clean_order:
        clean_order = ["brown_noise", "rain"]
    raw["preferred_intervention_order"] = clean_order

    # Validate via Pydantic — raises ValidationError if required fields are missing.
    return NightlyPlan.model_validate(raw)
