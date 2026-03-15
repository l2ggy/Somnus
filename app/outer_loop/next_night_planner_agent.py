"""
NextNightPlannerAgent
---------------------
Generates the NightlyPlan for the upcoming sleep session using the accumulated
UserPolicy and last night's NightSummary.

This is the "write-back" step of the outer loop: it produces the NightlyPlan
that will be loaded by the inner loop's strategist and intervention agents
when the next session starts.

Two execution paths:

  run(policy, summary, prefs)      — deterministic.  Derives plan from policy
                                     scores and risk posture.  Fast and safe.

  run_gpt(policy, summary, prefs)  — LLM builds a richer plan with context-
                                     aware reasoning.  Falls back on failure.

The output is a NightlyPlan (existing model) so it can be injected directly
into SharedState without any adapter layer.

Integration note:
  When the user calls POST /session/start the next evening, if an outer-loop
  next-night plan exists for that user it should be passed as the initial
  nightly_plan.  The run.py orchestrator handles this; the strategist will
  refine it further if ?mode=gpt is requested.
"""

import logging

from app import llm_client
from app.models.outer_loop import NightSummary, UserPolicy
from app.models.userstate import NightlyPlan, UserPreferences

logger = logging.getLogger(__name__)

_VALID_TYPES = {
    "brown_noise", "white_noise", "pink_noise",
    "rain", "waves", "breathing_pace", "wake_ramp",
}
_DEFAULT_ORDER = ["brown_noise", "breathing_pace", "rain"]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(
    policy: UserPolicy,
    summary: NightSummary,
    prefs: UserPreferences,
) -> tuple[NightlyPlan, str]:
    """
    Deterministic next-night plan.

    Returns:
        (NightlyPlan, mode_used) where mode_used is "deterministic".
    """
    plan = _build_deterministic(policy, summary, prefs)
    return plan, "deterministic"


def run_gpt(
    policy: UserPolicy,
    summary: NightSummary,
    prefs: UserPreferences,
) -> tuple[NightlyPlan, str]:
    """
    GPT-backed next-night plan with automatic fallback.

    Returns:
        (NightlyPlan, mode_used) where mode_used is "gpt" or "fallback".
    """
    try:
        plan = _build_gpt(policy, summary, prefs)
        return plan, "gpt"
    except Exception as exc:
        logger.warning("NextNightPlannerAgent GPT path failed (%s), using fallback", exc)
        plan, _ = run(policy, summary, prefs)
        return plan, "fallback"


# ---------------------------------------------------------------------------
# Deterministic planner
# ---------------------------------------------------------------------------

def _build_deterministic(
    policy: UserPolicy,
    summary: NightSummary,
    prefs: UserPreferences,
) -> NightlyPlan:
    disliked = set(prefs.disliked_audio)

    # Derive intervention order from policy scores, filtered by preferences
    order = _resolve_intervention_order(policy, prefs, disliked)

    # Derive sleep goal from risk posture + user goals
    goal = _resolve_sleep_goal(policy.risk_posture, prefs)

    # Notes surface key policy context for debugging / display
    notes = (
        f"Outer-loop plan. Risk posture: {policy.risk_posture}. "
        f"Nights tracked: {policy.nights_tracked}. "
        f"{policy.policy_notes}"
    )

    return NightlyPlan(
        target_bedtime="23:00",
        target_wake_time=prefs.target_wake_time,
        sleep_goal=goal,
        notes=notes,
        preferred_intervention_order=order,
    )


def _resolve_intervention_order(
    policy: UserPolicy,
    prefs: UserPreferences,
    disliked: set[str],
) -> list[str]:
    # Start from policy's ranked order
    policy_order = [t for t in policy.preferred_intervention_order
                    if t in _VALID_TYPES and t not in disliked]
    if policy_order:
        return policy_order[:4]

    # Fall back to user preferences
    pref_order = [t for t in prefs.preferred_audio
                  if t in _VALID_TYPES and t not in disliked]
    if pref_order:
        return pref_order[:4]

    return [t for t in _DEFAULT_ORDER if t not in disliked] or _DEFAULT_ORDER


def _resolve_sleep_goal(risk_posture: str, prefs: UserPreferences) -> str:
    if risk_posture == "protective":
        return "reduce_awakenings"

    # Check explicit user goals
    lower_goals = [g.lower() for g in prefs.goals]
    if any("deep" in g for g in lower_goals):
        return "deep_sleep_focus"
    if any("rem" in g or "dream" in g for g in lower_goals):
        return "rem_focus"
    if any("awaken" in g or "interrupt" in g for g in lower_goals):
        return "reduce_awakenings"

    if risk_posture == "exploratory":
        return "recovery"

    return "balanced_sleep"


# ---------------------------------------------------------------------------
# GPT planner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a personalized sleep planning assistant for Somnus.
Based on a user's accumulated sleep policy and last night's session summary,
create a practical plan for tonight's sleep session.

Rules:
- Respond with a single JSON object. No prose, no markdown, no extra keys.
- Do not make medical claims.
- Respect disliked_audio — never include those types.
- preferred_intervention_order must have 2–4 items from:
  brown_noise, white_noise, pink_noise, rain, waves, breathing_pace, wake_ramp
- sleep_goal must be one of:
  deep_sleep_focus, rem_focus, balanced_sleep, recovery, reduce_awakenings
- notes must be 2–3 practical sentences grounded in the policy data.

Output schema (JSON only):
{
  "target_bedtime": "HH:MM",
  "target_wake_time": "HH:MM",
  "sleep_goal": "...",
  "notes": "...",
  "preferred_intervention_order": ["...", "..."]
}"""


def _build_gpt(
    policy: UserPolicy,
    summary: NightSummary,
    prefs: UserPreferences,
) -> NightlyPlan:
    disliked = set(prefs.disliked_audio)

    # Format policy scores for the prompt
    score_lines = [
        f"  {iv}: {score:+.2f}"
        for iv, score in sorted(policy.intervention_scores.items(), key=lambda x: -x[1])
    ]
    scores_text = "\n".join(score_lines) if score_lines else "  none"

    # Format last night summary
    iv_text = ", ".join(iv.type for iv in summary.interventions_used) or "none"
    disturbance_text = ", ".join(summary.disturbances_detected) or "none"

    user_prompt = f"""\
User profile:
  Goals: {', '.join(prefs.goals) or 'none'}
  Disliked audio: {', '.join(disliked) or 'none'}
  Target wake time: {prefs.target_wake_time}
  Intervention aggressiveness: {prefs.intervention_aggressiveness}

Accumulated policy (after {policy.nights_tracked} nights):
  Risk posture: {policy.risk_posture}
  Avg rested score: {policy.avg_rested_score or 'n/a'}
  Policy notes: {policy.policy_notes}

Intervention preference scores (higher = more preferred):
{scores_text}

Last night summary ({summary.date}):
  Total ticks: {summary.total_ticks}
  Phases observed: {', '.join(summary.phases_observed) or 'unknown'}
  Disturbances: {disturbance_text}
  Interventions used: {iv_text}
  Peak wake-risk: {summary.peak_wake_risk:.2f}

Create tonight's sleep plan informed by the policy."""

    raw = llm_client.chat_json(_SYSTEM_PROMPT, user_prompt)

    # Sanitise and validate
    order = raw.get("preferred_intervention_order", [])
    clean_order = [t for t in order if t in _VALID_TYPES and t not in disliked]
    if not clean_order:
        clean_order = _resolve_intervention_order(policy, prefs, disliked)
    raw["preferred_intervention_order"] = clean_order[:4]

    return NightlyPlan.model_validate(raw)
