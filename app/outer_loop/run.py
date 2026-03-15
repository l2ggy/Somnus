"""
Outer loop orchestrator.

The outer loop runs once at the end of a sleep session (or when the user
submits morning feedback).  It chains the four outer-loop agents in order
and persists the results.

Agent pipeline:
  SharedState + optional MorningFeedback
    → NightSummaryAgent       → NightSummary
    → InterventionReviewAgent → InterventionReview
    → PolicyUpdateAgent       → UserPolicy (updated)
    → NextNightPlannerAgent   → NightlyPlan

All agents have deterministic fallbacks.  The orchestrator never lets an
agent failure propagate — it catches, logs, and continues with a degraded
result.

Public interface:
  run_outer_loop(state, feedback, mode)  →  OuterLoopReport
  apply_policy_to_state(state, policy)   →  SharedState   ← runtime injection
  get_user_policy(user_id)              →  UserPolicy
"""

import logging

from app.models.outer_loop import (
    DecisionTrace,
    InterventionReview,
    MorningFeedback,
    NightSummary,
    OuterLoopReport,
    ScoreChange,
    UserPolicy,
)
from app.models.userstate import NightlyPlan, SharedState
from app.outer_loop import (
    intervention_review_agent,
    next_night_planner_agent,
    night_summary_agent,
    policy_update_agent,
    store as outer_store,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_outer_loop(
    state: SharedState,
    feedback: MorningFeedback | None = None,
    mode: str = "deterministic",
) -> OuterLoopReport:
    """
    Run the full outer loop pipeline end-to-end.

    Args:
        state:    SharedState from the completed sleep session.
        feedback: Optional morning feedback from the user.
        mode:     "deterministic" (default) or "gpt".
                  In GPT mode, NightSummaryAgent and InterventionReviewAgent
                  use the LLM for narrative/review_notes.  NextNightPlannerAgent
                  always uses GPT in this mode.
                  Falls back gracefully on any LLM failure.

    Returns:
        OuterLoopReport with all four agent outputs plus persisted updates.
    """
    use_gpt = mode == "gpt"
    modes_used: list[str] = []

    # ---- Step 1: Night Summary -------------------------------------------
    summary, summary_mode = _run_summary(state, feedback, use_gpt)
    modes_used.append(f"summary:{summary_mode}")

    # ---- Step 2: Intervention Review ------------------------------------
    review, review_mode = _run_review(summary, state.hypotheses, feedback, use_gpt)
    modes_used.append(f"review:{review_mode}")

    # ---- Step 3: Policy Update ------------------------------------------
    # Hold a reference to the pre-update policy so the trace can show deltas.
    pre_update_policy = outer_store.load_or_default_policy(state.user_id)
    updated_policy, policy_mode = _run_policy_update(pre_update_policy, review, feedback, use_gpt)
    outer_store.save_policy(updated_policy)
    modes_used.append(f"policy:{policy_mode}")

    # ---- Step 4: Next Night Plan ----------------------------------------
    next_plan, plan_mode = _run_next_plan(updated_policy, summary, state, use_gpt)
    modes_used.append(f"plan:{plan_mode}")

    # ---- Persist night report -------------------------------------------
    outer_store.save_night_report(state.user_id, summary.date, summary, review)

    # ---- Build decision trace -------------------------------------------
    step_modes = {
        "summary": modes_used[0].split(":")[1],
        "review":  modes_used[1].split(":")[1],
        "policy":  modes_used[2].split(":")[1],
        "plan":    modes_used[3].split(":")[1],
    }
    trace = _build_decision_trace(
        pre_policy=pre_update_policy,
        post_policy=updated_policy,
        review=review,
        next_plan_dict=next_plan.model_dump(),
        step_modes=step_modes,
    )
    _log_decision_trace(state.user_id, trace)

    # ---- Build report ---------------------------------------------------
    combined_mode = _summarise_mode(modes_used)

    return OuterLoopReport(
        user_id=state.user_id,
        date=summary.date,
        night_summary=summary,
        intervention_review=review,
        user_policy=updated_policy,
        next_night_plan=next_plan.model_dump(),
        mode_used=combined_mode,
        decision_trace=trace,
    )


def get_user_policy(user_id: str) -> UserPolicy:
    """Return the current UserPolicy for a user, or a fresh default."""
    return outer_store.load_or_default_policy(user_id)


# ---------------------------------------------------------------------------
# Runtime injection — called at session start to seed the next night
# ---------------------------------------------------------------------------

# Score thresholds that map policy scores to runtime enforcement levels.
_BLOCK_THRESHOLD       = -0.50   # Score ≤ this  → blocked entirely from the night
_DEPRIORITIZE_THRESHOLD = -0.10  # Score ≤ this  → moved to the end of candidate list
_PREFER_THRESHOLD       =  0.10  # Score ≥ this  → moved to the front of candidate list


def apply_policy_to_state(state: SharedState, policy: UserPolicy) -> SharedState:
    """
    Apply the stored UserPolicy to the session's NightlyPlan before the night begins.

    Called at session start, AFTER the strategist has built the initial plan.
    This function re-ranks the plan's preferred_intervention_order and populates
    blocked_interventions based on accumulated cross-night policy scores.

    Precedence order enforced here (lowest precedence first, overrides stack up):
      1. Strategist/GPT plan order          (already in nightly_plan)
      2. Policy-preferred types             (moved to front, score ≥ +0.10)
      3. Policy-deprioritized types         (moved to back,  score ≤ -0.10)
      4. Policy-blocked types               (removed from order, added to blocked list)

    The intervention agent then enforces its own layer on top at runtime:
      5. Disliked audio (user prefs)        — handled in intervention.py (existing)
      6. Phase blocklist (safety)           — handled in intervention.py (existing)

    If no policy exists (nights_tracked == 0) this function is a no-op.

    Args:
        state:  SharedState after strategist has set nightly_plan.
        policy: UserPolicy loaded from outer_store (may be a zero-state default).

    Returns:
        Updated SharedState with nightly_plan modified by policy guidance.
        Returns state unchanged if policy has no nights tracked.
    """
    if policy.nights_tracked == 0:
        logger.debug(
            "apply_policy_to_state: no prior nights for user=%s, skipping injection",
            state.user_id,
        )
        return state

    scores = policy.intervention_scores
    current_order = list(state.nightly_plan.preferred_intervention_order)

    # ---- Categorise by score threshold ------------------------------------
    blocked: list[str] = []
    deprioritized: list[str] = []
    preferred_front: list[str] = []
    neutral: list[str] = []

    for iv in current_order:
        score = scores.get(iv, 0.0)
        if score <= _BLOCK_THRESHOLD:
            blocked.append(iv)
        elif score <= _DEPRIORITIZE_THRESHOLD:
            deprioritized.append(iv)
        elif score >= _PREFER_THRESHOLD:
            preferred_front.append(iv)
        else:
            neutral.append(iv)

    # Also check for any types in the policy that aren't in the plan order yet
    # but have strongly positive scores — candidates to inject.
    disliked = set(state.preferences.disliked_audio)
    for iv, score in sorted(scores.items(), key=lambda x: -x[1]):
        if score >= _PREFER_THRESHOLD and iv not in current_order and iv not in disliked:
            preferred_front.append(iv)

    # ---- Rebuild order: preferred → neutral → deprioritized ---------------
    # (blocked types are removed entirely)
    new_order = preferred_front + neutral + deprioritized

    # ---- Build guidance note for traceability ----------------------------
    changes: list[str] = []
    if blocked:
        changes.append(f"blocked={','.join(blocked)}")
    if preferred_front:
        changes.append(f"front={','.join(preferred_front)}")
    if deprioritized:
        changes.append(f"back={','.join(deprioritized)}")

    if changes:
        note = (
            f"outer_loop policy applied (nights={policy.nights_tracked}, "
            f"posture={policy.risk_posture}): " + "; ".join(changes)
        )
        logger.info(
            "apply_policy_to_state user=%s: %s | old_order=%s new_order=%s blocked=%s",
            state.user_id, "; ".join(changes), current_order, new_order, blocked,
        )
    else:
        note = (
            f"outer_loop policy loaded (nights={policy.nights_tracked}), "
            "no reordering required"
        )
        logger.debug(
            "apply_policy_to_state user=%s: no changes to plan order", state.user_id
        )

    updated_plan = state.nightly_plan.model_copy(update={
        "preferred_intervention_order": new_order if new_order else current_order,
        "blocked_interventions": blocked,
        "policy_guidance_note": note,
    })
    return state.model_copy(update={"nightly_plan": updated_plan})


# ---------------------------------------------------------------------------
# Decision trace construction
# ---------------------------------------------------------------------------

def _build_decision_trace(
    pre_policy: UserPolicy,
    post_policy: UserPolicy,
    review: InterventionReview,
    next_plan_dict: dict,
    step_modes: dict[str, str],
) -> DecisionTrace:
    """
    Build a compact, self-contained trace of what the outer loop decided.

    Uses pre/post policy snapshots so callers can see exactly what changed
    without needing to diff two full UserPolicy objects themselves.
    """
    # ---- Score changes -------------------------------------------------------
    all_types = set(pre_policy.intervention_scores) | set(post_policy.intervention_scores)
    score_changes: dict[str, ScoreChange] = {}

    for iv in sorted(all_types):
        before = round(pre_policy.intervention_scores.get(iv, 0.0), 4)
        after  = round(post_policy.intervention_scores.get(iv, 0.0), 4)
        delta  = round(after - before, 4)
        if abs(delta) < 0.0001:
            source = "unchanged"
        elif iv in post_policy.gpt_score_adjustments:
            source = "gpt+deterministic"
        else:
            source = "deterministic"
        score_changes[iv] = ScoreChange(before=before, after=after, delta=delta, source=source)

    # ---- Tomorrow's guidance buckets ----------------------------------------
    blocked_tomorrow:     list[str] = []
    discouraged_tomorrow: list[str] = []
    preferred_tomorrow:   list[str] = []

    for iv, sc in score_changes.items():
        if sc.after <= _BLOCK_THRESHOLD:
            blocked_tomorrow.append(iv)
        elif sc.after <= _DEPRIORITIZE_THRESHOLD:
            discouraged_tomorrow.append(iv)
        elif sc.after >= _PREFER_THRESHOLD:
            preferred_tomorrow.append(iv)

    # Sort preferred by score descending for readability
    preferred_tomorrow.sort(key=lambda iv: -post_policy.intervention_scores.get(iv, 0.0))

    # ---- Compact verdicts ---------------------------------------------------
    verdicts = [
        {"type": v.type, "verdict": v.verdict, "confidence": v.confidence}
        for v in review.verdicts
    ]

    return DecisionTrace(
        night_quality=review.overall_night_quality,
        verdicts=verdicts,
        score_changes=score_changes,
        risk_posture_before=pre_policy.risk_posture,
        risk_posture_after=post_policy.risk_posture,
        blocked_tomorrow=sorted(blocked_tomorrow),
        discouraged_tomorrow=sorted(discouraged_tomorrow),
        preferred_tomorrow=preferred_tomorrow,
        pattern_hypothesis=post_policy.pattern_hypothesis,
        next_night_goal=next_plan_dict.get("sleep_goal", ""),
        next_night_order=next_plan_dict.get("preferred_intervention_order", []),
        next_night_notes=next_plan_dict.get("notes", ""),
        step_modes=step_modes,
    )


def _log_decision_trace(user_id: str, trace: DecisionTrace) -> None:
    """Emit structured log lines that make the outer loop easy to follow in any log viewer."""
    # What changed in scores
    changed = {iv: sc for iv, sc in trace.score_changes.items() if sc.source != "unchanged"}
    if changed:
        score_str = "  ".join(
            f"{iv}:{sc.before:+.3f}→{sc.after:+.3f}({sc.source})"
            for iv, sc in sorted(changed.items(), key=lambda x: abs(x[1].delta), reverse=True)
        )
        logger.info("outer_loop user=%s score_changes: %s", user_id, score_str)
    else:
        logger.info("outer_loop user=%s score_changes: none (no interventions reviewed)", user_id)

    # Risk posture shift
    if trace.risk_posture_before != trace.risk_posture_after:
        logger.info(
            "outer_loop user=%s posture_shift: %s → %s",
            user_id, trace.risk_posture_before, trace.risk_posture_after,
        )

    # Tomorrow's guidance
    if trace.blocked_tomorrow:
        logger.info("outer_loop user=%s blocked_tomorrow: %s", user_id, trace.blocked_tomorrow)
    if trace.discouraged_tomorrow:
        logger.info("outer_loop user=%s discouraged_tomorrow: %s", user_id, trace.discouraged_tomorrow)
    if trace.preferred_tomorrow:
        logger.info("outer_loop user=%s preferred_tomorrow: %s", user_id, trace.preferred_tomorrow)

    # GPT pattern (if any)
    if trace.pattern_hypothesis:
        logger.info("outer_loop user=%s pattern_hypothesis: %s", user_id, trace.pattern_hypothesis)

    # Next plan
    logger.info(
        "outer_loop user=%s next_night: goal=%s order=%s",
        user_id, trace.next_night_goal, trace.next_night_order,
    )


# ---------------------------------------------------------------------------
# Per-agent wrappers with isolated error handling
# ---------------------------------------------------------------------------

def _run_summary(
    state: SharedState,
    feedback: MorningFeedback | None,
    use_gpt: bool,
) -> tuple[NightSummary, str]:
    try:
        if use_gpt:
            return night_summary_agent.run_gpt(state, feedback)
        return night_summary_agent.run(state, feedback), "deterministic"
    except Exception as exc:
        logger.error("NightSummaryAgent failed unexpectedly: %s", exc)
        # Return a minimal stub so the pipeline continues
        from app.outer_loop.night_summary_agent import _infer_date
        return NightSummary(
            user_id=state.user_id,
            date=_infer_date(state),
            total_ticks=0,
            narrative="Summary unavailable due to an error.",
        ), "error"


def _run_review(
    summary: NightSummary,
    hypotheses: list[dict],
    feedback: MorningFeedback | None,
    use_gpt: bool,
) -> tuple[InterventionReview, str]:
    try:
        if use_gpt:
            return intervention_review_agent.run_gpt(summary, hypotheses, feedback)
        return intervention_review_agent.run(summary, hypotheses, feedback), "deterministic"
    except Exception as exc:
        logger.error("InterventionReviewAgent failed unexpectedly: %s", exc)
        return InterventionReview(
            date=summary.date,
            overall_night_quality="fair",
            review_notes="Review unavailable due to an error.",
        ), "error"


def _run_policy_update(
    policy: UserPolicy,
    review: InterventionReview,
    feedback: MorningFeedback | None,
    use_gpt: bool = False,
) -> tuple[UserPolicy, str]:
    try:
        if use_gpt:
            updated, mode = policy_update_agent.run_gpt(policy, review, feedback)
        else:
            updated = policy_update_agent.run(policy, review, feedback)
            mode = "deterministic"
        return updated, mode
    except Exception as exc:
        logger.error("PolicyUpdateAgent failed unexpectedly: %s", exc)
        return policy, "error"  # Return unchanged policy on failure


def _run_next_plan(
    policy: UserPolicy,
    summary: NightSummary,
    state: SharedState,
    use_gpt: bool,
) -> tuple[NightlyPlan, str]:
    try:
        if use_gpt:
            return next_night_planner_agent.run_gpt(policy, summary, state.preferences)
        return next_night_planner_agent.run(policy, summary, state.preferences)
    except Exception as exc:
        logger.error("NextNightPlannerAgent failed unexpectedly: %s", exc)
        return NightlyPlan(), "error"


def _summarise_mode(modes: list[str]) -> str:
    """Collapse per-step mode labels into a single summary."""
    if all("deterministic" in m for m in modes):
        return "deterministic"
    if any("gpt" in m for m in modes):
        return "partial_gpt" if any("fallback" in m or "deterministic" in m for m in modes) else "gpt"
    return "deterministic"
