"""
PolicyUpdateAgent
-----------------
Updates the persistent UserPolicy based on last night's InterventionReview
and optional MorningFeedback.

Two execution paths:

  run(policy, review, feedback)      — deterministic.  Score nudges from
                                       verdict × confidence, feedback bonuses,
                                       risk posture from rolling avg.  Always
                                       succeeds and produces a valid policy.

  run_gpt(policy, review, feedback)  — deterministic base first, then GPT
                                       proposes bounded per-intervention score
                                       adjustments based on cross-night pattern
                                       analysis.  Falls back to deterministic
                                       result on any failure.

Deterministic update mechanics:

  Per-intervention score nudge (learning_rate = 0.12):
    - "helpful"  → +0.12
    - "neutral"  →  0.00  (no change)
    - "harmful"  → -0.12

  Scale by verdict confidence so low-confidence verdicts nudge less:
    delta = base_delta × confidence

  Morning feedback supplements (applied to ALL interventions used that night):
    - rested_score ≥ 7  → +0.04 bonus to interventions with helpful/neutral verdict
    - rested_score ≤ 3  → -0.04 penalty to interventions with harmful/neutral verdict
    - awakenings ≥ 3    → -0.03 penalty to harmful interventions

  All scores clamped to [-2.0, +2.0] after every operation.

GPT adjustment mechanics:

  GPT sees: current scores (before tonight), tonight's verdicts + feedback,
  last 7 nights of rested scores, and which interventions have a track record.

  GPT returns: per-intervention score_adjustments (bounded to ±_MAX_GPT_DELTA
  per intervention per night) plus a pattern_hypothesis string.

  Validation + clamping:
    1. Pydantic model validates the raw JSON shape
    2. Each adjustment is clamped to [-_MAX_GPT_DELTA, +_MAX_GPT_DELTA]
    3. Only known intervention types are accepted
    4. Final score = clamp(deterministic_score + gpt_delta, -2.0, +2.0)
    5. If GPT fails → deterministic result used unchanged

  This means GPT can move a score up to ±0.20 per night on top of the
  deterministic ±0.12, giving it real but bounded influence.

Risk posture and preferred order are always derived deterministically from
the final scores, regardless of which path was used.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from app import llm_client
from app.models.outer_loop import (
    InterventionReview,
    MorningFeedback,
    UserPolicy,
)

logger = logging.getLogger(__name__)

_LEARNING_RATE = 0.12
_FEEDBACK_BONUS = 0.04
_AWAKENING_PENALTY = 0.03
_SCORE_FLOOR = -2.0
_SCORE_CEIL = 2.0
_SCORE_HISTORY_CAP = 14        # Keep last 14 nights of rested scores
_DEFAULT_ORDER = ["brown_noise", "breathing_pace", "rain"]
_MAX_GPT_DELTA = 0.20          # Hard cap on any single GPT score adjustment

_VALID_INTERVENTION_TYPES = {
    "brown_noise", "white_noise", "pink_noise",
    "rain", "waves", "breathing_pace", "wake_ramp",
}


# ---------------------------------------------------------------------------
# Pydantic model for GPT output validation
# ---------------------------------------------------------------------------

class _GptPolicyAdjustment(BaseModel):
    """
    Validates raw JSON from the LLM before any of it enters the policy.

    score_adjustments: Per-intervention deltas the model believes are warranted
        based on cross-night pattern analysis.  Values must be numeric; we clamp
        them to ±_MAX_GPT_DELTA after validation regardless of what GPT returns.

    pattern_hypothesis: A single observation about a pattern GPT detected across
        nights.  Stored in UserPolicy for auditability and display.

    confidence_note: Optional sentence about how confident GPT is in its
        assessment, given the amount of data available.
    """
    score_adjustments: Dict[str, float] = {}
    pattern_hypothesis: str = ""
    confidence_note: str = ""

    @field_validator("score_adjustments", mode="before")
    @classmethod
    def coerce_adjustments(cls, v):
        """Accept only dict[str, numeric]; drop non-numeric and unknown types."""
        if not isinstance(v, dict):
            return {}
        cleaned = {}
        for key, val in v.items():
            if not isinstance(key, str):
                continue
            try:
                cleaned[key] = float(val)
            except (TypeError, ValueError):
                continue
        return cleaned

    @field_validator("pattern_hypothesis", "confidence_note", mode="before")
    @classmethod
    def coerce_str(cls, v):
        return str(v).strip() if v is not None else ""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(
    policy: UserPolicy,
    review: InterventionReview,
    feedback: MorningFeedback | None = None,
) -> UserPolicy:
    """
    Deterministic policy update.  Always succeeds.

    Args:
        policy:   Current UserPolicy loaded from the outer loop store.
        review:   InterventionReview from this night.
        feedback: Optional MorningFeedback from the user.

    Returns:
        New UserPolicy with updated scores, posture, and order.
        The original policy is never mutated.
    """
    updated_scores, recent_scores = _apply_deterministic_updates(policy, review, feedback)
    return _build_policy(
        policy, updated_scores, recent_scores, review,
        pattern_hypothesis="",
        gpt_score_adjustments={},
    )


def run_gpt(
    policy: UserPolicy,
    review: InterventionReview,
    feedback: MorningFeedback | None = None,
) -> tuple[UserPolicy, str]:
    """
    GPT-assisted policy update with automatic fallback.

    Runs the deterministic base first, then asks GPT to propose bounded
    per-intervention score adjustments based on cross-night pattern analysis.
    The deterministic result is always the floor — GPT can only nudge, not
    override.

    Returns:
        (UserPolicy, mode_used) where mode_used is "gpt" or "fallback".
    """
    # Step 1: deterministic base — always valid, used as fallback
    updated_scores, recent_scores = _apply_deterministic_updates(policy, review, feedback)

    # Step 2: GPT proposes adjustments on top of the deterministic scores
    try:
        adjustment = _generate_gpt_adjustment(policy, review, feedback, updated_scores)
        accepted_deltas = _apply_gpt_deltas(updated_scores, adjustment.score_adjustments)

        if accepted_deltas:
            logger.info(
                "PolicyUpdateAgent GPT applied adjustments for user=%s: %s | hypothesis: %s",
                policy.user_id,
                {k: f"{v:+.3f}" for k, v in accepted_deltas.items()},
                adjustment.pattern_hypothesis[:80] if adjustment.pattern_hypothesis else "(none)",
            )
        else:
            logger.debug(
                "PolicyUpdateAgent GPT proposed no valid adjustments for user=%s", policy.user_id
            )

        updated_policy = _build_policy(
            policy, updated_scores, recent_scores, review,
            pattern_hypothesis=adjustment.pattern_hypothesis,
            gpt_score_adjustments=accepted_deltas,
        )
        return updated_policy, "gpt"

    except Exception as exc:
        logger.warning(
            "PolicyUpdateAgent GPT path failed (%s); using deterministic result", exc
        )
        fallback = _build_policy(
            policy, updated_scores, recent_scores, review,
            pattern_hypothesis="",
            gpt_score_adjustments={},
        )
        return fallback, "fallback"


# ---------------------------------------------------------------------------
# Deterministic update core
# ---------------------------------------------------------------------------

def _apply_deterministic_updates(
    policy: UserPolicy,
    review: InterventionReview,
    feedback: MorningFeedback | None,
) -> tuple[dict[str, float], list[int]]:
    """
    Compute updated scores and rolling history without building the full policy.
    Returns (updated_scores, recent_scores) so the GPT path can add deltas before
    calling _build_policy.
    """
    updated_scores = dict(policy.intervention_scores)

    # Per-verdict nudges
    for verdict in review.verdicts:
        iv_type = verdict.type
        if iv_type not in updated_scores:
            updated_scores[iv_type] = 0.0
        base_delta = _base_delta(verdict.verdict)
        weighted_delta = base_delta * verdict.confidence
        updated_scores[iv_type] = _clamp(updated_scores[iv_type] + weighted_delta)

    # Morning feedback adjustments
    if feedback is not None:
        for verdict in review.verdicts:
            iv_type = verdict.type
            if iv_type not in updated_scores:
                continue
            score = feedback.rested_score
            awakenings = feedback.awakenings_reported
            if score is not None and score >= 7 and verdict.verdict in ("helpful", "neutral"):
                updated_scores[iv_type] = _clamp(updated_scores[iv_type] + _FEEDBACK_BONUS)
            if score is not None and score <= 3 and verdict.verdict in ("harmful", "neutral"):
                updated_scores[iv_type] = _clamp(updated_scores[iv_type] - _FEEDBACK_BONUS)
            if awakenings is not None and awakenings >= 3 and verdict.verdict == "harmful":
                updated_scores[iv_type] = _clamp(updated_scores[iv_type] - _AWAKENING_PENALTY)

    # Rolling rested score history
    recent_scores = list(policy.recent_rested_scores)
    if feedback is not None and feedback.rested_score is not None:
        recent_scores.append(feedback.rested_score)
    elif review.overall_night_quality == "good":
        recent_scores.append(8)
    elif review.overall_night_quality == "poor":
        recent_scores.append(3)
    else:
        recent_scores.append(5)
    recent_scores = recent_scores[-_SCORE_HISTORY_CAP:]

    return updated_scores, recent_scores


def _build_policy(
    policy: UserPolicy,
    updated_scores: dict[str, float],
    recent_scores: list[int],
    review: InterventionReview,
    pattern_hypothesis: str,
    gpt_score_adjustments: dict[str, float],
) -> UserPolicy:
    avg_rested = round(sum(recent_scores) / len(recent_scores), 1) if recent_scores else None
    risk_posture = _derive_risk_posture(avg_rested, review)
    preferred_order = _derive_preferred_order(updated_scores)
    policy_notes = _build_notes(updated_scores, risk_posture, avg_rested)

    return UserPolicy(
        user_id=policy.user_id,
        updated_at=_now(),
        intervention_scores=updated_scores,
        risk_posture=risk_posture,
        preferred_intervention_order=preferred_order,
        nights_tracked=policy.nights_tracked + 1,
        avg_rested_score=avg_rested,
        last_review_date=review.date,
        policy_notes=policy_notes,
        recent_rested_scores=recent_scores,
        pattern_hypothesis=pattern_hypothesis,
        gpt_score_adjustments=gpt_score_adjustments,
    )


# ---------------------------------------------------------------------------
# GPT adjustment logic
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a sleep policy analyst for Somnus, an AI sleep assistant.
Your job is to look at a user's cross-night intervention history and propose
small, evidence-based adjustments to intervention preference scores.

You will receive:
- Current intervention scores (accumulated from prior nights, range -2.0 to +2.0)
- Tonight's verdict for each intervention (helpful/neutral/harmful + confidence)
- Morning feedback if available (rested score, awakenings, whether intervention felt helpful)
- Recent rested score history (last few nights)

Your task:
1. Identify any cross-night pattern the deterministic rules might miss
   (e.g. an intervention that always follows bad nights, a disturbance type
   that keeps recurring, a score that seems wrong given the full history)
2. Propose small bounded adjustments to intervention scores where justified
3. Write a single sentence describing the pattern you detected (or "No clear pattern yet.")

Strict rules:
- Respond with a single JSON object only. No prose, no markdown.
- score_adjustments values must be between -0.20 and +0.20 (you cannot exceed this)
- Only include interventions where you have genuine evidence for adjustment
- If you see no clear pattern, return empty score_adjustments and say so
- Do not invent data. Work only with what is provided.
- Do not make medical claims.

Output schema (JSON only):
{
  "score_adjustments": {
    "intervention_type": <float between -0.20 and +0.20>
  },
  "pattern_hypothesis": "<one sentence observation, or 'No clear pattern yet.'>",
  "confidence_note": "<one sentence about your confidence given available data>"
}"""


def _generate_gpt_adjustment(
    policy: UserPolicy,
    review: InterventionReview,
    feedback: MorningFeedback | None,
    post_deterministic_scores: dict[str, float],
) -> _GptPolicyAdjustment:
    """
    Call the LLM and parse + validate its score adjustment proposal.
    Raises on any failure so run_gpt() can fall back cleanly.
    """
    # ---- Format current scores (before tonight's deterministic update) ----
    score_lines = [
        f"  {iv}: {score:+.3f}"
        for iv, score in sorted(policy.intervention_scores.items())
    ]
    scores_text = "\n".join(score_lines) if score_lines else "  (no prior scores)"

    post_det_lines = [
        f"  {iv}: {score:+.3f}"
        for iv, score in sorted(post_deterministic_scores.items())
    ]
    post_det_text = "\n".join(post_det_lines)

    # ---- Format tonight's verdicts ---------------------------------------
    verdict_lines = []
    for v in review.verdicts:
        verdict_lines.append(
            f"  {v.type}: {v.verdict} (confidence {v.confidence:.2f}) — {v.reason}"
        )
    verdicts_text = "\n".join(verdict_lines) if verdict_lines else "  (no interventions recorded)"

    # ---- Format morning feedback -----------------------------------------
    if feedback is None:
        feedback_text = "  none submitted"
    else:
        parts = []
        if feedback.rested_score is not None:
            parts.append(f"rested_score={feedback.rested_score}/10")
        if feedback.awakenings_reported is not None:
            parts.append(f"awakenings={feedback.awakenings_reported}")
        if feedback.intervention_helpful is not None:
            parts.append(f"intervention_felt_helpful={feedback.intervention_helpful}")
        if feedback.notes:
            parts.append(f'notes="{feedback.notes}"')
        feedback_text = "  " + ", ".join(parts) if parts else "  (submitted but empty)"

    # ---- Format rested score history ------------------------------------
    history = policy.recent_rested_scores[-7:]  # last 7 nights
    if history:
        history_text = "  " + ", ".join(str(s) for s in history) + " (oldest → newest)"
    else:
        history_text = "  (no history yet)"

    # ---- Nights tracked context -----------------------------------------
    context_note = (
        f"This is night {policy.nights_tracked + 1} of tracking. "
        f"Current risk posture: {policy.risk_posture}. "
        f"Nights with data: {policy.nights_tracked}."
    )

    user_prompt = f"""\
{context_note}

Scores before tonight (accumulated from prior nights):
{scores_text}

After tonight's deterministic update:
{post_det_text}

Tonight's intervention verdicts:
{verdicts_text}

Morning feedback:
{feedback_text}

Recent rested score history (1=worst, 10=best):
{history_text}

Propose score adjustments and identify any pattern."""

    raw = llm_client.chat_json(_SYSTEM_PROMPT, user_prompt)
    return _GptPolicyAdjustment.model_validate(raw)


def _apply_gpt_deltas(
    scores: dict[str, float],
    proposed: dict[str, float],
) -> dict[str, float]:
    """
    Apply validated GPT score adjustments to the post-deterministic scores.

    Each delta is:
      1. Filtered to known intervention types only
      2. Clamped to [-_MAX_GPT_DELTA, +_MAX_GPT_DELTA]
      3. Added to the score, then the result is clamped to [-2.0, +2.0]

    Returns a dict of {type: actual_delta_applied} for auditability.
    """
    accepted: dict[str, float] = {}
    for iv_type, raw_delta in proposed.items():
        if iv_type not in _VALID_INTERVENTION_TYPES:
            logger.debug("GPT proposed adjustment for unknown type %r, skipping", iv_type)
            continue
        clamped_delta = max(-_MAX_GPT_DELTA, min(_MAX_GPT_DELTA, float(raw_delta)))
        if abs(clamped_delta) < 0.001:
            continue  # Skip negligible adjustments
        old_score = scores.get(iv_type, 0.0)
        new_score = _clamp(old_score + clamped_delta)
        actual_delta = round(new_score - old_score, 4)
        scores[iv_type] = new_score
        if abs(actual_delta) >= 0.001:
            accepted[iv_type] = actual_delta
    return accepted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_delta(verdict: str) -> float:
    if verdict == "helpful":
        return _LEARNING_RATE
    if verdict == "harmful":
        return -_LEARNING_RATE
    return 0.0


def _clamp(value: float) -> float:
    return max(_SCORE_FLOOR, min(_SCORE_CEIL, round(value, 4)))


def _derive_risk_posture(
    avg_rested: float | None,
    review: InterventionReview,
) -> str:
    if avg_rested is not None:
        if avg_rested < 5.0:
            return "protective"
        if avg_rested > 7.0:
            return "exploratory"
        return "balanced"

    # Fallback: use last night's quality
    if review.overall_night_quality == "poor":
        return "protective"
    if review.overall_night_quality == "good":
        return "exploratory"
    return "balanced"


def _derive_preferred_order(scores: dict[str, float]) -> list[str]:
    # Sort by score descending; only include non-negative
    ranked = sorted(
        [(iv, s) for iv, s in scores.items() if s >= 0.0],
        key=lambda x: (-x[1], x[0]),  # desc score, asc name for ties
    )
    order = [iv for iv, _ in ranked[:4]]  # cap at 4
    return order if order else _DEFAULT_ORDER


def _build_notes(
    scores: dict[str, float],
    posture: str,
    avg_rested: float | None,
) -> str:
    top = sorted(scores.items(), key=lambda x: -x[1])[:2]
    top_str = ", ".join(f"{iv}={s:+.2f}" for iv, s in top) if top else "none"
    avg_str = f"{avg_rested:.1f}/10" if avg_rested is not None else "unknown"
    return (
        f"Risk posture: {posture}. "
        f"Top interventions: {top_str}. "
        f"Avg rested score: {avg_str}."
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
