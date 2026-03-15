"""
InterventionReviewAgent
-----------------------
Evaluates which interventions used during a night were helpful, neutral, or
harmful.  Runs after NightSummaryAgent and optionally consumes MorningFeedback.

Two execution paths:

  run(summary)       — deterministic.  Uses wake-risk trajectory from the
                       hypothesis log to evaluate each intervention.

  run_gpt(summary)   — LLM evaluates the same data and returns richer reasoning.
                       Falls back to deterministic on failure.

Evaluation heuristic (deterministic):
  The hypothesis log records wake_risk_after at each disturbance tick.  We
  compare wake-risk before and after an intervention was active to estimate
  whether it helped.

  For each intervention type that ran:
    - Compute avg wake_risk during ticks when it was active
    - Compare to avg wake_risk when it was NOT active (or the night average)
    - delta < -0.05  →  helpful   (risk dropped while it ran)
    - delta > +0.05  →  harmful   (risk rose while it ran)
    - otherwise      →  neutral

  Morning feedback overrides / supplements when available:
    - intervention_helpful=True  → boost verdict toward "helpful"
    - intervention_helpful=False → boost verdict toward "harmful"
    - rested_score ≥ 7           → treat all neutral as "helpful"
    - rested_score ≤ 3           → treat all neutral as "harmful"

Without explicit per-tick intervention data (current inner loop), we use
the disturbance hypothesis wake_risk trajectory as a proxy.
"""

import logging
from typing import Any

from app import llm_client
from app.models.outer_loop import (
    InterventionReview,
    InterventionVerdict,
    MorningFeedback,
    NightSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(
    summary: NightSummary,
    hypotheses: list[dict],
    feedback: MorningFeedback | None = None,
) -> InterventionReview:
    """Deterministic review.  Always succeeds."""
    verdicts = _evaluate_deterministic(summary, hypotheses, feedback)
    quality = _overall_quality(summary, feedback)
    primary_disturbance = summary.disturbances_detected[0] if summary.disturbances_detected else None

    return InterventionReview(
        date=summary.date,
        overall_night_quality=quality,
        verdicts=verdicts,
        primary_disturbance=primary_disturbance,
        review_notes="",
    )


def run_gpt(
    summary: NightSummary,
    hypotheses: list[dict],
    feedback: MorningFeedback | None = None,
) -> tuple[InterventionReview, str]:
    """
    GPT-backed review with automatic fallback.

    Returns:
        (InterventionReview, mode_used) where mode_used is "gpt" or "fallback".
    """
    base = run(summary, hypotheses, feedback)
    try:
        review_notes = _generate_review_notes(summary, base, feedback)
        return base.model_copy(update={"review_notes": review_notes}), "gpt"
    except Exception as exc:
        logger.warning("InterventionReviewAgent GPT path failed (%s), using fallback", exc)
        return base, "fallback"


# ---------------------------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------------------------

def _evaluate_deterministic(
    summary: NightSummary,
    hypotheses: list[dict],
    feedback: MorningFeedback | None,
) -> list[InterventionVerdict]:
    if not summary.interventions_used:
        return []

    # Build a timeline of wake_risk from disturbance hypotheses
    wake_risk_timeline = _extract_wake_risk_timeline(hypotheses)
    night_avg_risk = (
        sum(wake_risk_timeline) / len(wake_risk_timeline)
        if wake_risk_timeline else summary.avg_wake_risk
    )

    verdicts = []
    for iv in summary.interventions_used:
        verdict, confidence, reason = _verdict_for_intervention(
            iv_type=iv.type,
            iv_ticks=iv.ticks_active,
            night_avg_risk=night_avg_risk,
            summary=summary,
            feedback=feedback,
        )
        verdicts.append(InterventionVerdict(
            type=iv.type,
            verdict=verdict,
            confidence=confidence,
            reason=reason,
        ))

    return verdicts


def _verdict_for_intervention(
    iv_type: str,
    iv_ticks: int,
    night_avg_risk: float,
    summary: NightSummary,
    feedback: MorningFeedback | None,
) -> tuple[str, float, str]:
    """
    Returns (verdict, confidence, reason).

    Without per-tick intervention records we can't do a before/after comparison,
    so we use the overall night quality as a proxy.  When feedback is present
    it overrides the heuristic.
    """
    # Feedback-based override (direct signal, highest confidence)
    if feedback is not None and feedback.intervention_helpful is not None:
        if feedback.intervention_helpful:
            return "helpful", 0.80, "User reported the intervention was helpful."
        else:
            return "harmful", 0.75, "User reported the intervention was not helpful."

    # Night quality proxy (lower confidence, no direct attribution)
    quality = _overall_quality(summary, feedback)

    # Ticks active as a proxy for how prominent the intervention was
    prominence = "prominent" if iv_ticks > max(summary.total_ticks // 4, 1) else "brief"

    if quality == "good":
        verdict = "helpful"
        confidence = 0.55 if prominence == "prominent" else 0.40
        reason = (
            f"Night quality was good and {iv_type} was {prominence}. "
            "Tentatively attributed as helpful."
        )
    elif quality == "poor":
        verdict = "harmful" if prominence == "prominent" else "neutral"
        confidence = 0.45 if prominence == "prominent" else 0.35
        reason = (
            f"Night quality was poor and {iv_type} was {prominence}. "
            "Cannot confirm helpfulness."
        )
    else:
        verdict = "neutral"
        confidence = 0.40
        reason = (
            f"Night quality was fair. Insufficient signal to attribute {iv_type} effect."
        )

    # Rested-score modifier (when feedback is partial)
    if feedback is not None and feedback.rested_score is not None:
        if feedback.rested_score >= 8:
            verdict = "helpful"
            confidence = min(confidence + 0.10, 0.90)
            reason += f" High rested score ({feedback.rested_score}/10) supports helpful verdict."
        elif feedback.rested_score <= 3:
            if verdict == "neutral":
                verdict = "harmful"
            confidence = min(confidence + 0.10, 0.90)
            reason += f" Low rested score ({feedback.rested_score}/10) suggests poor outcome."

    return verdict, round(confidence, 2), reason


def _extract_wake_risk_timeline(hypotheses: list[dict]) -> list[float]:
    """Extract ordered wake_risk_after values from disturbance hypotheses."""
    risks = []
    for h in hypotheses:
        if h.get("source") != "disturbance":
            continue
        risk = h.get("wake_risk_after")
        if isinstance(risk, (int, float)):
            risks.append(float(risk))
    return risks


def _overall_quality(
    summary: NightSummary,
    feedback: MorningFeedback | None,
) -> str:
    # Direct score from feedback wins
    if feedback is not None and feedback.rested_score is not None:
        if feedback.rested_score <= 3:
            return "poor"
        if feedback.rested_score <= 6:
            return "fair"
        return "good"

    # Proxy from wake-risk
    if summary.avg_wake_risk <= 0.25:
        return "good"
    if summary.avg_wake_risk <= 0.50:
        return "fair"
    return "poor"


# ---------------------------------------------------------------------------
# GPT review notes
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a sleep intervention analyst for Somnus.
Given a structured evaluation of a sleep session's interventions, write a
brief review explaining the verdicts.

Rules:
- Return a single JSON object with one key: "review_notes"
- "review_notes" must be 2–3 plain sentences
- Be honest about confidence levels — low confidence = less certain language
- Do not make medical claims
- Mention the overall night quality and highlight the most notable intervention

Output schema (JSON only):
{"review_notes": "..."}"""


def _generate_review_notes(
    summary: NightSummary,
    review: InterventionReview,
    feedback: MorningFeedback | None,
) -> str:
    verdict_lines = [
        f"  {v.type}: {v.verdict} (confidence {v.confidence:.2f}) — {v.reason}"
        for v in review.verdicts
    ]
    verdict_text = "\n".join(verdict_lines) if verdict_lines else "  no interventions to review"

    feedback_text = "none"
    if feedback:
        parts = []
        if feedback.rested_score is not None:
            parts.append(f"rested={feedback.rested_score}/10")
        if feedback.awakenings_reported is not None:
            parts.append(f"awakenings={feedback.awakenings_reported}")
        if feedback.intervention_helpful is not None:
            parts.append(f"helpful={feedback.intervention_helpful}")
        feedback_text = ", ".join(parts) if parts else "submitted but empty"

    user_prompt = f"""\
Night: {summary.date}
Overall quality: {review.overall_night_quality}
Primary disturbance: {review.primary_disturbance or 'none'}
Morning feedback: {feedback_text}

Intervention verdicts:
{verdict_text}

Write 2–3 sentences reviewing the interventions."""

    raw = llm_client.chat_json(_SYSTEM_PROMPT, user_prompt)
    return str(raw.get("review_notes", "")).strip()
