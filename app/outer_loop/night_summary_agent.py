"""
NightSummaryAgent
-----------------
Consumes a completed SharedState (after the last night tick) and produces a
compact NightSummary that downstream outer-loop agents can reason about.

Two execution paths (matching the pattern used by strategist and journal_reflection):

  run(state)      — deterministic, always available.
                    Extracts structured data from the hypothesis log.

  run_gpt(state)  — adds a short narrative summary via LLM.
                    Falls back to deterministic on failure; narrative="" on fallback.

Input contract:
  - state.hypotheses contains tick-level entries from the inner loop:
      sensor_interpreter: {source, timestamp, signal_summary, ...}
      disturbance:        {source, timestamp, phase, primary_reason, wake_risk_after, ...}
      intervention:       not in hypotheses — read from state.active_intervention
  - state.nightly_plan.preferred_intervention_order tells us which interventions
    were planned (used as a cross-reference, not the only source of truth)

Because the inner loop stores per-tick hypotheses, NightSummary is derived
entirely from state without touching the database.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app import llm_client
from app.models.outer_loop import (
    InterventionUsage,
    MorningFeedback,
    NightSummary,
)
from app.models.userstate import SharedState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(state: SharedState, feedback: MorningFeedback | None = None) -> NightSummary:
    """Deterministic summary.  Always succeeds."""
    return _summarise(state, feedback, narrative="")


def run_gpt(
    state: SharedState,
    feedback: MorningFeedback | None = None,
) -> tuple[NightSummary, str]:
    """
    GPT-enriched summary.  Adds a narrative; falls back gracefully.

    Returns:
        (NightSummary, mode_used) where mode_used is "gpt" or "fallback".
    """
    base = _summarise(state, feedback, narrative="")
    try:
        narrative = _generate_narrative(base, state)
        return base.model_copy(update={"narrative": narrative}), "gpt"
    except Exception as exc:
        logger.warning("NightSummaryAgent GPT path failed (%s), skipping narrative", exc)
        return base, "fallback"


# ---------------------------------------------------------------------------
# Deterministic extraction
# ---------------------------------------------------------------------------

def _summarise(
    state: SharedState,
    feedback: MorningFeedback | None,
    narrative: str,
) -> NightSummary:
    hypotheses = state.hypotheses

    # ---- Collect disturbance ticks ----------------------------------------
    disturbance_ticks = [h for h in hypotheses if h.get("source") == "disturbance"]
    disturbance_reasons = []
    wake_risks: list[float] = []
    for h in disturbance_ticks:
        reason = h.get("primary_reason")
        if reason and reason not in disturbance_reasons:
            disturbance_reasons.append(reason)
        risk_after = h.get("wake_risk_after")
        if isinstance(risk_after, (int, float)):
            wake_risks.append(float(risk_after))

    peak_wake_risk = max(wake_risks, default=0.0)
    avg_wake_risk = round(sum(wake_risks) / len(wake_risks), 3) if wake_risks else 0.0

    # ---- Collect sleep phases from disturbance hypotheses -----------------
    # disturbance writes the phase at each tick, giving us a timeline
    phases_seen: list[str] = []
    for h in disturbance_ticks:
        p = h.get("phase")
        if p and p not in phases_seen:
            phases_seen.append(p)

    # Fallback: current sleep_state phase
    if not phases_seen and state.sleep_state.phase != "unknown":
        phases_seen = [state.sleep_state.phase]

    # ---- Count total ticks (sensor_interpreter fires once per tick) --------
    total_ticks = sum(1 for h in hypotheses if h.get("source") == "sensor_interpreter")

    # ---- Intervention usage -------------------------------------------------
    interventions = _extract_intervention_usage(hypotheses, total_ticks)

    # ---- Date ---------------------------------------------------------------
    date = _infer_date(state)

    return NightSummary(
        user_id=state.user_id,
        date=date,
        total_ticks=total_ticks,
        interventions_used=interventions,
        phases_observed=phases_seen,
        disturbances_detected=disturbance_reasons,
        peak_wake_risk=round(peak_wake_risk, 3),
        avg_wake_risk=avg_wake_risk,
        narrative=narrative,
        morning_feedback=feedback,
    )


def _extract_intervention_usage(hypotheses: list[dict], total_ticks: int) -> list[InterventionUsage]:
    """
    Reconstruct per-intervention usage from the hypothesis log.

    The intervention agent doesn't write to hypotheses — we infer usage from
    disturbance hypotheses which record the tick count, combined with the
    assumption that interventions run between disturbance events.

    Since the inner loop doesn't currently log per-tick intervention choices
    to hypotheses, we use a best-effort approach: look for any intervention-
    tagged hypotheses, and fall back to reading state.active_intervention as
    a proxy for "what ran last / most prominently".

    TODO: When the inner loop appends per-tick intervention hypotheses (with
    source="intervention"), this function can be made exact.
    """
    # Look for explicit intervention hypotheses (future-proof)
    iv_hypotheses = [h for h in hypotheses if h.get("source") == "intervention"]

    if iv_hypotheses:
        # Aggregate by type
        usage_map: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"ticks": 0, "intensity_sum": 0.0, "tick_indices": []}
        )
        for i, h in enumerate(iv_hypotheses):
            iv_type = h.get("type", "none")
            if iv_type == "none":
                continue
            intensity = float(h.get("intensity", 0.0))
            usage_map[iv_type]["ticks"] += 1
            usage_map[iv_type]["intensity_sum"] += intensity
            usage_map[iv_type]["tick_indices"].append(i)

        result = []
        for iv_type, data in usage_map.items():
            ticks = data["ticks"]
            avg_intensity = round(data["intensity_sum"] / ticks, 3) if ticks > 0 else 0.0
            peak_period = _estimate_peak_period(data["tick_indices"], total_ticks)
            result.append(InterventionUsage(
                type=iv_type,
                ticks_active=ticks,
                avg_intensity=avg_intensity,
                peak_period=peak_period,
            ))
        return result

    # Fallback: nothing logged, return empty (caller will use active_intervention
    # as a best-effort single entry if desired)
    return []


def _estimate_peak_period(
    tick_indices: list[int],
    total_ticks: int,
) -> str:
    if not tick_indices or total_ticks == 0:
        return "unknown"
    avg_idx = sum(tick_indices) / len(tick_indices)
    frac = avg_idx / total_ticks
    if frac < 0.35:
        return "early"
    if frac < 0.65:
        return "middle"
    return "late"


def _infer_date(state: SharedState) -> str:
    """Best-effort date: from last journal entry, last hypothesis, or today."""
    if state.journal_history:
        return state.journal_history[-1].date

    for h in reversed(state.hypotheses):
        ts = h.get("timestamp") or h.get("date")
        if ts and len(ts) >= 10:
            return ts[:10]

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# GPT narrative generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a concise sleep-session analyst for Somnus, an AI sleep assistant.
Given structured data from a completed sleep session, write a 2–3 sentence
factual narrative summarising what happened.

Rules:
- Return a single JSON object with one key: "narrative"
- "narrative" must be a plain string, 2–3 sentences, no bullet points
- Do not make medical claims
- Mention which interventions ran and whether sleep was disrupted
- If data is sparse, say so briefly and avoid speculation

Output schema (JSON only):
{"narrative": "..."}"""


def _generate_narrative(summary: NightSummary, state: SharedState) -> str:
    iv_lines = []
    for iv in summary.interventions_used:
        iv_lines.append(
            f"  {iv.type}: {iv.ticks_active} ticks, avg intensity {iv.avg_intensity:.2f}, "
            f"peak in {iv.peak_period} of night"
        )
    iv_text = "\n".join(iv_lines) if iv_lines else "  none recorded"

    feedback_text = "none submitted"
    if summary.morning_feedback:
        fb = summary.morning_feedback
        parts = []
        if fb.rested_score is not None:
            parts.append(f"rested_score={fb.rested_score}/10")
        if fb.awakenings_reported is not None:
            parts.append(f"awakenings={fb.awakenings_reported}")
        if fb.intervention_helpful is not None:
            parts.append(f"intervention_helpful={fb.intervention_helpful}")
        feedback_text = ", ".join(parts) if parts else "submitted (no details)"

    user_prompt = f"""\
Session date: {summary.date}
Total ticks: {summary.total_ticks}
Phases observed: {', '.join(summary.phases_observed) or 'unknown'}
Disturbances detected: {', '.join(summary.disturbances_detected) or 'none'}
Peak wake-risk: {summary.peak_wake_risk:.2f}
Average wake-risk: {summary.avg_wake_risk:.2f}

Interventions used:
{iv_text}

Morning feedback: {feedback_text}

Write a 2–3 sentence summary of this sleep session."""

    raw = llm_client.chat_json(_SYSTEM_PROMPT, user_prompt)
    return str(raw.get("narrative", "")).strip()
