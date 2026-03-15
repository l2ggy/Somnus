"""
Outer loop data models for Somnus.

These models represent the slower, cross-night learning layer that sits above
the real-time tick pipeline.  They are intentionally kept simple and
JSON-serializable so they can be persisted to SQLite as blobs alongside
SharedState, following the same pattern as the existing session store.

Lifecycle:
  End of night  → NightSummary (what happened) + InterventionReview (how well it worked)
  Policy update → UserPolicy (accumulated learning, updated per night)
  Next planning → NightlyPlan (reuses existing model, informed by UserPolicy)

MorningFeedback is optional at every step.  All agents degrade gracefully when
it is absent so the outer loop runs even if the user doesn't fill in a report.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Incoming API model
# ---------------------------------------------------------------------------

class MorningFeedback(BaseModel):
    """
    Optional user-submitted feedback at the end of a sleep session.

    Richer than JournalEntry — includes direct signal about whether the
    intervention felt helpful so the policy update agent can use it.
    All fields are optional so callers can submit partial information.
    """
    rested_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="1–10 self-reported restedness. None if not submitted.",
    )
    awakenings_reported: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of remembered wake-ups. None if not submitted.",
    )
    intervention_helpful: Optional[bool] = Field(
        default=None,
        description="Whether the user felt the intervention helped. None = unknown.",
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Night summary (output of NightSummaryAgent)
# ---------------------------------------------------------------------------

class InterventionUsage(BaseModel):
    """Record of one intervention type used during a night."""
    type: str
    ticks_active: int                   # How many ticks it was running
    avg_intensity: float
    # Rough timing: which third of the night this intervention was most active
    peak_period: Literal["early", "middle", "late", "unknown"] = "unknown"


class NightSummary(BaseModel):
    """
    Compact summary of a completed sleep session, produced by NightSummaryAgent.

    Derived entirely from SharedState — no external calls needed for the
    deterministic path.  The GPT path adds a human-readable narrative.
    """
    user_id: str
    date: str                           # ISO date of the night (YYYY-MM-DD)
    total_ticks: int
    interventions_used: List[InterventionUsage] = Field(default_factory=list)
    phases_observed: List[str] = Field(default_factory=list)   # e.g. ["light","deep","rem"]
    disturbances_detected: List[str] = Field(default_factory=list)  # primary reasons
    peak_wake_risk: float = 0.0
    avg_wake_risk: float = 0.0
    # Narrative summary — empty string if GPT path was skipped
    narrative: str = ""
    # Pass-through of morning feedback if submitted before end-of-night run
    morning_feedback: Optional[MorningFeedback] = None


# ---------------------------------------------------------------------------
# Intervention review (output of InterventionReviewAgent)
# ---------------------------------------------------------------------------

class InterventionVerdict(BaseModel):
    """Evaluation of one intervention type from the night."""
    type: str
    verdict: Literal["helpful", "neutral", "harmful"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class InterventionReview(BaseModel):
    """
    Structured evaluation of all interventions used during a night,
    produced by InterventionReviewAgent.
    """
    date: str
    overall_night_quality: Literal["poor", "fair", "good"]
    verdicts: List[InterventionVerdict] = Field(default_factory=list)
    primary_disturbance: Optional[str] = None
    # Free-text summary (GPT path only, empty string otherwise)
    review_notes: str = ""


# ---------------------------------------------------------------------------
# User policy (output of PolicyUpdateAgent, persisted across nights)
# ---------------------------------------------------------------------------

class UserPolicy(BaseModel):
    """
    Persistent per-user intervention policy, updated after each night.

    Stores accumulated preference scores for each intervention type plus
    a rolling risk posture that informs the next night's planner.

    Score semantics:
      - Range [-2.0, +2.0], starts at 0.0
      - Positive: user benefits from this intervention
      - Negative: this intervention correlates with worse nights
      - Magnitude reflects how many nights of evidence accumulated

    This is intentionally simple — no neural nets, no regression.
    Bounded nudges accumulate naturally into meaningful bias over weeks.
    """
    user_id: str
    updated_at: str                     # ISO datetime of last update

    # Per-intervention accumulated preference scores
    intervention_scores: Dict[str, float] = Field(
        default_factory=lambda: {
            "brown_noise": 0.0,
            "white_noise": 0.0,
            "pink_noise":  0.0,
            "rain":        0.0,
            "waves":       0.0,
            "breathing_pace": 0.0,
            "wake_ramp":   0.0,
        }
    )
    # Derived from recent nights — informs planner risk posture
    risk_posture: Literal["protective", "balanced", "exploratory"] = "balanced"
    # Ordered list inferred from scores (highest positive first)
    preferred_intervention_order: List[str] = Field(
        default_factory=lambda: ["brown_noise", "breathing_pace", "rain"]
    )
    nights_tracked: int = 0
    avg_rested_score: Optional[float] = None
    last_review_date: Optional[str] = None
    # Human-readable one-line policy summary for debugging / display
    policy_notes: str = ""

    # Raw rolling rested scores (last N nights, capped at 14)
    recent_rested_scores: List[int] = Field(default_factory=list)

    # GPT-generated cross-night pattern observation (empty if GPT path not used).
    # Stored for display and auditability; does not affect score computation directly.
    pattern_hypothesis: str = ""
    # Per-intervention deltas that GPT proposed and were accepted this update cycle.
    # Keys: intervention type. Values: bounded delta applied on top of deterministic base.
    gpt_score_adjustments: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Outer loop report (returned from the orchestrator)
# ---------------------------------------------------------------------------

class OuterLoopReport(BaseModel):
    """
    Combined result returned when the outer loop runs end-of-night.

    Bundles the four agent outputs so callers get everything in one response.
    """
    user_id: str
    date: str
    night_summary: NightSummary
    intervention_review: InterventionReview
    user_policy: UserPolicy
    next_night_plan: Dict[str, Any]     # NightlyPlan as dict (avoids circular import)
    mode_used: str                      # "deterministic", "gpt", or "partial_gpt"
