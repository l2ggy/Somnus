"""Morning journal feedback loop for personalizing intervention preferences.

Role in the system
------------------
LLM path  →  preferred planner  (agents/llm_planner_agent.py)
Rule-based →  baseline/fallback (agents/world_model_agent.py)
This module →  slowly shapes UserProfile.action_preferences over time
               based on how the sleeper felt the morning after.

No training, no model updates. Each feedback session applies a small,
bounded nudge to the relevant action's preference score so that repeated
positive/negative experiences accumulate into a meaningful personal bias
over weeks of nightly use.

Update magnitude reference (with default learning_rate=0.15):
  intervention helpful     : +0.15
  intervention unhelpful   : -0.15
  felt rested              : +0.06  (smaller — less directly attributable)
  did not feel rested      : -0.06
  awakenings ≥ 3           : -0.05  (mild environmental/arousal signal)
  awakenings ≥ 5           : -0.10

All preferences are clamped to [-2.0, +2.0] after each update.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from simulation.user_profile import UserProfile

# Preference range — keeps values interpretable and prevents runaway drift
_PREF_MIN = -2.0
_PREF_MAX = 2.0

# Fraction of learning_rate applied to secondary signals
_RESTED_SCALE = 0.4
_AWAKENING_THRESHOLD_MILD = 3     # ≥ this many: mild penalty
_AWAKENING_THRESHOLD_HIGH = 5     # ≥ this many: stronger penalty
_AWAKENING_MILD_SCALE = 0.33
_AWAKENING_HIGH_SCALE = 0.67


# ── Feedback dataclass ────────────────────────────────────────────────────────

@dataclass
class JournalFeedback:
    """Morning self-report capturing the sleeper's experience of last night.

    Attributes:
        action_used:           The intervention Somnus applied (from ALL_ACTIONS).
        intervention_helpful:  Whether the sleeper felt the intervention helped.
                               None = not reported.
        felt_rested:           Whether the sleeper woke feeling rested.
                               None = not reported.
        awakenings:            Number of times the sleeper remembers waking.
                               None = not reported.
        notes:                 Optional free-text comment (logged, not processed).
    """

    action_used: str
    intervention_helpful: bool | None = None
    felt_rested: bool | None = None
    awakenings: int | None = None
    notes: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp_preference(value: float) -> float:
    """Clamp a preference score to the valid range [_PREF_MIN, _PREF_MAX]."""
    return max(_PREF_MIN, min(_PREF_MAX, value))


def _feedback_delta(feedback: JournalFeedback, learning_rate: float) -> float:
    """Compute the total preference delta for feedback.action_used.

    Each reported signal contributes independently. Signals that were not
    reported (None) contribute 0 so partial feedback is handled gracefully.

    Args:
        feedback:      The morning journal entry.
        learning_rate: Base magnitude for the primary helpful/unhelpful signal.

    Returns:
        A signed float to add to the current preference for feedback.action_used.
    """
    delta = 0.0

    # Primary signal: did the intervention help?
    if feedback.intervention_helpful is True:
        delta += learning_rate
    elif feedback.intervention_helpful is False:
        delta -= learning_rate

    # Secondary signal: overall restedness (less directly attributable to the action)
    if feedback.felt_rested is True:
        delta += learning_rate * _RESTED_SCALE
    elif feedback.felt_rested is False:
        delta -= learning_rate * _RESTED_SCALE

    # Mild signal: awakenings suggest environmental or arousal instability
    if feedback.awakenings is not None:
        if feedback.awakenings >= _AWAKENING_THRESHOLD_HIGH:
            delta -= learning_rate * _AWAKENING_HIGH_SCALE
        elif feedback.awakenings >= _AWAKENING_THRESHOLD_MILD:
            delta -= learning_rate * _AWAKENING_MILD_SCALE

    return delta


# ── Main update function ──────────────────────────────────────────────────────

def update_user_profile_from_feedback(
    profile: UserProfile,
    feedback: JournalFeedback,
    learning_rate: float = 0.15,
) -> UserProfile:
    """Return a new UserProfile with action preferences nudged by morning feedback.

    The original profile is never mutated.

    Only the preference for feedback.action_used is updated; all other action
    preferences are copied unchanged. This keeps each night's update targeted —
    we only know how the sleeper felt about the action that was actually used.

    Args:
        profile:       The current user profile.
        feedback:      Morning journal entry for last night.
        learning_rate: Base step size for the primary helpful/unhelpful signal.
                       Smaller values (e.g. 0.05) give smoother long-term drift;
                       larger values (e.g. 0.3) respond faster to strong feedback.

    Returns:
        A new UserProfile with the updated preference for feedback.action_used.
    """
    delta = _feedback_delta(feedback, learning_rate)

    current_pref = profile.action_preferences.get(feedback.action_used, 0.0)
    new_pref = _clamp_preference(current_pref + delta)

    updated_preferences = {**profile.action_preferences, feedback.action_used: new_pref}

    return replace(profile, action_preferences=updated_preferences)


# ── Example (run directly) ────────────────────────────────────────────────────

def _example() -> None:
    """Demonstrates before/after preference update from a single feedback session.

    Run:  python -m simulation.journal_feedback
    """
    from simulation.actions import ALL_ACTIONS

    profile = UserProfile(
        action_preferences={
            "play_brown_noise":  0.2,
            "play_rain":         0.0,
            "breathing_pacing":  0.1,
            "gradual_wake":     -0.3,
            "do_nothing":        0.0,
        }
    )

    feedback = JournalFeedback(
        action_used="play_brown_noise",
        intervention_helpful=True,
        felt_rested=True,
        awakenings=1,
        notes="Slept through most of the night. Noise felt soothing.",
    )

    updated = update_user_profile_from_feedback(profile, feedback, learning_rate=0.15)

    action = feedback.action_used
    before = profile.action_preferences[action]
    after  = updated.action_preferences[action]
    delta  = after - before

    print(f"Action     : {action}")
    print(f"Before     : {before:+.3f}")
    print(f"Delta      : {delta:+.3f}")
    print(f"After      : {after:+.3f}")
    print(f"Clamped    : {_PREF_MIN} ≤ {after:.3f} ≤ {_PREF_MAX}")
    print()

    # Show a negative feedback case
    bad_feedback = JournalFeedback(
        action_used="play_rain",
        intervention_helpful=False,
        felt_rested=False,
        awakenings=5,
        notes="Woke up multiple times. Rain sound felt irritating.",
    )
    updated2 = update_user_profile_from_feedback(profile, bad_feedback)
    before2 = profile.action_preferences["play_rain"]
    after2  = updated2.action_preferences["play_rain"]
    print(f"Action     : play_rain")
    print(f"Before     : {before2:+.3f}")
    print(f"Delta      : {after2 - before2:+.3f}")
    print(f"After      : {after2:+.3f}")


if __name__ == "__main__":
    _example()
