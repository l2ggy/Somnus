"""Multi-night orchestration loop.

Ties together the full Somnus pipeline for one or more nights:

    SleepState + UserProfile
          │
    choose_best_action_hybrid()    ← LLM preferred, rule-based fallback
          │
    selected_action + PlannerResult
          │
    JournalFeedback (morning self-report)
          │
    update_user_profile_from_feedback()
          │
    updated UserProfile  →  carried into the next night

LLM path:   agents/llm_planner_agent.py   (preferred)
Fallback:   agents/world_model_agent.py   (always available)
Feedback:   simulation/journal_feedback.py (rule-based, no training)

Nothing here is persisted or mutated. Each call returns plain Python objects;
the caller decides whether to save profiles to disk, a database, or memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulation.sleep_state import SleepState
from simulation.user_profile import UserProfile
from simulation.journal_feedback import JournalFeedback, update_user_profile_from_feedback
from agents.hybrid_planner_agent import choose_best_action_hybrid, PlannerResult


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class NightRunResult:
    """Full record of one night's planning and feedback cycle.

    Attributes:
        night:           1-based night index (set by run_multiple_nights).
        input_state:     The SleepState the planner received.
        selected_action: The intervention the planner chose.
        planner_type:    "llm" or "rule_based_fallback".
        score:           Numeric score assigned to the selected action.
        rationale:       LLM rationale if the LLM path ran, else None.
        fallback_reason: Exception string if the LLM path failed, else None.
        feedback:        The JournalFeedback applied after this night.
        updated_profile: The UserProfile after incorporating the feedback.
    """

    night: int
    input_state: SleepState
    selected_action: str
    planner_type: str
    score: float
    rationale: str | None
    fallback_reason: str | None
    feedback: JournalFeedback
    updated_profile: UserProfile

    def summary(self) -> str:
        """One-line summary suitable for loop progress output."""
        pref = self.updated_profile.action_preferences.get(self.selected_action, 0.0)
        tag = "✓" if self.planner_type == "llm" else "↩"
        return (
            f"Night {self.night:>2} {tag}  "
            f"action={self.selected_action:<24}  "
            f"score={self.score:+.3f}  "
            f"pref_after={pref:+.3f}"
            + (f"  [fallback: {self.fallback_reason}]" if self.fallback_reason else "")
        )


# ── Single night ──────────────────────────────────────────────────────────────

def run_single_night(
    state: SleepState,
    profile: UserProfile,
    feedback: JournalFeedback,
    llm_client: Any | None = None,
    horizon: int = 5,
    fallback_steps: int = 10,
    fallback_rollouts: int = 20,
    night: int = 1,
) -> NightRunResult:
    """Run one night of planning and apply morning feedback to the profile.

    The profile is never mutated — an updated copy is returned in the result.

    Steps:
      1. Call the hybrid planner to choose the best intervention.
      2. Override feedback.action_used with the planner's actual choice so
         the feedback is always correctly attributed to what was applied.
      3. Update the profile using the morning journal entry.
      4. Return the full record.

    Args:
        state:            Current biometric sleep state.
        profile:          User preference profile for this night.
        feedback:         Morning journal entry describing how the night felt.
                          feedback.action_used is overwritten with the planner's
                          selected action so attribution is always correct.
        llm_client:       Callable (prompt: str) -> str, or None for direct fallback.
        horizon:          Trajectory horizon for the LLM planner.
        fallback_steps:   Rollout horizon for the rule-based planner.
        fallback_rollouts: Monte Carlo samples for the rule-based planner.
        night:            Night index for labelling in multi-night results.

    Returns:
        NightRunResult with the planner decision, feedback, and updated profile.
    """
    # Plan
    plan: PlannerResult = choose_best_action_hybrid(
        state,
        user_profile=profile,
        horizon=horizon,
        llm_client=llm_client,
        fallback_steps=fallback_steps,
        fallback_rollouts=fallback_rollouts,
    )

    # Ensure feedback is attributed to the action that was actually chosen,
    # not whatever the caller pre-filled in feedback.action_used.
    from dataclasses import replace as _replace
    attributed_feedback = _replace(feedback, action_used=plan.selected_action)

    # Update profile from morning journal
    updated_profile = update_user_profile_from_feedback(profile, attributed_feedback)

    return NightRunResult(
        night=night,
        input_state=state,
        selected_action=plan.selected_action,
        planner_type=plan.planner_type,
        score=plan.score,
        rationale=plan.rationale,
        fallback_reason=plan.fallback_reason,
        feedback=attributed_feedback,
        updated_profile=updated_profile,
    )


# ── Multi-night loop ──────────────────────────────────────────────────────────

def run_multiple_nights(
    states: list[SleepState],
    initial_profile: UserProfile,
    feedbacks: list[JournalFeedback],
    llm_client: Any | None = None,
    horizon: int = 5,
    fallback_steps: int = 10,
    fallback_rollouts: int = 20,
) -> list[NightRunResult]:
    """Run multiple nights in sequence, carrying the updated profile forward.

    Each night's updated profile becomes the next night's starting profile,
    so preferences accumulate gradually based on the journal entries.

    Args:
        states:           One SleepState per night.
        initial_profile:  Starting user profile (not mutated).
        feedbacks:        One JournalFeedback per night.
        llm_client:       Callable (prompt: str) -> str, or None for direct fallback.
        horizon:          Trajectory horizon for the LLM planner.
        fallback_steps:   Rollout horizon for the rule-based planner.
        fallback_rollouts: Monte Carlo samples for the rule-based planner.

    Returns:
        List of NightRunResult, one per night, in order.

    Raises:
        ValueError: If states and feedbacks have different lengths.
    """
    if len(states) != len(feedbacks):
        raise ValueError(
            f"states and feedbacks must have the same length, "
            f"got {len(states)} states and {len(feedbacks)} feedbacks."
        )

    results: list[NightRunResult] = []
    profile = initial_profile

    for i, (state, feedback) in enumerate(zip(states, feedbacks), start=1):
        result = run_single_night(
            state=state,
            profile=profile,
            feedback=feedback,
            llm_client=llm_client,
            horizon=horizon,
            fallback_steps=fallback_steps,
            fallback_rollouts=fallback_rollouts,
            night=i,
        )
        results.append(result)
        profile = result.updated_profile  # carry forward

    return results


# ── Example (run directly) ────────────────────────────────────────────────────

def _example() -> None:
    """Three-night demo using the mock LLM client.

    Shows how brown-noise preference drifts upward after positive feedback
    and rain preference drifts downward after negative feedback.

    Run:  python -m simulation.night_loop
    """
    from agents.mock_llm_client import MockLLMClient

    initial_profile = UserProfile(
        action_preferences={
            "play_brown_noise":  0.1,
            "play_rain":         0.0,
            "breathing_pacing":  0.1,
            "gradual_wake":     -0.2,
            "do_nothing":        0.0,
        }
    )

    states = [
        SleepState("light", sleep_depth=0.42, movement=0.30, noise_level=0.55, time_until_alarm=210),
        SleepState("light", sleep_depth=0.38, movement=0.35, noise_level=0.60, time_until_alarm=200),
        SleepState("light", sleep_depth=0.45, movement=0.28, noise_level=0.50, time_until_alarm=220),
    ]

    # Feedback objects — action_used will be overwritten with the planner's choice
    feedbacks = [
        JournalFeedback(action_used="", intervention_helpful=True,  felt_rested=True,  awakenings=1),
        JournalFeedback(action_used="", intervention_helpful=True,  felt_rested=True,  awakenings=0),
        JournalFeedback(action_used="", intervention_helpful=False, felt_rested=False, awakenings=4,
                        notes="Kept waking up despite the intervention."),
    ]

    client = MockLLMClient(seed=7)
    results = run_multiple_nights(states, initial_profile, feedbacks, llm_client=client)

    # ── Print night-by-night summary ──────────────────────────────────────────
    print("=== Multi-night Somnus loop ===\n")
    for r in results:
        print(r.summary())

    # ── Show how the chosen action's preference evolved ───────────────────────
    print("\n=== Preference drift for chosen action ===\n")
    profile = initial_profile
    for r in results:
        action = r.selected_action
        before = profile.action_preferences.get(action, 0.0)
        after  = r.updated_profile.action_preferences.get(action, 0.0)
        helpful = r.feedback.intervention_helpful
        print(
            f"  Night {r.night}: {action:<24}  "
            f"{before:+.3f} → {after:+.3f}  "
            f"({'helpful' if helpful else 'not helpful' if helpful is False else 'no report'})"
        )
        profile = r.updated_profile

    # ── Final profile preferences ─────────────────────────────────────────────
    print("\n=== Final user profile preferences ===\n")
    final = results[-1].updated_profile
    for action, pref in final.action_preferences.items():
        bar = ("+" if pref >= 0 else "-") * max(1, int(abs(pref) * 6))
        print(f"  {action:<24} {pref:+.3f}  {bar}")


if __name__ == "__main__":
    _example()
