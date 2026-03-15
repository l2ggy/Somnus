"""Trajectory critic: code-based scoring of LLM-generated sleep trajectories.

Role in the system
------------------
The LLM world model (agents/llm_world_model_agent.py) generates candidate
futures — plausible sequences of sleep states for a given intervention.
This module scores those futures using deterministic code so that action
quality can be compared quantitatively, the same way the rule-based simulator
(simulation/simulator.py) produces cumulative rewards.

The rule-based simulator remains the baseline/fallback for planning.
This critic bridges the LLM path back to the same reward vocabulary.

Scoring pipeline
----------------
  TrajectoryStep
      │
      ├── to_sleep_state()  →  SleepState  →  sleep_reward()   (base score)
      └── wake_risk penalty                                      (LLM-only signal)
      │
  score_trajectory_step()          per-step float
      │
  score_candidate_trajectory()     sum across steps
      │
  score_action_bundle()            probability-weighted expected score
                                   + optional UserProfile preference
"""

from __future__ import annotations

from agents.llm_world_model_agent import (
    ActionTrajectoryBundle,
    CandidateTrajectory,
    TrajectoryStep,
)
from simulation.reward import sleep_reward
from simulation.user_profile import UserProfile

# Additional penalty for LLM-estimated wake risk (not present in SleepState).
# Kept separate from reward.py so the baseline scoring is untouched.
_WAKE_RISK_PENALTY_WEIGHT = 0.15


# ── Step-level scoring ────────────────────────────────────────────────────────

def score_trajectory_step(step: TrajectoryStep, time_until_alarm: int = 0) -> float:
    """Score a single TrajectoryStep.

    Reuses sleep_reward() via the to_sleep_state() bridge, then layers on a
    penalty for the LLM-estimated wake_risk signal that has no equivalent in
    the rule-based SleepState.

    Scoring breakdown (same scale as simulation/reward.py):
      Base (from sleep_reward):
        deep=+1.0, rem=+0.6, light=+0.2, wake=−1.0
        noise penalty:    up to −0.30
        movement penalty: up to −0.20
      Additional critic penalty:
        wake_risk:        up to −0.15 at wake_risk=1.0

    Args:
        step:              The trajectory step to score.
        time_until_alarm:  Passed to to_sleep_state(); affects nothing in the
                           current reward function but keeps the bridge honest.

    Returns:
        A float score. Range: approximately [−1.65, +1.0].
    """
    state = step.to_sleep_state(time_until_alarm)
    base_score = sleep_reward(state)
    wake_risk_penalty = _WAKE_RISK_PENALTY_WEIGHT * step.wake_risk
    return base_score - wake_risk_penalty


# ── Trajectory-level scoring ──────────────────────────────────────────────────

def score_candidate_trajectory(
    traj: CandidateTrajectory,
    starting_time_until_alarm: int | None = None,
) -> float:
    """Aggregate step scores across a single candidate trajectory.

    Each step is scored independently and the results are summed (same
    convention as simulate() in simulation/simulator.py).

    Args:
        traj:                      The candidate trajectory to score.
        starting_time_until_alarm: If provided, time_until_alarm is decremented
                                   by step_index for each step so later steps
                                   correctly reflect being closer to the alarm.
                                   If None, all steps use time_until_alarm=0.

    Returns:
        Cumulative score across all steps.
    """
    total = 0.0
    for step in traj.steps:
        if starting_time_until_alarm is not None:
            tta = max(0, starting_time_until_alarm - step.step_index)
        else:
            tta = 0
        total += score_trajectory_step(step, time_until_alarm=tta)
    return total


# ── Bundle-level scoring ──────────────────────────────────────────────────────

def _normalize_probabilities(trajectories: list[CandidateTrajectory]) -> list[float]:
    """Return normalized probabilities for a list of trajectories.

    If probabilities already sum to ~1.0 (within 1e-6) they are returned
    unchanged. Otherwise they are renormalized so downstream math is correct.
    Raises ValueError on an empty list or all-zero probabilities.
    """
    if not trajectories:
        raise ValueError("Cannot score an ActionTrajectoryBundle with no trajectories.")

    raw = [t.probability for t in trajectories]
    total = sum(raw)

    if total <= 0.0:
        raise ValueError(
            "All trajectory probabilities are zero or negative — cannot normalize. "
            "Check the LLM output or re-run generation."
        )

    if abs(total - 1.0) < 1e-6:
        return raw

    return [p / total for p in raw]


def score_action_bundle(
    bundle: ActionTrajectoryBundle,
    user_profile: UserProfile | None = None,
    starting_time_until_alarm: int | None = None,
) -> float:
    """Compute the expected score of an action across all candidate trajectories.

    Expected score = Σ (probability_i × score_i)

    This is the LLM-path analogue of the Monte Carlo average in
    agents/world_model_agent.py, but using LLM-assigned probabilities instead
    of sampling frequency.

    An optional UserProfile preference is added after the expectation so that
    personal taste nudges — but does not override — sleep quality.

    Args:
        bundle:                    The action's full trajectory bundle from the LLM.
        user_profile:              Optional user preferences. None = pure sleep quality.
        starting_time_until_alarm: Passed through to score_candidate_trajectory().

    Returns:
        Probability-weighted expected score + preference adjustment.

    Raises:
        ValueError: If the trajectory list is empty or all probabilities are zero.
    """
    probs = _normalize_probabilities(bundle.trajectories)

    expected_sleep_score = sum(
        p * score_candidate_trajectory(traj, starting_time_until_alarm)
        for p, traj in zip(probs, bundle.trajectories)
    )

    preference = (
        user_profile.preference_for(bundle.action)
        if user_profile is not None
        else 0.0
    )

    return expected_sleep_score + preference


# ── Example (run directly) ────────────────────────────────────────────────────

def _example() -> None:
    """Demonstrates scoring a CandidateTrajectory and an ActionTrajectoryBundle."""
    from agents.llm_world_model_agent import TrajectoryStep, CandidateTrajectory, ActionTrajectoryBundle
    from simulation.user_profile import UserProfile

    # Build two candidate trajectories by hand (normally from parse_action_trajectory_bundle)
    traj_good = CandidateTrajectory(
        probability=0.60,
        summary="Brown noise masks ambient sound; sleeper deepens into deep sleep.",
        steps=[
            TrajectoryStep(1, "light", 0.50, 0.25, 0.35, wake_risk=0.08),
            TrajectoryStep(2, "deep",  0.68, 0.18, 0.22, wake_risk=0.03),
            TrajectoryStep(3, "deep",  0.72, 0.15, 0.18, wake_risk=0.02),
        ],
    )

    traj_mediocre = CandidateTrajectory(
        probability=0.40,
        summary="Noise reduction helps but sleeper stays in light sleep.",
        steps=[
            TrajectoryStep(1, "light", 0.44, 0.28, 0.38, wake_risk=0.12),
            TrajectoryStep(2, "light", 0.46, 0.26, 0.25, wake_risk=0.10),
            TrajectoryStep(3, "light", 0.48, 0.25, 0.20, wake_risk=0.09),
        ],
    )

    bundle = ActionTrajectoryBundle(
        action="play_brown_noise",
        trajectories=[traj_good, traj_mediocre],
        rationale="Brown noise masks ambient sound, lowering noise level and reducing arousal risk.",
    )

    profile = UserProfile(
        action_preferences={
            "play_brown_noise": 0.8,
            "play_rain": -0.5,
            "breathing_pacing": 0.3,
            "gradual_wake": -1.0,
            "do_nothing": 0.0,
        }
    )

    print("=== CandidateTrajectory scores ===")
    for i, traj in enumerate(bundle.trajectories, 1):
        score = score_candidate_trajectory(traj, starting_time_until_alarm=180)
        print(f"  Trajectory {i} (p={traj.probability:.2f}): {score:+.3f}  — {traj.summary}")

    print()
    print("=== ActionTrajectoryBundle expected scores ===")
    raw_score = score_action_bundle(bundle, starting_time_until_alarm=180)
    personalized = score_action_bundle(bundle, user_profile=profile, starting_time_until_alarm=180)
    print(f"  Without profile : {raw_score:+.3f}")
    print(f"  With profile    : {personalized:+.3f}  (pref adj: {profile.preference_for(bundle.action):+.1f})")


if __name__ == "__main__":
    _example()
