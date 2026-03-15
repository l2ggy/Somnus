"""World-model agent: Monte Carlo rollout planner with optional user preferences.

Planning strategy
-----------------
Each candidate action is evaluated by running it forward in simulation
n_rollouts times and averaging the cumulative sleep reward. Averaging across
rollouts smooths out disturbance randomness so the planner picks the action
that is robustly good — not just lucky on one sample.

Personalization
---------------
If a UserProfile is supplied, a small preference adjustment is added on top
of the expected sleep score before ranking. This lets the planner favour
interventions the user tolerates or enjoys without overriding sleep quality
entirely. The split is explicit:

    final_score = expected_sleep_score + preference_adjustment

Where preference_adjustment is typically in [-1.0, +1.0] and expected sleep
scores span roughly [-10, +10] over a 10-step horizon, so preferences nudge
without dominating.
"""

from simulation.sleep_state import SleepState
from simulation.actions import ALL_ACTIONS
from simulation.simulator import simulate
from simulation.user_profile import UserProfile


def _preference_adjustment(action: str, user_profile: UserProfile | None) -> float:
    """Return the preference bonus/penalty for an action given a user profile.

    Returns 0.0 when no profile is provided, preserving original behavior.

    Args:
        action: The action being evaluated.
        user_profile: Optional user preference profile.

    Returns:
        A signed float to add to the expected sleep score.
    """
    if user_profile is None:
        return 0.0
    return user_profile.preference_for(action)


def _score_action(
    state: SleepState,
    action: str,
    steps: int,
    n_rollouts: int,
    user_profile: UserProfile | None = None,
) -> float:
    """Estimate the personalised expected value of taking `action` from `state`.

    Runs n_rollouts independent simulations and averages cumulative sleep
    reward, then adds the user's preference adjustment for that action.

    Args:
        state: Current sleep state (start of every rollout).
        action: Candidate action to evaluate.
        steps: Rollout horizon — number of transition steps per rollout.
        n_rollouts: Number of independent Monte Carlo samples to average.
        user_profile: Optional preferences. None = pure sleep quality scoring.

    Returns:
        Mean cumulative sleep reward + preference adjustment.
    """
    sleep_score = sum(simulate(state, action, steps) for _ in range(n_rollouts)) / n_rollouts
    preference = _preference_adjustment(action, user_profile)
    return sleep_score + preference


def rank_actions(
    state: SleepState,
    steps: int = 10,
    n_rollouts: int = 20,
    user_profile: UserProfile | None = None,
) -> list[tuple[str, float]]:
    """Score every available action and return them sorted best-first.

    Useful for demos and debugging — shows the full preference ordering,
    not just the winner.

    Args:
        state: The current sleep state to plan from.
        steps: Rollout horizon per simulation.
        n_rollouts: Monte Carlo samples per action.
        user_profile: Optional user preferences to mix into scoring.

    Returns:
        List of (action, final_score) tuples, descending by score.
    """
    scores = {
        action: _score_action(state, action, steps, n_rollouts, user_profile)
        for action in ALL_ACTIONS
    }
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def choose_best_action(
    state: SleepState,
    steps: int = 10,
    n_rollouts: int = 20,
    user_profile: UserProfile | None = None,
) -> tuple[str, float]:
    """Return the action with the highest personalised expected reward.

    Each candidate is evaluated over n_rollouts independent simulations.
    If a UserProfile is provided, preference adjustments are added after
    averaging so they nudge — but do not override — sleep quality.

    Args:
        state: The current sleep state to plan from.
        steps: Rollout horizon — how many minutes ahead to simulate.
        n_rollouts: Monte Carlo samples per action. 20 is a reasonable default;
                    raise to 50–100 for more stable rankings at the cost of speed.
        user_profile: Optional user preferences. None = unchanged behavior.

    Returns:
        (best_action, best_score) — the chosen action and its final score.
    """
    ranked = rank_actions(state, steps=steps, n_rollouts=n_rollouts, user_profile=user_profile)
    return ranked[0]
