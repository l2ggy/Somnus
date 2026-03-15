"""Reward function scoring how beneficial a sleep state is for the sleeper."""

from simulation.sleep_state import SleepState

# Stage base rewards — easy to tune
_STAGE_REWARD: dict[str, float] = {
    "deep": 1.0,
    "rem": 0.6,
    "light": 0.2,
    "wake": -1.0,
}

# Penalty weights — scale 0.0 to 1.0
_NOISE_PENALTY_WEIGHT = 0.3
_MOVEMENT_PENALTY_WEIGHT = 0.2


def sleep_reward(state: SleepState) -> float:
    """Score a single sleep state from the sleeper's perspective.

    Scoring breakdown:
      - Stage base score:    deep=+1.0, rem=+0.6, light=+0.2, wake=−1.0
      - Noise penalty:       up to −0.3 at maximum noise
      - Movement penalty:    up to −0.2 at maximum movement

    The minimum possible score is −1.5 (wake + full noise + full movement).
    The maximum possible score is  +1.0 (deep + no noise + no movement).

    Args:
        state: The sleep state to evaluate.

    Returns:
        A float reward value.
    """
    base = _STAGE_REWARD[state.sleep_stage]
    noise_penalty = _NOISE_PENALTY_WEIGHT * state.noise_level
    movement_penalty = _MOVEMENT_PENALTY_WEIGHT * state.movement
    return base - noise_penalty - movement_penalty
