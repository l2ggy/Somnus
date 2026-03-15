"""Rollout simulator: apply an action repeatedly and accumulate reward.

simulate(...)            — cumulative reward only (used by the agent for scoring)
simulate_trajectory(...) — full per-step debug records (state, reward, action, disturbance)
"""

from dataclasses import dataclass

from simulation.sleep_state import SleepState
from simulation.sleep_dynamics import transition
from simulation.disturbance import Disturbance, QUIET
from simulation.reward import sleep_reward


# ── Debug record ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepRecord:
    """All information about a single simulation step.

    Attributes:
        step:        1-based step index.
        state:       The sleep state *after* this step's transition.
        reward:      Reward earned in this step.
        action:      The action that was applied.
        disturbance: The disturbance event that fired (QUIET if none).
    """

    step: int
    state: SleepState
    reward: float
    action: str
    disturbance: Disturbance

    def __str__(self) -> str:
        d_str = str(self.disturbance) if self.disturbance.kind != "no_disturbance" else "—"
        return (
            f"step {self.step:02d} | {self.action:<22} | "
            f"stage={self.state.sleep_stage:<5} "
            f"depth={self.state.sleep_depth:.2f} "
            f"noise={self.state.noise_level:.2f} "
            f"move={self.state.movement:.2f} | "
            f"reward={self.reward:+.3f} | "
            f"disturbance: {d_str}"
        )


# ── Core rollout ──────────────────────────────────────────────────────────────

def simulate(state: SleepState, action: str, steps: int = 10) -> float:
    """Roll out a fixed action for `steps` transitions and return cumulative reward.

    This is the primary function used by the agent to score candidate actions.
    Disturbance details are discarded here — only the reward matters.

    Args:
        state: The starting sleep state.
        action: The action to apply at each step.
        steps: Number of simulation steps to run.

    Returns:
        Total accumulated reward over the rollout.
    """
    total_reward = 0.0
    current = state

    for _ in range(steps):
        current, _ = transition(current, action)
        total_reward += sleep_reward(current)

    return total_reward


# ── Debug trajectory ──────────────────────────────────────────────────────────

def simulate_trajectory(
    state: SleepState,
    action: str,
    steps: int = 10,
    simple: bool = False,
) -> list[StepRecord] | list[tuple[SleepState, float]]:
    """Roll out a fixed action and return per-step records.

    Args:
        state:  The starting sleep state.
        action: The action to apply at each step.
        steps:  Number of simulation steps to run.
        simple: If True, return lightweight (state, reward) tuples instead of
                full StepRecord objects (preserves the original interface).

    Returns:
        If simple=False (default): list of StepRecord with full debug info.
        If simple=True:            list of (SleepState, float) tuples.
    """
    records: list[StepRecord] = []
    current = state

    for i in range(1, steps + 1):
        current, disturbance = transition(current, action)
        reward = sleep_reward(current)
        records.append(StepRecord(
            step=i,
            state=current,
            reward=reward,
            action=action,
            disturbance=disturbance,
        ))

    if simple:
        return [(r.state, r.reward) for r in records]

    return records
