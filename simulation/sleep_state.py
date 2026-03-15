"""Sleep state dataclass representing a snapshot of a sleeper's biometric conditions."""

from dataclasses import dataclass, replace
from typing import Self


VALID_STAGES = {"wake", "light", "deep", "rem"}


@dataclass
class SleepState:
    """A snapshot of the sleeper's current biometric and environmental conditions.

    Attributes:
        sleep_stage: Current sleep stage — one of "wake", "light", "deep", "rem".
        sleep_depth: Continuous measure of sleep depth, 0.0 (fully awake) to 1.0 (deepest sleep).
        movement: Physical movement level, 0.0 (still) to 1.0 (very restless).
        noise_level: Ambient or perceived noise level, 0.0 (silent) to 1.0 (very loud).
        time_until_alarm: Minutes remaining until the wake alarm fires.
    """

    sleep_stage: str
    sleep_depth: float
    movement: float
    noise_level: float
    time_until_alarm: int

    def __post_init__(self) -> None:
        if self.sleep_stage not in VALID_STAGES:
            raise ValueError(f"sleep_stage must be one of {VALID_STAGES}, got '{self.sleep_stage}'")
        self.clamp()

    def clamp(self) -> None:
        """Clamp all continuous fields to their valid ranges in place."""
        self.sleep_depth = max(0.0, min(1.0, self.sleep_depth))
        self.movement = max(0.0, min(1.0, self.movement))
        self.noise_level = max(0.0, min(1.0, self.noise_level))
        self.time_until_alarm = max(0, self.time_until_alarm)

    def updated(self, **changes) -> "SleepState":
        """Return a new SleepState with the given fields replaced, then clamped.

        Example:
            next_state = state.updated(sleep_depth=0.8, movement=0.1)
        """
        new_state = replace(self, **changes)
        new_state.clamp()
        return new_state
