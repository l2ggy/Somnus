"""Disturbance events that can stochastically disrupt sleep.

A disturbance represents a random environmental or physiological event
that modifies the sleeper's noise, movement, or depth signals before
final stage inference. Interventions are evaluated partly by how well
they absorb or prevent these disruptions.
"""

from dataclasses import dataclass


# ── Disturbance type constants ────────────────────────────────────────────────

NO_DISTURBANCE = "no_disturbance"
SUDDEN_NOISE_SPIKE = "sudden_noise_spike"
MOVEMENT_SPIKE = "movement_spike"
SNORE_LIKE_DISTURBANCE = "snore_like_disturbance"

ALL_DISTURBANCES = [
    NO_DISTURBANCE,
    SUDDEN_NOISE_SPIKE,
    MOVEMENT_SPIKE,
    SNORE_LIKE_DISTURBANCE,
]


@dataclass(frozen=True)
class Disturbance:
    """A disturbance event and its measured signal impact.

    Attributes:
        kind: Which disturbance occurred (one of the constants above).
        delta_noise: Change applied to noise_level (positive = louder).
        delta_movement: Change applied to movement (positive = more restless).
        delta_depth: Change applied to sleep_depth (negative = shallower).
    """

    kind: str
    delta_noise: float = 0.0
    delta_movement: float = 0.0
    delta_depth: float = 0.0

    def __str__(self) -> str:
        parts = [self.kind]
        if self.delta_noise:
            parts.append(f"noise{self.delta_noise:+.2f}")
        if self.delta_movement:
            parts.append(f"move{self.delta_movement:+.2f}")
        if self.delta_depth:
            parts.append(f"depth{self.delta_depth:+.2f}")
        return "  ".join(parts)


# ── Sentinel for when nothing happens ────────────────────────────────────────

QUIET = Disturbance(kind=NO_DISTURBANCE)
