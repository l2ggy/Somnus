"""User preference profile for personalizing action selection.

The simulator estimates which action best promotes sleep quality.
The UserProfile nudges that ranking toward interventions the user
personally tolerates or enjoys — without overriding sleep quality entirely.
"""

from dataclasses import dataclass, field
from simulation.actions import ALL_ACTIONS


def _default_preferences() -> dict[str, float]:
    return {action: 0.0 for action in ALL_ACTIONS}


@dataclass
class UserProfile:
    """Lightweight preference model for a single user.

    Attributes:
        action_preferences: Per-action bonus/penalty added to the expected
            sleep score. Positive = user likes this intervention; negative =
            user dislikes it. Scale loosely matches the reward function
            (deep sleep ≈ +1.0), so values in [-1.0, +1.0] are sensible.
        sensitivity_to_noise: Multiplier on noise penalties in preference
            adjustments. >1.0 means the user is more bothered by noise than
            average; <1.0 means they tolerate it better. Currently used as
            metadata — can be wired into reward weighting later.
        intervention_tolerance: Global scale on all preference bonuses.
            1.0 = normal; 0.0 = ignore all preferences (pure sleep quality).
            Useful for gradually introducing personalization.
    """

    action_preferences: dict[str, float] = field(default_factory=_default_preferences)
    sensitivity_to_noise: float = 1.0
    intervention_tolerance: float = 1.0

    def preference_for(self, action: str) -> float:
        """Return the scaled preference bonus for a given action.

        Applies intervention_tolerance as a global dampener so the caller
        doesn't need to remember to scale manually.

        Args:
            action: One of the action constants from simulation.actions.

        Returns:
            Scaled preference adjustment (can be negative).
        """
        raw = self.action_preferences.get(action, 0.0)
        return raw * self.intervention_tolerance
