"""Rule-based sleep dynamics: given a state and an action, produce the next state.

Sleep cycle overview (used as the basis for transition logic):
  - A typical night runs roughly:  wake → light → deep → light → REM → light → …
  - Deep sleep dominates the first third of the night (slow-wave sleep).
  - REM sleep dominates the final third, growing longer toward morning.
  - Light sleep acts as the universal transition buffer between all other stages.
  - Disturbances (noise, movement) push toward wake or light regardless of cycle position.

We use `time_until_alarm` as a coarse proxy for night progression:
  - High time_until_alarm  →  earlier in the night  →  favor deep sleep
  - Low  time_until_alarm  →  later  in the night  →  favor REM

Disturbance system:
  - At each step, a random event may occur before final stage inference.
  - Disturbance probability scales with instability (noise, movement, shallow depth).
  - Deep sleep provides mild resistance; being near wake time raises instability slightly.
  - Interventions (e.g. play_brown_noise) reduce the signals disturbances act on,
    so they implicitly help absorb or prevent disruptions.
"""

import random

from simulation.sleep_state import SleepState
from simulation.actions import (
    DO_NOTHING,
    PLAY_BROWN_NOISE,
    PLAY_RAIN,
    BREATHING_PACING,
    GRADUAL_WAKE,
)
from simulation.disturbance import (
    Disturbance,
    QUIET,
    NO_DISTURBANCE,
    SUDDEN_NOISE_SPIKE,
    MOVEMENT_SPIKE,
    SNORE_LIKE_DISTURBANCE,
)

# Small random drift applied at every step regardless of action
_DRIFT = 0.03

# Assumed total sleep window in minutes (used to normalise night position)
_TOTAL_SLEEP_MINUTES = 480  # 8 hours


# ── Night-position helper ─────────────────────────────────────────────────────

def _night_progress(time_until_alarm: int) -> float:
    """Return a 0.0–1.0 value representing how far through the night we are.

    0.0 = just fell asleep (alarm far away)
    1.0 = alarm is imminent
    """
    elapsed = _TOTAL_SLEEP_MINUTES - time_until_alarm
    return max(0.0, min(1.0, elapsed / _TOTAL_SLEEP_MINUTES))


# ── Stage inference ───────────────────────────────────────────────────────────

def _stage_weights(
    depth: float,
    movement: float,
    noise: float,
    time_until_alarm: int,
    current_stage: str,
) -> dict[str, float]:
    """Compute unnormalised weights for each sleep stage given current signals.

    Design rules:
      - wake   is favoured by high noise, high movement, low depth
      - deep   is favoured by high depth, low noise, early-in-night
      - rem    is favoured by moderate depth, low movement, late-in-night
      - light  always has a nonzero baseline (transition buffer)
      - Current stage gets a small inertia bonus to avoid instant flipping.
    """
    progress = _night_progress(time_until_alarm)

    # wake: disturbances and shallow depth both pull here
    wake_w = (
        max(0.0, 1.0 - depth * 2.5)
        + noise * 0.6
        + movement * 0.5
    )

    # deep: needs quiet, still, and is natural early in the night
    deep_possible = depth >= 0.55 and noise < 0.5 and movement < 0.35
    early_night_bonus = max(0.0, 1.0 - progress * 2.0)
    deep_w = (depth * 1.5 + early_night_bonus * 0.8) if deep_possible else 0.0

    # rem: moderate depth, low movement, rises through the second half
    rem_possible = 0.30 <= depth <= 0.75 and movement < 0.45
    late_night_bonus = max(0.0, progress * 2.0 - 0.5)
    rem_w = (0.4 + late_night_bonus * 1.0) if rem_possible else 0.0

    # light: always available
    light_w = 0.5 + max(0.0, 0.5 - abs(depth - 0.4))

    weights: dict[str, float] = {
        "wake": wake_w,
        "deep": deep_w,
        "rem": rem_w,
        "light": light_w,
    }

    # Slight inertia: staying in the current stage gets a small boost
    if current_stage in weights:
        weights[current_stage] *= 1.2

    return weights


def _infer_stage(
    depth: float,
    movement: float,
    noise: float,
    time_until_alarm: int,
    current_stage: str,
) -> str:
    """Sample the next sleep stage using signal-derived weights.

    Hard overrides first for extreme values, then a weighted random draw.
    """
    # Hard overrides for extreme states
    if depth < 0.12 or movement > 0.85:
        return "wake"
    if depth > 0.85 and movement < 0.2 and noise < 0.3:
        return "deep"

    weights = _stage_weights(depth, movement, noise, time_until_alarm, current_stage)
    total = sum(weights.values())
    if total == 0.0:
        return "light"

    roll = random.random() * total
    cumulative = 0.0
    for stage, w in weights.items():
        cumulative += w
        if roll <= cumulative:
            return stage

    return "light"


# ── Disturbance system ────────────────────────────────────────────────────────

def _disturbance_probability(
    depth: float,
    movement: float,
    noise: float,
    current_stage: str,
    time_until_alarm: int,
) -> float:
    """Compute the probability that any disturbance event occurs this step.

    Higher instability → higher chance of a disturbance.

    Contributing factors:
      - Shallow depth: harder to stay asleep
      - High noise: louder environment is inherently disruptive
      - High movement: restlessness compounds instability
      - Near alarm: natural arousal threshold rises toward morning
      - Deep sleep: provides mild resistance (discount applied)
    """
    progress = _night_progress(time_until_alarm)

    base_prob = (
        (1.0 - depth) * 0.25   # shallow sleep → more vulnerable
        + noise * 0.20           # noisy environment → more likely disrupted
        + movement * 0.15        # restlessness → more likely disrupted
        + progress * 0.10        # near alarm → natural arousal creep
    )

    # Deep sleep is more resistant, but not immune
    if current_stage == "deep":
        base_prob *= 0.5

    return min(base_prob, 0.85)  # cap so disturbances never feel deterministic


def _sample_disturbance(
    depth: float,
    movement: float,
    noise: float,
    current_stage: str,
    time_until_alarm: int,
) -> Disturbance:
    """Decide whether a disturbance occurs and, if so, which kind.

    Returns QUIET (no_disturbance) most of the time.
    When a disturbance fires, its magnitude scales mildly with instability.

    Disturbance menu:
      sudden_noise_spike      — environmental noise burst (traffic, neighbour, etc.)
      movement_spike          — restless limb or body shift
      snore_like_disturbance  — rhythmic noise + slight movement, partial arousal
    """
    prob = _disturbance_probability(depth, movement, noise, current_stage, time_until_alarm)

    if random.random() > prob:
        return QUIET

    # Instability scalar: louder/shallower → stronger events
    strength = 0.5 + (noise * 0.3) + ((1.0 - depth) * 0.2)

    kind = random.choice([SUDDEN_NOISE_SPIKE, MOVEMENT_SPIKE, SNORE_LIKE_DISTURBANCE])

    if kind == SUDDEN_NOISE_SPIKE:
        return Disturbance(
            kind=kind,
            delta_noise=round(random.uniform(0.15, 0.30) * strength, 3),
            delta_depth=round(-random.uniform(0.04, 0.10), 3),
        )

    if kind == MOVEMENT_SPIKE:
        return Disturbance(
            kind=kind,
            delta_movement=round(random.uniform(0.12, 0.25) * strength, 3),
            delta_depth=round(-random.uniform(0.03, 0.08), 3),
        )

    # snore_like_disturbance: noisy + slightly restless + partial arousal
    return Disturbance(
        kind=kind,
        delta_noise=round(random.uniform(0.08, 0.18) * strength, 3),
        delta_movement=round(random.uniform(0.05, 0.12) * strength, 3),
        delta_depth=round(-random.uniform(0.05, 0.12), 3),
    )


def _apply_disturbance(
    depth: float,
    movement: float,
    noise: float,
    disturbance: Disturbance,
) -> tuple[float, float, float]:
    """Add a disturbance's deltas to the current signal values.

    Clamping is handled later by SleepState.updated().

    Returns:
        Updated (depth, movement, noise) after the disturbance.
    """
    return (
        depth + disturbance.delta_depth,
        movement + disturbance.delta_movement,
        noise + disturbance.delta_noise,
    )


# ── Main transition function ──────────────────────────────────────────────────

def transition(
    state: SleepState,
    action: str,
) -> tuple["SleepState", Disturbance]:
    """Apply an action to the current state and return the next state + disturbance.

    Steps:
      1. Copy continuous signals from the current state.
      2. Apply action-specific deltas (interventions).
      3. Add small random drift.
      4. Sample a stochastic disturbance event.
      5. Apply the disturbance to the (already-modified) signals.
      6. Infer the next sleep stage using sleep-cycle-aware weights.
      7. Return a brand-new SleepState and the disturbance that occurred.

    The original state is never mutated.

    Args:
        state: The current sleep state.
        action: One of the action constants from simulation.actions.

    Returns:
        (next_state, disturbance) — the resulting state and what event fired.
    """
    depth = state.sleep_depth
    movement = state.movement
    noise = state.noise_level
    time_until_alarm = max(0, state.time_until_alarm - 1)

    # ── Step 2: action effects ────────────────────────────────────────────────

    if action == DO_NOTHING:
        # Natural drift — sleep evolves on its own
        depth += random.uniform(-_DRIFT, _DRIFT)
        movement += random.uniform(-_DRIFT, _DRIFT)
        noise += random.uniform(-_DRIFT / 2, _DRIFT / 2)

    elif action == PLAY_BROWN_NOISE:
        # Strong masking: significantly reduces perceived noise,
        # which also shrinks the disturbance probability at the next step
        noise -= 0.12 + random.uniform(0.0, 0.05)
        depth += random.uniform(0.0, 0.04)
        movement += random.uniform(-_DRIFT, _DRIFT)

    elif action == PLAY_RAIN:
        # Gentler masking: moderate noise reduction
        noise -= 0.06 + random.uniform(0.0, 0.03)
        depth += random.uniform(0.0, 0.02)
        movement += random.uniform(-_DRIFT, _DRIFT)

    elif action == BREATHING_PACING:
        # Slow guided breathing: calms body (↓ movement) and slightly deepens sleep
        movement -= 0.08 + random.uniform(0.0, 0.04)
        depth += 0.04 + random.uniform(0.0, 0.03)
        noise += random.uniform(-_DRIFT / 2, _DRIFT / 2)

    elif action == GRADUAL_WAKE:
        # Only ramps wakefulness when the alarm is close
        if time_until_alarm <= 30:
            depth -= 0.10 + random.uniform(0.0, 0.05)
            movement += 0.06 + random.uniform(0.0, 0.04)
        else:
            depth -= 0.01
            movement += random.uniform(0.0, _DRIFT)

    else:
        raise ValueError(f"Unknown action: '{action}'")

    # ── Step 4–5: stochastic disturbance ─────────────────────────────────────
    # Sampled *after* action effects so that noise-reducing actions
    # lower the disturbance probability for this same step.

    disturbance = _sample_disturbance(depth, movement, noise, state.sleep_stage, time_until_alarm)
    depth, movement, noise = _apply_disturbance(depth, movement, noise, disturbance)

    # ── Step 6: infer stage from updated signals ──────────────────────────────

    new_stage = _infer_stage(depth, movement, noise, time_until_alarm, state.sleep_stage)

    next_state = state.updated(
        sleep_stage=new_stage,
        sleep_depth=depth,
        movement=movement,
        noise_level=noise,
        time_until_alarm=time_until_alarm,
    )

    return next_state, disturbance
