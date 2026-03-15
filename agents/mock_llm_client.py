"""Mock LLM client for testing the LLM pipeline without a real API call.

This module is a local test harness only.
The rule-based simulator (simulation/simulator.py) remains the baseline/fallback.
The real LLM path (agents/llm_world_model_agent.py) calls a live model backend.
This mock approximates action-conditioned trajectories just well enough to
exercise parse_action_trajectory_bundle(), score_action_bundle(), and the
full llm_planner_agent flow end-to-end.

Usage:
    from agents.mock_llm_client import MockLLMClient
    client = MockLLMClient(seed=42)
    # pass as llm_client= anywhere the planner expects a (prompt)->str callable
    bundle = generate_action_trajectories(state, "play_brown_noise", llm_client=client)
"""

from __future__ import annotations

import json
import random
import re


# ── Per-action signal templates ───────────────────────────────────────────────
# Each entry describes the *direction* of effect on the three key signals.
# Values are (delta_noise, delta_movement, delta_depth) per step, with small
# random noise added during generation.

_ACTION_EFFECTS: dict[str, dict] = {
    "do_nothing": {
        "rationale": (
            "No intervention applied. Sleep evolves naturally with random drift. "
            "Existing instability tends to persist or worsen without external support."
        ),
        "delta_noise":    0.00,   # no masking; noise drifts randomly
        "delta_movement": 0.00,
        "delta_depth":    0.00,
        "wake_risk_base": 0.20,   # moderate baseline risk
        "stage_bias": "light",    # most likely stage outcome
    },
    "play_brown_noise": {
        "rationale": (
            "Brown noise provides strong acoustic masking, substantially reducing "
            "perceived noise level each step. Lower noise lets the sleeper consolidate "
            "deeper stages and reduces disturbance-triggered arousals."
        ),
        "delta_noise":    -0.12,
        "delta_movement": -0.02,
        "delta_depth":     0.05,
        "wake_risk_base":  0.06,
        "stage_bias": "deep",
    },
    "play_rain": {
        "rationale": (
            "Rain sounds provide gentle acoustic masking. Noise reduction is moderate "
            "compared to brown noise. Sleep depth improves slightly; wake risk is "
            "reduced but not eliminated."
        ),
        "delta_noise":    -0.06,
        "delta_movement": -0.01,
        "delta_depth":     0.02,
        "wake_risk_base":  0.11,
        "stage_bias": "light",
    },
    "breathing_pacing": {
        "rationale": (
            "Guided slow breathing reduces physical restlessness each step and "
            "mildly deepens sleep by promoting parasympathetic arousal. Wake risk "
            "drops as movement decreases."
        ),
        "delta_noise":     0.00,
        "delta_movement": -0.07,
        "delta_depth":     0.04,
        "wake_risk_base":  0.08,
        "stage_bias": "rem",
    },
    "gradual_wake": {
        "rationale": (
            "Gradual wake gently escalates light and sound to raise the sleeper "
            "toward wakefulness. Wake risk increases each step. Effective only when "
            "the alarm is close; otherwise the effect is minimal."
        ),
        "delta_noise":     0.03,
        "delta_movement":  0.05,
        "delta_depth":    -0.08,
        "wake_risk_base":  0.35,
        "stage_bias": "wake",
    },
}

# Trajectory scenarios: (probability_weight, scenario_label, modifier)
# Modifier shifts the per-step deltas to create meaningfully different futures.
_SCENARIOS: list[tuple[float, str, dict]] = [
    (0.45, "best_case",   {"depth": +0.05, "noise": -0.03, "move": -0.02, "risk": -0.06}),
    (0.30, "typical",     {"depth":  0.00, "noise":  0.00, "move":  0.00, "risk":  0.00}),
    (0.15, "mediocre",    {"depth": -0.03, "noise": +0.04, "move": +0.03, "risk": +0.08}),
    (0.10, "disruption",  {"depth": -0.07, "noise": +0.10, "move": +0.08, "risk": +0.18}),
]

_SCENARIO_SUMMARIES: dict[str, dict[str, str]] = {
    "do_nothing": {
        "best_case":  "Sleep stabilises naturally; no disturbances fire during the window.",
        "typical":    "Sleep drifts with minor instability; stage stays light throughout.",
        "mediocre":   "Noise fluctuations prevent deeper sleep; brief light-to-wake transitions.",
        "disruption": "Unaddressed environmental noise causes repeated micro-arousals.",
    },
    "play_brown_noise": {
        "best_case":  "Brown noise masks all disturbances; sleeper consolidates into deep sleep.",
        "typical":    "Noise drops steadily; sleeper remains in light sleep with reduced risk.",
        "mediocre":   "Masking helps but underlying restlessness limits depth improvement.",
        "disruption": "A loud spike breaks through masking; sleeper briefly wakes before settling.",
    },
    "play_rain": {
        "best_case":  "Rain sounds gently mask disturbances; sleep depth improves moderately.",
        "typical":    "Noise reduces at a mild rate; sleeper stays in light or REM sleep.",
        "mediocre":   "Partial masking is insufficient against a noisy environment.",
        "disruption": "Rain sound is too gentle; a disturbance causes a transient arousal.",
    },
    "breathing_pacing": {
        "best_case":  "Guided breathing calms the body fully; sleeper enters restful REM.",
        "typical":    "Movement reduces each step; sleep depth gradually improves.",
        "mediocre":   "Breathing pacing helps movement but cannot address ambient noise.",
        "disruption": "A noise spike interrupts the pacing rhythm; movement spikes briefly.",
    },
    "gradual_wake": {
        "best_case":  "Sleeper transitions smoothly from light sleep into wakefulness.",
        "typical":    "Sleep lightens each step; wake risk rises toward the alarm window.",
        "mediocre":   "Gradual wake is slightly premature; sleeper resists arousal.",
        "disruption": "Stimulation is too abrupt; sleeper wakes with elevated movement.",
    },
}

_STAGE_ORDER = ["wake", "light", "rem", "deep"]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Prompt parsing helpers ────────────────────────────────────────────────────

def _extract_action(prompt: str) -> str:
    """Pull the action name out of the prompt via simple pattern matching."""
    match = re.search(r"Proposed intervention:\s*(\S+)", prompt)
    if match:
        candidate = match.group(1).strip()
        from simulation.actions import ALL_ACTIONS
        if candidate in ALL_ACTIONS:
            return candidate
    # Fallback: search for any known action name
    from simulation.actions import ALL_ACTIONS
    for action in ALL_ACTIONS:
        if action in prompt:
            return action
    return "do_nothing"


def _extract_time_until_alarm(prompt: str) -> int:
    """Extract time_until_alarm from the prompt; defaults to 240 if not found."""
    match = re.search(r"time_until_alarm[:\s]+(\d+)", prompt)
    return int(match.group(1)) if match else 240


def _extract_noise_level(prompt: str) -> float:
    """Extract noise_level from the prompt; defaults to 0.3 if not found."""
    match = re.search(r"noise_level[:\s]+([\d.]+)", prompt)
    return float(match.group(1)) if match else 0.3


def _extract_horizon(prompt: str) -> int:
    """Extract the requested trajectory horizon from the prompt."""
    match = re.search(r"Generate (\d+) steps", prompt)
    return int(match.group(1)) if match else 5


# ── Step generation ───────────────────────────────────────────────────────────

def _infer_stage(depth: float, wake_risk: float, stage_bias: str) -> str:
    """Choose a sleep stage consistent with depth and wake_risk."""
    if depth < 0.12 or wake_risk > 0.75:
        return "wake"
    if depth > 0.70 and wake_risk < 0.10:
        return "deep"
    if stage_bias in ("rem", "deep") and 0.35 <= depth <= 0.75 and wake_risk < 0.20:
        return stage_bias
    return "light"


def _build_steps(
    action: str,
    horizon: int,
    modifier: dict,
    starting_noise: float,
    near_alarm: bool,
    rng: random.Random,
) -> list[dict]:
    """Generate one trajectory's worth of steps for a given action + scenario."""
    effects = _ACTION_EFFECTS[action]

    # Starting values — moderate defaults, anchored to extracted noise
    depth = 0.45
    movement = 0.28
    noise = _clamp(starting_noise)
    wake_risk = effects["wake_risk_base"]

    # gradual_wake is only impactful near the alarm; dampen otherwise
    gradual_wake_scale = 1.0 if near_alarm else 0.15

    steps = []
    for i in range(1, horizon + 1):
        # Apply per-step action effect + scenario modifier + small jitter
        d_noise    = (effects["delta_noise"]    + modifier["noise"]) * (1.0 if action != "gradual_wake" else gradual_wake_scale)
        d_movement = (effects["delta_movement"] + modifier["move"])  * (1.0 if action != "gradual_wake" else gradual_wake_scale)
        d_depth    = (effects["delta_depth"]    + modifier["depth"]) * (1.0 if action != "gradual_wake" else gradual_wake_scale)
        d_risk     = modifier["risk"]

        noise     = _clamp(noise     + d_noise    + rng.uniform(-0.02, 0.02))
        movement  = _clamp(movement  + d_movement + rng.uniform(-0.02, 0.02))
        depth     = _clamp(depth     + d_depth    + rng.uniform(-0.02, 0.02))
        wake_risk = _clamp(wake_risk + d_risk     + rng.uniform(-0.02, 0.02))

        stage = _infer_stage(depth, wake_risk, effects["stage_bias"])

        steps.append({
            "step_index":  i,
            "sleep_stage": stage,
            "sleep_depth": round(depth, 3),
            "movement":    round(movement, 3),
            "noise_level": round(noise, 3),
            "wake_risk":   round(wake_risk, 3),
            "note":        None,
        })

    return steps


# ── MockLLMClient ─────────────────────────────────────────────────────────────

class MockLLMClient:
    """Fake LLM client that returns valid action-trajectory JSON.

    Intended only for local testing of the LLM pipeline. Parses the action
    and a few state hints from the prompt text so the output is plausibly
    action-conditioned without calling a real model.

    Implements both __call__(prompt) and generate(prompt) so it can be passed
    wherever the planner expects a callable or an object with a generate method.

    Args:
        seed: Optional random seed for reproducible mock outputs.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    def generate(self, prompt: str) -> str:
        """Generate a mock trajectory bundle JSON string for the given prompt.

        Args:
            prompt: The full prompt string built by build_world_model_prompt().

        Returns:
            A JSON string that parse_action_trajectory_bundle() will accept.
        """
        rng = random.Random(self._seed)

        action = _extract_action(prompt)
        time_until_alarm = _extract_time_until_alarm(prompt)
        starting_noise = _extract_noise_level(prompt)
        horizon = _extract_horizon(prompt)
        near_alarm = time_until_alarm <= 45

        effects = _ACTION_EFFECTS[action]
        summaries = _SCENARIO_SUMMARIES[action]

        # Normalise scenario weights to probabilities
        total_weight = sum(w for w, *_ in _SCENARIOS)
        trajectories = []
        for weight, label, modifier in _SCENARIOS:
            prob = round(weight / total_weight, 4)
            steps = _build_steps(action, horizon, modifier, starting_noise, near_alarm, rng)
            trajectories.append({
                "probability": prob,
                "summary":     summaries[label],
                "steps":       steps,
            })

        payload = {
            "action":      action,
            "rationale":   effects["rationale"],
            "trajectories": trajectories,
        }
        return json.dumps(payload)

    def __call__(self, prompt: str) -> str:
        """Allow the client to be passed directly as a callable llm_client."""
        return self.generate(prompt)


# ── Example (run directly) ────────────────────────────────────────────────────

def _example() -> None:
    """End-to-end test of the mock through the full LLM planner pipeline.

    Run:  python -m agents.mock_llm_client
    """
    from simulation.sleep_state import SleepState
    from simulation.user_profile import UserProfile
    from agents.llm_planner_agent import rank_actions_with_llm, choose_best_action_with_llm

    state = SleepState(
        sleep_stage="light",
        sleep_depth=0.42,
        movement=0.30,
        noise_level=0.55,
        time_until_alarm=180,
    )

    profile = UserProfile(
        action_preferences={
            "play_brown_noise":  0.8,
            "play_rain":        -0.5,
            "breathing_pacing":  0.3,
            "gradual_wake":     -1.0,
            "do_nothing":        0.0,
        }
    )

    client = MockLLMClient(seed=42)

    print("=== rank_actions_with_llm (mock) ===")
    ranked = rank_actions_with_llm(state, user_profile=profile, horizon=5, llm_client=client)
    for action, score, bundle in ranked:
        print(f"  {action:<24} score={score:+.3f}  ({len(bundle.trajectories)} trajectories)")

    print()
    best_action, best_score, best_bundle = choose_best_action_with_llm(
        state, user_profile=profile, horizon=5, llm_client=client
    )
    print(f"=== choose_best_action_with_llm (mock) ===")
    print(f"  Best action : {best_action}")
    print(f"  Score       : {best_score:+.3f}")
    print(f"  Rationale   : {best_bundle.rationale}")
    print()
    print("  Trajectories:")
    for i, traj in enumerate(best_bundle.trajectories, 1):
        first, last = traj.steps[0], traj.steps[-1]
        print(
            f"    [{i}] p={traj.probability:.2f}  "
            f"stage: {first.sleep_stage}→{last.sleep_stage}  "
            f"noise: {first.noise_level:.2f}→{last.noise_level:.2f}  "
            f"— {traj.summary}"
        )


if __name__ == "__main__":
    _example()
