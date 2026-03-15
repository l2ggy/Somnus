"""LLM-powered world model agent.

This module is the new trajectory-generation path for Somnus.
The original rule-based simulator (simulation/sleep_dynamics.py,
simulation/simulator.py) remains fully intact as the fallback/baseline.

Architecture
------------
                        SleepState + action
                               │
               ┌───────────────┴────────────────┐
               │                                │
    build_world_model_prompt()        [rule-based simulator]
               │                      simulation/simulator.py
         LLM call via                          │
         llm_client(prompt)           simulate() / simulate_trajectory()
               │                                │
    parse_action_trajectory_bundle()            │
               │                                │
         ActionTrajectoryBundle          cumulative float score
               │                                │
               └───────────────┬────────────────┘
                                │
                    reward.py can score either path:
                    sleep_reward(TrajectoryStep → SleepState)

LLM client contract
-------------------
llm_client is any callable with the signature:
    (prompt: str) -> str
It should return the raw model output text. The caller is responsible for
instantiating whatever SDK they use (Anthropic, OpenAI, local, etc.) and
wrapping it into this single-argument callable before passing it here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from simulation.sleep_state import SleepState, VALID_STAGES
from simulation.actions import ALL_ACTIONS
from simulation.user_profile import UserProfile


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """One minute-step in a simulated sleep trajectory.

    Fields mirror SleepState so that steps can be trivially converted for
    scoring with the existing reward function.

    Attributes:
        step_index:  1-based position in the trajectory.
        sleep_stage: One of "wake", "light", "deep", "rem".
        sleep_depth: 0.0 (fully awake) to 1.0 (deepest sleep).
        movement:    0.0 (still) to 1.0 (very restless).
        noise_level: 0.0 (silent) to 1.0 (very loud).
        wake_risk:   0.0 (no risk) to 1.0 (certain wake). LLM-estimated.
        note:        Optional free-text explanation for this step.
    """

    step_index: int
    sleep_stage: str
    sleep_depth: float
    movement: float
    noise_level: float
    wake_risk: float
    note: str | None = None

    def to_sleep_state(self, time_until_alarm: int) -> SleepState:
        """Convert this step back into a SleepState for reward scoring.

        Args:
            time_until_alarm: Passed through unchanged (not tracked per-step).

        Returns:
            A SleepState equivalent of this trajectory step.
        """
        return SleepState(
            sleep_stage=self.sleep_stage,
            sleep_depth=self.sleep_depth,
            movement=self.movement,
            noise_level=self.noise_level,
            time_until_alarm=time_until_alarm,
        )


@dataclass
class CandidateTrajectory:
    """One plausible future, as imagined by the LLM world model.

    The LLM is asked to generate several of these per action, reflecting
    genuine uncertainty about how sleep will evolve.

    Attributes:
        probability: LLM-assigned likelihood of this scenario (should sum
                     to ~1.0 across all candidates for the same action).
        steps:       Ordered sequence of TrajectorySteps.
        summary:     Short natural-language description of this scenario,
                     e.g. "sleeper remains in light sleep throughout".
    """

    probability: float
    steps: list[TrajectoryStep]
    summary: str


@dataclass
class ActionTrajectoryBundle:
    """All candidate trajectories for a single candidate action.

    Attributes:
        action:       The intervention being evaluated (from ALL_ACTIONS).
        trajectories: 3–5 candidate futures, each with its own probability.
        rationale:    LLM explanation of why this action affects sleep the
                      way it does (useful for logging and user-facing output).
    """

    action: str
    trajectories: list[CandidateTrajectory]
    rationale: str


# ── Prompt builder ─────────────────────────────────────────────────────────────

# Compact action descriptions injected into the prompt so the LLM understands
# the intervention semantics without needing access to the codebase.
_ACTION_DESCRIPTIONS: dict[str, str] = {
    "do_nothing":        "no intervention; sleep evolves naturally with random drift",
    "play_brown_noise":  "strong masking sound; significantly reduces perceived noise level",
    "play_rain":         "gentle masking sound; moderately reduces noise level",
    "breathing_pacing":  "slow guided breathing audio; reduces movement, slightly deepens sleep",
    "gradual_wake":      "soft light/sound ramp; only meaningful when time_until_alarm is low (≤30 min)",
}

_JSON_SCHEMA = """\
{
  "action": "<action_name>",
  "rationale": "<why this action affects sleep this way>",
  "trajectories": [
    {
      "probability": <float 0-1>,
      "summary": "<one-sentence scenario description>",
      "steps": [
        {
          "step_index": <int, starting at 1>,
          "sleep_stage": "<wake|light|deep|rem>",
          "sleep_depth": <float 0.0-1.0>,
          "movement": <float 0.0-1.0>,
          "noise_level": <float 0.0-1.0>,
          "wake_risk": <float 0.0-1.0>,
          "note": "<optional short explanation or null>"
        }
      ]
    }
  ]
}"""


def build_world_model_prompt(
    state: SleepState,
    action: str,
    user_profile: UserProfile | None = None,
    horizon: int = 5,
) -> str:
    """Build the LLM prompt for imagining future sleep trajectories.

    The prompt instructs the model to act as a probabilistic sleep world model
    and return a structured JSON bundle of 3–5 candidate futures for the given
    action, starting from the provided sleep state.

    Args:
        state:        Current sleep state (the starting point for all trajectories).
        action:       The intervention to evaluate.
        user_profile: Optional user preferences, injected as context so the
                      LLM can reflect personal sensitivities in its reasoning.
        horizon:      Number of trajectory steps to generate (each step ≈ 1 minute).

    Returns:
        A self-contained prompt string ready to send to any LLM.
    """
    if action not in ALL_ACTIONS:
        raise ValueError(f"Unknown action '{action}'. Must be one of: {ALL_ACTIONS}")

    action_desc = _ACTION_DESCRIPTIONS[action]

    # Build optional user context block
    user_context = ""
    if user_profile is not None:
        pref = user_profile.action_preferences.get(action, 0.0)
        pref_label = "positive" if pref > 0.1 else ("negative" if pref < -0.1 else "neutral")
        user_context = f"""\

User profile context:
- User preference for this action: {pref_label} (score {pref:+.1f})
- Noise sensitivity multiplier: {user_profile.sensitivity_to_noise:.1f}
- Intervention tolerance: {user_profile.intervention_tolerance:.1f}
Reflect these preferences in which trajectories you weight as more likely.\
"""

    prompt = f"""\
You are a probabilistic world model for a sleep-tracking system called Somnus.
Your job is to imagine plausible future sleep trajectories given the current
biometric state and a proposed intervention.

Current sleep state:
- sleep_stage: {state.sleep_stage}
- sleep_depth: {state.sleep_depth:.2f}  (0.0 = fully awake, 1.0 = deepest sleep)
- movement:    {state.movement:.2f}     (0.0 = still, 1.0 = very restless)
- noise_level: {state.noise_level:.2f}  (0.0 = silent, 1.0 = very loud)
- time_until_alarm: {state.time_until_alarm} minutes
{user_context}
Proposed intervention: {action}
Intervention description: {action_desc}

Sleep cycle priors to reflect in your trajectories:
- Deep sleep is more likely earlier in the night (high time_until_alarm).
- REM sleep is more likely later in the night (low time_until_alarm).
- Light sleep acts as the natural transition state between all other stages.
- High noise or high movement increase the probability of waking.
- Deep sleep is somewhat resistant to disturbances but not immune.

Task:
Generate {horizon} steps ({horizon} minutes) of simulated sleep for this intervention.
Produce 3 to 5 candidate trajectories that reflect genuine uncertainty about
how sleep will evolve. Trajectories should differ meaningfully — not just be
small random perturbations of each other.

Probability rules:
- All trajectory probabilities must be non-negative and sum to approximately 1.0.
- Weight trajectories by how plausible they are given the current state and intervention.

Return ONLY valid JSON matching this schema exactly. No markdown, no commentary:

{_JSON_SCHEMA}
"""
    return prompt.strip()


# ── Parser / validator ────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _parse_step(raw: dict[str, Any], action: str) -> TrajectoryStep:
    """Parse and validate a single trajectory step dict."""
    stage = raw.get("sleep_stage", "")
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid sleep_stage '{stage}' in action '{action}'. "
            f"Must be one of {sorted(VALID_STAGES)}."
        )
    return TrajectoryStep(
        step_index=int(raw["step_index"]),
        sleep_stage=stage,
        sleep_depth=_clamp(float(raw["sleep_depth"])),
        movement=_clamp(float(raw["movement"])),
        noise_level=_clamp(float(raw["noise_level"])),
        wake_risk=_clamp(float(raw["wake_risk"])),
        note=raw.get("note") or None,
    )


def _parse_trajectory(raw: dict[str, Any], action: str) -> CandidateTrajectory:
    """Parse and validate a single candidate trajectory dict."""
    prob = _clamp(float(raw["probability"]))
    steps = [_parse_step(s, action) for s in raw["steps"]]
    if not steps:
        raise ValueError(f"Trajectory for action '{action}' has no steps.")
    return CandidateTrajectory(
        probability=prob,
        steps=steps,
        summary=str(raw.get("summary", "")),
    )


def parse_action_trajectory_bundle(raw_text: str) -> ActionTrajectoryBundle:
    """Parse and validate the LLM's raw JSON response into an ActionTrajectoryBundle.

    Performs structural validation, sleep-stage validation, and numeric
    clamping. Raises ValueError with a descriptive message on any failure
    so callers can fall back to the rule-based simulator cleanly.

    Args:
        raw_text: Raw text returned by the LLM (should be pure JSON).

    Returns:
        A validated ActionTrajectoryBundle.

    Raises:
        ValueError: If the JSON is malformed, fields are missing, or values
                    are outside valid ranges that cannot be clamped.
    """
    # Strip markdown code fences if the model wrapped its output
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\n\nRaw text:\n{raw_text}") from exc

    action = str(data.get("action", ""))
    if action not in ALL_ACTIONS:
        raise ValueError(
            f"LLM returned unknown action '{action}'. "
            f"Expected one of: {ALL_ACTIONS}."
        )

    raw_trajectories = data.get("trajectories")
    if not isinstance(raw_trajectories, list) or len(raw_trajectories) == 0:
        raise ValueError(f"Expected a non-empty list of trajectories for action '{action}'.")

    trajectories = [_parse_trajectory(t, action) for t in raw_trajectories]

    return ActionTrajectoryBundle(
        action=action,
        trajectories=trajectories,
        rationale=str(data.get("rationale", "")),
    )


# ── Generation entry point ────────────────────────────────────────────────────

def generate_action_trajectories(
    state: SleepState,
    action: str,
    user_profile: UserProfile | None = None,
    horizon: int = 5,
    llm_client: Any | None = None,
) -> ActionTrajectoryBundle:
    """Generate LLM-imagined future trajectories for a candidate action.

    This is the main entry point for the LLM world model path. The rule-based
    simulator in simulation/simulator.py remains available as a fallback and
    can score or validate the trajectories produced here via sleep_reward().

    Args:
        state:        Current sleep state.
        action:       The intervention to evaluate (must be in ALL_ACTIONS).
        user_profile: Optional user preferences injected into the prompt.
        horizon:      Number of steps (minutes) to simulate.
        llm_client:   A callable (prompt: str) -> str. Wrap your SDK of
                      choice into this shape before passing it here.
                      If None, raises NotImplementedError.

    Returns:
        A parsed and validated ActionTrajectoryBundle.

    Raises:
        NotImplementedError: If llm_client is None.
        ValueError:          If the LLM response cannot be parsed or validated.

    Example llm_client wrappers
    ---------------------------
    # Anthropic
    import anthropic
    client = anthropic.Anthropic()
    def llm_client(prompt: str) -> str:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    # OpenAI-compatible
    from openai import OpenAI
    client = OpenAI()
    def llm_client(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    """
    if llm_client is None:
        raise NotImplementedError(
            "llm_client is required. Wrap your LLM SDK into a callable "
            "(prompt: str) -> str and pass it as llm_client=. "
            "See the docstring for Anthropic and OpenAI examples. "
            "To use the rule-based fallback, call simulate() or "
            "simulate_trajectory() from simulation/simulator.py instead."
        )

    prompt = build_world_model_prompt(state, action, user_profile=user_profile, horizon=horizon)
    raw_response = llm_client(prompt)
    return parse_action_trajectory_bundle(raw_response)


# ── Example usage (not executed on import) ────────────────────────────────────

def _example() -> None:
    """Illustrates the intended usage pattern.

    Run directly:  python -m agents.llm_world_model_agent
    """
    from simulation.sleep_state import SleepState
    from simulation.user_profile import UserProfile
    from simulation.reward import sleep_reward

    state = SleepState(
        sleep_stage="light",
        sleep_depth=0.42,
        movement=0.30,
        noise_level=0.50,
        time_until_alarm=180,
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

    # ── Inspect the prompt without calling an LLM ──────────────────────────
    prompt = build_world_model_prompt(
        state, "play_brown_noise", user_profile=profile, horizon=5
    )
    print("=== Prompt (first 800 chars) ===")
    print(prompt[:800])
    print("...\n")

    # ── Wire up a real LLM (Anthropic example) ─────────────────────────────
    # import anthropic
    # client = anthropic.Anthropic()
    # def llm_client(prompt: str) -> str:
    #     msg = client.messages.create(
    #         model="claude-opus-4-6",
    #         max_tokens=2048,
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     return msg.content[0].text
    #
    # bundle = generate_action_trajectories(
    #     state, "play_brown_noise", user_profile=profile, horizon=5,
    #     llm_client=llm_client,
    # )
    # print(f"Action: {bundle.action}")
    # print(f"Rationale: {bundle.rationale}\n")
    # for i, traj in enumerate(bundle.trajectories, 1):
    #     score = sum(
    #         sleep_reward(step.to_sleep_state(state.time_until_alarm - step.step_index))
    #         for step in traj.steps
    #     )
    #     print(f"  Trajectory {i} (p={traj.probability:.2f}, score={score:.3f}): {traj.summary}")

    # ── Manual parse test with a synthetic response ────────────────────────
    synthetic = json.dumps({
        "action": "play_brown_noise",
        "rationale": "Brown noise masks ambient sound, lowering noise level and reducing arousal risk.",
        "trajectories": [
            {
                "probability": 0.55,
                "summary": "Sleeper deepens into deep sleep as noise drops.",
                "steps": [
                    {"step_index": 1, "sleep_stage": "light", "sleep_depth": 0.48,
                     "movement": 0.27, "noise_level": 0.35, "wake_risk": 0.10, "note": "noise masking begins"},
                    {"step_index": 2, "sleep_stage": "deep",  "sleep_depth": 0.68,
                     "movement": 0.20, "noise_level": 0.22, "wake_risk": 0.04, "note": None},
                ],
            },
            {
                "probability": 0.45,
                "summary": "Sleeper stays in light sleep despite noise reduction.",
                "steps": [
                    {"step_index": 1, "sleep_stage": "light", "sleep_depth": 0.44,
                     "movement": 0.30, "noise_level": 0.38, "wake_risk": 0.15, "note": None},
                    {"step_index": 2, "sleep_stage": "light", "sleep_depth": 0.46,
                     "movement": 0.28, "noise_level": 0.25, "wake_risk": 0.12, "note": None},
                ],
            },
        ],
    })

    bundle = parse_action_trajectory_bundle(synthetic)
    print(f"Parsed bundle — action: {bundle.action}")
    print(f"Rationale: {bundle.rationale}")
    for i, traj in enumerate(bundle.trajectories, 1):
        score = sum(
            sleep_reward(step.to_sleep_state(state.time_until_alarm - step.step_index))
            for step in traj.steps
        )
        print(f"  Trajectory {i}: p={traj.probability:.2f}  score={score:.3f}  → {traj.summary}")


if __name__ == "__main__":
    import json  # noqa: F811 — re-import for __main__ block clarity
    _example()
