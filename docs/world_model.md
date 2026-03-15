# Somnus — World Model Documentation

## Overview

Somnus uses a **world model** to select sleep interventions during the night. The world model predicts how a sleeper's biometric state will evolve under each possible action, then picks the action with the best expected outcome.

Two planning paths exist side by side:

| Path | Module | When used |
|---|---|---|
| **Rule-based (baseline)** | `agents/world_model_agent.py` | Always available; used as fallback |
| **LLM-powered** | `agents/llm_planner_agent.py` | Preferred when an LLM client is available |
| **Hybrid** | `agents/hybrid_planner_agent.py` | Normal runtime entry point |

The rule-based simulator is never removed. Every LLM-path component is additive.

---

## Architecture

```
SleepState + UserProfile
        │
choose_best_action_hybrid()          ← normal entry point
        │
   ┌────┴────────────────────────────┐
   │  LLM path (preferred)           │  Rule-based path (fallback)
   │                                 │
   │  build_world_model_prompt()     │  simulate() × n_rollouts
   │  llm_client(prompt)             │  sleep_reward() per step
   │  parse_action_trajectory_bundle │  mean score per action
   │  score_action_bundle()          │
   └────────────────┬────────────────┘
                    │
             PlannerResult
                    │
          JournalFeedback (morning)
                    │
   update_user_profile_from_feedback()
                    │
          updated UserProfile  →  next night
```

---

## Simulation Package

### `simulation/sleep_state.py`

**`SleepState`** — snapshot of the sleeper's biometric and environmental conditions at one moment in time.

```python
@dataclass
class SleepState:
    sleep_stage: str    # "wake" | "light" | "deep" | "rem"
    sleep_depth: float  # 0.0 (awake) → 1.0 (deepest sleep)
    movement: float     # 0.0 (still) → 1.0 (very restless)
    noise_level: float  # 0.0 (silent) → 1.0 (very loud)
    time_until_alarm: int  # minutes remaining
```

Key methods:
- `clamp()` — clamps all continuous fields to valid ranges in place (called automatically on init)
- `updated(**changes) -> SleepState` — returns a new clamped instance with the given fields replaced; the original is never mutated

---

### `simulation/actions.py`

The five available interventions, as string constants and a list:

| Constant | Value | Effect |
|---|---|---|
| `DO_NOTHING` | `"do_nothing"` | Sleep evolves with random drift |
| `PLAY_BROWN_NOISE` | `"play_brown_noise"` | Strongly masks environmental noise |
| `PLAY_RAIN` | `"play_rain"` | Gently masks noise |
| `BREATHING_PACING` | `"breathing_pacing"` | Reduces movement, slightly deepens sleep |
| `GRADUAL_WAKE` | `"gradual_wake"` | Eases toward wakefulness (only near alarm) |

`ALL_ACTIONS` — ordered list used by all planners to iterate candidates.

---

### `simulation/disturbance.py`

Stochastic events that can disrupt sleep at any step, sampled inside `transition()`.

**`Disturbance`** dataclass:
```python
@dataclass(frozen=True)
class Disturbance:
    kind: str           # one of the constants below
    delta_noise: float
    delta_movement: float
    delta_depth: float
```

| Constant | Description |
|---|---|
| `NO_DISTURBANCE` | Nothing happened this step |
| `SUDDEN_NOISE_SPIKE` | Environmental noise burst |
| `MOVEMENT_SPIKE` | Restless limb or body shift |
| `SNORE_LIKE_DISTURBANCE` | Rhythmic noise + slight movement |

`QUIET` — singleton `Disturbance(kind=NO_DISTURBANCE)` used as the no-event sentinel.

Disturbance probability at each step scales with instability: high noise + high movement + shallow depth all increase the chance of an event. Deep sleep applies a ×0.5 resistance discount.

---

### `simulation/sleep_dynamics.py`

**`transition(state, action) -> tuple[SleepState, Disturbance]`**

The core simulation step. Given a state and an action, returns the next state and the disturbance that fired.

Execution order:
1. Apply action-specific signal deltas (noise reduction, movement calming, depth change)
2. Add small random drift
3. Sample a stochastic `Disturbance` on the already-modified signals (so noise-reducing actions lower disturbance probability this same step)
4. Apply the disturbance deltas
5. Infer the next `sleep_stage` via weighted sampling

Sleep cycle priors encoded in stage inference:
- **Deep sleep** is favoured early in the night (high `time_until_alarm`)
- **REM** is favoured in the second half (low `time_until_alarm`)
- **Light sleep** always has a nonzero weight — acts as the universal transition buffer
- **Wake** weight rises with noise, movement, and shallow depth
- Inertia: current stage gets a small ×1.2 boost to prevent instant stage flipping

---

### `simulation/reward.py`

**`sleep_reward(state: SleepState) -> float`**

Scores a single sleep state. Used by both the rule-based simulator and the trajectory critic.

| Component | Formula | Range |
|---|---|---|
| Stage base | `deep=+1.0, rem=+0.6, light=+0.2, wake=−1.0` | |
| Noise penalty | `−0.3 × noise_level` | 0 to −0.30 |
| Movement penalty | `−0.2 × movement` | 0 to −0.20 |
| **Total** | | **−1.50 to +1.00** |

---

### `simulation/simulator.py`

**`simulate(state, action, steps=10) -> float`**

Rolls out a fixed action for `steps` steps and returns the cumulative reward. Disturbance details are discarded — only the reward is returned. This is the primary function the rule-based planner calls for scoring.

**`simulate_trajectory(state, action, steps=10, simple=False) -> list[StepRecord] | list[tuple[SleepState, float]]`**

Same rollout, but returns per-step records for debugging.

- `simple=False` (default) — returns `list[StepRecord]`
- `simple=True` — returns `list[(SleepState, float)]` (original interface)

**`StepRecord`** fields: `step`, `state`, `reward`, `action`, `disturbance`. `__str__()` renders a compact table row.

---

### `simulation/user_profile.py`

**`UserProfile`** — lightweight preference model that personalises action selection.

```python
@dataclass
class UserProfile:
    action_preferences: dict[str, float]  # action → bonus/penalty, scale ≈ [-1, +1]
    sensitivity_to_noise: float = 1.0     # metadata; wired into future reward weighting
    intervention_tolerance: float = 1.0   # global scale on all preference bonuses
```

**`preference_for(action) -> float`** — returns `action_preferences[action] × intervention_tolerance`. Always call this rather than accessing the dict directly, so `intervention_tolerance` is applied consistently.

Setting `intervention_tolerance=0.0` disables all personalisation without removing the profile.

---

### `simulation/journal_feedback.py`

Morning self-report that slowly personalises `UserProfile.action_preferences` over time.

**`JournalFeedback`** dataclass:
```python
@dataclass
class JournalFeedback:
    action_used: str
    intervention_helpful: bool | None = None
    felt_rested: bool | None = None
    awakenings: int | None = None
    notes: str | None = None
```

**`update_user_profile_from_feedback(profile, feedback, learning_rate=0.15) -> UserProfile`**

Returns a new profile with the preference for `feedback.action_used` nudged. The original is never mutated.

Per-session delta breakdown (at default `learning_rate=0.15`):

| Signal | Condition | Delta |
|---|---|---|
| `intervention_helpful=True` | primary | +0.150 |
| `intervention_helpful=False` | primary | −0.150 |
| `felt_rested=True` | secondary (×0.4) | +0.060 |
| `felt_rested=False` | secondary (×0.4) | −0.060 |
| `awakenings ≥ 3` | mild penalty (×0.33) | −0.050 |
| `awakenings ≥ 5` | stronger penalty (×0.67) | −0.100 |

Preferences are clamped to `[−2.0, +2.0]` after each update. At `learning_rate=0.15`, roughly 7 consistent bad nights push a preference from neutral to the floor.

---

### `simulation/night_loop.py`

Orchestrates one or more nights end-to-end.

**`NightRunResult`** dataclass — full record of one night:

| Field | Description |
|---|---|
| `night` | 1-based index |
| `input_state` | Starting `SleepState` |
| `selected_action` | Action the planner chose |
| `planner_type` | `"llm"` or `"rule_based_fallback"` |
| `score` | Numeric score for the selected action |
| `rationale` | LLM rationale if LLM path ran, else `None` |
| `fallback_reason` | Exception string if fallback occurred, else `None` |
| `feedback` | `JournalFeedback` applied this night |
| `updated_profile` | `UserProfile` after incorporating feedback |

`summary() -> str` — compact one-liner for loop output.

**`run_single_night(state, profile, feedback, llm_client=None, ...) -> NightRunResult`**

Runs the hybrid planner, overwrites `feedback.action_used` with the planner's actual choice (so attribution is always correct), then applies the feedback to the profile.

**`run_multiple_nights(states, initial_profile, feedbacks, llm_client=None, ...) -> list[NightRunResult]`**

Iterates nights in sequence. Each night's `updated_profile` becomes the next night's `profile`. Raises `ValueError` immediately if `states` and `feedbacks` have different lengths.

---

## Agents Package

### `agents/world_model_agent.py` — Rule-based planner (baseline)

Monte Carlo rollout planner. Scores each action by simulating it `n_rollouts` times and averaging the cumulative reward.

**`rank_actions(state, steps=10, n_rollouts=20, user_profile=None) -> list[tuple[str, float]]`**

Returns all actions sorted descending by score.

**`choose_best_action(state, steps=10, n_rollouts=20, user_profile=None) -> tuple[str, float]`**

Returns `(best_action, best_score)`. With `n_rollouts=20` the ranking is stable across calls despite stochastic disturbances.

Scoring formula per action:
```
mean(simulate(state, action, steps) × n_rollouts) + user_profile.preference_for(action)
```

---

### `agents/llm_world_model_agent.py` — LLM world model interface

Prompt builder, response parser, and generation entry point.

**Data model:**

```
TrajectoryStep          one minute of simulated sleep
    └── to_sleep_state()  bridges back to reward.py

CandidateTrajectory     one plausible future
    ├── probability
    ├── steps: list[TrajectoryStep]
    └── summary: str

ActionTrajectoryBundle  all candidates for one action
    ├── action
    ├── trajectories: list[CandidateTrajectory]
    └── rationale: str
```

**`build_world_model_prompt(state, action, user_profile=None, horizon=5) -> str`**

Builds the full LLM prompt. Injects: current `SleepState`, action semantics, sleep cycle priors (deep early / REM late / light as buffer), and optional `UserProfile` context. Requests 3–5 candidate trajectories with normalised probabilities and a `rationale`.

**`parse_action_trajectory_bundle(raw_text) -> ActionTrajectoryBundle`**

Parses and validates the LLM's JSON response. Strips markdown fences defensively. Validates `action` against `ALL_ACTIONS`, `sleep_stage` against `VALID_STAGES`, clamps all floats. Raises `ValueError` with a precise message on any failure.

**`generate_action_trajectories(state, action, user_profile=None, horizon=5, llm_client=None) -> ActionTrajectoryBundle`**

Main entry point. `llm_client` must be a `(prompt: str) -> str` callable. Raises `NotImplementedError` with fallback guidance if `None`.

---

### `agents/trajectory_critic.py` — Code-based trajectory scorer

Scores LLM-generated trajectories using the same reward vocabulary as the rule-based simulator.

**`score_trajectory_step(step, time_until_alarm=0) -> float`**

Bridges via `step.to_sleep_state()` then calls `sleep_reward()`, then adds a `wake_risk` penalty unique to the LLM path:
```
score = sleep_reward(state) − 0.15 × wake_risk
```

Range: approximately `[−1.65, +1.00]`.

**`score_candidate_trajectory(traj, starting_time_until_alarm=None) -> float`**

Sums step scores. If `starting_time_until_alarm` is given, decrements alarm distance per step so later steps are scored correctly.

**`score_action_bundle(bundle, user_profile=None, starting_time_until_alarm=None) -> float`**

Probability-weighted expected score across all trajectories, then adds the user preference:
```
expected = Σ (pᵢ × score_candidate_trajectoryᵢ)
final    = expected + user_profile.preference_for(action)
```

Probabilities are normalised silently if they do not sum to 1.0. Raises `ValueError` on empty list or all-zero weights.

---

### `agents/llm_planner_agent.py` — LLM planner

Evaluates all actions via the LLM world model and ranks them.

**`evaluate_action_with_llm(state, action, user_profile=None, horizon=5, llm_client=None) -> tuple[ActionTrajectoryBundle, float]`**

Single-action evaluation: `generate_action_trajectories()` → `score_action_bundle()`.

**`rank_actions_with_llm(state, user_profile=None, horizon=5, llm_client=None) -> list[tuple[str, float, ActionTrajectoryBundle]]`**

Calls `evaluate_action_with_llm` for every action, sorts descending. Returns the full bundle alongside the score so rationales are available.

**`choose_best_action_with_llm(state, user_profile=None, horizon=5, llm_client=None) -> tuple[str, float, ActionTrajectoryBundle]`**

Returns `ranked[0]`. Call `rank_actions_with_llm` directly for the full ordering.

---

### `agents/mock_llm_client.py` — Local test harness

Fake LLM client for exercising the full LLM pipeline without an API call.

**`MockLLMClient(seed=None)`**

Parses `action`, `time_until_alarm`, `noise_level`, and `horizon` from the prompt via regex, then generates four candidate trajectories (best-case/typical/mediocre/disruption) with action-conditioned signal deltas.

- `generate(prompt) -> str` — returns valid JSON matching the expected schema
- `__call__(prompt) -> str` — allows passing the instance directly as `llm_client=`

Behaviour is action-sensitive: `play_brown_noise` produces trajectories with falling noise; `gradual_wake` produces rising wake risk, dampened to 15% when far from the alarm.

---

### `agents/openai_compatible_llm_client.py` — Real LLM backend

Thin wrapper around the OpenAI SDK for any OpenAI-compatible endpoint.

**`OpenAICompatibleLLMClient(base_url=None, api_key=None, model_name=None, temperature=0.7, timeout=30.0)`**

Configuration priority (highest → lowest):
1. Constructor arguments
2. `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` (loaded from `.env` automatically)
3. `SOMNUS_LLM_BASE_URL`, `SOMNUS_LLM_API_KEY`, `SOMNUS_LLM_MODEL`

The `.env` at the repo root is loaded at import time via `python-dotenv`, so no manual `load_dotenv()` call is needed.

- `generate(prompt) -> str` — sends a single user message, returns raw assistant text
- `__call__(prompt) -> str` — callable interface

Error handling: `APITimeoutError`, `APIError`, and empty responses each raise with a descriptive message pointing at the likely cause.

---

### `agents/hybrid_planner_agent.py` — Normal runtime entry point

Tries the LLM path first; falls back to the rule-based planner on any exception.

**`PlannerResult`** dataclass:

| Field | LLM success | Fallback |
|---|---|---|
| `selected_action` | ✓ | ✓ |
| `score` | ✓ | ✓ |
| `planner_type` | `"llm"` | `"rule_based_fallback"` |
| `rationale` | LLM rationale | `None` |
| `bundle` | `ActionTrajectoryBundle` | `None` |
| `fallback_reason` | `None` | exception string |

**`choose_best_action_hybrid(state, user_profile=None, horizon=5, llm_client=None, fallback_steps=10, fallback_rollouts=20) -> PlannerResult`**

- If `llm_client` is provided, tries `choose_best_action_with_llm()`
- On any exception, captures the error as `fallback_reason` and runs `choose_best_action()`
- If `llm_client=None`, goes directly to the rule-based planner with an explanatory `fallback_reason`

Failures are never silently hidden. `planner_type` always tells you which path ran.

---

## End-to-End Usage

### Minimal example

```python
from simulation.sleep_state import SleepState
from simulation.user_profile import UserProfile
from agents.hybrid_planner_agent import choose_best_action_hybrid
from agents.mock_llm_client import MockLLMClient

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

result = choose_best_action_hybrid(state, user_profile=profile, llm_client=MockLLMClient())
print(result.selected_action)   # e.g. "play_brown_noise"
print(result.planner_type)      # "llm"
```

### Real LLM backend

```python
from agents.openai_compatible_llm_client import OpenAICompatibleLLMClient

# Reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL from .env automatically
client = OpenAICompatibleLLMClient()
result = choose_best_action_hybrid(state, user_profile=profile, llm_client=client)
```

### Multi-night feedback loop

```python
from simulation.journal_feedback import JournalFeedback
from simulation.night_loop import run_multiple_nights

results = run_multiple_nights(
    states=[state_night1, state_night2, state_night3],
    initial_profile=profile,
    feedbacks=[
        JournalFeedback(action_used="", intervention_helpful=True, felt_rested=True, awakenings=1),
        JournalFeedback(action_used="", intervention_helpful=True, felt_rested=True, awakenings=0),
        JournalFeedback(action_used="", intervention_helpful=False, felt_rested=False, awakenings=4),
    ],
    llm_client=MockLLMClient(),
)

for r in results:
    print(r.summary())
# Night  1 ✓  action=play_brown_noise   score=+3.34  pref_after=+0.31
# Night  2 ✓  action=play_brown_noise   score=+3.49  pref_after=+0.52
# Night  3 ✓  action=play_brown_noise   score=+3.82  pref_after=+0.26
```

---

## File Map

```
simulation/
    sleep_state.py          SleepState dataclass
    actions.py              5 action constants + ALL_ACTIONS
    disturbance.py          Disturbance dataclass + 3 event types + QUIET sentinel
    sleep_dynamics.py       transition() — rule-based dynamics + disturbance sampling
    reward.py               sleep_reward() — single-state scoring
    simulator.py            simulate() + simulate_trajectory() + StepRecord
    user_profile.py         UserProfile — per-action preference model
    journal_feedback.py     JournalFeedback + update_user_profile_from_feedback()
    night_loop.py           run_single_night() + run_multiple_nights() + NightRunResult

agents/
    world_model_agent.py            Rule-based Monte Carlo planner (baseline)
    llm_world_model_agent.py        Prompt builder, parser, data model
    trajectory_critic.py            Code-based scorer for LLM trajectories
    llm_planner_agent.py            LLM action ranker + best-action selector
    mock_llm_client.py              Local test harness (no API key required)
    openai_compatible_llm_client.py Real LLM backend (OpenAI-compatible)
    hybrid_planner_agent.py         LLM-first with rule-based fallback

test_llm_world_model.py     End-to-end demo script (mock client, two scenarios)
```
