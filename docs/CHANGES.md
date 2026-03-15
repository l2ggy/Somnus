# Somnus — World Model Layer: What Changed and Why

This document describes the new `simulation/` and `agents/` packages added alongside the original Somnus backend. It is meant to be read next to [`ARCHITECTURE.md`](ARCHITECTURE.md), which covers the production `app/` stack.

---

## What Existed Before

The original Somnus system (documented in `ARCHITECTURE.md`) is a **real-time FastAPI backend** for a sleep assistant. Its core loop is:

```
Wearable sensor reading
    → intake (validate + clamp)
    → sensor_interpreter (label signals)
    → sleep_state (classify phase)
    → disturbance (detect threats)
    → intervention (choose audio)
    → save to SQLite
```

All of this lives in `app/`. It is tick-based, deterministic in the real-time path, and uses GPT only twice per night: once for pre-sleep planning (`strategist`) and once for morning reflection (`journal_reflection`).

**What it does not have:** any forward-looking simulation. The intervention agent picks an action using rule-based heuristics against the current state, with no model of how sleep will evolve under that action. The strategist builds a plan once at bedtime; the tick pipeline executes it blindly.

---

## What Was Added in This Session

A standalone **world model layer**: a simulation environment that can predict future sleep trajectories given a candidate intervention, and a planner that uses those predictions to choose the best action.

The new code lives in two top-level packages that are entirely separate from `app/`:

```
simulation/       ← sleep environment model
agents/           ← planners that query the environment
```

Nothing in `app/` was modified. The new layer is additive.

---

## The Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  app/ — Production runtime (unchanged)                  │
│                                                         │
│  Wearable → intake → interpreter → sleep_state →        │
│  disturbance → intervention → SQLite                    │
│                                                         │
│  GPT used for: pre-sleep planning, morning reflection   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  simulation/ + agents/ — World model layer (new)        │
│                                                         │
│  SleepState → planner → simulated futures →             │
│  scored outcomes → best action                          │
│                                                         │
│  GPT used for: imagining sleep trajectories per action  │
└─────────────────────────────────────────────────────────┘
```

The two layers use the same vocabulary (sleep stages, actions, noise/movement signals) but are not wired together yet. The world model layer is currently a standalone planning and research environment.

---

## New File Map (Full Delta)

### `simulation/` — Environment model

| File | What it adds |
|---|---|
| `sleep_state.py` | `SleepState` dataclass — snapshot of biometric + environmental signals at one moment |
| `actions.py` | Five intervention constants (`DO_NOTHING`, `PLAY_BROWN_NOISE`, `PLAY_RAIN`, `BREATHING_PACING`, `GRADUAL_WAKE`) and `ALL_ACTIONS` list |
| `disturbance.py` | `Disturbance` dataclass + event constants (`SUDDEN_NOISE_SPIKE`, `MOVEMENT_SPIKE`, `SNORE_LIKE_DISTURBANCE`) sampled stochastically per step |
| `sleep_dynamics.py` | `transition(state, action) → (SleepState, Disturbance)` — the core simulation step with sleep-cycle-aware stage transitions |
| `reward.py` | `sleep_reward(state) → float` — scores a single state (deep=+1.0, rem=+0.6, light=+0.2, wake=−1.0, minus noise/movement penalties) |
| `simulator.py` | `simulate()` — cumulative reward rollout; `simulate_trajectory()` — full step-by-step debug trace via `StepRecord` |
| `user_profile.py` | `UserProfile` — per-action preference bonuses/penalties that personalise action selection |
| `journal_feedback.py` | `JournalFeedback` + `update_user_profile_from_feedback()` — morning self-report nudges action preferences over time |
| `night_loop.py` | `run_single_night()` + `run_multiple_nights()` — multi-night orchestration that carries the updated profile forward each night |

### `agents/` — Planners

| File | What it adds |
|---|---|
| `world_model_agent.py` | Rule-based Monte Carlo planner — simulates each action `n_rollouts` times, returns the highest average reward action |
| `llm_world_model_agent.py` | LLM world model interface — prompt builder, JSON response parser, `TrajectoryStep` / `CandidateTrajectory` / `ActionTrajectoryBundle` data model |
| `trajectory_critic.py` | Code-based scorer for LLM-generated trajectories — probability-weighted expected score using the same reward vocabulary as `reward.py` |
| `llm_planner_agent.py` | LLM planner — evaluates all actions via the LLM world model and ranks them |
| `mock_llm_client.py` | Fake LLM client — returns valid action-conditioned trajectory JSON without an API call; used for local testing |
| `openai_compatible_llm_client.py` | Real LLM backend — thin OpenAI SDK wrapper that reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` from `.env` |
| `hybrid_planner_agent.py` | Normal runtime entry point — tries the LLM path first, falls back to the rule-based planner on any failure, always returns a `PlannerResult` with `planner_type` set |

### Root

| File | What it adds |
|---|---|
| `test_llm_world_model.py` | End-to-end demo script — two scenarios (mid-night noisy state, near-alarm state), printed trajectory breakdown |

---

## How Each Piece Evolved

The world model layer was built incrementally. Each step preserved what came before.

### Step 1 — Rule-based baseline

`SleepState`, `actions.py`, `sleep_dynamics.py`, `reward.py`, `simulator.py`, and `world_model_agent.py` were created together. The planner scored each action by running `simulate()` once and returning the highest reward. Sleep dynamics were simple signal deltas; stage was inferred from thresholds.

### Step 2 — Realistic sleep cycle transitions

`sleep_dynamics.py` was updated to replace hard threshold-based stage inference with **weighted probabilistic sampling** that encodes sleep cycle structure:
- Deep sleep is favoured early in the night (`time_until_alarm` is large)
- REM is favoured in the second half (`time_until_alarm` is small)
- Light sleep always has a nonzero weight as the universal transition buffer
- Wake weight rises with noise, movement, and shallow depth
- A ×1.2 inertia bonus on the current stage prevents instant flipping

### Step 3 — Stochastic disturbances

`disturbance.py` was created and `sleep_dynamics.py` was updated to sample a `Disturbance` event at each step. Disturbance probability scales with current instability (higher noise + higher movement + shallower depth all increase risk). Deep sleep applies a ×0.5 resistance factor. `simulator.py` was updated to expose disturbance events in `StepRecord` for debugging.

### Step 4 — Monte Carlo averaging

`world_model_agent.py` was updated to run `simulate()` `n_rollouts` times per action and average the results. One rollout is too noisy under stochastic disturbances; averaging over 20 rollouts produces stable rankings. A `rank_actions()` helper was added for demo/debug use.

### Step 5 — User preferences

`user_profile.py` was created. `world_model_agent.py` was updated so planners add a `preference_for(action)` bonus after averaging rollout scores. The sleep quality signal and personal preference signal are kept separate and inspectable.

### Step 6 — LLM world model path

`llm_world_model_agent.py`, `trajectory_critic.py`, and `llm_planner_agent.py` were created. Instead of the rule-based simulator, the LLM imagines 3–5 candidate futures per action; the critic scores them using the same reward vocabulary. The rule-based simulator becomes the fallback/baseline.

### Step 7 — Mock client + demo script

`mock_llm_client.py` and `test_llm_world_model.py` were created so the full LLM pipeline can be exercised locally without an API key.

### Step 8 — Real LLM backend

`openai_compatible_llm_client.py` was created. It reads the existing `.env` variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`) so no new configuration is needed.

### Step 9 — Hybrid planner (safe fallback)

`hybrid_planner_agent.py` was created as the recommended runtime entry point. It tries the LLM path and falls back silently to the rule-based planner on any failure. `PlannerResult.planner_type` always records which path ran.

### Step 10 — Feedback loop

`journal_feedback.py` was created. `update_user_profile_from_feedback()` takes a morning `JournalFeedback` and nudges the `UserProfile` action preference for the action that was used. Updates are bounded to `[−2.0, +2.0]`. The original profile is never mutated.

### Step 11 — Multi-night orchestration

`night_loop.py` was created. `run_multiple_nights()` carries the updated `UserProfile` forward from night to night, so preferences evolve across a simulated run without any training.

---

## Key Design Decisions

### Rule-based path is never removed

Every LLM-path component was added alongside the existing simulator, never replacing it. `hybrid_planner_agent.py` makes this explicit at runtime: if the LLM path fails for any reason, the rule-based planner runs instead.

### Separation from `app/`

The world model layer does not import anything from `app/`. It is a standalone research and planning environment. When the time comes to wire it into the production tick pipeline, the integration point is clear: replace the intervention agent's heuristic lookup with a call to `choose_best_action_hybrid()`.

### `SleepState` vs `app/models/userstate.SleepState`

There are now two `SleepState` types in the repo:

| | `simulation/sleep_state.SleepState` | `app/models/userstate.SleepState` |
|---|---|---|
| Purpose | World model input/output | App domain model (Pydantic) |
| Fields | `sleep_stage`, `sleep_depth`, `movement`, `noise_level`, `time_until_alarm` | `phase`, `confidence`, `wake_risk`, `disturbance_reason` |
| Used by | `simulation/`, `agents/` | `app/` |

These are intentionally separate. Merging them would require either polluting the clean world model dataclass with Pydantic or making the app model carry simulation-internal fields. A thin adapter can map between them when the layers are eventually integrated.

### Trajectory probabilities are normalised by the critic

The LLM is asked to produce roughly normalised probabilities, but the critic normalises them silently if they do not sum to 1.0. This keeps the LLM path robust to minor model output drift without hiding the issue (the raw probabilities are still visible in `ActionTrajectoryBundle`).

### Preference bonus is added after simulation scoring

The user preference term is always added after the sleep quality score, never folded into it. This keeps the two signals inspectable and easy to retune independently.

---

## Data Flow: End-to-End

```
SleepState (current biometric snapshot)
    │
    ▼
choose_best_action_hybrid(state, user_profile, llm_client)
    │
    ├── LLM path (preferred)
    │       │
    │       ├── for each action in ALL_ACTIONS:
    │       │       build_world_model_prompt(state, action, user_profile, horizon)
    │       │       llm_client(prompt) → raw JSON
    │       │       parse_action_trajectory_bundle(raw_text) → ActionTrajectoryBundle
    │       │       score_action_bundle(bundle, user_profile) → float
    │       │
    │       └── rank_actions_with_llm() → choose best
    │
    └── Rule-based fallback (on any LLM error)
            │
            ├── for each action in ALL_ACTIONS:
            │       simulate(state, action, steps) × n_rollouts → mean reward
            │       + user_profile.preference_for(action)
            │
            └── choose_best_action() → choose best
    │
    ▼
PlannerResult
    ├── selected_action
    ├── score
    ├── planner_type: "llm" | "rule_based_fallback"
    ├── rationale (LLM path only)
    ├── bundle (LLM path only)
    └── fallback_reason (fallback path only)
    │
    ▼ (morning)
JournalFeedback
    │
update_user_profile_from_feedback(profile, feedback)
    │
    ▼
Updated UserProfile → next night's planner
```

---

## Relationship to the Roadmap in ARCHITECTURE.md

`ARCHITECTURE.md` section 14 lists an LLM expansion path:

> Every agent already has the deterministic path as fallback. Adding GPT to any agent means:
> 1. Write the GPT prompt + Pydantic output model
> 2. Add `run_gpt()` with the standard try/except fallback pattern
> 3. Add `mode` param to the relevant orchestrator function
> 4. Add `?mode=gpt` to the relevant route

The world model layer follows exactly this pattern at the planner level. The intervention selector in `app/agents/intelligence/intervention.py` currently applies a fixed heuristic. The integration path when ready:

1. Map `app/models/userstate.SleepState` → `simulation/sleep_state.SleepState` at the API boundary
2. Call `choose_best_action_hybrid()` inside the intervention agent
3. Map the result back to `ActiveIntervention`
4. Add `?mode=world_model` to the sensor endpoint

No other changes to `app/` are required.

---

## Quick Reference

```python
# Rule-based planner only
from simulation.sleep_state import SleepState
from agents.world_model_agent import choose_best_action

state = SleepState(sleep_stage="light", sleep_depth=0.4,
                   movement=0.3, noise_level=0.55, time_until_alarm=180)
action, score = choose_best_action(state)

# Hybrid planner (LLM preferred, rule-based fallback)
from agents.hybrid_planner_agent import choose_best_action_hybrid
from agents.mock_llm_client import MockLLMClient        # local testing
# from agents.openai_compatible_llm_client import OpenAICompatibleLLMClient  # production

result = choose_best_action_hybrid(state, llm_client=MockLLMClient())
print(result.selected_action, result.planner_type)

# Multi-night feedback loop
from simulation.journal_feedback import JournalFeedback
from simulation.night_loop import run_multiple_nights
from simulation.user_profile import UserProfile

profile = UserProfile(action_preferences={
    "play_brown_noise": 0.5, "play_rain": -0.3,
    "breathing_pacing": 0.2, "gradual_wake": -1.0, "do_nothing": 0.0,
})
results = run_multiple_nights(
    states=[state] * 3,
    initial_profile=profile,
    feedbacks=[
        JournalFeedback(action_used="", intervention_helpful=True, felt_rested=True),
        JournalFeedback(action_used="", intervention_helpful=True, felt_rested=True),
        JournalFeedback(action_used="", intervention_helpful=False, felt_rested=False, awakenings=4),
    ],
    llm_client=MockLLMClient(),
)
for r in results:
    print(r.summary())
```

For full API reference of each module, see [`world_model.md`](world_model.md).
