# Somnus — Architecture & Developer Reference

Somnus is an agentic sleep assistant backend. It monitors real-time sensor data during sleep, detects disturbances, activates audio interventions, and builds a learning feedback loop through nightly planning and morning reflection.

---

## Table of Contents

1. [Project layout](#1-project-layout)
2. [Core concept: shared state](#2-core-concept-shared-state)
3. [Data models](#3-data-models)
4. [Agents](#4-agents)
5. [The tick pipeline](#5-the-tick-pipeline)
6. [Session lifecycle](#6-session-lifecycle)
7. [Orchestrator](#7-orchestrator)
8. [GPT integration](#8-gpt-integration)
9. [Persistence (SQLite)](#9-persistence-sqlite)
10. [API reference](#10-api-reference)
11. [Configuration](#11-configuration)
12. [Running the server](#12-running-the-server)
13. [Testing](#13-testing)
14. [Roadmap](#14-roadmap)

---

## 1. Project layout

```
Somnus/
├── app/
│   ├── main.py                  FastAPI app, routes, lifespan hook
│   ├── orchestrator.py          Session lifecycle coordinator
│   ├── store.py                 SQLite persistence layer
│   ├── llm_client.py            OpenAI-compatible API wrapper
│   ├── models/
│   │   ├── userstate.py         Domain models (SharedState and children)
│   │   └── api.py               HTTP request models
│   ├── agents/
│   │   ├── backend/
│   │   │   └── intake.py        Sensor ingestion + validation
│   │   ├── intelligence/
│   │   │   ├── sensor_interpreter.py  Feature extraction
│   │   │   ├── sleep_state.py   Sleep phase classifier
│   │   │   ├── disturbance.py   Disturbance detector
│   │   │   ├── intervention.py  Intervention selector
│   │   │   ├── strategist.py    Pre-sleep planner (deterministic + GPT)
│   │   │   └── journal_reflection.py  Morning reflection (deterministic + GPT)
│   │   └── *.py wrappers         Backward-compatible import aliases
│   └── demo_support/
│       └── demo_states.py       Demo/user-flow fixtures for UI endpoints
├── docs/
│   └── ARCHITECTURE.md          This file
├── .env                         OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
├── requirements.txt
└── somnus.db                    SQLite database (created on first run)
```

---

## 1.1 Ownership boundaries for parallel development

To reduce merge conflicts, code is split into sub-folders aligned with the 3-person team model and encoded in `app/pipeline_contracts.py`.

- **team_a (Product + Frontend + Pitch)** owns demo/user-flow fixtures in `app/demo_support/` used by `/state/example`, `/pipeline/example`, `/plan/example`, and `/journal/example`.
- **team_b (Backend + Orchestration)** owns ingestion at the API boundary (`app/agents/backend/intake.py`) plus runtime lifecycle coordination (`app/orchestrator.py`, `app/store.py`, `app/main.py`).
- **team_c (Sleep Intelligence + Personalization)** owns all modeling/policy agents in `app/agents/intelligence/` (sensor interpretation, sleep-state estimation, disturbance, intervention, planning, reflection).

`AgentContract` is the source of truth for lifecycle, reads/writes, and owner metadata. The orchestrator validates contracts at runtime before ticks so incompatible merges (like duplicate agent names) fail fast.

---

## 2. Core concept: shared state

Every agent in Somnus reads from and writes to a single `SharedState` object. No agent talks directly to another. All coordination flows through the state tree.

```
                    ┌─────────────────────────────┐
                    │         SharedState          │
                    │                             │
                    │  preferences                │
                    │  latest_sensor              │
                    │  sleep_state                │
                    │  active_intervention        │
                    │  nightly_plan               │
                    │  hypotheses                 │
                    │  journal_history            │
                    └─────────────────────────────┘
                           ↑          ↓
               agents read from   agents write to
                    (immutable — each agent returns a new copy)
```

Agents are **pure functions**: `SharedState → SharedState`. Each agent receives the output of the previous agent and returns an updated copy using Pydantic's `.model_copy(update={...})`. Nothing is mutated in place. This makes agents easy to test in isolation, safe to reason about, and straightforward to replace with LLM-backed versions later.

The orchestrator owns the sequencing. It calls agents in order, threads state through, and is the only place that knows what runs when.

---

## 3. Data models

All models live in `app/models/userstate.py`. They are Pydantic v2 `BaseModel` subclasses.

### UserPreferences

Persisted user configuration, loaded once at session start and consulted by the strategist and intervention agent throughout the night.

| Field | Type | Description |
|---|---|---|
| `goals` | `list[str]` | Sleep goals in plain text, e.g. `["maximize deep sleep"]` |
| `preferred_audio` | `list[str]` | Ordered list of audio types the user likes |
| `disliked_audio` | `list[str]` | Types that must never be selected |
| `target_wake_time` | `str` | HH:MM format, e.g. `"07:00"` |
| `intervention_aggressiveness` | `"low" \| "medium" \| "high"` | Controls intervention intensity scaling |

### SensorSnapshot

One moment-in-time reading from the wearable and environment sensors. All fields except `timestamp` are optional — agents fall back to safe defaults when a value is absent.

| Field | Type | Range | Notes |
|---|---|---|---|
| `timestamp` | `str` | ISO 8601 | Auto-filled by intake if missing |
| `heart_rate` | `float \| None` | 20–220 bpm | Clamped by intake |
| `hrv` | `float \| None` | 0–300 ms | Heart-rate variability |
| `movement` | `float \| None` | 0–1 | Normalized accelerometer score |
| `noise_db` | `float \| None` | dB | Not clamped — raw env reading |
| `light_level` | `float \| None` | 0–1 | Normalized lux |
| `breathing_rate` | `float \| None` | 4–60 bpm | Clamped by intake |

### SleepState

The output of the `sleep_state` agent. Updated on every tick.

| Field | Type | Description |
|---|---|---|
| `phase` | `"awake" \| "light" \| "deep" \| "rem" \| "unknown"` | Inferred sleep phase |
| `confidence` | `float` | 0–1 model confidence |
| `wake_risk` | `float` | 0–1 probability of waking in the next window |
| `disturbance_reason` | `str \| None` | Set by the disturbance agent, e.g. `"noise_spike:62dB"` |

### ActiveIntervention

Whatever the system is currently playing or suggesting. Cleared when `wake_risk` drops below threshold.

| Field | Type | Description |
|---|---|---|
| `type` | literal union | `"none"`, `"brown_noise"`, `"white_noise"`, `"pink_noise"`, `"rain"`, `"waves"`, `"breathing_pace"`, `"wake_ramp"` |
| `intensity` | `float` | 0–1 volume/strength |
| `started_at` | `str \| None` | ISO timestamp |
| `rationale` | `str \| None` | Human-readable explanation from the agent |

### JournalEntry

Filled in by the user each morning.

| Field | Type | Description |
|---|---|---|
| `date` | `str` | ISO date, e.g. `"2026-03-14"` |
| `rested_score` | `int` | 1–10 self-reported restedness |
| `awakenings_reported` | `int` | Number of times the user recalls waking |
| `notes` | `str` | Free-text notes |

### NightlyPlan

Built once by the strategist before sleep. Consulted by the intervention agent throughout the night.

| Field | Type | Description |
|---|---|---|
| `target_bedtime` | `str` | HH:MM |
| `target_wake_time` | `str` | HH:MM |
| `sleep_goal` | `str` | e.g. `"deep_sleep_focus"`, `"reduce_awakenings"` |
| `notes` | `str` | Planning context and rationale |
| `preferred_intervention_order` | `list[str]` | Ordered list of intervention types to try first |

### SharedState

The root object that flows through the entire system.

```python
SharedState(
    user_id         = "alice",
    preferences     = UserPreferences(...),
    latest_sensor   = SensorSnapshot(...),   # None until first tick
    sleep_state     = SleepState(...),
    active_intervention = ActiveIntervention(...),
    nightly_plan    = NightlyPlan(...),
    hypotheses      = [...],                 # free-form dicts from agents
    journal_history = [JournalEntry(...)],
)
```

`hypotheses` is an append-only list where agents log intermediate reasoning — sensor signals, disturbance detections, reflections. It is the mechanism by which agents communicate non-binding context to each other and to future LLM prompts.

---

## 4. Agents

Every agent file exports at minimum one function: `run(state: SharedState) -> SharedState`. Planning and reflection agents also export `run_gpt(state, ...) -> tuple[SharedState, str]`.

### intake

**File:** `app/agents/intake.py`
**Lifecycle:** on sensor arrival (before the tick)
**Reads:** raw dict from HTTP request
**Writes:** `latest_sensor`

Converts a raw sensor dict into a validated `SensorSnapshot`. Clamps physiological values to safe ranges (`heart_rate` 20–220, `breathing_rate` 4–60, normalized fields 0–1). Auto-fills `timestamp` with UTC now if absent or null.

```python
state = intake.run(state, {
    "heart_rate": 72, "hrv": 44, "movement": 0.3,
    "noise_db": 62,   "light_level": 0.04
})
```

---

### sensor_interpreter

**File:** `app/agents/sensor_interpreter.py`
**Lifecycle:** tick step 1
**Reads:** `latest_sensor`
**Writes:** appends to `hypotheses`

Translates every raw sensor number into a human-readable label. Always emits all six signals; uses `"unknown"` when a sensor field is absent.

| Signal | Labels |
|---|---|
| `hr_status` | `very_low` / `low_resting` / `normal` / `mildly_elevated` / `elevated` |
| `hrv_quality` | `excellent` / `good` / `moderate` / `low` |
| `movement_level` | `still` / `minimal` / `restless` / `active` |
| `noise_alert` | `quiet` / `moderate` / `loud` / `very_loud` |
| `light_status` | `dark` / `dim` / `moderate` / `bright` |
| `breathing_status` | `very_slow` / `normal` / `slightly_elevated` / `elevated` |

Thresholds are based on published sleep-research norms and can be tuned per-user in a later iteration.

---

### sleep_state

**File:** `app/agents/sleep_state.py`
**Lifecycle:** tick step 2
**Reads:** `latest_sensor`
**Writes:** `sleep_state`

Six-signal heuristic classifier. Rules run in priority order; the first match wins.

| Rule | Condition | Phase | Confidence | Wake risk |
|---|---|---|---|---|
| 1 | `light_level > 0.5` | `awake` | 0.85 | 0.90 |
| 2 | `movement > 0.55` | `awake` | 0.80 | 0.85 |
| 3 | `hr > 85` and `movement > 0.15` | `awake` | 0.70 | 0.80 |
| 4 | `hr < 60` and `movement < 0.1` and `hrv ≥ 55` and `br ≤ 15` | `deep` | 0.55–0.87 | 0.10 |
| 5 | `movement < 0.15` and `hrv ≥ 50` and `14 ≤ br ≤ 19` | `rem` | 0.58 | 0.20 |
| 6 | Default | `light` | 0.55 | 0.30 + noise penalty |

Deep sleep confidence scales with the number of signals available (0.55 base + 0.08 per signal). Loud noise adds up to +0.20 to `wake_risk` in the default case via a proportional penalty.

Real examples from the running system:

| Scenario | Phase | Confidence | Wake risk |
|---|---|---|---|
| HR 55, HRV 65, movement 0.05, noise 36 dB | `deep` | 0.87 | 0.10 |
| HR 62, HRV 58, movement 0.10, br 16 | `rem` | 0.58 | 0.20 |
| HR 68, HRV 44, movement 0.28, noise 62 dB | `light` | 0.55 | 0.44 |
| HR 85, movement 0.70, light 0.60 | `awake` | 0.85 | 0.90 |

---

### disturbance

**File:** `app/agents/disturbance.py`
**Lifecycle:** tick step 3
**Reads:** `latest_sensor`, `sleep_state.phase`
**Writes:** `sleep_state.disturbance_reason`, `sleep_state.wake_risk`

Scans for four threat types. Checks in priority order — only the highest-priority active threat is reported per tick.

| Threat | Trigger | Max risk bump |
|---|---|---|
| `light_spike` | `light_level > 0.35` | +0.25 |
| `noise_spike` | `noise_db > threshold` (phase-dependent) | +0.20 |
| `high_movement` | `movement > 0.60` | +0.30 |
| `elevated_hr` | `heart_rate > 85` | +0.15 |

Noise thresholds are **phase-aware** — deep sleep is more sensitive:

| Phase | Noise threshold |
|---|---|
| `deep` | 45 dB |
| `rem` | 50 dB |
| `light` | 58 dB |
| `awake` | 70 dB |
| `unknown` | 55 dB |

Risk bumps scale proportionally to severity (e.g. 80 dB noise = larger bump than 62 dB). `wake_risk` is capped at 1.0. When no disturbance is detected, `disturbance_reason` is set to `None` and `wake_risk` stays unchanged — allowing the intervention agent to ramp down on the next tick.

---

### intervention

**File:** `app/agents/intervention.py`
**Lifecycle:** tick step 4
**Reads:** `sleep_state`, `nightly_plan`, `preferences`
**Writes:** `active_intervention`

Decides what to play (or whether to stop playing anything).

**Decision logic:**

1. If `wake_risk < 0.38` and `disturbance_reason` is `None` → clear intervention.
2. Otherwise build a candidate list from `nightly_plan.preferred_intervention_order`, filtering out anything in `preferences.disliked_audio`.
3. Pick the most appropriate type for the current sleep phase:
   - `deep` → prefer `brown_noise` or `pink_noise`; avoid `white_noise` and `wake_ramp`
   - `rem` → prefer `breathing_pace`
   - `light` / `awake` → first non-stimulating candidate
4. Scale intensity: `base_intensity[aggressiveness] × wake_risk`, floored at 0.15.

Base intensity by aggressiveness setting:

| Aggressiveness | Base intensity |
|---|---|
| `low` | 0.35 |
| `medium` | 0.60 |
| `high` | 0.85 |

`white_noise` and `wake_ramp` are classified as "stimulating" and are excluded from deep-sleep selection even if they appear in the plan.

---

### strategist

**File:** `app/agents/strategist.py`
**Lifecycle:** pre-sleep (once per night)
**Reads:** `preferences`, `journal_history`
**Writes:** `nightly_plan`

Two execution paths, same output type.

**Deterministic path (`run`):**
- Keyword-matches goals to a `sleep_goal` string
- Copies `preferred_audio` directly into `preferred_intervention_order`
- Always succeeds, zero latency

**GPT path (`run_gpt`):**
- Sends last 5 journal entries + full preferences to the LLM
- Asks for a structured JSON `NightlyPlan`
- Validates the response with `NightlyPlan.model_validate()`
- Sanitises `preferred_intervention_order` — strips disliked types, rejects unknown values
- Falls back to deterministic on any exception; returns `mode_used="fallback"`

GPT system prompt rules (enforced in prompt, validated in code):
- JSON-only response
- No medical diagnoses or health claims
- `sleep_goal` must be one of: `deep_sleep_focus`, `rem_focus`, `balanced_sleep`, `recovery`, `reduce_awakenings`
- `preferred_intervention_order` must be 2–4 items from the valid set, excluding disliked types

---

### journal_reflection

**File:** `app/agents/journal_reflection.py`
**Lifecycle:** morning (once per night)
**Reads:** `journal_history`, `active_intervention`, `hypotheses`
**Writes:** appends to `journal_history`, appends to `hypotheses`

Two execution paths.

**Deterministic path (`run`):**
- Computes quality label: `poor` (≤3), `fair` (≤6), `good` (>6)
- Computes rolling average `rested_score` across all prior journal entries
- Returns `insights: []` and `suggestion: null`

**GPT path (`run_gpt`):**
- Sends last 7 journal entries + intervention context to the LLM
- Validates response with `_GptReflectionOutput` Pydantic model before use:

```python
class _GptReflectionOutput(BaseModel):
    quality:    Literal["poor", "fair", "good"]
    notes:      str
    insights:   List[str] = []      # capped at 3 by field_validator
    suggestion: Optional[str] = None
```

- Merges GPT-generated `notes`, `insights`, `suggestion` with deterministically computed `date`, `rested_score`, `awakenings`, `rolling_avg_score`
- Falls back to deterministic on any exception

**Why GPT output is never trusted for immutable fields:** even if the model returns a wrong `rested_score`, the reflection always uses the value from the `JournalEntry` supplied by the user. GPT can only affect the narrative fields.

---

## 5. The tick pipeline

The real-time pipeline runs on every sensor reading — approximately every 30–60 seconds during an active sleep session.

```
sensor reading arrives (POST /session/{id}/sensor)
    │
    ▼
intake.run(state, raw_dict)
    └─ clamps values, fills timestamp
    └─ writes: latest_sensor
    │
    ▼
sensor_interpreter.run(state)
    └─ labels all 6 sensor signals
    └─ writes: hypotheses (appends)
    │
    ▼
sleep_state.run(state)
    └─ 6-rule heuristic classifier
    └─ writes: sleep_state.phase, confidence, wake_risk
    │
    ▼
disturbance.run(state)
    └─ checks 4 threat types, phase-aware thresholds
    └─ writes: sleep_state.disturbance_reason, wake_risk
    │
    ▼
intervention.run(state)
    └─ selects type, scales intensity, filters disliked
    └─ writes: active_intervention
    │
    ▼
state saved to SQLite
result returned to caller (state + trace + warnings)
```

**The tick pipeline is fully deterministic.** No LLM calls, no randomness, no I/O beyond SQLite. This means it is fast (~1 ms), testable without mocks, and reliable enough for a real-time loop.

**Why `intake` is not inside the orchestrator's tick function:** by the time `run_night_tick()` is called, `latest_sensor` is already a validated `SensorSnapshot`. The HTTP route calls `ingest_sensor()` (which wraps `intake.run()`) first, at the API boundary where the raw dict arrives. The tick pipeline only sees clean data.

---

## 6. Session lifecycle

A full night has three phases:

```
EVENING — user goes to bed
    POST /session/start?mode=deterministic|gpt
        → SharedState created with preferences + optional past journal history
        → strategist builds NightlyPlan
        → state written to SQLite

NIGHT — wearable sends readings every ~30–60s
    POST /session/{id}/sensor  (repeated)
        → intake → sensor_interpreter → sleep_state → disturbance → intervention
        → state updated in SQLite after every tick

MORNING — user wakes up
    POST /session/{id}/journal?mode=deterministic|gpt
        → journal_reflection processes the morning entry
        → new JournalEntry appended to journal_history
        → reflection (with optional GPT insights) appended to hypotheses
        → state saved to SQLite
        → journal_history feeds into the next evening's planning
```

State from each phase persists to SQLite. The next night's `POST /session/start` can be called with a pre-seeded `journal_history` or the app can load the existing session and append to it.

---

## 7. Orchestrator

**File:** `app/orchestrator.py`

The orchestrator is the only module that knows the agent execution order. Routes call the orchestrator; they do not import agents directly.

### `ingest_sensor(state, raw_dict) → SharedState`

Thin wrapper around `intake.run()`. Call this at the API boundary before `run_night_tick()`.

### `run_night_tick(state, ) → dict`

Runs the four-agent tick pipeline.

```python
{
    "ok":       True,
    "state":    SharedState,
    "trace":    [
        {"step": 1, "agent": "sensor_interpreter", "output": {...}},
        {"step": 2, "agent": "sleep_state",        "output": {...}},
        {"step": 3, "agent": "disturbance",        "output": {...}},
        {"step": 4, "agent": "intervention",       "output": {...}},
    ],
    "warnings": ["sensor fields missing: ['hrv']"],   # if any
}
```

If `latest_sensor` is None, returns `{"ok": False, "error": "..."}` without calling any agents.
If some sensor fields are `None`, returns `ok=True` but includes a warnings list — agents use safe defaults.

### `run_pre_sleep_planning(state, mode="deterministic") → dict`

```python
{
    "state":     SharedState,   # nightly_plan populated
    "mode_used": "deterministic" | "gpt" | "fallback",
}
```

### `run_morning_reflection(state, entry, mode="deterministic") → dict`

```python
{
    "state":      SharedState,  # journal_history and hypotheses updated
    "reflection": dict,         # extracted reflection summary
    "mode_used":  "deterministic" | "gpt" | "fallback",
}
```

---

## 8. GPT integration

### LLM client (`app/llm_client.py`)

Wraps the OpenAI Python SDK. Works with any OpenAI-compatible endpoint via `OPENAI_BASE_URL`.

```python
from app.llm_client import chat_json, is_configured

result = chat_json(
    system="Your role and JSON schema rules.",
    user="The actual context to reason about.",
    temperature=0.3,
)
# → parsed dict from model's JSON response
```

**`response_format` handling:** the client first tries `response_format={"type": "json_object"}` (grammar-constrained JSON on supporting endpoints). If the endpoint rejects this with a `BadRequestError` mentioning "json" or "response_format", it retries without the parameter. The prompt still enforces JSON-only output, and `_parse_json()` strips any markdown code fences the model adds anyway.

**Fallback chain:**

```
is_configured() → False   →  raise RuntimeError  ─┐
API error / timeout        →  raise               ─┤  all caught by
Bad JSON                   →  raise JSONDecodeError─┤  run_gpt() → fallback
Pydantic ValidationError   →  raise               ─┘
                                                     mode_used = "fallback"
```

The `mode_used` field in every planning and reflection response tells you which path ran: `"gpt"`, `"fallback"`, or `"deterministic"`.

### Which agents use GPT

| Agent | Has GPT path | Guards output with Pydantic |
|---|---|---|
| intake | No | — |
| sensor_interpreter | No | — |
| sleep_state | No | — |
| disturbance | No | — |
| intervention | No | — |
| **strategist** | **Yes** | `NightlyPlan.model_validate()` |
| **journal_reflection** | **Yes** | `_GptReflectionOutput.model_validate()` |

The real-time tick pipeline (steps 1–4) is intentionally GPT-free. Planning and reflection run once per night where a 2–5 s LLM latency is acceptable and richer language understanding genuinely improves output.

### Adding a new GPT agent

1. Write a deterministic `run(state) → SharedState` that always works.
2. Add `run_gpt(state) → tuple[SharedState, str]` that:
   - Calls `llm_client.chat_json(system, user)`
   - Validates the response with a dedicated Pydantic model
   - Returns `(updated_state, "gpt")` on success
   - Catches all exceptions and returns `(run(state), "fallback")`
3. Add a `mode` parameter to the relevant orchestrator function.

---

## 9. Persistence (SQLite)

**File:** `app/store.py`
**Database:** `somnus.db` in the working directory (override: `SOMNUS_DB_PATH`)

### Schema

```sql
CREATE TABLE sessions (
    user_id    TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL      -- ISO 8601 UTC timestamp
);
```

### Why JSON blobs

`SharedState` is a deeply nested model that is still evolving. Storing the whole thing as one JSON blob means:

- No migrations needed when fields are added or renamed
- The full state tree is a single `SELECT` and a single `INSERT OR REPLACE`
- Pydantic handles deserialization with type coercion and defaults for missing fields

When the schema stabilises and cross-session queries are needed (e.g. analytics across all journal entries), `JournalEntry` and `SensorSnapshot` can be extracted into normalized tables. The four-function store interface (`save_state`, `load_state`, `delete_state`, `list_states`) stays identical.

### Store API

```python
store.init_db()                      # called at startup — creates table if missing
store.save_state(state)              # upsert: INSERT OR REPLACE
store.load_state("alice")            # → SharedState | None
store.delete_state("alice")          # → bool (True if row existed)
store.list_states()                  # → ["alice", "bob", ...]  (newest first)

# Backward-compatible aliases (used by routes):
store.create_session(state)          # → state
store.get_session("alice")           # → SharedState | None
store.save_session(state)            # → None
store.list_sessions()                # → list[str]
```

### Error handling

If a row exists but its JSON fails Pydantic validation (e.g. after a schema change), `load_state` logs the error and returns `None`. The caller sees it as "not found" rather than getting a 500.

---

## 10. API reference

### Meta

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check → `{"status":"ok","service":"somnus"}` |
| `GET` | `/agents` | Agent registry with name, purpose, lifecycle, reads, writes |

### Session (live, SQLite-backed)

| Method | Path | Body / Params | Description |
|---|---|---|---|
| `POST` | `/session/start` | `StartSessionRequest` + `?mode=` | Create session, run planning, store |
| `POST` | `/session/{id}/sensor` | `SensorPayload` | Ingest sensor, run tick, save |
| `GET` | `/session/{id}/state` | — | Return current `SharedState` |
| `POST` | `/session/{id}/journal` | `JournalEntry` + `?mode=` | Morning reflection, save |
| `GET` | `/session` | — | List all active user_ids |

`?mode=deterministic` (default) or `?mode=gpt` on `/session/start` and `/session/{id}/journal`.

#### StartSessionRequest

```json
{
  "user_id": "alice",
  "preferences": {
    "goals": ["maximize deep sleep"],
    "preferred_audio": ["brown_noise", "rain"],
    "disliked_audio": ["white_noise"],
    "target_wake_time": "07:00",
    "intervention_aggressiveness": "medium"
  },
  "journal_history": []
}
```

#### SensorPayload (all fields optional)

```json
{
  "timestamp":      "2026-03-14T03:15:00+00:00",
  "heart_rate":     68.0,
  "hrv":            44.0,
  "movement":       0.28,
  "noise_db":       62.0,
  "light_level":    0.04,
  "breathing_rate": 16.5
}
```

#### JournalEntry

```json
{
  "date":                 "2026-03-14",
  "rested_score":         7,
  "awakenings_reported":  1,
  "notes":                "Brown noise helped a lot."
}
```

### Demo endpoints (stateless, no SQLite)

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/state/example` | — | Static deep-sleep SharedState |
| `GET` | `/pipeline/example` | — | Run tick on noisy light-sleep scenario |
| `GET` | `/plan/example` | `?mode=` | Run planning on hardcoded pre-sleep state |
| `GET` | `/journal/example` | `?mode=` | Run reflection on hardcoded completed-night state |

### Response shapes

**Tick (`/sensor`):**
```json
{
  "ok": true,
  "state": { ... },
  "trace": [
    {"step": 1, "agent": "sensor_interpreter", "output": {"hr_status": "normal", ...}},
    {"step": 2, "agent": "sleep_state",        "output": {"phase": "light", "confidence": 0.55, "wake_risk": 0.44}},
    {"step": 3, "agent": "disturbance",        "output": {"disturbance_reason": "noise_spike:62dB", "wake_risk": 0.54}},
    {"step": 4, "agent": "intervention",       "output": {"type": "brown_noise", "intensity": 0.32, "rationale": "..."}}
  ],
  "warnings": []
}
```

**Planning (`/session/start`, `/plan/example`):**
```json
{
  "nightly_plan": { ... },
  "mode_used": "gpt",
  "final_state": { ... }
}
```

**Reflection (`/journal`, `/journal/example`):**
```json
{
  "state": { ... },
  "reflection": {
    "date": "2026-03-14",
    "quality": "good",
    "rested_score": 7,
    "awakenings": 1,
    "rolling_avg_score": 6.1,
    "notes": "...",
    "insights": ["Brown noise coincided with fewer awakenings.", "..."],
    "suggestion": "Continue using brown noise consistently."
  },
  "mode_used": "gpt"
}
```

---

## 11. Configuration

All configuration is via environment variables. Copy `.env` and edit as needed.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | For GPT mode | — | API key for the LLM endpoint |
| `OPENAI_BASE_URL` | No | OpenAI default | Base URL for any OpenAI-compatible server |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model name to use |
| `SOMNUS_DB_PATH` | No | `somnus.db` | Path to SQLite database file |

To use the hackathon GPT-OSS endpoint:
```
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=https://<hf-endpoint>.aws.endpoints.huggingface.cloud/v1
OPENAI_MODEL=openai/gpt-oss-120b
```

To disable GPT and run fully deterministic: remove or unset `OPENAI_API_KEY`. All `?mode=gpt` requests will return `mode_used="fallback"` with no error.

---

## 12. Running the server

```bash
# Install dependencies
pip install -r requirements.txt

# Start (database is created automatically on first run)
uvicorn app.main:app --reload

# Production (single worker, no reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Custom database location
SOMNUS_DB_PATH=/data/somnus.db uvicorn app.main:app
```

Interactive API docs (auto-generated by FastAPI):
```
http://localhost:8000/docs    ← Swagger UI
http://localhost:8000/redoc   ← ReDoc
```

Inspect the database directly:
```bash
sqlite3 somnus.db ".mode column" ".headers on" \
  "SELECT user_id, updated_at, length(state_json) as bytes FROM sessions;"
```

---

## 13. Testing

A full runthrough can be done with curl against a live server, or by calling route handlers directly in Python (no HTTP needed).

### Full night via curl

```bash
# 1. Start session (deterministic planning)
curl -s -X POST "http://localhost:8000/session/start" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "preferences": {
      "goals": ["maximize deep sleep"],
      "preferred_audio": ["brown_noise", "rain"],
      "disliked_audio": ["white_noise"],
      "target_wake_time": "07:00",
      "intervention_aggressiveness": "medium"
    }
  }' | python -m json.tool

# 2. Send a noisy-night sensor tick
curl -s -X POST "http://localhost:8000/session/alice/sensor" \
  -H "Content-Type: application/json" \
  -d '{
    "heart_rate": 70, "hrv": 44, "movement": 0.28,
    "noise_db": 64,   "light_level": 0.03, "breathing_rate": 17
  }' | python -m json.tool

# 3. Read state back
curl -s "http://localhost:8000/session/alice/state" | python -m json.tool

# 4. Submit morning journal
curl -s -X POST "http://localhost:8000/session/alice/journal?mode=gpt" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-03-15",
    "rested_score": 7,
    "awakenings_reported": 1,
    "notes": "Brown noise helped. Woke briefly around 4am."
  }' | python -m json.tool

# 5. Restart server, confirm state survives
# Ctrl-C
# uvicorn app.main:app --reload
curl -s "http://localhost:8000/session/alice/state" | python -m json.tool
# → state is still there
```

### GPT mode vs deterministic

```bash
curl "http://localhost:8000/plan/example?mode=deterministic" | python -m json.tool
curl "http://localhost:8000/plan/example?mode=gpt"           | python -m json.tool
# compare nightly_plan.notes and mode_used in both responses

curl "http://localhost:8000/journal/example?mode=gpt" | python -m json.tool
# look at reflection.insights and reflection.suggestion
```

### Expected tick output for the noisy-night scenario

Given: HR 70, HRV 44, movement 0.28, noise 64 dB, light 0.03:

```
sensor_interpreter → noise_alert=loud, movement_level=restless, hrv_quality=moderate
sleep_state        → phase=light, confidence=0.55, wake_risk=0.44
disturbance        → noise_spike:64dB, wake_risk=0.54
intervention       → brown_noise, intensity=0.32 (0.60 × 0.54, white_noise filtered out)
```

---

## 14. Roadmap

The architecture is designed so each of these is an additive change with no rewrites.

**Near-term:**
- `POST /session/{id}/tick` — explicit tick endpoint (currently tick is triggered by the sensor endpoint)
- LLM-backed `sleep_state` agent as an optional override (same `run_gpt` pattern)
- Rolling sensor window on `SharedState` for trend-aware classification
- Multi-worker safety: replace SQLite with a connection pool or Redis

**Medium-term:**
- Normalized tables: extract `JournalEntry` and `SensorSnapshot` into separate tables for analytics queries
- Background tick scheduler using APScheduler or a Celery worker
- User auth (JWT) — `user_id` is already the primary key everywhere
- Multi-user journal analytics: "users with similar profiles improve when..."

**LLM expansion path:**
Every agent already has the deterministic path as fallback. Adding GPT to any agent means:
1. Write the GPT prompt + Pydantic output model
2. Add `run_gpt()` with the standard try/except fallback pattern
3. Add `mode` param to the relevant orchestrator function
4. Add `?mode=gpt` to the relevant route

No other changes required.
