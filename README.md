# Somnus 🌙

**Somnus** is an AI sleep strategist that monitors live sleep signals, predicts disturbances before they wake the user, and responds with personalized interventions in real time.

Instead of only generating post-hoc sleep analytics, Somnus runs an **active overnight control loop**: it ingests sensor data, infers sleep state, detects risk, selects an intervention, and adapts over time using morning journal feedback.

> Built for the [GenAI Genesis Hackathon 2026](https://genai-genesis-2026.devpost.com/?_gl=1*1abvi59*_gcl_au*OTc1NjQzNzk4LjE3NzM0NjY0MTg.*_ga*MTk1ODM0NzUyNi4xNzU4NDI4MTg4*_ga_0YHJK3Y10M*czE3NzM1MzYzNTEkbzEyJGcxJHQxNzczNTM2NDQwJGo2MCRsMCRoMA..).

---

## Why this matters

Sleep apps usually tell users what went wrong **after** the night ends.
Somnus is designed to help **during** the night:

- Detects threats to sleep continuity (noise spikes, movement, wake risk)
- Applies context-aware interventions (brown/pink/white noise, rain/waves, breathing pace, gradual wake)
- Learns user preferences from journal feedback to improve future nights

The goal is simple: **fewer disruptions, smoother wakeups, and personalized sleep quality gains over time**.

---

## Live links

- **Frontend (Vercel):** https://somnus-gamma.vercel.app/
- **Backend API (Render):** https://somnus-api.onrender.com
- **Devpost:** https://devpost.com/software/somnus-udcfsk
- **Canva presentation:** https://www.canva.com/design/DAHD6B1R3jA/Z2kF_LeWoFk4smkhWoUUJw/view
- **Youtube demo:** https://www.youtube.com/watch?v=_3oaMIZYD38

---

## What Somnus does (end-to-end)

Somnus runs a three-phase nightly cycle that compounds over time.

### 1) Pre-sleep planning
Before bedtime, the **Strategist** agent builds a `NightlyPlan` — a structured goal, ordered intervention preference list, and sleep-phase targets — by reading the user's preferences and scanning the last 5 journal entries for patterns. In GPT mode, the LLM reasons over this history and returns a JSON-validated plan; in deterministic mode, the same output schema is produced from rule-based logic in under 1 ms.

### 2) Real-time overnight loop
Every sensor reading (targeted at ~30–60 s intervals) triggers a 5-agent deterministic pipeline — no LLM calls, no randomness, no blocking I/O:

| Step | Agent | What it does |
|---|---|---|
| 1 | **Intake** | Validates and clamps raw sensor values (HR, HRV, movement, noise, light, breathing rate); fills missing timestamps |
| 2 | **Sensor Interpreter** | Translates every numeric reading into a labeled signal (`quiet / moderate / loud / very_loud`, `still / restless / active`, etc.) and appends a structured hypothesis to shared state |
| 3 | **Sleep State** | 6-rule priority classifier: infers sleep phase (`awake / light / deep / rem`), confidence, and a `wake_risk` score (0–1) from the labeled signals |
| 4 | **Disturbance** | Scans 4 threat types (noise spike, light spike, high movement, elevated HR) with **phase-aware thresholds** (deep sleep triggers at 45 dB; light sleep at 58 dB); bumps `wake_risk` proportionally to severity |
| 5 | **Intervention** | If `wake_risk ≥ 0.38`, selects the best intervention type for the current phase (e.g. `brown_noise` for deep, `breathing_pace` for REM), filters out user-disliked types, and scales intensity as `base_intensity[aggressiveness] × wake_risk` |

The full tick completes in ~1 ms. State is persisted to SQLite after every tick.

### 3) Morning reflection
After the user submits their journal, the **Reflection** agent computes a quality label, rolling restedness average, and — in GPT mode — generates structured insights and a personalization suggestion. Critically, GPT output is only trusted for narrative fields; scores and timestamps are always pulled from the user's actual journal entry.

### 4) Cross-night learning (outer loop)
At end-of-night, Somnus runs a four-stage outer loop that accumulates learning across nights:

1. **NightSummaryAgent** — compiles tick-level data into a compact session summary (phases observed, disturbances detected, peak wake risk, intervention usage)
2. **InterventionReviewAgent** — evaluates each intervention with a `helpful / neutral / harmful` verdict and confidence score
3. **PolicyUpdateAgent** — updates a persistent `UserPolicy` with per-intervention scores in `[-2.0, +2.0]`; bounded nudges accumulate naturally over weeks. Interventions scoring ≤ −0.5 are blocked from the next night's candidates
4. **Re-planning** — the updated policy feeds directly into the next night's `NightlyPlan`, adjusting intervention ordering, risk posture (`protective / balanced / exploratory`), and sleep goal

Every outer-loop run returns a `DecisionTrace` — a single structured object that answers: *what did the system learn, what changed in the policy, and what will be different tomorrow night.*

---

## Technical architecture

Tech stack at `docs/Somnus-tech-stack.png`.

### Shared-state contract

Every agent in Somnus is a **pure function**: `SharedState → SharedState`. Agents never call each other directly. All coordination flows through a single Pydantic `SharedState` object, with each agent returning an immutable updated copy via `.model_copy(update={...})`. This makes agents trivially testable in isolation, safe to reason about, and straightforward to swap for LLM-backed versions.

```
SharedState
  ├── preferences          (UserPreferences)
  ├── latest_sensor        (SensorSnapshot)
  ├── sleep_state          (phase, confidence, wake_risk, disturbance_reason)
  ├── active_intervention  (type, intensity, rationale)
  ├── nightly_plan         (goal, intervention order, notes)
  ├── hypotheses           (append-only agent reasoning log)
  └── journal_history      (JournalEntry[])
```

`hypotheses` is the inter-agent communication channel: agents log structured reasoning payloads here, which planning and reflection agents can read on future ticks to build richer context for LLM prompts.

### Ownership and contract enforcement

Agent boundaries are formally declared in `app/pipeline_contracts.py` via `AgentContract` objects that specify each agent's lifecycle placement, `reads`, `writes`, and owning team. The orchestrator validates all contracts at startup — incompatible merges (duplicate names, stale read/write scopes) fail fast before a single request is served.

### Key modules

| Path | Role |
|---|---|
| `app/main.py` | FastAPI routes and lifespan hook |
| `app/orchestrator.py` | Sole keeper of agent execution order; routes never import agents directly |
| `app/agents/backend/intake.py` | API-boundary sensor validation and clamping |
| `app/agents/intelligence/` | 6 intelligence agents: sensor interpreter, sleep state, disturbance, intervention, strategist, reflection |
| `app/models/userstate.py` | All Pydantic v2 domain models with field-level validators |
| `app/models/outer_loop.py` | Cross-night learning models: `NightSummary`, `InterventionReview`, `UserPolicy`, `DecisionTrace`, `OuterLoopReport` |
| `app/store.py` | SQLite persistence — state stored as JSON blobs, no migrations needed as schema evolves |
| `app/llm_client.py` | OpenAI-compatible wrapper with `response_format` negotiation and markdown fence stripping |

### Persistence

`SharedState` is stored as a single JSON blob per user in SQLite (`INSERT OR REPLACE`). Pydantic handles deserialization with type coercion and field defaults, so schema evolution requires no migrations. The four-function store interface (`save_state`, `load_state`, `delete_state`, `list_states`) is stable regardless of model changes underneath.

---

## Model usage

### Design principle: LLM where latency is acceptable, determinism where it isn't

The real-time tick pipeline (sensor → interpretation → sleep phase → disturbance → intervention) is **intentionally LLM-free**. It runs in ~1 ms, produces fully auditable output, and never fails due to API availability. This is a deliberate architectural choice: waking someone up because a model call timed out is worse than no intervention at all.

LLMs are used only for the two once-per-night operations where latency is acceptable and richer language understanding genuinely improves output:

| Operation | Mode options | LLM benefit |
|---|---|---|
| Pre-sleep planning (Strategist) | `deterministic` or `gpt` | Reasons over journal history to produce goal-aware, context-rich plans |
| Morning reflection (journal_reflection) | `deterministic` or `gpt` | Generates structured insights and personalization suggestions from multi-night patterns |

Both operations fall back to deterministic silently on any error (`mode_used="fallback"` in the response). GPT output for planning is validated with `NightlyPlan.model_validate()` before use; reflection output is validated with a dedicated `_GptReflectionOutput` Pydantic model — **GPT can only affect narrative fields, never factual fields like rested scores or timestamps.**

### Fallback chain

```
API error / timeout    →  fallback
Bad JSON               →  fallback
Pydantic ValidationError →  fallback
                            mode_used = "fallback"
```

### Model in deployment

**`openai/gpt-oss-120b`** via an OpenAI-compatible endpoint. The `llm_client` automatically negotiates `response_format={"type": "json_object"}` where the endpoint supports it, and falls back to prompt-enforced JSON with markdown fence stripping otherwise — making the client compatible with any OpenAI-spec endpoint.

---

## API highlights

- `POST /session/start` — start/replace a session and generate nightly plan
- `POST /session/{user_id}/sensor` — process one overnight sensor tick
- `GET /session/{user_id}/state` — fetch live shared state
- `POST /session/{user_id}/journal` — submit morning journal and run reflection
- `GET /health` — liveness check
- `GET /docs` — interactive Swagger UI

Both planning and reflection endpoints support `?mode=deterministic|gpt`.

---

## Local development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:

- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## Future roadmap

- Expand from web experience to native mobile app
- Add additional physical/hardware interventions (e.g., smartwatch haptics)
- Deepen long-term personalization with durable user preference memory

---

## Documentation map

- Architecture + API reference: `docs/ARCHITECTURE.md`
- World-model technical notes: `docs/world_model.md`
- World-model change log: `docs/CHANGES.md`
- Project proposal: `PROJECT_PROPOSAL.md`
- Web/mobile deployment plan: `DEPLOYMENT_WEB_MOBILE.md`
