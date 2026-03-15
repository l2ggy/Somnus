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
- **Tech stack:** https://www.canva.com/design/DAHEBLwbQpk/LWZMTwcLAIbBNEK5P784XA/view?utm_content=DAHEBLwbQpk&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h984f96f9a9 

---

## What Somnus does (end-to-end)

### 1) Pre-sleep planning
Before bedtime, Somnus builds a nightly strategy from user preferences and history.

### 2) Real-time overnight loop
On each sensor tick, the backend executes:

1. Sensor intake + validation
2. Signal interpretation
3. Sleep phase inference
4. Disturbance detection
5. Intervention selection
6. State persistence

### 3) Morning reflection + personalization
User journal feedback is converted into reflection insights and preference updates that influence future planning.

---

## Technical architecture

Somnus is implemented as an **agentic FastAPI backend** with a shared-state contract:

- `app/main.py` — API routes and lifecycle
- `app/orchestrator.py` — pipeline sequencing
- `app/agents/backend/` — API-boundary ingestion
- `app/agents/intelligence/` — interpretation, sleep state, disturbance, intervention, planning, reflection
- `app/models/` — shared Pydantic domain models
- `app/store.py` — SQLite-backed session state

The repo also includes a standalone **world-model simulation layer** (`simulation/`, `agents/`) for trajectory-based planning and hybrid LLM/rule-based action selection research.

---

## Model usage

Somnus supports deterministic and LLM-assisted modes:

- **Deterministic mode** for stable, low-latency baseline behavior
- **GPT mode** for richer planning/reflection language with safe fallback

**Model used in deployment:** `openai/gpt-oss-120b` (GPT OSS 120b) via an OpenAI-compatible endpoint.

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
