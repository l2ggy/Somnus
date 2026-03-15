# Somnus Web + Mobile Deployment Plan

## 1) What exists right now (from this repo)
- **Backend framework:** FastAPI app with session lifecycle APIs (`/session/start`, `/session/{user_id}/sensor`, `/session/{user_id}/journal`, `/session/{user_id}/state`).
- **Agent orchestration:** deterministic multi-agent pipeline for pre-sleep, night tick, and morning reflection.
- **State model:** Pydantic `SharedState` carries preferences, latest sensor, sleep state, intervention, nightly plan, hypotheses, and journal history.
- **Current persistence:** in-memory Python dict (single-process only, state lost on restart).
- **Optional AI integration:** OpenAI-compatible client using env vars (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`) with deterministic fallback behavior.

## 2) Target production stack (exact recommendation)

### Client (mobile-first web app)
- **Framework:** Next.js 14 (App Router) + React 18 + TypeScript
- **UI:** Tailwind CSS + shadcn/ui (fast mobile component delivery)
- **Mobile delivery:**
  - Primary: **PWA** (installable on iOS/Android)
  - Optional packaging: Capacitor wrapper for app-store release later
- **Auth:** Clerk or Supabase Auth (email/social)
- **Data/API:** HTTPS calls from Next.js to FastAPI

### API / Orchestration
- **Runtime:** Python 3.11+
- **API framework:** FastAPI + Uvicorn/Gunicorn workers
- **Core logic:** existing `app/orchestrator.py` + agent modules
- **Containerization:** Docker

### Data layer
- **Session state (hot, low-latency):** Redis (Upstash or managed Redis)
- **Durable data (history/analytics):** Postgres (Supabase, Neon, or RDS)
- **Migration path:** replace `app/store.py` dict API with Redis-backed implementation now; optionally mirror key records into Postgres for long-term reporting.

### AI provider layer
- **Primary:** OpenAI-compatible endpoint via current `app/llm_client.py`
- **Fallback:** deterministic planner/reflection modes already implemented

### Hosting
- **Frontend:** Vercel
- **Backend API container:** Render, Railway, or Fly.io
- **Redis:** Upstash
- **Postgres:** Supabase/Neon
- **Observability:** Sentry + structured logs (JSON)

## 3) Deployment architecture

```text
Mobile Browser / PWA (Next.js on Vercel)
    |
    | HTTPS (JWT / session token)
    v
FastAPI (Docker on Render/Railway/Fly)
    |
    |-- Redis (active sessions + tick state)
    |-- Postgres (journal history, outcomes, metrics)
    |-- OpenAI-compatible LLM endpoint (optional GPT mode)
```

## 4) Implementation phases

### Phase A — productionize current backend
1. Add Dockerfile + `.dockerignore`.
2. Add `/health` and readiness checks (already has `/health`; keep it as probe target).
3. Replace in-memory `app/store.py` with an interface-backed store:
   - `InMemoryStore` for local dev
   - `RedisStore` for production
4. Add env-driven config:
   - `STORE_BACKEND=inmemory|redis`
   - `REDIS_URL=...`
   - Existing `OPENAI_*` variables remain unchanged.
5. Add CORS for frontend domain.

### Phase B — create mobile-first frontend
1. Build Next.js app with screens:
   - Onboarding/preferences
   - Live night view (risk/intervention state)
   - Morning journal + reflection
2. Connect to existing endpoints:
   - `POST /session/start`
   - `POST /session/{user_id}/sensor`
   - `GET /session/{user_id}/state`
   - `POST /session/{user_id}/journal`
3. Enable PWA capabilities:
   - manifest, icons, service worker, install prompt
4. Add basic auth and map user identity to `user_id`.

### Phase C — release pipeline
1. GitHub Actions:
   - backend: test + build Docker image + deploy
   - frontend: build + deploy Vercel
2. Environment promotion:
   - dev -> staging -> prod
3. Monitoring and rollback strategy:
   - health checks, error alerts, one-click rollback

## 5) Concrete deploy runbook (one practical path)

### Backend on Render (example)
1. Create `render.yaml` web service from Docker.
2. Set env vars:
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL` (if not using default OpenAI)
   - `OPENAI_MODEL`
   - `STORE_BACKEND=redis`
   - `REDIS_URL`
3. Set start command:
   - `gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:$PORT app.main:app`
4. Health check path: `/health`.

### Frontend on Vercel
1. Import Next.js repo.
2. Set env var `NEXT_PUBLIC_API_BASE_URL=https://somnus-api.onrender.com`.
3. Deploy; configure custom domain + HTTPS.
4. Add PWA plugin/config and verify installability on iOS/Android.

### Data services
1. Provision Upstash Redis.
2. Provision Supabase/Neon Postgres.
3. Store credentials as backend secrets.

## 6) Security and mobile reliability checklist
- Enforce HTTPS only.
- Tight CORS allowlist (frontend origin only).
- Rate limit session and sensor endpoints.
- Validate payloads (already Pydantic-backed in API models).
- Add per-user auth before allowing session mutation.
- Keep deterministic fallback path when LLM errors happen.

## 7) Why this stack fits this project proposal
- Proposal requires **monitor -> decide -> act -> learn** lifecycle.
- Current backend already implements these three phases and agents.
- Mobile requirement is satisfied quickly by a **PWA frontend**, without rewriting backend logic.
- Redis + Postgres closes the biggest production gap (current in-memory store).

## 8) Minimal MVP-to-production delta
- Keep all existing agent logic and endpoints.
- Add durable session store + frontend app + CI/CD.
- Ship as installable mobile web app first, then optionally package via Capacitor.
