# Somnus Backend API Deployment (No Local Computer Required)

This guide is for deploying the backend **directly from the GitHub repo** using Render.
You do **not** need to run Docker, Python, or terminal commands on your own machine.

## What this repo now includes

- `Dockerfile` (backend image build)
- `.dockerignore` (clean Docker context)
- `render.yaml` (Render Blueprint config)

With these files, Render can build and deploy the backend from GitHub only.

---

## 1) Accounts and tools you need

Because you said you do not have a computer, this process uses web UIs only.

Required:
1. GitHub account with access to this repo
2. Render account (https://render.com)

Optional:
- OpenAI API key (only needed if you want `?mode=gpt`; deterministic mode works without it)

---

## 2) Confirm files exist in default branch

In GitHub web UI, confirm these files are present in the repo root:
- `Dockerfile`
- `.dockerignore`
- `render.yaml`

If they are in a PR branch, merge that PR first.

---

## 3) Deploy using Render Blueprint (repo-driven)

1. Log in to **Render**.
2. Click **New +** → **Blueprint**.
3. Connect your GitHub account (if not already connected).
4. Select this repository.
5. Select the branch you want to deploy (usually `main`).
6. Render detects `render.yaml` and shows the planned web service (`somnus-api`).
7. Click **Apply** / **Create Resources**.

Render now builds from the repo `Dockerfile`, creates the service, attaches a persistent disk, and sets baseline env vars from `render.yaml`.

---

## 4) Set secret environment variables in Render UI

After service creation:

1. Open Render service: `somnus-api`.
2. Go to **Environment**.
3. Set:
   - `OPENAI_API_KEY` = your key (optional if deterministic-only)
   - `OPENAI_BASE_URL` = only if using an OpenAI-compatible provider other than default
4. Confirm existing value:
   - `SOMNUS_DB_PATH=/var/data/somnus.db`

Save changes; Render will redeploy.

---

## 5) Verify persistent storage (SQLite)

In Render service settings:
1. Open **Disks**.
2. Confirm disk exists:
   - Name: `somnus-data`
   - Mount path: `/var/data`
3. Confirm env var `SOMNUS_DB_PATH` points inside mount path (`/var/data/somnus.db`).

This ensures session data persists across restarts/deploys.

---

## 6) Get backend URL

In Render service overview, copy the public URL, e.g.:
- `https://somnus-api.onrender.com`

This is your backend API base URL.

---

## 7) Smoke test using only browser

Because you have no local machine, use browser URL checks first:

### Health check
Open in browser:
- `https://<your-render-url>/health`

Expected JSON:
```json
{"status":"ok","service":"somnus"}
```

### Agent registry check
Open:
- `https://<your-render-url>/agents`

Expected: JSON object with `agents` list.

> For POST endpoint testing without local tools, use Render's API docs (`/docs`) or any web-based API client from a mobile browser.

---

## 8) Use FastAPI Swagger UI from phone/tablet browser

1. Open:
   - `https://<your-render-url>/docs`
2. Test `POST /session/start`:
   - Click endpoint → **Try it out**
   - Paste body:
```json
{
  "user_id": "demo-user",
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
3. Execute.
4. Then test `GET /session/{user_id}/state` with `demo-user`.

If both succeed, backend is operational.

---

## 9) Frontend integration once frontend exists

Set frontend environment variable:
- `API_BASE_URL=https://<your-render-url>`

Frontend should call:
- `POST /session/start`
- `POST /session/{user_id}/sensor`
- `GET /session/{user_id}/state`
- `POST /session/{user_id}/journal`

---

## 10) Immediate production gaps to address

Current backend is deployable, but before real users:
1. Add CORS middleware with explicit allowed origins.
2. Add authentication/authorization (do not trust arbitrary user-provided `user_id`).
3. Add rate limiting and abuse protection.
4. Add monitoring/error reporting.
5. Consider moving session store from SQLite to managed Redis/Postgres for scale.

---

## 11) If you cannot do any API testing yourself

You can still launch backend from repo and hand this verification checklist to a teammate:
1. `GET /health` returns status JSON.
2. `POST /session/start` creates session.
3. `GET /session/{user_id}/state` returns created session.
4. Restart service; verify session still exists (disk persistence check).

That is sufficient to confirm deployment from repo is working.
