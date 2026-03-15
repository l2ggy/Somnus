# Somnus Backend API Deployment on Render (No Blueprint, No Local Terminal)

This guide deploys the backend directly from GitHub using Render's **Web Service** flow.
It avoids Render Blueprint (`render.yaml`) since Blueprint is not suitable for your free/no-cost constraint.

You do **not** need to run Docker or terminal commands locally.

---

## 1) What you need

Required:
1. GitHub account with access to this repo
2. Render account

Optional:
- OpenAI API key (only for `?mode=gpt`; deterministic mode works without it)

---

## 2) Repo files required (already present)

In the GitHub repo root, confirm:
- `Dockerfile`
- `.dockerignore`

Render will build from these directly.

---

## 3) Create the backend service in Render (manual UI flow)

1. Sign in to Render.
2. Click **New +** → **Web Service**.
3. Connect GitHub (if not already connected).
4. Select this repository.
5. Choose the branch to deploy (usually `main`).
6. Configure service:
   - **Name**: `somnus-api`
   - **Runtime**: `Docker`
   - **Region**: nearest your users
   - In the form section usually called **Advanced** (or **Health & Alerts**), find **Health Check Path** and enter: `/health`
7. Click **Create Web Service**.

Render builds image from `Dockerfile` and starts the API.

### If you missed the health check field during creation

You can set it after deploy:
1. Open your service in Render.
2. Click **Settings**.
3. Scroll to **Health Check Path**.
4. Enter `/health`.
5. Click **Save Changes**.

### How to confirm health check is correctly configured

1. Open service page → **Events** or **Logs** and confirm no repeated health-check failures.
2. Open `https://<your-url>/health` in browser.
3. Confirm response is:
   - `{"status":"ok","service":"somnus"}`

---

## 4) Configure environment variables in Render UI

Open service → **Environment** and set:

- `SOMNUS_DB_PATH=/var/data/somnus.db`
- `OPENAI_MODEL=gpt-4o-mini` (or preferred model)
- `OPENAI_API_KEY=<your_key>` (optional for deterministic-only mode)
- `OPENAI_BASE_URL=<optional_custom_provider_url>`

Save changes (Render redeploys automatically).

---

## 5) Add persistent disk for SQLite data

Without a disk, SQLite data can reset on redeploy/restart.

1. Open service → **Disks**.
2. Add disk:
   - **Mount path**: `/var/data`
   - Size: 1GB is enough to start
3. Confirm env var remains:
   - `SOMNUS_DB_PATH=/var/data/somnus.db`

---

## 6) Verify backend from mobile browser

Copy your Render URL from service overview, e.g.:
- `https://somnus-api.onrender.com`

Check:

1. `https://<your-url>/health`
   - Expect: `{"status":"ok","service":"somnus"}`
2. `https://<your-url>/agents`
   - Expect JSON with `agents`
3. `https://<your-url>/docs`
   - FastAPI Swagger UI opens

Use `/docs` to test POST endpoints without local tools.

---

## 7) API smoke test via `/docs` (no terminal)

In Swagger UI (`/docs`):

1. Execute `POST /session/start` with:

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

2. Execute `GET /session/{user_id}/state` with `demo-user`.

If both succeed, backend is working.

---

## 8) Connect frontend later

When frontend exists, set:
- `API_BASE_URL=https://<your-render-url>`

Frontend calls:
- `POST /session/start`
- `POST /session/{user_id}/sensor`
- `GET /session/{user_id}/state`
- `POST /session/{user_id}/journal`

---

## 9) Important caveat about cost

- This guide intentionally avoids Blueprint.
- Render pricing changes over time; if no free web service is available on your account, you will need a paid plan/trial.

If you need a strict zero-cost host, choose a provider with a current free web service tier and use the same `Dockerfile`.

---

## 10) Next production fixes (before real users)

1. Add CORS middleware (explicit frontend origins).
2. Add authentication/authorization.
3. Add rate limiting and abuse protection.
4. Add monitoring/error reporting.
5. Move from SQLite to managed Redis/Postgres if scaling.
