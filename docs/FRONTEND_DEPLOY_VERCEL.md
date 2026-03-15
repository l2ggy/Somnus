# Somnus Frontend Deployment Setup (Vercel, No Local Terminal)

This guide is for setting up **frontend deployment infrastructure only**.
It does **not** require frontend code to exist yet, and it does not require a local machine.

You can complete this from phone/tablet/desktop browser using GitHub + Vercel UI.

---

## 1) What you need

Required:
1. GitHub account with admin/write access to this repo
2. Vercel account (can sign in with GitHub)
3. A deployed backend URL (for later env var), for example:
   - `https://somnus-api.onrender.com`

Optional now, needed later:
- Custom domain (e.g. `somnus.app`)

---

## 2) Create a placeholder frontend project in Vercel first

Do this even before frontend code is ready, so the deployment target exists.

1. Log in to **Vercel**.
2. Click **Add New...** → **Project**.
3. Connect GitHub if asked.
4. Select the **Somnus** repo.
5. In "Framework Preset":
   - If your frontend folder exists later (e.g. `web/`), choose the actual framework then.
   - For now, leave defaults; we are just setting up deployment plumbing.
6. Set **Root Directory**:
   - If frontend will live at repo root, leave as `.`
   - If frontend will live in subfolder (recommended), set later to `web`
7. Click **Deploy**.

If build fails because frontend code does not exist yet, that is fine for now. The project/config still gets created.

---

## 3) Configure deployment target for a future frontend subfolder

Once project is created in Vercel:

1. Open project → **Settings** → **General**.
2. Find **Root Directory**.
3. Set it to the folder where frontend will live (example: `web`).
4. Save.

This prevents Vercel from trying to build backend files as if they were frontend.

---

## 4) Configure production environment variables now

Open project → **Settings** → **Environment Variables**.

Add these variables so frontend can call backend when code is ready:

1. `NEXT_PUBLIC_API_BASE_URL`
   - Value: your backend base URL
   - Example: `https://somnus-api.onrender.com`
   - Environments: **Production**, **Preview**, **Development**

Optional common vars (if you add auth later):
2. `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
3. `CLERK_SECRET_KEY`

Click **Save** after each variable.

---

## 5) Configure automatic deployments from GitHub

In Vercel project → **Settings** → **Git**:

1. Confirm repo is connected.
2. Confirm **Production Branch** is set (usually `main`).
3. Keep **Auto-deploy** enabled.

Result:
- Push/merge to `main` triggers Production deploy.
- Pull requests trigger Preview deploys.

---

## 6) Protect production with a simple branch strategy

In GitHub repo settings (recommended):

1. Open **Settings** → **Branches**.
2. Add protection rule for `main`:
   - Require pull request before merge
   - Require at least 1 approval

This prevents accidental frontend deploys to production.

---

## 7) Add a custom domain (optional, can be done now)

In Vercel project → **Settings** → **Domains**:

1. Add your domain (e.g. `app.somnus.ai`).
2. Follow Vercel DNS instructions (A/CNAME records).
3. Wait for verification.

Vercel provisions HTTPS automatically once DNS is correct.

---

## 8) Preview and production URL expectations

After setup, you will have:
- Production URL: `https://<project>.vercel.app`
- Preview URLs per PR: `https://<project>-<hash>-<team>.vercel.app`

Use preview URLs to validate frontend changes before merging.

---

## 9) First-time checks once frontend code exists

When you later add frontend code, verify in this order:

1. Vercel build succeeds.
2. Site loads at production URL.
3. Frontend can call backend health endpoint through configured base URL.
4. Session APIs work:
   - `POST /session/start`
   - `GET /session/{user_id}/state`

---

## 10) Common misconfigurations and fixes


### "No fastapi entrypoint found" during frontend deploy
- Cause: Vercel is trying to treat the deployment as a Python backend instead of a static/frontend app.
- Fix checklist:
  1. In Vercel project settings, set **Root Directory** to `web` (not repo root).
  2. Ensure `web/index.html` exists (Vercel needs a frontend entry file for static deploys).
  3. Keep `web/vercel.json` committed so all routes resolve to `index.html` for SPA-style routing.
  4. In Framework Preset, choose **Other** (or your actual frontend framework once added).

### Build runs in wrong folder
- Fix: set **Root Directory** to frontend folder (`web`, `frontend`, etc.).

### Frontend cannot reach backend
- Fix: set `NEXT_PUBLIC_API_BASE_URL` exactly, including `https://`.

### CORS errors in browser
- Cause: backend not allowing frontend origin.
- Fix: add frontend URL to backend CORS allowlist.

### Preview works but production fails
- Cause: env vars only set for Preview, not Production.
- Fix: set env var scope to all required environments.

---

## 11) What this setup gives you right now

Even before frontend is built, you now have:
1. A Vercel project connected to repo
2. Auto-deploy pipeline ready
3. Environment variable scaffolding ready
4. Optional domain/HTTPS path prepared

So when you add frontend code later, deployment is mostly plug-and-play.
