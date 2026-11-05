# Deployment

Two Python services, one Vercel project. No JavaScript build anywhere.

```text
browser ──HTTPS/WebSocket──▶ web  (Vercel Python function: Streamlit as an ASGI app, server.py)
                              │  private service binding: API_INTERNAL_URL (injected at runtime, server-side only)
                              ▼
                             api  (Vercel Container Runtime: root Dockerfile, FastAPI + XGBoost + SHAP, model baked in)
                              ├──▶ Supabase Postgres (users, predictions)
                              └──▶ managed Redis via REDIS_URL (optional; no Redis = no cache, never an error)
```

Render (`render.yaml`) stays as the rollback until a Vercel **preview and** production deployment both return a real prediction (see the end of this page).

> Vercel **Services** and **Container Runtime** are both labelled *Beta* in Vercel's docs (checked 2026-11-05). WebSocket support for functions is *public beta*. Re-check these pages before deploying:
> [Services](https://vercel.com/docs/services) · [Service bindings](https://vercel.com/docs/services/bindings) · [Container Runtime](https://vercel.com/docs/functions/runtimes/container) · [Python runtime](https://vercel.com/docs/functions/runtimes/python) · [Function limits](https://vercel.com/docs/functions/limitations) · [Container Registry limits](https://vercel.com/docs/container-registry/limits-and-pricing) · [Redis on Vercel](https://vercel.com/docs/redis)

## How it is wired (`vercel.json`)

| Service | Runtime | Entrypoint | Public? |
|---------|---------|------------|---------|
| `web` | `python` | `api.index:app` → `server.py` → `streamlit.App(streamlit_app.py)` | Yes, via the single top-level rewrite `/(.*) → web` |
| `api` | `container` | root `Dockerfile` | **No.** It has no rewrite, so the internet can't reach it. Only `web` can, through its binding. |

- **Binding:** `web` declares `{"type": "service", "service": "api", "format": "url", "env": "API_INTERNAL_URL"}`. Vercel generates the value per deployment, so a preview's UI talks to the same preview's API. **Never set `API_INTERNAL_URL` yourself.**
- **Bindings don't authenticate** (Vercel docs). The API still requires the user's JWT on every scoring route.
- **The URL never reaches the browser.** Streamlit code runs on the server; `ui/api.py` is the only module that reads the URL. Its error messages are generic, and the original exception (which would contain the host) is dropped (`from None`). This is tested in `tests/frontend/test_api_client.py`. The `/health` route of `web` reports only `api_binding_configured: true/false`.

## Size limits (checked before choosing this layout)

| Item | Limit (Vercel docs) | Measured here | Source of the measurement |
|------|---------------------|---------------|---------------------------|
| `web` Python bundle (uncompressed) | 500 MB; "large functions" beta up to 5 GB | ≈ 424 MB of dependencies (streamlit, pandas, plotly, requests + transitive; pyarrow alone ≈ 169 MB), Linux wheels, Python 3.12, excluding `__pycache__` | `pip install --target … --platform manylinux…` + `du`, run during this migration |
| `api` container image | 15 GB total, 2 GB per compressed layer, 2,000,000 files | 1.13 GB (local arm64 build) | `docs/verification/cold_start_local.json` and `docs/verification/ci.md` (amd64 size from the CI run) |
| Request / response body | 4.5 MB | Streamlit upload limit set to 4 MB (`server.py`, `.streamlit/config.toml`); `/predict/batch` capped at 500 rows | |
| Function duration | Hobby 300 s | `maxDuration: 300` for `web` | |

The `web` bundle is under the limit but not by much, which is why the API's heavy dependencies (xgboost, shap, scikit-learn, scipy) live only in the container (`requirements-api.txt`). The root `requirements.txt` is UI-only. If a future Streamlit release pushes the bundle over 500 MB, set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` (large-functions beta).

`xgboost-cpu` is used inside the image: the normal Linux `xgboost` wheel also pulls in `nvidia-nccl-cu12` (CUDA), which a CPU-only API doesn't need.

## Environment variables

Vercel environment variables are **project-wide**: both services receive them. Both run server-side, there's no client-side build that could inline them, and the UI code never reads the API's credentials.

| Variable | Needed by | Environments | Notes |
|----------|-----------|--------------|-------|
| `JWT_SECRET_KEY` | api | Production, Preview | **Required**; the API refuses to start without it. `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Mark it Sensitive. |
| `SUPABASE_URL` | api | Production, Preview | Your Supabase project URL. |
| `SUPABASE_KEY` | api | Production, Preview | Server-side only. Mark it Sensitive. Tables: see README. |
| `REDIS_URL` | api | Production, Preview | A TCP URL (`rediss://…`) from the managed Redis. Optional: unset or unreachable means no cache. The API also accepts `KV_URL`. Never logged (it contains the password). |
| `PORT` | api | Production, Preview | **Set to `8000`.** Vercel routes container traffic to port 80 unless `PORT` is set, and the image listens on `${PORT:-8000}` as a non-root user. |
| `API_INTERNAL_URL` | web | (automatic) | Injected by the binding. Do not create it. |
| `BACKEND_URL` | web | — | Don't set on Vercel. Used only locally and on Render. |
| `ALLOWED_ORIGINS` | api | — | Not needed: no browser calls the API directly. |

**Redis:** Vercel KV no longer exists, and Redis comes from a Marketplace integration (e.g. Upstash). Vercel's docs don't list which variable names a given integration injects. After connecting it, open *Settings → Environment Variables*:
- `REDIS_URL` or `KV_URL` holding a `redis://` / `rediss://` URL works as-is.
- If you only see REST variables (`…_REST_API_URL` / `…_TOKEN`), copy the **TCP** connection string from the provider's console into `REDIS_URL`. This app uses the `redis` Python client over TCP, not the REST API.

## Vercel project settings (dashboard)

- **Root Directory:** repository root (`.`).
- **Framework Preset / Build / Install / Output overrides:** leave empty. `vercel.json` services define the builds. A top-level `buildCommand` / `functions` / `framework` is invalid in services mode.
- **Fluid compute:** on (the default for new projects; large functions need it).
- **Function region:** pick the one closest to your Supabase project (default `iad1`).
- **Deployment Protection:** keep it on for Preview. It applies to the public `web` entry; internal binding calls skip it.

## Commands (you run these; I didn't deploy)

```bash
npm i -g vercel@latest
vercel login
vercel link                       # create/link the project from the repo root

# secrets: prompted for the value, never typed into shell history
vercel env add JWT_SECRET_KEY production
vercel env add JWT_SECRET_KEY preview
vercel env add SUPABASE_URL production
vercel env add SUPABASE_URL preview
vercel env add SUPABASE_KEY production
vercel env add SUPABASE_KEY preview
vercel env add PORT production    # value: 8000
vercel env add PORT preview       # value: 8000
# Redis: Dashboard → Storage / Marketplace → add a Redis store → connect to this project (Production + Preview),
# then check the variable names as described above.

vercel deploy                     # preview
# … run the checklist below against the preview URL …
vercel deploy --prod              # only after the preview passes
```

Git integration works too: every push to a PR branch creates a preview, and merging to `main` deploys production.

## Preview checklist (record the results in `docs/verification/vercel_preview.md`)

1. **Build:** in the deployment's *Build Logs*, both `web` and `api` built. Note the image size and the function bundle size.
2. **Binding:** `curl https://<preview>/health` → `"api_binding_configured": true`.
3. **API is private:** `curl -i https://<preview>/meta` returns the Streamlit page (HTML), **not** the API's JSON, and `curl -X POST https://<preview>/predict` doesn't reach FastAPI.
4. **No URL in the browser:** in DevTools → Network and *View source*, search for `API_INTERNAL_URL`, `.vercel.app` hosts other than the preview, and `supabase.co`. Expect no hits.
5. **End to end:** sign up → *Score a company* → "Distressed firm" preset → a PD, grade and SHAP chart appear. Then *History* shows the new row, and the row exists in Supabase `predictions`.
6. **Cache:** score the same preset again → the result shows "served from cache" (if Redis is configured).
7. **Cold start (measure, don't assume):** wait at least 1 minute (preview containers scale down after 30 s idle; production after 5 minutes), then score once and note the "API responded in X s" caption on the Score page. Repeat once warm. Record both numbers.
8. **Session state:** refresh the page → you're signed out. That's expected (see below).
9. **Portfolio:** upload `ui/sample_portfolio.csv` → 150 rows scored.

## Cold starts

Measured **locally** (`docs/verification/cold_start_local.json`, Docker on Apple Silicon): seconds from `docker run` to the first 200, plus in-container timings for the first prediction with SHAP and a counterfactual search. Vercel adds VM start and image fetch on top, and the docs don't publish a number. **Don't promise zero cold starts.** Measure with step 7 above and record it. After 5 idle minutes in production (30 s in preview) the container scales down, so the next user pays the start-up again. The UI waits up to 60 s for the API and shows a "still starting" message instead of a stack trace.

## Streamlit session state on serverless

`st.session_state` (your login token, the firm being edited, the last results) lives in the memory of one function instance. A refresh, a reconnect after the function's 300 s duration, or being routed to a new instance starts a **new, empty session**: you sign in again. Durable data (users, prediction history) lives in Supabase, never in the session. This is a demo limitation, not a bug.

## Rolling back / removing Render

- **Rollback now:** Render still deploys from `render.yaml`. Since this migration, its API builds from `requirements-api.txt` and the UI from `requirements.txt`. If your Render services were created from the Blueprint, the build-command change syncs automatically. If not, set the API's build command in the Render dashboard to `pip install -r requirements-api.txt` **before** merging. Otherwise Render's next API build installs only the UI packages, fails, and keeps serving the previous deploy.
- **Remove Render** once a Vercel preview *and* production both pass steps 2–6:
  1. In Render, **suspend** both services (reversible) and watch Vercel for a few days.
  2. Delete the two Render services, and any Render Redis.
  3. In git: delete `render.yaml`, update the README's live-demo link to the Vercel URL, and remove the Render section of this page.
  4. Rotate `JWT_SECRET_KEY` if the Render and Vercel values were the same, so old tokens stop working.

## Local

```bash
docker compose up --build          # API on :8000 + Redis (needs JWT_SECRET_KEY in .env)
BACKEND_URL=http://localhost:8000 streamlit run streamlit_app.py
```
