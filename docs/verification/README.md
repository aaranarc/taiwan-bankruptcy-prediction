# Verification evidence

| File | What it is | How it was produced |
|------|------------|---------------------|
| `screenshots/*_desktop.png`, `*_mobile.png` | Every page at 1440 px and 390 px (phone) width | `python -m scripts.screenshots` (Playwright) clicking through the real Streamlit UI, which calls the real FastAPI app (`python -m scripts.demo_api`). **Only the database is an in-memory stand-in** for Supabase, so the History page shows that session's scores, not a real Supabase table. Not taken on Vercel. |
| `cold_start_local.json` | Container start-up time and in-container model timings | Measured locally with Docker on the machine named in the file; **not** a Vercel number. |
| `ci.md` | Test count, coverage and Docker image size from GitHub Actions | Copied from the CI run linked in the file. |
| `vercel_preview.md` | *(to be created)* Results of the preview checklist in `docs/DEPLOYMENT.md` | Filled in after you deploy. |

The sample-portfolio screenshot scores firms from the training data, so those PDs are in-sample.
