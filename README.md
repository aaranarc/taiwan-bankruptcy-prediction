# Corporate Bankruptcy Prediction System

An XGBoost-powered corporate bankruptcy prediction system benchmarked against the classical Altman Z-Score, interpreted with SHAP, utilizing Redis caching, and integrated with Supabase database for user authentication and prediction history logging.

**[Live Demo Link](https://bankruptcy-predictor-ui.onrender.com/)**

![Dashboard Screenshot](assets/asset.png)

---

## Architecture Overview

This decoupled application separates frontend presentation from model inference and database layers:

- **Streamlit UI**: Pure presentation layer. Communicates with the backend exclusively via JSON HTTP APIs with JWT bearer authentication.
- **FastAPI Backend**: Hosts the XGBoost classifier pipeline, handles password hashing and JWT validation, interacts with Supabase, and manages Redis caching.
- **Redis Cache**: Caches predictions to bypass model inference for identical inputs.
- **Supabase Postgres**: Stores hashed user credentials and historical predictions.

---

## Model Performance

Held-out test split: 1,364 firms, 44 bankrupt (base rate 3.2%). Recall is for the bankrupt class at a 0.50 threshold. Baseline rows come from `taiwan_bankruptcy_prediction.ipynb`; the deployed row comes from `python training/evaluate.py`.

| Model | ROC-AUC | PR-AUC | Recall (Bankrupt) |
|-------|---------|--------|-------------------|
| Altman Z-Score (1968) | 0.92 (using −Z as the score) | — | 0% at its 1.81 / 2.99 cut-offs (6,813 of 6,819 firms "Safe") |
| Logistic Regression | 0.45 | 0.03 | 20% |
| Random Forest | 0.93 | 0.47 | 27% |
| RF + SMOTE | 0.95 | 0.48 | 55% |
| **XGBoost + `scale_pos_weight` (deployed)** | **0.95** | **0.47** | **80%** |

> **Logistic Regression** scores below 0.5 ROC-AUC, which points to a convergence/scaling problem in that run (default solver settings), not a real baseline.
>
> **Altman Z-Score:** the ranking works (ROC-AUC 0.92 on the test set), but the calibration doesn't transfer. The 1.81 / 2.99 cut-offs were set for 1960s US manufacturers; on this Taiwanese data almost every firm scores above 2.99, so the classic zones flag no bankruptcies.

### Decision thresholds (deployed model)

| Threshold | Recall | Precision | Bankrupt firms caught (of 44) | False alarms (of 1,320) |
|-----------|--------|-----------|-------------------------------|-------------------------|
| 0.15 | 0.98 | 0.12 | 43 | 306 |
| 0.35 | 0.89 | 0.18 | 39 | 172 |
| 0.50 | 0.80 | 0.23 | 35 | 116 |
| 0.60 | 0.77 | 0.27 | 34 | 91 |

The API maps the predicted probability to risk tiers with the same cut-offs: **Low** < 0.15 ≤ **Medium** < 0.35 ≤ **High** < 0.60 ≤ **Critical**. Every response also returns `top_factors`, the five features with the largest per-prediction SHAP contribution.

---

## Project Structure

```text
app/
├── __init__.py
├── main.py                     # Entry point combining routers, middlewares, and CORS
├── models/
│   ├── model.joblib            # Serialized XGBoost ML model
│   └── robust_scaler.pkl, winsorize_bounds.pkl, etc.
├── api/
│   ├── routes_predict.py       # Predict endpoints with Redis & Supabase prediction logging
│   └── routes_auth.py          # /auth/signup & /auth/login with Supabase users
├── core/
│   ├── config.py               # Env var loading via dotenv
│   ├── security.py             # Password hashing (bcrypt) & JWT token helpers
│   ├── dependencies.py         # DB (Supabase client) and Current User retrieval
│   └── exceptions.py           # Custom exception handler classes
├── services/
│   └── model_service.py        # Pipeline: load artifacts, winsorize, scale, predict, compute Altman Z
├── middleware/
│   └── logging_middleware.py   # Global HTTP request/response logging
├── cache/
│   └── redis_cache.py          # Redis client cache helper
└── utils/
    └── logger.py               # Clean stdout logging formatter
streamlit_app.py                # Streamlit UI client
training/
├── __init__.py
├── train_utils.py              # Winsorization & scaling helper functions
└── train_model.py              # XGBoost training pipeline script (outputs to app/models/)
```

---

## Setup & Running Locally

### 1. Configure the Environment
Create a `.env` file at the root of the project:
```ini
SUPABASE_URL=https://your-supabase-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your-custom-jwt-signing-secret   # required: the API refuses to start without it
ALLOWED_ORIGINS=http://localhost:8501           # optional: comma-separated CORS origins (default shown)
```
Generate a secret with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. The Streamlit UI calls the API server-to-server, so CORS only matters for browser clients.

### 2. Prepare Database Tables
Run the following SQL in your Supabase SQL Editor:
```sql
create table users (
    id uuid default gen_random_uuid() primary key,
    email text unique not null,
    password_hash text not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table predictions (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references users(id) on delete cascade not null,
    input_features jsonb not null,
    prediction_result jsonb not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);
```

### 3. Start the Backend API
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Start the Frontend UI
```bash
export BACKEND_URL="http://localhost:8000"
streamlit run streamlit_app.py
```
