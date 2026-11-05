# FastAPI model API. Used by docker compose, by Vercel's Container Runtime (service "api" in vercel.json)
# and as the image CI builds. Model artifacts are baked in, so a cold start never downloads anything.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
WORKDIR /app

# libgomp: OpenMP runtime that XGBoost's compiled library links against (not in the slim image).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ app/
COPY config/ config/

RUN useradd --create-home --uid 10001 api
USER api

# Vercel routes container traffic to $PORT (project env var, see docs/DEPLOYMENT.md); 8000 locally.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/meta')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
