import os
import hashlib
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis

from src.evaluation import (
    load_model_artifacts,
    preprocess_input,
    predict_bankruptcy,
    get_risk_color,
    get_risk_emoji,
)
from src.altman import (
    compute_altman_z,
    classify_zone,
    get_zone_color,
    get_zone_emoji,
    ALTMAN_FEATURES,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Taiwan Bankruptcy Prediction API",
    description="FastAPI service for predicting corporate bankruptcy using XGBoost and Altman Z-Score, with Redis caching.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts and Redis
artifacts = None
redis_client = None

@app.on_event("startup")
def startup_event():
    global artifacts, redis_client
    logger.info("Loading model artifacts...")
    artifacts = load_model_artifacts()
    if not artifacts or artifacts.get("model") is None:
        logger.error("Failed to load model artifacts. Ensure train_and_save.py has been run.")
    else:
        logger.info("Model artifacts loaded successfully.")

    # Initialize Redis connection
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    logger.info(f"Connecting to Redis at {redis_url}...")
    try:
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2.0)
        redis_client.ping()
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.")
        redis_client = None


class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(
        ..., 
        description="Dictionary of features where keys match the dataset columns.",
        example={
            "Industrial class": 1.0,
            "ROA(C) before interest and depreciation before interest": 0.46,
            "ROA(A) before interest and % after tax": 0.52,
            "ROA(B) before interest and depreciation after tax": 0.51,
            "Operating Gross Margin": 0.60,
            # ... all other required features
        }
    )


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    risk_tier: str
    risk_emoji: str
    risk_color: str
    altman_z_score: Optional[float] = None
    altman_zone: Optional[str] = None
    altman_emoji: Optional[str] = None
    altman_color: Optional[str] = None
    cached: bool


def get_cache_key(features: Dict[str, float]) -> str:
    """Generate a unique MD5 hash based on sorted feature key-values."""
    sorted_features = sorted(features.items())
    features_str = json.dumps(sorted_features)
    return f"predict:{hashlib.md5(features_str.encode('utf-8')).hexdigest()}"


def run_inference(features: Dict[str, float]) -> Dict[str, Any]:
    """Preprocess, predict, and compute Altman Z-Score for a single input."""
    # Convert to DataFrame
    df_raw = pd.DataFrame([features])
    
    # Preprocess
    df_proc = preprocess_input(
        df_raw, 
        artifacts["scaler"], 
        artifacts["bounds"], 
        artifacts["feature_names"]
    )
    
    # Predict
    pred_df = predict_bankruptcy(artifacts["model"], df_proc)
    row = pred_df.iloc[0]
    
    # Compute Altman Z-Score if Altman features are available
    z_score = None
    zone = None
    zone_emoji = None
    zone_color = None
    
    try:
        # Check if all required Altman features are in the raw features
        has_all_altman = all(col in features for col in ALTMAN_FEATURES.values())
        if has_all_altman:
            series = pd.Series(features)
            z_score = float(compute_altman_z(series))
            zone = classify_zone(z_score)
            zone_emoji = get_zone_emoji(zone)
            zone_color = get_zone_color(zone)
    except Exception as e:
        logger.warning(f"Error computing Altman Z-Score: {e}")

    return {
        "probability": float(row["probability"]),
        "prediction": int(row["prediction"]),
        "risk_tier": row["risk_tier"],
        "risk_emoji": get_risk_emoji(row["risk_tier"]),
        "risk_color": get_risk_color(row["risk_tier"]),
        "altman_z_score": z_score,
        "altman_zone": zone,
        "altman_emoji": zone_emoji,
        "altman_color": zone_color,
    }


@app.get("/health")
def health_check():
    """Check application health and Redis status."""
    redis_ok = False
    if redis_client:
        try:
            redis_ok = bool(redis_client.ping())
        except Exception:
            redis_ok = False
            
    model_ok = artifacts is not None and artifacts.get("model") is not None
    
    overall_status = "healthy" if (model_ok) else "unhealthy"
    
    return {
        "status": overall_status,
        "redis_connected": redis_ok,
        "model_loaded": model_ok,
    }


@app.get("/features", response_model=List[str])
def get_features():
    """Get the list of features expected by the prediction model."""
    if not artifacts or not artifacts.get("feature_names"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model features not loaded yet."
        )
    return artifacts["feature_names"]


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    """Predict corporate bankruptcy for a single company."""
    if not artifacts or not artifacts.get("model"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
        
    # Check if all required features are present
    missing = [f for f in artifacts["feature_names"] if f not in payload.features]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required features: {missing}"
        )
        
    cache_key = get_cache_key(payload.features)
    
    # Try fetching from Redis cache
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Prediction cache HIT")
                res = json.loads(cached_data)
                res["cached"] = True
                return res
        except Exception as e:
            logger.warning(f"Error accessing Redis cache: {e}")

    # Cache miss or Redis down
    logger.info("Prediction cache MISS. Running model inference...")
    try:
        inference_res = run_inference(payload.features)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
        
    # Attempt to cache the result
    if redis_client:
        try:
            # Cache for 24 hours (86400 seconds)
            redis_client.setex(cache_key, 86400, json.dumps(inference_res))
        except Exception as e:
            logger.warning(f"Error saving to Redis cache: {e}")
            
    inference_res["cached"] = False
    return inference_res


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(payloads: List[PredictionInput]):
    """Predict corporate bankruptcy for a batch of companies."""
    if not artifacts or not artifacts.get("model"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
        
    results = []
    for idx, payload in enumerate(payloads):
        # Validate features
        missing = [f for f in artifacts["feature_names"] if f not in payload.features]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Item {idx} is missing required features: {missing}"
            )
            
        cache_key = get_cache_key(payload.features)
        cached_result = None
        
        if redis_client:
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    cached_result = json.loads(cached_data)
                    cached_result["cached"] = True
            except Exception as e:
                logger.warning(f"Error accessing Redis cache for item {idx}: {e}")
                
        if cached_result:
            results.append(cached_result)
        else:
            try:
                inference_res = run_inference(payload.features)
                if redis_client:
                    try:
                        redis_client.setex(cache_key, 86400, json.dumps(inference_res))
                    except Exception as e:
                        logger.warning(f"Error saving to Redis cache for item {idx}: {e}")
                inference_res["cached"] = False
                results.append(inference_res)
            except Exception as e:
                logger.error(f"Inference error on item {idx}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed for item {idx}: {str(e)}"
                )
                
    return results
