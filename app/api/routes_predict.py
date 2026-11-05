import hashlib
import json
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from supabase import Client

from app.cache.redis_cache import redis_cache
from app.core.dependencies import get_current_user, get_supabase
from app.services.model_service import model_service
from app.utils.logger import get_logger

logger = get_logger("routes_predict")

router = APIRouter(prefix="", tags=["predictions"])

MAX_BATCH_ROWS = 500  # keeps a request well under Vercel's 4.5 MB body limit (~95 floats per row)


class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of features where keys match the dataset columns."
    )


class CounterfactualInput(PredictionInput):
    target_grade: Optional[str] = Field(None, description="Grade to reach, e.g. 'C'. Defaults to the policy's.")


class StressInput(PredictionInput):
    shocks: Dict[str, float] = Field(..., description="Shock name -> amount, see GET /meta for names and ranges.")


class BatchInput(BaseModel):
    rows: List[Dict[str, float]] = Field(..., min_length=1, max_length=MAX_BATCH_ROWS)


class TopFactor(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Calibrated probability of default (PD).")
    grade: str
    threshold: float = Field(..., description="Cost-based decision threshold; PD at or above it is flagged.")
    flagged: bool
    prediction: int
    model_version: str
    top_factors: Optional[List[TopFactor]] = None
    shap_base_value: Optional[float] = None
    shap_other: Optional[float] = None
    cached: bool


def get_cache_key(features: Dict[str, float], model_version: str) -> str:
    """Generate a unique MD5 hash based on sorted feature key-values, scoped to the model version."""
    sorted_features = sorted(features.items())
    features_str = json.dumps(sorted_features)
    return f"predict:{model_version}:{hashlib.md5(features_str.encode('utf-8')).hexdigest()}"


def require_features(features: Dict[str, float]) -> None:
    """400 unless every model feature is present and finite (JSON allows NaN/Infinity, the model doesn't)."""
    missing = [f for f in model_service.get_feature_names() if f not in features]
    if missing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Missing required features: {missing}")
    bad = [k for k, v in features.items() if not math.isfinite(v)]
    if bad:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Non-finite values for: {bad}")


def log_prediction(db: Client, user: dict, features: Dict[str, float], res: Dict[str, Any]) -> None:
    """Best effort: a Supabase outage must not fail the prediction itself."""
    try:
        db.table("predictions").insert({
            "user_id": user["id"],
            "input_features": features,
            "prediction_result": {k: res.get(k) for k in
                                  ("probability", "grade", "threshold", "flagged", "prediction", "model_version")},
        }).execute()
        logger.info(f"Logged prediction to Supabase for user: {user['email']}")
    except Exception as e:
        logger.error(f"Failed to log prediction to Supabase: {e}")


@router.get("/features", response_model=List[str])
def get_features():
    """Get list of features expected by the model."""
    features = model_service.get_feature_names()
    if not features:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model features not loaded yet."
        )
    return features


@router.get("/meta")
def get_meta():
    """Grades, threshold, adjustable ratios and stress shocks, so the UI never hard-codes policy."""
    return model_service.public_meta()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionInput,
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """Run model prediction with Redis caching and log the prediction to Supabase."""
    require_features(payload.features)
    cache_key = get_cache_key(payload.features, model_service.model_version)

    res = redis_cache.get(cache_key)
    if res:
        logger.info("Prediction cache HIT")
        cached = True
    else:
        logger.info("Prediction cache MISS. Running model inference...")
        try:
            res = model_service.predict(payload.features)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed.")
        redis_cache.set(cache_key, res, expire_seconds=86400)
        cached = False

    log_prediction(db, current_user, payload.features, res)
    return {**res, "cached": cached}


@router.post("/predict/batch")
def predict_batch(payload: BatchInput, current_user: dict = Depends(get_current_user)):
    """Score up to MAX_BATCH_ROWS firms. Not cached or logged per row (portfolio uploads would flood history)."""
    for row in payload.rows:
        require_features(row)
    return {"threshold": model_service.threshold, "model_version": model_service.model_version,
            "results": model_service.predict_batch(payload.rows)}


@router.post("/predict/counterfactual")
def predict_counterfactual(payload: CounterfactualInput, current_user: dict = Depends(get_current_user)):
    """Smallest realistic change to 1-3 adjustable ratios that brings the firm into the target grade."""
    require_features(payload.features)
    target = payload.target_grade or model_service.policy['counterfactual']['default_target_grade']
    try:
        return model_service.counterfactual(payload.features, target)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/predict/stress")
def predict_stress(payload: StressInput, current_user: dict = Depends(get_current_user)):
    """PD before and after applying shocks to the input ratios."""
    require_features(payload.features)
    try:
        return model_service.stress(payload.features, payload.shocks)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/predict/history", response_model=List[Dict[str, Any]])
def get_prediction_history(
    db: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user)
):
    """Retrieve authenticated user's prediction history from Supabase."""
    try:
        res = db.table("predictions").select("*").eq("user_id", current_user["id"]).order("created_at", desc=True).execute()
        return res.data
    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prediction history."
        )
