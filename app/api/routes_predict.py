import hashlib
import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from supabase import Client

from app.core.dependencies import get_supabase, get_current_user
from app.services.model_service import model_service
from app.cache.redis_cache import redis_cache
from app.utils.logger import get_logger

logger = get_logger("routes_predict")

router = APIRouter(prefix="", tags=["predictions"])

class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of features where keys match the dataset columns."
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

@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionInput,
    db: Client = Depends(get_supabase),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Run model prediction with Redis caching and optionally log prediction history to Supabase."""
    feature_names = model_service.get_feature_names()
    
    # Validate features
    missing = [f for f in feature_names if f not in payload.features]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required features: {missing}"
        )

    cache_key = get_cache_key(payload.features)
    
    # Try fetching from Redis cache
    cached_data = redis_cache.get(cache_key)
    if cached_data:
        logger.info("Prediction cache HIT")
        # Ensure we set cached=True
        cached_data["cached"] = True
        
        # If user is authenticated, still save the prediction to Supabase predictions table if not logged yet.
        if current_user:
            try:
                db.table("predictions").insert({
                    "user_id": current_user["id"],
                    "input_features": payload.features,
                    "prediction_result": {
                        "probability": cached_data["probability"],
                        "prediction": cached_data["prediction"],
                        "risk_tier": cached_data["risk_tier"],
                        "risk_emoji": cached_data["risk_emoji"],
                        "risk_color": cached_data["risk_color"],
                        "altman_z_score": cached_data.get("altman_z_score"),
                        "altman_zone": cached_data.get("altman_zone"),
                        "altman_emoji": cached_data.get("altman_emoji"),
                        "altman_color": cached_data.get("altman_color")
                    }
                }).execute()
            except Exception as e:
                logger.warning(f"Failed to log cached prediction to Supabase: {e}")
                
        return cached_data

    # Cache miss
    logger.info("Prediction cache MISS. Running model inference...")
    try:
        res = model_service.predict(payload.features)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

    # Set cache (expires in 24 hours)
    redis_cache.set(cache_key, res, expire_seconds=86400)
    
    # If user is authenticated, save the prediction history to Supabase Postgres
    if current_user:
        try:
            db.table("predictions").insert({
                "user_id": current_user["id"],
                "input_features": payload.features,
                "prediction_result": {
                    "probability": res["probability"],
                    "prediction": res["prediction"],
                    "risk_tier": res["risk_tier"],
                    "risk_emoji": res["risk_emoji"],
                    "risk_color": res["risk_color"],
                    "altman_z_score": res.get("altman_z_score"),
                    "altman_zone": res.get("altman_zone"),
                    "altman_emoji": res.get("altman_emoji"),
                    "altman_color": res.get("altman_color")
                }
            }).execute()
            logger.info(f"Log prediction to Supabase for user: {current_user['email']}")
        except Exception as e:
            logger.error(f"Failed to log prediction to Supabase: {e}")

    res["cached"] = False
    return res


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
