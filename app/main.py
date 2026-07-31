from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client

from app.core.config import settings
from app.core.dependencies import get_supabase
from app.api.routes_auth import router as auth_router
from app.api.routes_predict import router as predict_router
from app.middleware.logging_middleware import LoggingMiddleware
from app.cache.redis_cache import redis_cache
from app.utils.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Taiwan Bankruptcy Prediction API with Supabase and Redis integration.",
    version=settings.VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
app.add_middleware(LoggingMiddleware)

# Include routes
app.include_router(auth_router)
app.include_router(predict_router)

@app.get("/health")
def health_check(db: Client = Depends(get_supabase)):
    """Check application health, Redis, and Supabase status."""
    redis_ok = redis_cache.is_connected()
    
    supabase_ok = False
    try:
        # Perform simple query to verify Supabase DB connection
        db.table("users").select("count", count="exact").limit(1).execute()
        supabase_ok = True
    except Exception as e:
        logger.error(f"Supabase connection check failed: {e}")
        supabase_ok = False
        
    overall_status = "healthy" if supabase_ok else "degraded"
    
    return {
        "status": overall_status,
        "redis_connected": redis_ok,
        "supabase_connected": supabase_ok,
    }

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME}",
        "version": settings.VERSION
    }
