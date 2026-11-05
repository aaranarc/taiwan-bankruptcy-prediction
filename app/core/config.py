import os
from dotenv import load_dotenv

# Load environmental variables from .env file
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Taiwan Bankruptcy Prediction API"
    VERSION: str = "2.0.0"
    
    # Supabase config
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # Redis config
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Security config
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours for convenience

    # CORS: comma-separated browser origins allowed to call the API
    ALLOWED_ORIGINS: list = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()]

settings = Settings()

if not settings.JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is not set. Add it to .env or the environment before starting the API.")
