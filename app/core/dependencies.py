from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import create_client, Client
from app.core.config import settings
from app.core.security import decode_access_token
from app.core.exceptions import CredentialsException
from app.utils.logger import get_logger

logger = get_logger("dependencies")

# Create a singleton Supabase client
supabase_client: Client = None
if settings.SUPABASE_URL and settings.SUPABASE_KEY:
    try:
        supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY not configured. DB integrations will fail.")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

def get_supabase() -> Client:
    if not supabase_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase client is not configured."
        )
    return supabase_client

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Client = Depends(get_supabase)
) -> dict:
    if not token:
        raise CredentialsException("Not authenticated")
    
    email = decode_access_token(token)
    if not email:
        raise CredentialsException("Invalid token or token expired")
        
    try:
        res = db.table("users").select("*").eq("email", email).execute()
        if not res.data or len(res.data) == 0:
            raise CredentialsException("User not found")
        return res.data[0]
    except Exception as e:
        logger.error(f"Error fetching user in dependencies: {e}")
        raise CredentialsException("Could not validate credentials")
