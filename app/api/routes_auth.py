from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from supabase import Client
from app.core.dependencies import get_supabase
from app.core.security import hash_password, verify_password, create_access_token
from app.core.exceptions import EntityAlreadyExistsException, CredentialsException, DatabaseException
from app.utils.logger import get_logger

logger = get_logger("routes_auth")

router = APIRouter(prefix="/auth", tags=["auth"])

class UserRegister(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    email: str

@router.post("/signup", response_model=TokenResponse)
def signup(payload: UserRegister, db: Client = Depends(get_supabase)):
    email = payload.email.lower().strip()
    
    # Check if user already exists
    try:
        existing = db.table("users").select("*").eq("email", email).execute()
        if existing.data and len(existing.data) > 0:
            raise EntityAlreadyExistsException("A user with this email already exists.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Supabase check user error: {e}")
        raise DatabaseException("Database lookup failed.")

    # Hash the password
    pwd_hash = hash_password(payload.password)
    
    # Insert new user
    try:
        insert_res = db.table("users").insert({
            "email": email,
            "password_hash": pwd_hash
        }).execute()
        
        if not insert_res.data or len(insert_res.data) == 0:
            raise DatabaseException("Failed to register user.")
    except Exception as e:
        logger.error(f"Supabase insert user error: {e}")
        raise DatabaseException("Database insertion failed.")

    # Create JWT token
    token = create_access_token(subject=email)
    return TokenResponse(access_token=token, token_type="bearer", email=email)


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Client = Depends(get_supabase)):
    email = form_data.username.lower().strip()
    password = form_data.password
    
    try:
        res = db.table("users").select("*").eq("email", email).execute()
        if not res.data or len(res.data) == 0:
            raise CredentialsException("Incorrect email or password")
            
        user = res.data[0]
        if not verify_password(password, user["password_hash"]):
            raise CredentialsException("Incorrect email or password")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Supabase user lookup error: {e}")
        raise CredentialsException("Authentication process failed")
        
    token = create_access_token(subject=email)
    return TokenResponse(access_token=token, token_type="bearer", email=email)
