import os

# Set before any app import: no real Supabase, Redis unreachable (caching off), test-only JWT secret.
# load_dotenv() never overrides variables that are already set, so a local .env is ignored here.
os.environ["JWT_SECRET_KEY"] = "test-only-secret-at-least-32-bytes-long"
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_KEY"] = ""
os.environ["REDIS_URL"] = "redis://127.0.0.1:1"
