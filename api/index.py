"""Vercel Python function entrypoint (vercel.json -> services.web.entrypoint = "api.index:app")."""
from server import app  # noqa: F401  (Vercel imports this ASGI object)
