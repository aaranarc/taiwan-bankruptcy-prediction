"""Client for the FastAPI service. Runs inside the Streamlit *server* process, never in the browser.

On Vercel the API has no public route: the UI reaches it through the private service binding, which injects
API_INTERNAL_URL into this process only. Locally / on Render, BACKEND_URL is used instead.
Error messages shown to users never include the URL (a raw requests exception would contain the host).
"""
import os
from urllib.parse import urljoin

import requests

TIMEOUT = 60  # seconds; a scaled-to-zero container needs a few seconds to start


class ApiError(Exception):
    """An error whose message is safe to show in the UI."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def base_url() -> str:
    url = os.getenv("API_INTERNAL_URL") or os.getenv("BACKEND_URL") or "http://localhost:8000"
    return url if url.endswith("/") else url + "/"  # urljoin keeps any path prefix only with a trailing slash


def provider() -> str:
    return "vercel-binding" if os.getenv("API_INTERNAL_URL") else "backend-url"


def _request(method: str, path: str, token: str | None = None, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        res = requests.request(method, urljoin(base_url(), path), headers=headers, timeout=TIMEOUT, **kwargs)
    except requests.RequestException:
        raise ApiError("The scoring service is unreachable or still starting. Try again in a few seconds.") from None
    if res.status_code == 401:
        raise ApiError("Please sign in to use this feature.", 401)
    if not res.ok:
        try:
            detail = res.json().get("detail")
        except ValueError:
            detail = None
        message = detail if isinstance(detail, str) else f"Request failed ({res.status_code})."
        raise ApiError(message, res.status_code)
    return res.json()


def login(email: str, password: str) -> dict:
    return _request("POST", "auth/login", data={"username": email, "password": password})


def signup(email: str, password: str) -> dict:
    return _request("POST", "auth/signup", json={"email": email, "password": password})


def meta() -> dict:
    return _request("GET", "meta")


def features() -> list:
    return _request("GET", "features")


def predict(features: dict, token: str | None) -> dict:
    return _request("POST", "predict", token, json={"features": features})


def predict_batch(rows: list, token: str | None) -> dict:
    return _request("POST", "predict/batch", token, json={"rows": rows})


def counterfactual(features: dict, target_grade: str, token: str | None) -> dict:
    return _request("POST", "predict/counterfactual", token, json={"features": features, "target_grade": target_grade})


def stress(features: dict, shocks: dict, token: str | None) -> dict:
    return _request("POST", "predict/stress", token, json={"features": features, "shocks": shocks})


def history(token: str | None) -> list:
    return _request("GET", "predict/history", token)
