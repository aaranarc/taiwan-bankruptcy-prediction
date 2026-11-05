"""The UI's API client: binding URL handling, and that errors never expose the internal URL."""
import asyncio
import json

import pytest
import requests

from ui import api

SECRET_HOST = "http://api-internal.example.invalid:9999/svc"


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def test_binding_url_wins_and_keeps_path_prefix(monkeypatch):
    monkeypatch.setenv("API_INTERNAL_URL", SECRET_HOST)
    monkeypatch.setenv("BACKEND_URL", "http://ignored")
    seen = {}

    def fake(method, url, **kwargs):
        seen["url"] = url
        return Resp(200, {})

    monkeypatch.setattr(api.requests, "request", fake)
    api.meta()
    assert seen["url"] == SECRET_HOST + "/meta"
    assert api.provider() == "vercel-binding"


def test_falls_back_to_backend_url(monkeypatch):
    monkeypatch.delenv("API_INTERNAL_URL", raising=False)
    monkeypatch.setenv("BACKEND_URL", "http://localhost:8000")
    assert api.base_url() == "http://localhost:8000/"


def test_connection_error_message_hides_internal_url(monkeypatch):
    monkeypatch.setenv("API_INTERNAL_URL", SECRET_HOST)

    def boom(*a, **k):
        raise requests.ConnectionError(f"HTTPConnectionPool(host='{SECRET_HOST}'): refused")

    monkeypatch.setattr(api.requests, "request", boom)
    with pytest.raises(api.ApiError) as err:
        api.predict({}, "token")
    assert "example.invalid" not in str(err.value)
    assert err.value.__cause__ is None  # `from None`: the original exception is not chained either


def test_401_and_api_detail_are_mapped(monkeypatch):
    monkeypatch.setattr(api.requests, "request", lambda *a, **k: Resp(401, {"detail": "x"}))
    with pytest.raises(api.ApiError, match="sign in") as err:
        api.history(None)
    assert err.value.status == 401
    monkeypatch.setattr(api.requests, "request", lambda *a, **k: Resp(400, {"detail": "Missing required features"}))
    with pytest.raises(api.ApiError, match="Missing required features"):
        api.predict({}, "t")
    monkeypatch.setattr(api.requests, "request", lambda *a, **k: Resp(502, None))
    with pytest.raises(api.ApiError, match="502"):
        api.predict({}, "t")


def test_token_is_sent_as_bearer_header(monkeypatch):
    seen = {}

    def fake(method, url, headers=None, **kwargs):
        seen.update(headers)
        return Resp(200, {})

    monkeypatch.setattr(api.requests, "request", fake)
    api.predict({"a": 1.0}, "abc")
    assert seen == {"Authorization": "Bearer abc"}


def test_vercel_health_route_reveals_no_url(monkeypatch):
    monkeypatch.setenv("API_INTERNAL_URL", SECRET_HOST)
    import server
    body = json.loads(asyncio.run(server.health(None)).body)
    assert body["api_binding_configured"] is True
    assert "example.invalid" not in json.dumps(body)
