"""Auth routes and the JWT -> user dependency against a fake Supabase; Redis wrapper failure modes."""
import pytest
from fastapi.testclient import TestClient

from app.cache import redis_cache as redis_module
from app.core.dependencies import get_supabase
from app.core.security import create_access_token, hash_password
from app.main import app


class UsersDB:
    """Fake Supabase 'users' table supporting the select/eq/insert chains the routes use."""

    def __init__(self, users=None):
        self.users = users or []
        self._email, self._insert = None, None

    def table(self, name):
        return self

    def select(self, *args, **kwargs):
        return self

    def eq(self, column, value):
        self._email = value
        return self

    def order(self, *args, **kwargs):
        return self

    def insert(self, row):
        self._insert = row
        return self

    def execute(self):
        if self._insert is not None:
            row, self._insert = {"id": f"id-{len(self.users)}", **self._insert}, None
            self.users.append(row)
            return type("R", (), {"data": [row]})
        return type("R", (), {"data": [u for u in self.users if u["email"] == self._email]})


@pytest.fixture
def users():
    return UsersDB([{"id": "u1", "email": "a@example.com", "password_hash": hash_password("right-pass")}])


@pytest.fixture
def client(users):
    app.dependency_overrides[get_supabase] = lambda: users
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_signup_returns_token_and_stores_hash(client, users):
    res = client.post("/auth/signup", json={"email": "New@Example.com", "password": "pw-123456"})
    assert res.status_code == 200 and res.json()["email"] == "new@example.com"
    stored = users.users[-1]
    assert stored["password_hash"] != "pw-123456"


def test_signup_duplicate_email_is_400(client):
    assert client.post("/auth/signup", json={"email": "a@example.com", "password": "x"}).status_code == 400


def test_login_success_and_wrong_password(client):
    ok = client.post("/auth/login", data={"username": "a@example.com", "password": "right-pass"})
    assert ok.status_code == 200 and ok.json()["access_token"]
    bad = client.post("/auth/login", data={"username": "a@example.com", "password": "wrong"})
    assert bad.status_code == 401


def test_valid_token_reaches_protected_route(client):
    token = create_access_token("a@example.com")
    res = client.get("/predict/history", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200


def test_token_for_unknown_user_is_401(client):
    token = create_access_token("ghost@example.com")
    assert client.get("/predict/history", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_garbage_token_is_401(client):
    assert client.get("/predict/history", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401


# --- Redis wrapper: managed Redis down must mean "no cache", never an error ---

class BrokenRedis:
    def ping(self):
        raise ConnectionError("down")

    def get(self, key):
        raise ConnectionError("down")

    def setex(self, *args):
        raise ConnectionError("down")


def test_unreachable_redis_at_startup_disables_cache(monkeypatch):
    monkeypatch.setattr(redis_module.settings, "REDIS_URL", "rediss://user:secret@example.invalid:6379")
    monkeypatch.setattr(redis_module.redis.Redis, "from_url", staticmethod(lambda *a, **k: BrokenRedis()))
    cache = redis_module.RedisCache()
    assert cache.client is None and cache.get("k") is None and cache.set("k", {"a": 1}) is False


def test_redis_failing_mid_request_returns_miss():
    cache = redis_module.RedisCache.__new__(redis_module.RedisCache)
    cache.client = BrokenRedis()
    assert cache.get("k") is None
    assert cache.set("k", {"a": 1}) is False
    assert cache.is_connected() is False


def test_redis_round_trip_with_working_client():
    class MemoryRedis:
        def __init__(self):
            self.d = {}

        def get(self, k):
            return self.d.get(k)

        def setex(self, k, ttl, v):
            self.d[k] = v

        def ping(self):
            return True

    cache = redis_module.RedisCache.__new__(redis_module.RedisCache)
    cache.client = MemoryRedis()
    assert cache.set("k", {"pd": 0.1}) is True and cache.get("k") == {"pd": 0.1} and cache.is_connected()


def test_redis_failure_log_does_not_reveal_host(monkeypatch, caplog):
    """A managed Redis URL names the host (and may embed a password); neither may reach the logs."""
    monkeypatch.setattr(redis_module.settings, "REDIS_URL", "rediss://user:pw@secret-host.example.invalid:6379")

    def refuse(*a, **k):
        raise ConnectionError("Error -2 connecting to secret-host.example.invalid:6379")

    monkeypatch.setattr(redis_module.redis.Redis, "from_url", staticmethod(refuse))
    with caplog.at_level("WARNING"):
        redis_module.RedisCache()
    assert "secret-host" not in caplog.text and "pw" not in caplog.text
