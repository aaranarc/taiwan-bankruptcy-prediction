import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.routes_predict import get_cache_key
from app.core.dependencies import get_current_user, get_supabase
from app.main import app

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "data.csv")
TIERS = {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}


class FakeDB:
    """Stands in for the Supabase client: every query chain succeeds and returns no rows."""

    data = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self

    def execute(self):
        return self


@pytest.fixture
def client():
    app.dependency_overrides[get_supabase] = lambda: FakeDB()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def raw_features():
    df = pd.read_csv(DATA, nrows=1)
    df.columns = df.columns.str.strip()
    return {k: float(v) for k, v in df.drop(columns=["Bankrupt?"]).iloc[0].items()}


def login(user_id="u1"):
    app.dependency_overrides[get_current_user] = lambda: {"id": user_id, "email": "test@example.com"}


def test_cache_key_ignores_feature_order():
    a = {"x": 1.0, "y": 2.0}
    b = {"y": 2.0, "x": 1.0}
    assert get_cache_key(a, "v1") == get_cache_key(b, "v1")


def test_cache_key_changes_with_model_version():
    f = {"x": 1.0}
    assert get_cache_key(f, "v1") != get_cache_key(f, "v2")
    assert get_cache_key(f, "v1").startswith("predict:v1:")


def test_predict_returns_probability_tier_and_top_factors(client, raw_features):
    login()
    res = client.post("/predict", json={"features": raw_features})
    assert res.status_code == 200, res.text
    body = res.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_tier"] in TIERS
    assert len(body["top_factors"]) == 5
    assert all(f["direction"] in ("raises risk", "lowers risk") for f in body["top_factors"])


def test_predict_without_token_is_401(client, raw_features):
    assert client.post("/predict", json={"features": raw_features}).status_code == 401


def test_predict_missing_features_is_400(client):
    login()
    assert client.post("/predict", json={"features": {"Current Ratio": 1.5}}).status_code == 400
