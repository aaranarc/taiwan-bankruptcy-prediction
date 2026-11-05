"""API contract: /predict, auth, history, batch, counterfactual, stress, /meta. Supabase and Redis are faked."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import routes_predict
from app.api.routes_predict import get_cache_key
from app.core.dependencies import get_current_user, get_supabase
from app.main import app

GRADES = {"A", "B", "C", "D", "E"}


class FakeDB:
    """Stands in for the Supabase client: records inserts; queries return `rows`."""

    def __init__(self, rows=None, fail=False):
        self.rows, self.fail, self.inserted = rows or [], fail, []
        self.data = None

    def table(self, name):
        return self

    def insert(self, payload):
        self.inserted.append(payload)
        return self

    def __getattr__(self, name):  # select/eq/order/... chain
        return lambda *args, **kwargs: self

    def execute(self):
        if self.fail:
            raise ConnectionError("supabase down")
        self.data = self.rows
        return self


class FakeCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, expire_seconds=0):
        self.store[key] = value
        return True


@pytest.fixture
def db():
    return FakeDB()


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr(routes_predict, "redis_cache", FakeCache())
    app.dependency_overrides[get_supabase] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


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


def test_predict_returns_pd_grade_threshold_and_shap(client, raw_features):
    login()
    res = client.post("/predict", json={"features": raw_features})
    assert res.status_code == 200, res.text
    body = res.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["grade"] in GRADES
    assert 0.0 < body["threshold"] < 1.0
    assert body["flagged"] == (body["probability"] >= body["threshold"])
    assert len(body["top_factors"]) == 10
    assert body["cached"] is False


def test_shap_waterfall_adds_up_to_model_log_odds(raw_features):
    """base + top factors + other == the base XGBoost margin, so the UI waterfall is exact."""
    from app.services.model_service import model_service
    res = model_service.predict(raw_features)
    total = res["shap_base_value"] + sum(f["shap"] for f in res["top_factors"]) + res["shap_other"]
    df = model_service.preprocess_input(pd.DataFrame([raw_features]))
    margin = model_service.base_estimator.predict(df, output_margin=True)[0]
    assert total == pytest.approx(float(margin), abs=1e-3)


def test_second_identical_request_is_served_from_cache_and_still_logged(client, db, raw_features):
    login()
    first = client.post("/predict", json={"features": raw_features}).json()
    second = client.post("/predict", json={"features": raw_features}).json()
    assert second["cached"] is True
    assert second["probability"] == first["probability"]
    assert len(db.inserted) == 2


def test_prediction_is_logged_to_supabase(client, db, raw_features):
    login("user-42")
    client.post("/predict", json={"features": raw_features})
    row = db.inserted[0]
    assert row["user_id"] == "user-42"
    assert set(row["prediction_result"]) >= {"probability", "grade", "threshold", "flagged"}


def test_supabase_outage_does_not_fail_prediction(raw_features, monkeypatch):
    monkeypatch.setattr(routes_predict, "redis_cache", FakeCache())
    app.dependency_overrides[get_supabase] = lambda: FakeDB(fail=True)
    login()
    try:
        assert TestClient(app).post("/predict", json={"features": raw_features}).status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_redis_disabled_behaves_as_no_cache():
    """The real RedisCache with no REDIS_URL must behave as 'no cache', not error."""
    from app.cache.redis_cache import redis_cache
    assert redis_cache.client is None
    assert redis_cache.get("k") is None and redis_cache.set("k", 1) is False


def test_predict_without_token_is_401(client, raw_features):
    assert client.post("/predict", json={"features": raw_features}).status_code == 401


@pytest.mark.parametrize("path", ["/predict/counterfactual", "/predict/stress", "/predict/batch"])
def test_other_endpoints_require_auth(client, raw_features, path):
    body = {"features": raw_features, "shocks": {}, "rows": [raw_features]}
    assert client.post(path, json=body).status_code == 401


def test_history_requires_auth(client):
    assert client.get("/predict/history").status_code == 401


def test_predict_missing_features_is_400(client):
    login()
    assert client.post("/predict", json={"features": {"Current Ratio": 1.5}}).status_code == 400


def test_predict_non_finite_value_is_rejected(client, raw_features):
    login()
    first = next(iter(raw_features))
    body = '{"features": {' + ", ".join(
        f'"{k}": {"NaN" if k == first else v}' for k, v in raw_features.items()) + "}}"
    res = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert res.status_code in (400, 422)


def test_history_returns_users_rows():
    rows = [{"id": "p1", "created_at": "2026-01-01T00:00:00Z", "prediction_result": {"probability": 0.1}}]
    app.dependency_overrides[get_supabase] = lambda: FakeDB(rows=rows)
    login()
    try:
        res = TestClient(app).get("/predict/history")
    finally:
        app.dependency_overrides.clear()
    assert res.status_code == 200 and res.json() == rows


def test_batch_scores_every_row(client, dataset):
    login()
    rows = [{k: float(v) for k, v in r.items()}
            for _, r in dataset.drop(columns=["Bankrupt?"]).head(5).iterrows()]
    res = client.post("/predict/batch", json={"rows": rows})
    assert res.status_code == 200, res.text
    results = res.json()["results"]
    assert len(results) == 5 and all(r["grade"] in GRADES for r in results)


def test_batch_rejects_too_many_rows(client, raw_features):
    login()
    rows = [raw_features] * (routes_predict.MAX_BATCH_ROWS + 1)
    assert client.post("/predict/batch", json={"rows": rows}).status_code == 422


def test_counterfactual_endpoint(client, risky_features):
    login()
    res = client.post("/predict/counterfactual", json={"features": risky_features, "target_grade": "D"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["target_grade"] == "D" and "changes" in body


def test_counterfactual_unknown_grade_is_400(client, risky_features):
    login()
    res = client.post("/predict/counterfactual", json={"features": risky_features, "target_grade": "Z"})
    assert res.status_code == 400


def test_stress_endpoint_and_validation(client, raw_features):
    login()
    ok = client.post("/predict/stress", json={"features": raw_features, "shocks": {"debt_change_pct": 50}})
    assert ok.status_code == 200
    assert ok.json()["pd_after"] >= ok.json()["pd_before"]  # more debt can never lower PD (monotone)
    unknown = {"features": raw_features, "shocks": {"made_up": 1}}
    too_big = {"features": raw_features, "shocks": {"debt_change_pct": 1000}}
    assert client.post("/predict/stress", json=unknown).status_code == 400
    assert client.post("/predict/stress", json=too_big).status_code == 400


def test_meta_is_public_and_has_policy(client):
    body = client.get("/meta").json()
    assert {g["grade"] for g in body["grades"]} == GRADES
    assert set(body["stress_shocks"]) == {"revenue_change_pct", "interest_rate_change_pct", "debt_change_pct",
                                          "profitability_shift_pctile"}
