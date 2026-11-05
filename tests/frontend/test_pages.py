"""Streamlit AppTest for the shell and each page. ui.api is replaced by an in-process fake that calls the real
model service, so pages render real model output without a server, Supabase or Redis."""
import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from ui import api

TIMEOUT = 60
ROOT = Path(__file__).resolve().parents[2]  # AppTest resolves relative paths against this test file


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    from app.services.model_service import model_service as ms

    calls = {"history": []}
    monkeypatch.setattr(api, "meta", ms.public_meta)
    monkeypatch.setattr(api, "features", ms.get_feature_names)
    monkeypatch.setattr(api, "predict", lambda f, t: {**ms.predict(f), "cached": False})
    monkeypatch.setattr(api, "counterfactual", lambda f, g, t: ms.counterfactual(f, g))
    monkeypatch.setattr(api, "stress", lambda f, s, t: ms.stress(f, s))
    monkeypatch.setattr(api, "predict_batch", lambda rows, t: {"results": ms.predict_batch(rows)})
    monkeypatch.setattr(api, "history", lambda t: calls["history"])
    monkeypatch.setattr(api, "login", lambda e, p: {"access_token": "tok", "email": e})
    st.cache_data.clear()
    return calls


def page(path: str, signed_in: bool = True) -> AppTest:
    at = AppTest.from_file(str(ROOT / path), default_timeout=TIMEOUT)
    if signed_in:
        at.session_state["token"], at.session_state["email"] = "tok", "analyst@example.com"
    return at.run()


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def html_text(at: AppTest) -> str:
    return " ".join(str(el.proto) for el in at.get("html"))


def test_shell_renders_navigation_and_sign_in():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any(b.label == "Sign in" for b in at.sidebar.button)  # form submit button
    assert any(b.label == "Score company" for b in at.button)  # default page is Score


def test_sign_in_stores_token_in_session():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    at.sidebar.text_input(key="si_email").input("analyst@example.com")
    at.sidebar.text_input(key="si_password").input("secret-pass")
    next(b for b in at.sidebar.button if b.label == "Sign in").click().run()
    assert at.session_state["token"] == "tok"


def test_score_page_empty_then_scores_distressed_preset():
    at = page("ui/pages/score.py")
    assert not at.exception and "No score yet" in html_text(at)
    button(at, "Distressed firm").click().run()
    button(at, "Score company").click().run()
    assert not at.exception and not at.error
    res = at.session_state["result"]
    assert res["grade"] in "ABCDE" and 0 <= res["probability"] <= 1
    assert len(at.get("plotly_chart")) == 2  # gauge + SHAP waterfall
    assert "Path to safety" in html_text(at)


def test_score_page_requires_sign_in():
    at = page("ui/pages/score.py", signed_in=False)
    button(at, "Score company").click().run()
    assert any("Sign in" in i.value for i in at.info)
    assert at.session_state["result"] is None


def test_score_page_shows_api_error(monkeypatch):
    def down(*a):
        raise api.ApiError("The scoring service is unreachable or still starting. Try again in a few seconds.")

    monkeypatch.setattr(api, "predict", down)
    at = page("ui/pages/score.py")
    button(at, "Score company").click().run()
    assert any("unreachable" in e.value for e in at.error)


def test_portfolio_sample_is_scored():
    at = page("ui/pages/portfolio.py")
    assert "No portfolio scored yet" in html_text(at)
    button(at, "Use the sample portfolio").click().run()
    assert not at.exception and not at.error
    assert len(at.session_state["portfolio"]) == 150
    assert len(at.dataframe) == 2  # top-10 + all firms


def test_stress_page_runs_default_shocks():
    at = page("ui/pages/stress.py")
    assert len(at.slider) == 4
    button(at, "Run stress test").click().run()
    assert not at.exception and not at.error
    res = at.session_state["stress"]
    assert {"pd_before", "pd_after", "changed_features"} <= set(res)


def test_model_card_reads_report():
    at = page("ui/pages/model_card.py", signed_in=False)
    assert not at.exception
    with open(ROOT / "reports" / "metrics.json") as fh:
        auc = json.load(fh)["deployed_model"]["roc_auc"]["mean"]
    assert f"{auc:.3f}" in html_text(at)  # card value comes straight from the report
    assert len(at.get("plotly_chart")) == 2  # calibration + grades


def test_history_handles_old_and_new_rows(fake_api):
    fake_api["history"] += [
        {"created_at": "2026-11-01T10:00:00Z", "prediction_result": {"probability": 0.4, "grade": "E",
                                                                     "flagged": True, "model_version": "abc"}},
        {"created_at": "2026-10-01T10:00:00Z", "prediction_result": {"probability": 0.1, "risk_tier": "Low Risk"}},
    ]
    at = page("ui/pages/history.py")
    assert not at.exception and not at.error
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 2


def test_history_empty_and_signed_out():
    assert "Nothing scored yet" in html_text(page("ui/pages/history.py"))
    at = page("ui/pages/history.py", signed_in=False)
    assert any("Sign in" in i.value for i in at.info)
