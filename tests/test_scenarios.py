"""Counterfactual bounds and stress-test mechanics, on the real model."""
import pytest

from app.services.model_service import model_service
from app.services.risk import grade_max_pd

POLICY = model_service.policy
ADJUSTABLE = set(POLICY["counterfactual"]["adjustable_features"])
LINKED = set(POLICY["linked_features"])


@pytest.mark.parametrize("target", ["C", "D"])
def test_counterfactual_only_moves_adjustable_ratios_within_training_range(risky_features, target):
    res = model_service.counterfactual(risky_features, target)
    assert res["pd_before"] >= grade_max_pd(target, model_service.grades)  # fixture firm starts above target
    assert 1 <= len(res["changes"]) <= POLICY["counterfactual"]["max_features_changed"] or not res["reachable"]
    q = model_service.artifacts["quantiles"]
    for ch in res["changes"]:
        f = ch["feature"]
        assert f in ADJUSTABLE
        assert q[f].loc[1] - 1e-12 <= ch["to_value"] <= q[f].loc[99] + 1e-12  # 1st-99th percentile
        safer = ch["to_value"] - ch["from_value"]
        assert safer * model_service.directions[f] < 0  # moved in the risk-reducing direction
    if res["reachable"]:
        assert res["pd_after"] < grade_max_pd(target, model_service.grades)


def test_counterfactual_result_reproduces_when_rescored(risky_features):
    res = model_service.counterfactual(risky_features, "D")
    assert res["reachable"], "the riskiest fixture firm should be able to reach grade D"
    changed = dict(risky_features)
    for ch in res["changes"]:
        changed[ch["feature"]] = ch["to_value"]
    if "Debt ratio %" in {c["feature"] for c in res["changes"]}:
        changed["Net worth/Assets"] = 1 - changed["Debt ratio %"]  # linked ratio stays consistent
    assert model_service.predict(changed)["probability"] == pytest.approx(res["pd_after"])


def test_counterfactual_for_safe_firm_needs_no_change(dataset):
    healthy = dataset[dataset["Bankrupt?"] == 0].drop(columns=["Bankrupt?"])
    pds = model_service.score(healthy.head(200))
    firm = {k: float(v) for k, v in healthy.head(200).iloc[int(pds.argmin())].items()}
    res = model_service.counterfactual(firm, "C")
    assert res["already_at_target"] and res["changes"] == []


def test_stress_touches_only_configured_columns(raw_features):
    shocks = {"revenue_change_pct": -20, "interest_rate_change_pct": 50}
    res = model_service.stress(raw_features, shocks)
    allowed = {c for name in shocks for c in POLICY["stress_shocks"][name]["columns"]} | LINKED
    assert {c["feature"] for c in res["changed_features"]} <= allowed


def test_stress_keeps_linked_ratio_consistent_and_within_range(raw_features):
    res = model_service.stress(raw_features, {"debt_change_pct": 100})
    after = {c["feature"]: c["after"] for c in res["changed_features"]}
    assert after["Net worth/Assets"] == pytest.approx(1 - after["Debt ratio %"])
    q = model_service.artifacts["quantiles"]
    for f, v in after.items():
        assert q[f].iloc[0] <= v <= q[f].iloc[-1]


def test_profitability_shock_moves_down_by_percentiles(raw_features):
    res = model_service.stress(raw_features, {"profitability_shift_pctile": -30})
    for ch in res["changed_features"]:
        assert ch["after"] <= ch["before"]
    assert res["pd_after"] >= res["pd_before"]  # all four columns are monotone-constrained (-1)


def test_zero_shocks_change_nothing(raw_features):
    res = model_service.stress(raw_features, {"revenue_change_pct": 0, "debt_change_pct": 0})
    assert res["changed_features"] == [] and res["pd_after"] == pytest.approx(res["pd_before"])
