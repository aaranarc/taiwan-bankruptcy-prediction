"""Counterfactual bounds and stress-test mechanics, on the real model."""
import pytest

from app.services.model_service import model_service
from app.services.risk import grade_max_pd

POLICY = model_service.policy
ADJUSTABLE = set(POLICY["counterfactual"]["adjustable_features"])
LINKED = set(POLICY["linked_features"])


@pytest.fixture(scope="module")
def reachable_case(dataset):
    """A real firm (grade C or worse) for which a path one grade up exists, so the bound checks below
    test actual changes instead of passing on an empty list."""
    order = [g["grade"] for g in model_service.grades]
    firms = dataset.drop(columns=["Bankrupt?"])
    pds = model_service.score(firms.head(600))
    for i in pds.argsort()[::-1]:
        feats = {k: float(v) for k, v in firms.iloc[int(i)].items()}
        grade = model_service.predict(feats)["grade"]
        if grade in ("A", "B"):
            break
        target = order[order.index(grade) - 1]
        res = model_service.counterfactual(feats, target)
        if res["reachable"] and res["changes"]:
            return feats, target, res
    pytest.fail("no firm among the first 600 has a reachable path one grade up")


def test_counterfactual_only_moves_adjustable_ratios_within_training_range(reachable_case):
    _, target, res = reachable_case
    assert res["pd_before"] >= grade_max_pd(target, model_service.grades)  # started above the target grade
    assert 1 <= len(res["changes"]) <= POLICY["counterfactual"]["max_features_changed"]
    q = model_service.artifacts["quantiles"]
    for ch in res["changes"]:
        f = ch["feature"]
        assert f in ADJUSTABLE
        assert q[f].loc[1] - 1e-12 <= ch["to_value"] <= q[f].loc[99] + 1e-12  # 1st-99th percentile
        safer = ch["to_value"] - ch["from_value"]
        assert safer * model_service.directions[f] < 0  # moved in the risk-reducing direction
    assert res["pd_after"] < grade_max_pd(target, model_service.grades)


def test_counterfactual_result_reproduces_when_rescored(reachable_case):
    feats, _, res = reachable_case
    changed = dict(feats)
    for ch in res["changes"]:
        changed[ch["feature"]] = ch["to_value"]
    if "Debt ratio %" in {c["feature"] for c in res["changes"]}:
        changed["Net worth/Assets"] = 1 - changed["Debt ratio %"]  # linked ratio stays consistent
    assert model_service.predict(changed)["probability"] == pytest.approx(res["pd_after"])


def test_severe_risk_firm_can_have_no_three_ratio_path(risky_features):
    """Known behaviour of the nested-tuned model: for grade-E firms, three adjustable ratios within the training
    range are usually not enough to reach grade D. The API must then say so cleanly, not invent a path."""
    res = model_service.counterfactual(risky_features, "D")
    if not res["reachable"]:
        assert res["changes"] == [] and res["pd_after"] is None and res["grade_after"] is None
    else:  # if a future model can reach D, the path must still respect the limits
        assert len(res["changes"]) <= POLICY["counterfactual"]["max_features_changed"]


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


def test_profitability_shock_stops_at_first_percentile(dataset):
    """A firm already near the bottom must not be pushed to the single most extreme value in the data."""
    bankrupt = dataset[dataset["Bankrupt?"] == 1].drop(columns=["Bankrupt?"]).median()
    firm = {k: float(v) for k, v in bankrupt.items()}
    res = model_service.stress(firm, {"profitability_shift_pctile": -50})
    q = model_service.artifacts["quantiles"]
    for ch in res["changed_features"]:
        assert ch["after"] >= q[ch["feature"]].loc[1] - 1e-12 or ch["after"] == ch["before"]
