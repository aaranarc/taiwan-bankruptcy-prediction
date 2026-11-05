"""Preprocessing, monotone behaviour of the trained model, threshold and grade logic."""
import json

import numpy as np
import pandas as pd
import pytest

from app.services.model_service import model_service
from app.services.risk import choose_threshold, expected_cost, grade_for, monotone_directions
from training.train_utils import ROOT, apply_winsorization, get_extreme_columns

GRADES = [{"grade": "A", "max_pd": 0.01}, {"grade": "B", "max_pd": 0.03}, {"grade": "C", "max_pd": 0.10},
          {"grade": "D", "max_pd": 0.30}, {"grade": "E", "max_pd": 1.0}]


# --- preprocessing ---

def test_winsor_bounds_come_from_train_only():
    train = pd.DataFrame({"a": np.arange(1000, dtype=float) * 1000})  # max > 100 -> "extreme"
    test = pd.DataFrame({"a": [-1e12, 5e5, 1e12]})
    tr, te, bounds = apply_winsorization(train, test, get_extreme_columns(train))
    lo, hi = bounds["a"]
    assert (lo, hi) == (train["a"].quantile(0.01), train["a"].quantile(0.99))
    assert te["a"].tolist() == [lo, 5e5, hi]  # test outliers clipped to *train* bounds
    assert test["a"].iloc[0] == -1e12  # inputs are not mutated


def test_columns_without_extremes_are_not_winsorised():
    df = pd.DataFrame({"ratio": np.linspace(0, 1, 50)})
    assert get_extreme_columns(df) == []


def test_preprocess_clips_then_scales_like_training(raw_features):
    col, (lo, hi) = next(iter(model_service.artifacts["bounds"].items()))
    out = model_service.preprocess_input(pd.DataFrame([{**raw_features, col: hi * 1e6}]))
    capped = model_service.preprocess_input(pd.DataFrame([{**raw_features, col: hi}]))
    assert out[col].iloc[0] == pytest.approx(capped[col].iloc[0])  # extreme value capped at the bound
    assert list(out.columns) == model_service.get_feature_names()


def test_preprocess_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing features"):
        model_service.preprocess_input(pd.DataFrame([{"Current Ratio": 0.1}]))


# --- monotone constraints on the trained, calibrated model ---

@pytest.mark.parametrize("feature,direction", sorted(monotone_directions().items()))
def test_pd_is_monotone_in_constrained_feature(dataset, feature, direction):
    """Sweep one constrained feature over its observed range for several real firms; PD must never move
    against the configured direction (calibration is order-preserving, so this holds end to end)."""
    firms = dataset.drop(columns=["Bankrupt?"]).sample(5, random_state=0)
    grid = np.quantile(dataset[feature], np.linspace(0, 1, 25))
    for _, firm in firms.iterrows():
        rows = pd.DataFrame([firm] * len(grid))
        rows[feature] = grid
        steps = np.diff(model_service.score(rows)) * direction
        assert (steps >= -1e-9).all(), f"{feature}: PD moved against direction {direction}"


# --- threshold and grades ---

def test_expected_cost_counts_misses_at_cost_ratio():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1])
    # threshold 0.5: one miss (cost 10) + one false alarm (cost 1) over 4 firms
    assert expected_cost(y, p, 0.5, cost_ratio=10) == pytest.approx(11 / 4)


def test_threshold_minimises_cost_and_moves_with_cost_ratio():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)  # perfectly calibrated toy scores
    t10 = choose_threshold(y, p, cost_ratio=10)
    t1 = choose_threshold(y, p, cost_ratio=1)
    # With calibrated PDs the cost-optimal threshold is 1/(1+ratio): ~0.09 for 10:1, ~0.5 for 1:1.
    assert t10 == pytest.approx(1 / 11, abs=0.03)
    assert t1 == pytest.approx(0.5, abs=0.05)
    assert expected_cost(y, p, t10, 10) <= expected_cost(y, p, 0.5, 10)


@pytest.mark.parametrize("pd_value,grade", [(0.0, "A"), (0.0099, "A"), (0.01, "B"), (0.05, "C"),
                                            (0.10, "D"), (0.29, "D"), (0.30, "E"), (1.0, "E")])
def test_grade_boundaries(pd_value, grade):
    assert grade_for(pd_value, GRADES) == grade


def test_api_threshold_comes_from_cv_report():
    with open(f"{ROOT}/reports/metrics.json") as fh:
        assert model_service.threshold == json.load(fh)["threshold"]["chosen"]
