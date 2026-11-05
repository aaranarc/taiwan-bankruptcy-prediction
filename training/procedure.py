"""The complete model-building procedure, as one function.

fit_procedure(x, y) does everything that involves choosing something from data:
  1. XGBoost hyperparameter search (inner 3-fold CV, monotone constraints on, PR-AUC as the score)
  2. calibration method: isotonic only if its paired Brier gain over sigmoid beats the fold-to-fold noise
  3. cost-based decision threshold, chosen on out-of-fold calibrated PDs
  4. final calibrated model refit on all of x
Nested CV (training/nested_cv.py) runs it inside every outer training fold; training/train_model.py runs it once on
all data. So the deployed model is produced by exactly the procedure the nested CV evaluates.

Inputs are already preprocessed (winsorised + RobustScaler). Trees are invariant to order-preserving transforms, so
fitting that preprocessing once per outer fold instead of once per inner fold changes nothing except the clip bounds.
"""
import json
from itertools import product

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from app.services.risk import choose_threshold

SEED = 42
# The previously deployed (notebook) setting is one point of this grid: depth 3, lr 0.01, 300 trees.
GRID = {"max_depth": [3, 5], "learning_rate": [0.01, 0.05], "n_estimators": [300, 600]}
FIXED = {"subsample": 0.7, "colsample_bytree": 0.9}


def make_xgb(params: dict, constraints: tuple, scale_pos_weight: float) -> XGBClassifier:
    # n_jobs=1: the outer folds run in parallel processes, so each model uses one core.
    return XGBClassifier(**params, **FIXED, monotone_constraints=constraints, scale_pos_weight=scale_pos_weight,
                         random_state=SEED, eval_metric="logloss", n_jobs=1)


def _calibrated(params, constraints, spw, method, n_splits):
    return CalibratedClassifierCV(make_xgb(params, constraints, spw), method=method, ensemble=False,
                                  cv=StratifiedKFold(n_splits, shuffle=True, random_state=SEED))


def search_params(x, y, constraints, spw) -> tuple[dict, dict]:
    """Grid search scored by mean PR-AUC over an inner 3-fold CV. Returns (best params, all scores)."""
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, y))
    results = []
    for values in product(*GRID.values()):
        params = dict(zip(GRID, values))
        fold_scores = [average_precision_score(y[te], make_xgb(params, constraints, spw).fit(x.iloc[tr], y[tr])
                                               .predict_proba(x.iloc[te])[:, 1]) for tr, te in folds]
        results.append((float(np.mean(fold_scores)), params))
    _, best = max(results, key=lambda r: r[0])
    return best, {json.dumps(p, sort_keys=True): s for s, p in results}


def fit_procedure(x, y, constraints: tuple, cost_ratio: float) -> dict:
    spw = float((y == 0).sum() / (y == 1).sum())
    params, search_scores = search_params(x, y, constraints, spw)

    # Calibration choice + threshold: one inner 5-fold pass gives per-fold Brier scores for both methods and
    # out-of-fold PDs for every training row.
    oof = {m: np.zeros(len(y)) for m in ("isotonic", "sigmoid")}
    brier = {m: [] for m in oof}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y):
        for m in oof:
            p = _calibrated(params, constraints, spw, m, 3).fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[te])[:, 1]
            oof[m][te] = p
            brier[m].append(brier_score_loss(y[te], p))
    gain = np.array(brier["sigmoid"]) - np.array(brier["isotonic"])
    method = "isotonic" if gain.mean() > gain.std(ddof=1) else "sigmoid"
    threshold = choose_threshold(y, oof[method], cost_ratio)

    model = _calibrated(params, constraints, spw, method, 5).fit(x, y)
    return {"model": model, "params": {**params, **FIXED}, "method": method, "threshold": threshold,
            "isotonic_brier_gain": {"mean": float(gain.mean()), "std": float(gain.std(ddof=1))},
            "search_scores_pr_auc": search_scores, "scale_pos_weight": spw}
