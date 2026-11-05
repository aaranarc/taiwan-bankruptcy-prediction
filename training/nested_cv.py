"""Nested cross-validation of the whole model-building procedure (training/procedure.py).

Outer loop: the same 5x5 RepeatedStratifiedKFold (same seed) as training/cv_evaluate.py, so results compare fold for
fold. Inside each outer training fold: preprocessing, hyperparameter search, calibration-method choice and threshold
choice. Each outer test fold is scored exactly once and never influences any choice.

Adds a "nested" block to reports/metrics.json (everything else in the file is left as it is).
Usage: python -m training.nested_cv
"""
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler

from app.services.risk import expected_cost, grade_for, load_config, monotone_tuple
from training.procedure import GRID, fit_procedure
from training.train_utils import ROOT, apply_winsorization, get_extreme_columns, git_info, load_data

OUT = os.path.join(ROOT, "reports", "metrics.json")
N_SPLITS, N_REPEATS, SEED = 5, 5, 42


def outer_fold(i, tr, te, x, y, constraints, cost_ratio):
    x_tr, x_te, y_tr, y_te = x.iloc[tr], x.iloc[te], y[tr], y[te]
    w_tr, w_te, _ = apply_winsorization(x_tr, x_te, get_extreme_columns(x_tr))
    scaler = RobustScaler().fit(w_tr)
    r_tr = pd.DataFrame(scaler.transform(w_tr), columns=x.columns)
    r_te = pd.DataFrame(scaler.transform(w_te), columns=x.columns)

    fit = fit_procedure(r_tr, y_tr, constraints, cost_ratio)  # every choice made on the outer training fold only
    p = fit["model"].predict_proba(r_te)[:, 1]
    p_raw = fit["model"].calibrated_classifiers_[0].estimator.predict_proba(r_te)[:, 1]  # same model, uncalibrated
    t = fit["threshold"]
    flag, raw_flag = (p >= t).astype(int), (p_raw >= 0.5).astype(int)
    print(f"outer fold {i:>2}: {fit['params']} {fit['method']} t={t}", flush=True)
    return {
        "y": y_te, "p": p,
        "roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p), "brier": brier_score_loss(y_te, p),
        "threshold": t, "precision_at_threshold": precision_score(y_te, flag, zero_division=0),
        "recall_at_threshold": recall_score(y_te, flag), "flagged_share": float(flag.mean()),
        "expected_cost_per_firm": expected_cost(y_te, p, t, cost_ratio),
        "uncalibrated_precision_at_0_50": precision_score(y_te, raw_flag, zero_division=0),
        "uncalibrated_recall_at_0_50": recall_score(y_te, raw_flag),
        "params": fit["params"], "method": fit["method"],
    }


def summarise(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "folds": [float(v) for v in arr]}


def main():
    provenance = git_info()  # captured before the run: the code state the numbers come from
    x, y = load_data()
    policy = load_config("risk_policy.json")
    cost_ratio, grades = policy["cost_ratio_fn_to_fp"], policy["grades"]
    constraints = monotone_tuple(list(x.columns))
    splits = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(x, y)

    folds = Parallel(n_jobs=-1)(delayed(outer_fold)(i, tr, te, x, y, constraints, cost_ratio)
                                for i, (tr, te) in enumerate(splits, start=1))

    y_pool = np.concatenate([f["y"] for f in folds])
    p_pool = np.concatenate([f["p"] for f in folds])
    grade_of = np.array([grade_for(v, grades) for v in p_pool])
    grade_rows = []
    for g in grades:
        m = grade_of == g["grade"]
        grade_rows.append({"grade": g["grade"], "max_pd": g["max_pd"], "n": int(m.sum()), "defaults": int(y_pool[m].sum()),
                           "observed_default_rate": float(y_pool[m].mean()) if m.any() else None,
                           "mean_predicted_pd": float(p_pool[m].mean()) if m.any() else None})
    rates = [r["observed_default_rate"] for r in grade_rows if r["observed_default_rate"] is not None]

    metric_keys = ["roc_auc", "pr_auc", "brier", "threshold", "precision_at_threshold", "recall_at_threshold",
                   "flagged_share", "expected_cost_per_firm", "uncalibrated_precision_at_0_50",
                   "uncalibrated_recall_at_0_50"]
    nested = {
        "label": "NESTED CV: tuning, calibration choice and threshold choice all happen inside each outer training fold",
        "generated_by": "training/nested_cv.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": provenance,
        "outer_cv": {"scheme": "RepeatedStratifiedKFold", "n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seed": SEED,
                     "same_splits_as": "training/cv_evaluate.py"},
        "inner": {"hyperparameter_grid": GRID, "search": "inner 3-fold CV, score = PR-AUC",
                  "calibration_rule": "isotonic only if its mean paired Brier gain over sigmoid exceeds the std of "
                                      "that gain (inner 5-fold)",
                  "threshold": f"minimise expected cost at {cost_ratio}:1 on inner out-of-fold calibrated PDs"},
        "metrics": {k: summarise([f[k] for f in folds]) for k in metric_keys},
        "choices": {
            "params": dict(Counter(json.dumps(f["params"], sort_keys=True) for f in folds)),
            "method": dict(Counter(f["method"] for f in folds)),
        },
        "grades": grade_rows,
        "grades_rank_ordered": all(a < b for a, b in zip(rates, rates[1:])),
        "grades_note": "Grades are rank-ordered (observed default rate rises with grade); they are not a calibration "
                       "claim. Compare mean_predicted_pd with observed_default_rate per grade. Counts pool all repeats.",
    }

    with open(OUT) as fh:
        report = json.load(fh)
    report["nested"] = nested
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Added nested results to {OUT}")


if __name__ == "__main__":
    main()
