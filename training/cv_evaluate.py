"""Repeated stratified k-fold CV: baselines, constrained vs unconstrained XGBoost, calibration,
cost-based threshold and risk grades.

Writes reports/metrics.json and reports/calibration.png. Every metric in the README and UI comes from
those files. Never writes to app/models/ (training/train_model.py does that, after this script).

Usage: python -m training.cv_evaluate
"""
import json
import os
from datetime import datetime, timezone

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

from app.services.risk import choose_threshold, expected_cost, grade_for, load_config, monotone_tuple
from training.train_utils import ROOT, apply_winsorization, get_extreme_columns, git_info, load_data

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

OUT = os.path.join(ROOT, 'reports', 'metrics.json')
PLOT = os.path.join(ROOT, 'reports', 'calibration.png')
N_SPLITS, N_REPEATS, SEED = 5, 5, 42
RAW_THRESHOLD = 0.5  # only for the uncalibrated comparison rows

# Altman (1968) inputs mapped to this dataset's columns. Only the ranking (AUC) is used:
# the columns are 0-1 normalised, so Z's absolute value and its 1.81/2.99 zones mean nothing here.
ALTMAN = {
    'Working Capital to Total Assets': 1.2,
    'Retained Earnings to Total Assets': 1.4,
    'ROA(B) before interest and depreciation after tax': 3.3,
    'Net worth/Assets': 0.6,
    'Total Asset Turnover': 1.0,
}


def xgb(params, scale_pos_weight, constraints=None):
    extra = {'monotone_constraints': constraints} if constraints else {}
    return XGBClassifier(**params, **extra, scale_pos_weight=scale_pos_weight, random_state=SEED,
                         eval_metric='logloss')


def calibrated(params, scale_pos_weight, constraints, method):
    # Calibration data comes from an inner CV *inside the training fold* (ensemble=False: out-of-fold
    # predictions fit the calibrator, then the model is refit on the whole training fold). The outer test
    # fold is never seen, so there is no leakage.
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(xgb(params, scale_pos_weight, constraints), method=method, cv=inner,
                                  ensemble=False)


def summarise(values):
    arr = np.asarray(values, dtype=float)
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)), 'folds': [float(v) for v in arr]}


def main():
    x, y = load_data()
    policy = load_config('risk_policy.json')
    params, cost_ratio, grades = policy['xgb_params'], policy['cost_ratio_fn_to_fp'], policy['grades']
    constraints = monotone_tuple(list(x.columns))

    # How small the old single 80/20 test split was (same seed/stratification as the old train_model.py).
    _, _, _, y_old_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    raw_names = ['logistic_regression', 'random_forest', 'xgboost_unconstrained', 'xgboost_monotone']
    raw = {n: {'roc_auc': [], 'pr_auc': [], 'brier': [], 'recall': [], 'precision': []} for n in raw_names}
    cal = {m: {'roc_auc': [], 'pr_auc': [], 'brier': []} for m in ('isotonic', 'sigmoid')}
    altman = {'roc_auc': [], 'pr_auc': []}
    oof = []  # (y_test, raw monotone PD, {method: calibrated PD}) per fold
    bankrupt_per_test_fold = []

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for i, (tr, te) in enumerate(cv.split(x, y), start=1):
        x_tr, x_te, y_tr, y_te = x.iloc[tr], x.iloc[te], y[tr], y[te]
        bankrupt_per_test_fold.append(int(y_te.sum()))

        # Altman uses raw ratios and has nothing to fit. Lower Z = more risk, so -Z is the risk score.
        z = sum(coef * x_te[col] for col, coef in ALTMAN.items())
        altman['roc_auc'].append(roc_auc_score(y_te, -z))
        altman['pr_auc'].append(average_precision_score(y_te, -z))

        # Preprocessing is fitted on the training fold only, so nothing leaks from the test fold.
        w_tr, w_te, _ = apply_winsorization(x_tr, x_te, get_extreme_columns(x_tr))
        scaler = RobustScaler().fit(w_tr)  # same preprocessing as the deployed model
        r_tr = pd.DataFrame(scaler.transform(w_tr), columns=x.columns)
        r_te = pd.DataFrame(scaler.transform(w_te), columns=x.columns)
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()

        models = {
            # LR gets StandardScaler on winsorised data instead of the RobustScaler output: 22 columns have a
            # near-zero IQR, so RobustScaler pushes them to ~1e13. Trees don't care; a linear model collapses.
            'logistic_regression': make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
            'random_forest': RandomForestClassifier(
                n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=SEED),
            'xgboost_unconstrained': xgb(params, spw),
            'xgboost_monotone': xgb(params, spw, constraints),
        }
        for name, model in models.items():
            a, b = (w_tr, w_te) if name == 'logistic_regression' else (r_tr, r_te)
            prob = model.fit(a, y_tr).predict_proba(b)[:, 1]
            pred = (prob >= RAW_THRESHOLD).astype(int)
            raw[name]['roc_auc'].append(roc_auc_score(y_te, prob))
            raw[name]['pr_auc'].append(average_precision_score(y_te, prob))
            raw[name]['brier'].append(brier_score_loss(y_te, prob))
            raw[name]['recall'].append(recall_score(y_te, pred))
            raw[name]['precision'].append(precision_score(y_te, pred, zero_division=0))
            if name == 'xgboost_monotone':
                raw_monotone = prob

        cal_probs = {}
        for method in cal:
            p = calibrated(params, spw, constraints, method).fit(r_tr, y_tr).predict_proba(r_te)[:, 1]
            cal_probs[method] = p
            cal[method]['roc_auc'].append(roc_auc_score(y_te, p))
            cal[method]['pr_auc'].append(average_precision_score(y_te, p))
            cal[method]['brier'].append(brier_score_loss(y_te, p))
        oof.append((y_te, raw_monotone, cal_probs))
        print(f"fold {i:>2}/{N_SPLITS * N_REPEATS}  monotone ROC-AUC {raw['xgboost_monotone']['roc_auc'][-1]:.4f}"
              f"  isotonic Brier {cal['isotonic']['brier'][-1]:.4f}")

    # --- Calibration method ---
    # Platt (sigmoid) is strictly increasing, so it keeps the model's ranking (AUC) exactly; isotonic is a step
    # function that creates ties. Use isotonic only if its Brier gain beats the fold-to-fold noise.
    brier_gain = np.array(cal['sigmoid']['brier']) - np.array(cal['isotonic']['brier'])
    method = 'isotonic' if brier_gain.mean() > brier_gain.std(ddof=1) else 'sigmoid'
    y_pool = np.concatenate([o[0] for o in oof])
    p_pool = np.concatenate([o[2][method] for o in oof])
    p_raw_pool = np.concatenate([o[1] for o in oof])

    # --- Cost-based threshold on pooled out-of-fold calibrated PDs ---
    threshold = choose_threshold(y_pool, p_pool, cost_ratio)
    at_t = {'precision': [], 'recall': [], 'flagged_share': [], 'expected_cost_per_firm': []}
    at_half = {'precision': [], 'recall': [], 'expected_cost_per_firm': []}
    for y_te, _, probs in oof:
        p = probs[method]
        for store, t in ((at_t, threshold), (at_half, 0.5)):
            flag = (p >= t).astype(int)
            store['precision'].append(precision_score(y_te, flag, zero_division=0))
            store['recall'].append(recall_score(y_te, flag))
            store['expected_cost_per_firm'].append(expected_cost(y_te, p, t, cost_ratio))
        at_t['flagged_share'].append(float(np.mean(p >= threshold)))

    # --- Risk grades: observed default rate per grade on pooled out-of-fold PDs ---
    grade_of = np.array([grade_for(v, grades) for v in p_pool])
    grade_rows = []
    for g in grades:
        mask = grade_of == g['grade']
        grade_rows.append({'grade': g['grade'], 'max_pd': g['max_pd'], 'n': int(mask.sum()),
                           'defaults': int(y_pool[mask].sum()),
                           'observed_default_rate': float(y_pool[mask].mean()) if mask.any() else None,
                           'mean_pd': float(p_pool[mask].mean()) if mask.any() else None})
    rates = [r['observed_default_rate'] for r in grade_rows if r['observed_default_rate'] is not None]
    grades_rank_correctly = all(a < b for a, b in zip(rates, rates[1:]))

    # --- Reliability (calibration) curve + plot ---
    obs_cal, pred_cal = calibration_curve(y_pool, p_pool, n_bins=10, strategy='quantile')
    obs_raw, pred_raw = calibration_curve(y_pool, p_raw_pool, n_bins=10, strategy='quantile')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls='--', c='grey', lw=1, label='Perfect calibration')
    ax.plot(pred_raw, obs_raw, 'o-', c='#d97706', label='Monotone XGBoost, uncalibrated')
    ax.plot(pred_cal, obs_cal, 'o-', c='#2563eb', label=f'Calibrated ({method})')
    ax.set(xscale='log', yscale='log', xlim=(1e-3, 1), ylim=(1e-3, 1),
           xlabel='Mean predicted PD (decile bin)', ylabel='Observed default rate',
           title=f'Reliability, out-of-fold, {N_REPEATS}x{N_SPLITS} CV')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)

    def paired_cost(metric):
        diff = np.array(raw['xgboost_monotone'][metric]) - np.array(raw['xgboost_unconstrained'][metric])
        return {'mean_difference': float(diff.mean()), 'std_difference': float(diff.std(ddof=1)),
                'folds_where_constrained_is_worse': int((diff < 0).sum() if metric != 'brier' else (diff > 0).sum())}

    report = {
        'generated_by': 'training/cv_evaluate.py',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git': git_info(),
        'versions': {'xgboost': xgboost.__version__, 'scikit-learn': sklearn.__version__},
        'data': {'rows': int(len(y)), 'features': int(x.shape[1]), 'bankrupt': int(y.sum()),
                 'base_rate': float(y.mean())},
        'cv': {'scheme': 'RepeatedStratifiedKFold', 'n_splits': N_SPLITS, 'n_repeats': N_REPEATS,
               'seed': SEED, 'bankrupt_per_test_fold': summarise(bankrupt_per_test_fold)},
        'old_single_split': {'test_rows': int(len(y_old_test)), 'test_bankrupt': int(y_old_test.sum())},
        'raw_threshold': RAW_THRESHOLD,
        'caveats': [
            'XGBoost hyperparameters (config/risk_policy.json) were tuned in the notebook on an 80/20 split that '
            'overlaps these CV test folds, so XGBoost rows are slightly optimistic (no nested CV).',
            'The cost threshold is chosen on the same pooled out-of-fold PDs it is evaluated on; compare it with '
            'the Bayes-optimal threshold 1/(1+cost_ratio), which uses no data.',
            'Grade boundaries are policy (config/risk_policy.json); grade counts pool all 5 repeats, so each firm '
            'appears 5 times.',
            'Recall at a fixed 0.50 threshold is not comparable across uncalibrated models; compare ROC-AUC / PR-AUC.',
        ],
        'xgboost_params': {**params, 'scale_pos_weight': 'n_negative / n_positive of each training fold'},
        'monotone_constraints': {'n_constrained': int(sum(c != 0 for c in constraints)),
                                 'config': 'config/monotone_constraints.json'},
        'models': {name: {m: summarise(v) for m, v in s.items()} for name, s in raw.items()},
        'constraint_cost': {m: paired_cost(m) for m in ('roc_auc', 'pr_auc', 'brier')},
        'altman_z_benchmark': {
            'score': '-Z (lower Z = higher risk), raw ratios, Altman 1968 coefficients',
            'caveat': 'Ratios are 0-1 normalised in this dataset, so Z values and the 1.81/2.99 zones are '
                      'not interpretable; only the ranking (AUC) is reported.',
            **{m: summarise(v) for m, v in altman.items()},
        },
        'calibration': {
            'chosen_method': method,
            'rule': 'isotonic only if its mean paired Brier gain over sigmoid exceeds the std of that gain',
            'isotonic_brier_gain': {'mean': float(brier_gain.mean()), 'std': float(brier_gain.std(ddof=1))},
            'how': 'CalibratedClassifierCV(ensemble=False) with a 5-fold inner CV inside each outer training fold',
            'methods': {m: {k: summarise(v) for k, v in s.items()} for m, s in cal.items()},
            'reliability_bins': {
                'calibrated': {'mean_predicted': pred_cal.tolist(), 'observed': obs_cal.tolist()},
                'uncalibrated_monotone': {'mean_predicted': pred_raw.tolist(), 'observed': obs_raw.tolist()},
            },
            'plot': 'reports/calibration.png',
        },
        'deployed_model': {
            'description': f'Monotone-constrained XGBoost + {method} calibration',
            **{k: summarise(v) for k, v in cal[method].items()},
        },
        'threshold': {
            'cost_ratio_fn_to_fp': cost_ratio,
            'chosen': threshold,
            'bayes_optimal_if_perfectly_calibrated': 1 / (1 + cost_ratio),
            'at_chosen': {k: summarise(v) for k, v in at_t.items()},
            'at_0_50_for_comparison': {k: summarise(v) for k, v in at_half.items()},
        },
        'grades': grade_rows,
        'grades_rank_risk_in_order': grades_rank_correctly,
    }
    with open(OUT, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {OUT} and {PLOT}")


if __name__ == '__main__':
    main()
