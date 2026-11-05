"""Repeated stratified k-fold CV for the deployed XGBoost config, two baselines and Altman Z.

Writes reports/metrics.json. Every metric in the README comes from that file
(see training/render_readme.py). Never writes to app/models/.

Usage: python -m training.cv_evaluate
"""
import json
import os
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from training.train_utils import apply_winsorization, get_extreme_columns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'raw', 'data.csv')
OUT = os.path.join(ROOT, 'reports', 'metrics.json')
N_SPLITS, N_REPEATS, SEED = 5, 5, 42
THRESHOLD = 0.5  # Phase 0 keeps the default; Phase 1 replaces it with a cost-based threshold

# Altman (1968) inputs mapped to this dataset's columns. Only the ranking (AUC) is used:
# the columns are 0-1 normalised, so Z's absolute value and its 1.81/2.99 zones mean nothing here.
ALTMAN = {
    'Working Capital to Total Assets': 1.2,
    'Retained Earnings to Total Assets': 1.4,
    'ROA(B) before interest and depreciation after tax': 3.3,
    'Net worth/Assets': 0.6,
    'Total Asset Turnover': 1.0,
}


def git_info():
    def run(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain'))}


def models(xgb_params, scale_pos_weight):
    # All three handle class imbalance the same way (re-weighting), so the comparison is fair.
    return {
        'logistic_regression': LogisticRegression(max_iter=5000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=SEED),
        'xgboost': XGBClassifier(
            **xgb_params, scale_pos_weight=scale_pos_weight, random_state=SEED, eval_metric='logloss'),
    }


def summarise(values):
    arr = np.asarray(values, dtype=float)
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)), 'folds': [float(v) for v in arr]}


def main():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    x, y = df.drop(columns=['Bankrupt?']), df['Bankrupt?'].to_numpy()
    xgb_params = joblib.load(os.path.join(ROOT, 'app', 'models', 'best_params.pkl'))

    # How small the old single 80/20 test split was (same seed/stratification as train_model.py).
    _, _, _, y_old_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    scores = {name: {'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': []}
              for name in ['logistic_regression', 'random_forest', 'xgboost']}
    altman = {'roc_auc': [], 'pr_auc': []}
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
        x_tr, x_te, _ = apply_winsorization(x_tr, x_te, get_extreme_columns(x_tr))
        scaler = RobustScaler().fit(x_tr)
        x_tr = pd.DataFrame(scaler.transform(x_tr), columns=x.columns)
        x_te = pd.DataFrame(scaler.transform(x_te), columns=x.columns)

        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        for name, model in models(xgb_params, spw).items():
            prob = model.fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
            pred = (prob >= THRESHOLD).astype(int)
            scores[name]['roc_auc'].append(roc_auc_score(y_te, prob))
            scores[name]['pr_auc'].append(average_precision_score(y_te, prob))
            scores[name]['recall'].append(recall_score(y_te, pred))
            scores[name]['precision'].append(precision_score(y_te, pred, zero_division=0))
        print(f"fold {i:>2}/{N_SPLITS * N_REPEATS}  xgb ROC-AUC {scores['xgboost']['roc_auc'][-1]:.4f}")

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
        'threshold': THRESHOLD,
        'xgboost_params': {**xgb_params, 'scale_pos_weight': 'n_negative / n_positive of each training fold'},
        'models': {name: {m: summarise(v) for m, v in s.items()} for name, s in scores.items()},
        'altman_z_benchmark': {
            'score': '-Z (lower Z = higher risk), raw ratios, Altman 1968 coefficients',
            'caveat': 'Ratios are 0-1 normalised in this dataset, so Z values and the 1.81/2.99 zones are '
                      'not interpretable; only the ranking (AUC) is reported.',
            **{m: summarise(v) for m, v in altman.items()},
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()
