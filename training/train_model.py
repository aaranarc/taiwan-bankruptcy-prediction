"""Fit the deployed model on all data and write its artifacts to app/models/.

Run training/cv_evaluate.py first: this script reads the calibration method and the cost-based threshold
that the cross-validation chose (reports/metrics.json), so no decision is made on the final training data.

Usage: python -m training.cv_evaluate && python -m training.train_model
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from app.services.risk import apply_linked, load_config, monotone_tuple
from training.train_utils import ROOT, apply_winsorization, get_extreme_columns, git_info, load_data

MODELS_DIR = os.path.join(ROOT, 'app', 'models')
UI_DIR = os.path.join(ROOT, 'ui')
SEED = 42
OBSOLETE = ['xgb_final_model.pkl', 'best_params.pkl', 'test_metrics.pkl', 'shap_values_test.pkl',
            'x_test.pkl', 'y_test.pkl']


def main():
    x, y = load_data()
    policy = load_config('risk_policy.json')
    with open(os.path.join(ROOT, 'reports', 'metrics.json')) as fh:
        metrics = json.load(fh)
    method, threshold = metrics['calibration']['chosen_method'], metrics['threshold']['chosen']

    # Same preprocessing as in CV, now fitted on all rows (train and "test" are the same frame here).
    x_w, _, bounds = apply_winsorization(x, x, get_extreme_columns(x))
    scaler = RobustScaler().fit(x_w)
    x_s = pd.DataFrame(scaler.transform(x_w), columns=x.columns)

    base = XGBClassifier(**policy['xgb_params'], monotone_constraints=monotone_tuple(list(x.columns)),
                         scale_pos_weight=(y == 0).sum() / (y == 1).sum(), random_state=SEED,
                         eval_metric='logloss')
    model = CalibratedClassifierCV(base, method=method, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                   ensemble=False).fit(x_s, y)

    # Percentiles 0..100 of the raw training data: bounds for counterfactuals and stress tests.
    quantiles = x.quantile(np.linspace(0, 1, 101))
    quantiles.index = np.arange(101)

    meta = {
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git': git_info(),
        'metrics_commit': metrics['git']['commit'],
        'calibration_method': method,
        'threshold': threshold,
        'cost_ratio_fn_to_fp': policy['cost_ratio_fn_to_fp'],
        'grades': policy['grades'],
        'n_rows': int(len(y)),
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(UI_DIR, exist_ok=True)
    for name in OBSOLETE:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            os.remove(path)
    artifacts = {'model.joblib': model, 'robust_scaler.pkl': scaler, 'winsorize_bounds.pkl': bounds,
                 'feature_names.pkl': list(x.columns), 'feature_quantiles.pkl': quantiles}
    for fname, obj in artifacts.items():
        joblib.dump(obj, os.path.join(MODELS_DIR, fname))
    with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)

    # UI demo inputs. Presets are the feature-wise medians of each class (not real firms).
    presets = {}
    for name, label in (('Healthy firm (median of non-bankrupt)', 0), ('Distressed firm (median of bankrupt)', 1)):
        row = apply_linked(x[y == label].median().to_frame().T, policy['linked_features']).iloc[0]
        presets[name] = {c: float(v) for c, v in row.items()}
    with open(os.path.join(UI_DIR, 'presets.json'), 'w') as fh:
        json.dump(presets, fh, indent=1)
    sample = x.sample(150, random_state=SEED)
    sample.insert(0, 'firm_id', [f'TW-{i:04d}' for i in sample.index])
    sample.to_csv(os.path.join(UI_DIR, 'sample_portfolio.csv'), index=False)

    # Reload through the API's own loader: proves the saved artifacts work end to end.
    from app.services.model_service import ModelService
    service = ModelService()
    for name, feats in presets.items():
        res = service.predict(feats)
        print(f"   preset {name!r}: PD {res['probability']:.4f}, grade {res['grade']}")
    print(f"Saved model ({method} calibration, threshold {threshold}) and artifacts to {MODELS_DIR}")


if __name__ == '__main__':
    main()
