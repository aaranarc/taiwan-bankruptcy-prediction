"""Evaluate the deployed model on the saved test split. Read-only: never writes to app/models/.

Usage: python training/evaluate.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'models')
THRESHOLDS = [0.15, 0.35, 0.50, 0.60]  # 0.15 / 0.35 / 0.60 are the API's risk-tier cut-offs


def load(name):
    return joblib.load(os.path.join(MODELS_DIR, name))


def main():
    model = load('model.joblib')
    x_test = load('x_test.pkl')
    y_test = np.asarray(load('y_test.pkl'))
    prob = model.predict_proba(x_test)[:, 1]

    print(f"Test set: {len(y_test)} firms, {int(y_test.sum())} bankrupt (base rate {y_test.mean():.4f})")
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, prob):.4f}")

    print(f"\n{'Threshold':<10} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5} {'Recall':>8} {'Precision':>10}")
    for t in THRESHOLDS:
        tn, fp, fn, tp = confusion_matrix(y_test, (prob >= t).astype(int), labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else float('nan')
        print(f"{t:<10.2f} {tn:>5} {fp:>5} {fn:>5} {tp:>5} {tp / (tp + fn):>8.2f} {precision:>10.2f}")

    shap_values = np.asarray(load('shap_values_test.pkl'))
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=x_test.columns).sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP| (test set):")
    for rank, (feature, value) in enumerate(importance.head(10).items(), start=1):
        print(f"{rank:>2}. {feature:<55} {value:.4f}")


if __name__ == '__main__':
    main()
