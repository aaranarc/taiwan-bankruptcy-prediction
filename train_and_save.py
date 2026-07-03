"""
train_and_save.py — Standalone training script for Taiwan Bankruptcy Prediction.

Replicates the notebook pipeline and saves all model artifacts to models/.
Run once: python train_and_save.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("Taiwan Bankruptcy Prediction — Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'data.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    print(f"\n✅ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Bankrupt: {(df['Bankrupt?'] == 1).sum()} ({(df['Bankrupt?'] == 1).mean()*100:.2f}%)")
    print(f"   Healthy:  {(df['Bankrupt?'] == 0).sum()} ({(df['Bankrupt?'] == 0).mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    x = df.drop(columns=['Bankrupt?'])
    y = df['Bankrupt?']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n✅ Train/test split: {x_train.shape[0]} train / {x_test.shape[0]} test")

    # ------------------------------------------------------------------
    # 3. Winsorization (bounds from training data only)
    # ------------------------------------------------------------------
    feature_cols = [c for c in x_train.select_dtypes(include='number').columns]
    extreme_cols = [c for c in feature_cols if x_train[c].max() > 100]

    winsorize_bounds_dict = {}
    for col in extreme_cols:
        lb = x_train[col].quantile(0.01)
        ub = x_train[col].quantile(0.99)
        winsorize_bounds_dict[col] = (lb, ub)
        x_train[col] = x_train[col].clip(lower=lb, upper=ub)
        x_test[col] = x_test[col].clip(lower=lb, upper=ub)

    print(f"✅ Winsorized {len(extreme_cols)} columns with extreme outliers")

    # ------------------------------------------------------------------
    # 4. Scaling
    # ------------------------------------------------------------------
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train_scaled, columns=x.columns)
    x_test = pd.DataFrame(x_test_scaled, columns=x.columns)
    print("✅ RobustScaler fitted and applied")

    # ------------------------------------------------------------------
    # 5. GridSearchCV for XGBoost
    # ------------------------------------------------------------------
    print("\n🔍 Running GridSearchCV (this may take a few minutes)...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    print(f"\n✅ Best parameters: {best_params}")
    print(f"   Best ROC-AUC (CV): {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 6. Train final model with scale_pos_weight
    # ------------------------------------------------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    )
    xgb.fit(x_train, y_train)

    # Quick evaluation
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    xgb_prob = xgb.predict_proba(x_test)[:, 1]
    xgb_pred = xgb.predict(x_test)
    roc_auc = roc_auc_score(y_test, xgb_prob)
    pr_auc = average_precision_score(y_test, xgb_prob)

    print(f"\n✅ Final XGBoost model trained")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC:  {pr_auc:.4f}")
    print(f"\n{classification_report(y_test, xgb_pred, target_names=['Healthy', 'Bankrupt'])}")

    # ------------------------------------------------------------------
    # 7. Compute SHAP values on test set for global importance
    # ------------------------------------------------------------------
    print("🔍 Computing SHAP values for global feature importance...")
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(x_test)
    print("✅ SHAP values computed")

    # ------------------------------------------------------------------
    # 8. Save all artifacts
    # ------------------------------------------------------------------
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    artifacts = {
        'xgb_final_model.pkl': xgb,
        'robust_scaler.pkl': scaler,
        'winsorize_bounds.pkl': winsorize_bounds_dict,
        'feature_names.pkl': list(x.columns),
        'best_params.pkl': best_params,
        'test_metrics.pkl': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'scale_pos_weight': scale_pos_weight,
        },
        'shap_values_test.pkl': shap_values,
        'x_test.pkl': x_test,
        'y_test.pkl': y_test,
    }

    for fname, obj in artifacts.items():
        path = os.path.join(models_dir, fname)
        joblib.dump(obj, path)
        print(f"   Saved: {path}")

    print(f"\n{'=' * 60}")
    print(f"✅ All artifacts saved to {models_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
