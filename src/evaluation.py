"""
src/evaluation.py — Model loading, preprocessing, and prediction utilities.

Used by app.py to load saved model artifacts and run inference.
"""

import os
import joblib
import pandas as pd
import numpy as np


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def load_model_artifacts():
    """
    Load all model artifacts from the models/ directory.

    Returns
    -------
    dict
        Dictionary with keys: 'model', 'scaler', 'bounds', 'feature_names',
        'best_params', 'test_metrics', 'shap_values', 'x_test', 'y_test'.
    """
    artifacts = {}

    file_map = {
        'model': 'xgb_final_model.pkl',
        'scaler': 'robust_scaler.pkl',
        'bounds': 'winsorize_bounds.pkl',
        'feature_names': 'feature_names.pkl',
        'best_params': 'best_params.pkl',
        'test_metrics': 'test_metrics.pkl',
        'shap_values': 'shap_values_test.pkl',
        'x_test': 'x_test.pkl',
        'y_test': 'y_test.pkl',
    }

    for key, fname in file_map.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            artifacts[key] = None

    return artifacts


def preprocess_input(df: pd.DataFrame, scaler, bounds: dict, feature_names: list) -> pd.DataFrame:
    """
    Preprocess raw input data for model prediction.

    Applies winsorization (using training-derived bounds) and RobustScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame with feature columns.
    scaler : RobustScaler
        Fitted scaler from training.
    bounds : dict
        Winsorization bounds {column_name: (lower, upper)}.
    feature_names : list
        Expected feature names in order.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for prediction.
    """
    # Ensure correct columns and order
    df = df.copy()

    # Strip column names
    df.columns = df.columns.str.strip()

    # Keep only model features (drop target if present)
    if 'Bankrupt?' in df.columns:
        df = df.drop(columns=['Bankrupt?'])

    # Reorder columns to match training
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing features: {missing_cols}")

    df = df[feature_names]

    # Apply winsorization with training bounds
    for col, (lb, ub) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lb, upper=ub)

    # Scale
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_names, index=df.index)


def predict_bankruptcy(model, X_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Run bankruptcy prediction on preprocessed data.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model.
    X_processed : pd.DataFrame
        Preprocessed feature DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'probability', 'prediction', 'risk_tier'.
    """
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = model.predict(X_processed)

    risk_tiers = []
    for p in probabilities:
        if p < 0.15:
            risk_tiers.append('Low Risk')
        elif p < 0.35:
            risk_tiers.append('Medium Risk')
        elif p < 0.60:
            risk_tiers.append('High Risk')
        else:
            risk_tiers.append('Critical Risk')

    return pd.DataFrame({
        'probability': probabilities,
        'prediction': predictions,
        'risk_tier': risk_tiers,
    }, index=X_processed.index)


def get_risk_color(risk_tier: str) -> str:
    """Return color for a risk tier."""
    return {
        'Low Risk': '#10b981',
        'Medium Risk': '#f59e0b',
        'High Risk': '#f97316',
        'Critical Risk': '#ef4444',
    }.get(risk_tier, '#6b7280')


def get_risk_emoji(risk_tier: str) -> str:
    """Return emoji for a risk tier."""
    return {
        'Low Risk': '🟢',
        'Medium Risk': '🟡',
        'High Risk': '🟠',
        'Critical Risk': '🔴',
    }.get(risk_tier, '⚪')
