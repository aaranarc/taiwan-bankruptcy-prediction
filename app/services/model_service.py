import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from app.utils.logger import get_logger

logger = get_logger("model_service")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

class ModelService:
    def __init__(self):
        self.artifacts = {}
        self.load_artifacts()

    def load_artifacts(self):
        file_map = {
            'model': 'model.joblib',
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
                self.artifacts[key] = joblib.load(path)
                logger.info(f"Loaded model artifact {fname} for key '{key}'")
            else:
                self.artifacts[key] = None
                logger.warning(f"Model artifact not found at {path}")

    def get_feature_names(self) -> List[str]:
        return self.artifacts.get("feature_names", [])

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()

        if 'Bankrupt?' in df.columns:
            df = df.drop(columns=['Bankrupt?'])

        feature_names = self.get_feature_names()
        missing_cols = set(feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")

        df = df[feature_names]

        # Apply winsorization
        bounds = self.artifacts.get("bounds") or {}
        for col, (lb, ub) in bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lb, upper=ub)

        # Scale
        scaler = self.artifacts.get("scaler")
        if scaler:
            scaled = scaler.transform(df)
            return pd.DataFrame(scaled, columns=feature_names, index=df.index)
        return df

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        if not self.artifacts.get("model"):
            raise RuntimeError("Model is not loaded.")

        df_raw = pd.DataFrame([features])
        df_proc = self.preprocess_input(df_raw)

        model = self.artifacts["model"]
        probabilities = model.predict_proba(df_proc)[:, 1]
        predictions = model.predict(df_proc)

        prob = float(probabilities[0])
        pred = int(predictions[0])

        if prob < 0.15:
            risk_tier = 'Low Risk'
            risk_emoji = '🟢'
            risk_color = '#10b981'
        elif prob < 0.35:
            risk_tier = 'Medium Risk'
            risk_emoji = '🟡'
            risk_color = '#f59e0b'
        elif prob < 0.60:
            risk_tier = 'High Risk'
            risk_emoji = '🟠'
            risk_color = '#f97316'
        else:
            risk_tier = 'Critical Risk'
            risk_emoji = '🔴'
            risk_color = '#ef4444'

        # Compute Altman Z-Score if features are present
        altman_z = None
        altman_zone = None
        altman_emoji = None
        altman_color = None

        altman_features = {
            'X1': 'Working Capital to Total Assets',
            'X2': 'Retained Earnings to Total Assets',
            'X3': 'ROA(B) before interest and depreciation after tax',
            'X4': 'Net worth/Assets',
            'X5': 'Total Asset Turnover',
        }
        altman_coeffs = {'X1': 1.2, 'X2': 1.4, 'X3': 3.3, 'X4': 0.6, 'X5': 1.0}

        try:
            if all(col in features for col in altman_features.values()):
                z = sum(altman_coeffs[k] * features[altman_features[k]] for k in altman_features)
                altman_z = float(z)
                if altman_z > 2.99:
                    altman_zone = 'Safe'
                    altman_emoji = '🟢'
                    altman_color = '#10b981'
                elif altman_z > 1.81:
                    altman_zone = 'Grey'
                    altman_emoji = '🟡'
                    altman_color = '#f59e0b'
                else:
                    altman_zone = 'Distress'
                    altman_emoji = '🔴'
                    altman_color = '#ef4444'
        except Exception as e:
            logger.warning(f"Error computing Altman Z-Score: {e}")

        return {
            "probability": prob,
            "prediction": pred,
            "risk_tier": risk_tier,
            "risk_emoji": risk_emoji,
            "risk_color": risk_color,
            "altman_z_score": altman_z,
            "altman_zone": altman_zone,
            "altman_emoji": altman_emoji,
            "altman_color": altman_color,
        }

model_service = ModelService()

