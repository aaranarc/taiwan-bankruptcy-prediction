import hashlib
import json
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

from app.services import scenarios
from app.services.risk import grade_for, load_config, monotone_directions
from app.utils.logger import get_logger

logger = get_logger("model_service")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


class ModelService:
    """Loads the trained artifacts once and turns raw ratios into PDs, grades and explanations."""

    def __init__(self):
        self.artifacts = {}
        self.meta = {}
        self.explainer = None
        self.model_version = "unknown"
        self.policy = load_config('risk_policy.json')
        self.directions = monotone_directions()
        self.load_artifacts()

    def load_artifacts(self):
        file_map = {
            'model': 'model.joblib',
            'scaler': 'robust_scaler.pkl',
            'bounds': 'winsorize_bounds.pkl',
            'feature_names': 'feature_names.pkl',
        }
        for key, fname in file_map.items():
            path = os.path.join(MODELS_DIR, fname)
            if os.path.exists(path):
                self.artifacts[key] = joblib.load(path)
                logger.info(f"Loaded model artifact {fname} for key '{key}'")
            else:
                self.artifacts[key] = None
                logger.warning(f"Model artifact not found at {path}")

        q_path = os.path.join(MODELS_DIR, 'feature_quantiles.json')
        if os.path.exists(q_path):
            with open(q_path) as fh:
                self.artifacts['quantiles'] = pd.DataFrame(json.load(fh))  # index = percentile 0..100

        meta_path = os.path.join(MODELS_DIR, 'model_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                self.meta = json.load(fh)

        if self.artifacts['model'] is not None:
            with open(os.path.join(MODELS_DIR, file_map['model']), 'rb') as fh:
                self.model_version = hashlib.sha256(fh.read()).hexdigest()[:8]
            try:
                # SHAP explains the underlying XGBoost (log-odds), not the calibration step on top of it.
                self.explainer = shap.TreeExplainer(self.base_estimator)
            except Exception as e:
                logger.warning(f"SHAP explainer unavailable; predictions will omit top_factors: {e}")

    @property
    def base_estimator(self):
        return self.artifacts['model'].calibrated_classifiers_[0].estimator

    @property
    def threshold(self) -> float:
        return self.meta['threshold']

    @property
    def grades(self) -> list:
        return self.meta.get('grades') or self.policy['grades']

    def get_feature_names(self) -> List[str]:
        return self.artifacts.get("feature_names") or []

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

    def score(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default for each row of raw ratios."""
        if self.artifacts.get("model") is None:
            raise RuntimeError("Model is not loaded.")
        return self.artifacts["model"].predict_proba(self.preprocess_input(df_raw))[:, 1]

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        df_raw = pd.DataFrame([features])
        prob = float(self.score(df_raw)[0])
        flagged = prob >= self.threshold
        result = {
            "probability": prob,
            "grade": grade_for(prob, self.grades),
            "threshold": self.threshold,
            "flagged": flagged,
            "prediction": int(flagged),
            "model_version": self.model_version,
            "top_factors": None,
            "shap_base_value": None,
            "shap_other": None,
        }
        if self.explainer:
            result.update(self.explain(self.preprocess_input(df_raw), features))
        return result

    def predict_batch(self, rows: List[Dict[str, float]], k: int = 3) -> List[Dict[str, Any]]:
        """Score many firms at once; each row gets its PD, grade, flag and top-k SHAP factors."""
        df_raw = pd.DataFrame(rows)
        df_proc = self.preprocess_input(df_raw)
        probs = self.artifacts["model"].predict_proba(df_proc)[:, 1]
        shap_rows = np.asarray(self.explainer.shap_values(df_proc)) if self.explainer else None
        out = []
        for i, prob in enumerate(probs):
            factors = None
            if shap_rows is not None:
                top = np.argsort(-np.abs(shap_rows[i]))[:k]
                factors = [{"feature": df_proc.columns[j], "shap": float(shap_rows[i][j])} for j in top]
            out.append({"probability": float(prob), "grade": grade_for(float(prob), self.grades),
                        "flagged": bool(prob >= self.threshold), "top_factors": factors})
        return out

    def explain(self, df_proc: pd.DataFrame, features: Dict[str, float], k: int = 10) -> Dict[str, Any]:
        """Top-k SHAP contributions (log-odds) plus the rest summed, so a waterfall adds up exactly."""
        shap_row = np.asarray(self.explainer.shap_values(df_proc))[0]
        top = np.argsort(-np.abs(shap_row))[:k]
        factors = [
            {
                "feature": df_proc.columns[i],
                "value": float(features[df_proc.columns[i]]),
                "shap": float(shap_row[i]),
                "direction": "raises risk" if shap_row[i] > 0 else "lowers risk",
            }
            for i in top
        ]
        return {
            "top_factors": factors,
            "shap_base_value": float(np.ravel(self.explainer.expected_value)[0]),
            "shap_other": float(shap_row.sum() - shap_row[top].sum()),
        }

    def counterfactual(self, features: Dict[str, float], target_grade: str) -> Dict[str, Any]:
        return scenarios.counterfactual(features, self.score, self.artifacts['quantiles'], self.policy,
                                        self.directions, self.grades, target_grade)

    def stress(self, features: Dict[str, float], shocks: Dict[str, float]) -> Dict[str, Any]:
        return scenarios.stress(features, self.score, self.artifacts['quantiles'], self.policy, self.grades,
                                shocks)

    def public_meta(self) -> Dict[str, Any]:
        """Policy the UI needs to render inputs and labels. Contains no secrets."""
        return {
            "model_version": self.model_version,
            "threshold": self.threshold,
            "cost_ratio_fn_to_fp": self.meta.get('cost_ratio_fn_to_fp'),
            "calibration_method": self.meta.get('calibration_method'),
            "grades": self.grades,
            "adjustable_features": self.policy['counterfactual']['adjustable_features'],
            "default_target_grade": self.policy['counterfactual']['default_target_grade'],
            "stress_shocks": {k: {f: v[f] for f in ('label', 'min', 'max', 'default', 'columns', 'doc')}
                              for k, v in self.policy['stress_shocks'].items()},
        }


model_service = ModelService()
