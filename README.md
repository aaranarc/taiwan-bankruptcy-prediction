# Taiwan Bankruptcy Prediction

An XGBoost-powered corporate bankruptcy prediction system with SHAP explanations and Altman Z-Score comparison, deployed as an interactive Streamlit dashboard.

## Dataset

**UCI Taiwanese Economic Journal (1999–2009)**
- 6,819 companies × 95 financial ratio features
- Binary target: `Bankrupt?` (0 = Healthy, 1 = Bankrupt)
- Severely imbalanced: ~3.2% bankrupt

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.951 |
| PR-AUC | 0.471 |
| Recall (Bankrupt) | 0.80 |

## Project Structure

```
taiwan-bankruptcy/
├── app.py                  # Streamlit dashboard
├── train_and_save.py       # Model training & artifact generation
├── data/
│   └── raw/data.csv        # Raw dataset
├── models/                 # Saved model artifacts (generated)
│   ├── xgb_final_model.pkl
│   ├── robust_scaler.pkl
│   ├── winsorize_bounds.pkl
│   └── feature_names.pkl
├── src/
│   ├── altman.py           # Altman Z-Score utilities
│   └── evaluation.py       # Model loading & inference
├── taiwan_bankruptcy_prediction.ipynb  # Analysis notebook
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time)

```bash
python train_and_save.py
```

This trains the XGBoost model and saves all artifacts to `models/`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

## Features

- **🏠 Overview**: Model metrics, class distribution, feature categories
- **🔍 Predict Bankruptcy**: Manual input or CSV batch upload with risk scoring
- **📊 SHAP Explanations**: Global & local feature importance visualizations
- **📈 Dataset Explorer**: Interactive EDA with filters and correlation heatmaps

## Methodology

1. **Preprocessing**: Winsorization (train-derived bounds) → RobustScaler
2. **Baseline**: Logistic Regression + Random Forest
3. **Imbalance handling**: SMOTE, class weights, threshold tuning
4. **Final model**: XGBoost with `scale_pos_weight` + GridSearchCV hyperparameter tuning
5. **Benchmark**: Altman Z-Score (1968) — achieves ~0.92 ROC-AUC
6. **Explainability**: SHAP TreeExplainer for feature attribution

## License

See [LICENSE](LICENSE) for details.
