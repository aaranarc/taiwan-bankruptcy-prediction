import pandas as pd
from sklearn.preprocessing import RobustScaler

def get_extreme_columns(df: pd.DataFrame) -> list:
    """Find numeric columns that have extreme values (outliers)."""
    feature_cols = [c for c in df.select_dtypes(include='number').columns]
    return [c for c in feature_cols if df[c].max() > 100]

def apply_winsorization(df_train: pd.DataFrame, df_test: pd.DataFrame, extreme_cols: list) -> tuple:
    """Compute winsorization bounds from train data and apply to both train and test data."""
    bounds = {}
    df_train_copied = df_train.copy()
    df_test_copied = df_test.copy()
    
    for col in extreme_cols:
        lb = df_train[col].quantile(0.01)
        ub = df_train[col].quantile(0.99)
        bounds[col] = (lb, ub)
        df_train_copied[col] = df_train_copied[col].clip(lower=lb, upper=ub)
        df_test_copied[col] = df_test_copied[col].clip(lower=lb, upper=ub)
        
    return df_train_copied, df_test_copied, bounds
