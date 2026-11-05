import os
import subprocess

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'raw', 'data.csv')
# Files the training scripts themselves write; changes there don't make the *code* dirty.
GENERATED = ['reports', 'README.md', 'app/models', 'ui/presets.json', 'ui/sample_portfolio.csv']


def load_data():
    """Features (DataFrame) and labels (numpy array) with column names stripped of stray spaces."""
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    return df.drop(columns=['Bankrupt?']), df['Bankrupt?'].to_numpy()


def git_info() -> dict:
    def run(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = run('status', '--porcelain', '--', '.', *[f':!{p}' for p in GENERATED])
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(dirty)}


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
