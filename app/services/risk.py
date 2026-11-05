"""Credit-risk policy helpers shared by training and the API: config, grades, cost threshold, linked ratios.

Pure functions with no side effects at import, so training scripts can use them without loading the model.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(ROOT, 'config')


def load_config(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name)) as fh:
        return json.load(fh)


def monotone_directions() -> dict:
    """{feature: +1 or -1} from the reviewed config."""
    return {f: spec['direction'] for f, spec in load_config('monotone_constraints.json')['constraints'].items()}


def monotone_tuple(feature_names: list) -> tuple:
    """XGBoost's monotone_constraints format: one entry per column, 0 = unconstrained."""
    directions = monotone_directions()
    return tuple(directions.get(f, 0) for f in feature_names)


def grade_for(pd_value: float, grades: list) -> str:
    """First grade whose max_pd the PD is strictly below; the last grade catches everything else."""
    for g in grades:
        if pd_value < g['max_pd']:
            return g['grade']
    return grades[-1]['grade']


def grade_max_pd(grade: str, grades: list) -> float:
    for g in grades:
        if g['grade'] == grade:
            return g['max_pd']
    raise ValueError(f"Unknown grade {grade!r}; expected one of {[g['grade'] for g in grades]}")


def expected_cost(y, p, threshold: float, cost_ratio: float) -> float:
    """Average cost per firm when flagging PD >= threshold. A false alarm costs 1, a missed default cost_ratio."""
    y, flag = np.asarray(y), np.asarray(p) >= threshold
    false_neg = np.sum((y == 1) & ~flag)
    false_pos = np.sum((y == 0) & flag)
    return float((cost_ratio * false_neg + false_pos) / len(y))


def choose_threshold(y, p, cost_ratio: float, grid=None) -> float:
    """Threshold on the grid with the lowest expected cost (ties go to the lowest threshold)."""
    grid = np.round(np.arange(0.005, 1.0, 0.005), 3) if grid is None else np.asarray(grid)
    costs = [expected_cost(y, p, t, cost_ratio) for t in grid]
    return float(grid[int(np.argmin(costs))])


def apply_linked(df: pd.DataFrame, linked: dict) -> pd.DataFrame:
    """Keep exactly-linked ratios consistent, e.g. Net worth/Assets = 1 - Debt ratio %."""
    for col, spec in linked.items():
        if col in df.columns and spec['equals_one_minus'] in df.columns:
            df[col] = 1.0 - df[spec['equals_one_minus']]
    return df
