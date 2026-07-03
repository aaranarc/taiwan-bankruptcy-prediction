"""
src/altman.py — Altman Z-Score utilities.

Implements the classical Altman Z-Score (1968) for bankruptcy prediction.
Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

Thresholds: Z > 2.99 = Safe, 1.81 < Z < 2.99 = Grey, Z < 1.81 = Distress
"""

import pandas as pd
import numpy as np


# Altman Z-Score feature mapping to dataset column names
ALTMAN_FEATURES = {
    'X1': 'Working Capital to Total Assets',
    'X2': 'Retained Earnings to Total Assets',
    'X3': 'ROA(B) before interest and depreciation after tax',
    'X4': 'Net worth/Assets',
    'X5': 'Total Asset Turnover',
}

ALTMAN_COEFFICIENTS = {
    'X1': 1.2,
    'X2': 1.4,
    'X3': 3.3,
    'X4': 0.6,
    'X5': 1.0,
}


def compute_altman_z(row: pd.Series) -> float:
    """
    Compute the Altman Z-Score for a single company row.

    Parameters
    ----------
    row : pd.Series
        A row from the dataset containing all Altman feature columns.

    Returns
    -------
    float
        The Altman Z-Score.
    """
    z = 0.0
    for key, col_name in ALTMAN_FEATURES.items():
        z += ALTMAN_COEFFICIENTS[key] * row[col_name]
    return z


def compute_altman_z_batch(df: pd.DataFrame) -> pd.Series:
    """
    Compute Altman Z-Scores for a DataFrame of companies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all Altman feature columns.

    Returns
    -------
    pd.Series
        Series of Altman Z-Scores.
    """
    z = pd.Series(0.0, index=df.index)
    for key, col_name in ALTMAN_FEATURES.items():
        z += ALTMAN_COEFFICIENTS[key] * df[col_name]
    return z


def classify_zone(z: float) -> str:
    """
    Classify a company based on its Altman Z-Score.

    Parameters
    ----------
    z : float
        Altman Z-Score.

    Returns
    -------
    str
        'Safe', 'Grey', or 'Distress'.
    """
    if z > 2.99:
        return 'Safe'
    elif z > 1.81:
        return 'Grey'
    else:
        return 'Distress'


def get_zone_color(zone: str) -> str:
    """Return a color associated with the Altman zone."""
    return {
        'Safe': '#10b981',
        'Grey': '#f59e0b',
        'Distress': '#ef4444',
    }.get(zone, '#6b7280')


def get_zone_emoji(zone: str) -> str:
    """Return an emoji for the Altman zone."""
    return {
        'Safe': '🟢',
        'Grey': '🟡',
        'Distress': '🔴',
    }.get(zone, '⚪')
