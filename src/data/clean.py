"""Cleaning utilities for the heart failure readmission dataset.

The proposal's initial EDA flagged: ~3% missing in BMI / sodium / creatinine,
impossible negative creatinine values, and several extreme clinical outliers.
This module codifies clinically plausible ranges and reports missingness.
"""
import numpy as np
import pandas as pd

from src.config import ID_COL

# Clinically plausible ranges. Values outside these are treated as data-entry
# errors and converted to NaN so downstream imputation handles them.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 110),
    "bmi": (10, 70),
    "bnp": (0, 5000),
    "sodium": (110, 160),
    "creatinine": (0, 15),
    "heart_rate": (30, 200),
    "systolic_bp": (60, 250),
    "diastolic_bp": (30, 150),
}


def drop_identifier(df: pd.DataFrame, id_col: str = ID_COL) -> pd.DataFrame:
    """Drop the patient identifier column if present."""
    return df.drop(columns=[id_col], errors="ignore")


def flag_impossible_values(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Replace clinically impossible values with NaN.

    Only operates on columns that exist in the frame; range keys are matched
    case-insensitively against column names.
    """
    ranges = ranges if ranges is not None else PLAUSIBLE_RANGES
    df = df.copy()
    lower = {c.lower(): c for c in df.columns}
    for key, (lo, hi) in ranges.items():
        col = lower.get(key.lower())
        if col is None:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        df.loc[mask, col] = np.nan
    return df


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness summary, sorted descending by percentage."""
    n = len(df)
    missing = df.isna().sum()
    return pd.DataFrame(
        {"missing": missing, "pct": (missing / n * 100).round(2)}
    ).sort_values("pct", ascending=False)
