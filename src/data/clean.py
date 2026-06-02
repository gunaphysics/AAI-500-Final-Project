import numpy as np
import pandas as pd

from src.config import ID_COL

# Anything outside these ranges is almost certainly a data-entry error, so we
# null it out and let imputation deal with it later.
PLAUSIBLE_RANGES = {
    "age": (18, 110),
    "bmi": (10, 70),
    "bnp": (0, 5000),
    "sodium": (110, 160),
    "creatinine": (0, 15),
    "heart_rate": (30, 200),
    "systolic_bp": (60, 250),
}


def drop_identifier(df, id_col=ID_COL):
    return df.drop(columns=[id_col], errors="ignore")


def flag_impossible_values(df, ranges=PLAUSIBLE_RANGES):
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    for key, (lo, hi) in ranges.items():
        col = cols.get(key.lower())
        if col is None:
            continue
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def missingness_report(df):
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing": missing, "pct": pct})
    return report.sort_values("pct", ascending=False)
