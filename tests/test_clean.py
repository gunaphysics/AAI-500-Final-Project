import numpy as np
import pandas as pd

from src.data.clean import (
    drop_identifier,
    flag_impossible_values,
    missingness_report,
)


def test_drop_identifier_removes_patient_id():
    df = pd.DataFrame({"patient_id": [1, 2], "age": [50, 60]})
    out = drop_identifier(df)
    assert "patient_id" not in out.columns
    assert list(out.columns) == ["age"]


def test_drop_identifier_is_safe_when_missing():
    df = pd.DataFrame({"age": [50, 60]})
    out = drop_identifier(df)
    assert list(out.columns) == ["age"]


def test_flag_impossible_replaces_out_of_range_with_nan():
    df = pd.DataFrame({"creatinine": [1.0, -0.5, 2.0, 99.0]})
    out = flag_impossible_values(df)
    assert np.isnan(out.loc[1, "creatinine"])
    assert np.isnan(out.loc[3, "creatinine"])
    assert out.loc[0, "creatinine"] == 1.0
    assert out.loc[2, "creatinine"] == 2.0


def test_flag_impossible_is_case_insensitive():
    df = pd.DataFrame({"BMI": [25.0, 200.0]})
    out = flag_impossible_values(df)
    assert out.loc[0, "BMI"] == 25.0
    assert np.isnan(out.loc[1, "BMI"])


def test_missingness_report_shape():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, 2, 3]})
    rep = missingness_report(df)
    assert set(rep.columns) == {"missing", "pct"}
    assert rep.loc["a", "missing"] == 1
    assert rep.loc["b", "missing"] == 0
