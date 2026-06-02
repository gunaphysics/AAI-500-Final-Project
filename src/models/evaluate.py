"""Binary-classification evaluation helpers."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score(model, X, y) -> dict:
    """Return the standard binary-classification metrics for one model."""
    preds = model.predict(X)
    # ROC-AUC needs scores for the positive class. Baselines without
    # predict_proba fall back to hard predictions.
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = preds
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
    }


def compare(models: dict, X, y) -> pd.DataFrame:
    """Score a dictionary of fitted models and rank them by ROC-AUC."""
    rows = {name: score(m, X, y) for name, m in models.items()}
    return pd.DataFrame(rows).T.sort_values("roc_auc", ascending=False)
