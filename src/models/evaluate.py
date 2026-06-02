import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score(model, X, y):
    preds = model.predict(X)
    # ROC-AUC wants positive-class scores; the dummy baseline has no
    # predict_proba so fall back to its hard predictions.
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


def compare(models, X, y):
    rows = {name: score(m, X, y) for name, m in models.items()}
    return pd.DataFrame(rows).T.sort_values("roc_auc", ascending=False)
