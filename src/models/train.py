"""Candidate classifiers compared in this project."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.models.baseline import majority_baseline


def candidate_models() -> dict:
    """Return the model lineup.

    The proposal calls for interpretable models (logistic regression, decision
    tree) and machine-learning classifiers (random forest, gradient boosting)
    compared against a majority-class baseline.
    """
    return {
        "baseline": majority_baseline(),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Wrap a preprocessor and an estimator into a single Pipeline."""
    return Pipeline([("preprocess", preprocessor), ("model", model)])
