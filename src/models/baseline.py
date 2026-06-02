"""Majority-class baseline classifier."""
from sklearn.dummy import DummyClassifier

from src.config import RANDOM_STATE


def majority_baseline() -> DummyClassifier:
    """Always predicts the most frequent class in the training data."""
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
