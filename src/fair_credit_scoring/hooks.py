"""Extension contracts for dataset and fairness components."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import DatasetConfig
from .metrics import METRICS
from .prejudice_remover import prejudice_remover_factory
from .reweighing import reweighing


class Estimator(Protocol):
    classes_: np.ndarray

    def fit(self, x, y, *, a, sample_weight) -> None: ...
    def predict_proba(self, x, *, a) -> np.ndarray: ...


class Postprocessor(Protocol):
    def fit(self, score, y, a, threshold) -> None: ...
    def predict(self, score, a) -> np.ndarray: ...


Loader = Callable[[DatasetConfig], pd.DataFrame]
Weight = Callable[[np.ndarray, np.ndarray], np.ndarray]
EstimatorFactory = Callable[[dict, int, int], Estimator]
Metric = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]


@dataclass(frozen=True, slots=True)
class Hooks:
    """Treat pipeline-owned arrays and matrices passed to callables as read-only."""

    loaders: dict[str, Loader] = field(default_factory=dict)
    weights: dict[str, Weight] = field(default_factory=dict)
    estimators: dict[str, EstimatorFactory] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    postprocessors: dict[str, Callable[[], Postprocessor]] = field(default_factory=dict)


class SklearnEstimator:
    """Adapt a classifier with scikit-learn's fit and predict_proba interface."""

    def __init__(self, classifier):
        self.classifier = classifier

    @property
    def classes_(self):
        return self.classifier.classes_

    def fit(self, x, y, *, a, sample_weight):
        self.classifier.fit(x, y, sample_weight=sample_weight)

    def predict_proba(self, x, *, a):
        return self.classifier.predict_proba(x)


def _logistic_regression(params, seed, threads):
    return SklearnEstimator(
        LogisticRegression(**({"max_iter": 1000} | params), random_state=seed)
    )


def _random_forest(params, seed, threads):
    return SklearnEstimator(
        RandomForestClassifier(**params, random_state=seed, n_jobs=threads)
    )


BUILTIN_ESTIMATORS = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "prejudice_remover": prejudice_remover_factory,
}

BUILTIN_WEIGHTS = {"reweighing": reweighing}

BUILTIN_METRICS = dict(METRICS)
