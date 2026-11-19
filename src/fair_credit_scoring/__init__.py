"""Research code for the Fairness in Credit Scoring replication."""

from .config import DatasetConfig, ModelConfig, RunConfig, load_config
from .hooks import Hooks, SklearnEstimator
from .metrics import (
    ConfusionCounts,
    GroupRates,
    compute_profit_eq16,
    confusion_counts,
    evaluate_metrics,
    group_confusion_rates,
)
from .prejudice_remover import PrejudiceRemover
from .reject_option import RejectOptionClassifier
from .runner import run

__all__ = [
    "ConfusionCounts",
    "DatasetConfig",
    "GroupRates",
    "Hooks",
    "ModelConfig",
    "PrejudiceRemover",
    "RejectOptionClassifier",
    "RunConfig",
    "SklearnEstimator",
    "compute_profit_eq16",
    "confusion_counts",
    "evaluate_metrics",
    "group_confusion_rates",
    "load_config",
    "run",
]
