"""Shared classification, fairness, and economic metrics."""

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from ._validation import as_1d_array, as_binary_vector

DEFAULT_PROFIT_C = 0.2664
DEFAULT_PROFIT_P0 = 0.55
DEFAULT_PROFIT_P1 = 0.10


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Binary outcome/decision counts with y=1 (repaid) and decision=1 (approved)."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


@dataclass(frozen=True, slots=True)
class GroupRates:
    """Group-level confusion counts and derived approval/error/precision rates."""

    counts: ConfusionCounts
    approval_rate: float
    tpr: float
    fpr: float
    ppv: float


def _as_score_vector(values, expected_length: int) -> np.ndarray:
    array = as_1d_array(values, "score")
    if array.shape[0] != expected_length:
        raise ValueError("score must have the same length as y.")
    if not (
        array.dtype == np.bool_
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("score must be a finite numeric 1-D vector.")
    scores = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError("score must contain only finite values.")
    return scores


def _validate_metric_inputs(
    y, a, score, decision
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_arr = as_binary_vector(y, "y")
    a_arr = as_binary_vector(a, "a")
    decision_arr = as_binary_vector(decision, "decision")
    if a_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("a must have the same length as y.")
    if decision_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("decision must have the same length as y.")
    score_arr = _as_score_vector(score, y_arr.shape[0])
    return y_arr, a_arr, score_arr, decision_arr


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def _counts_from_binary(y: np.ndarray, decision: np.ndarray) -> ConfusionCounts:
    if y.size == 0:
        return ConfusionCounts(tp=0, fp=0, fn=0, tn=0)
    repaid = y == 1
    approved = decision == 1
    tp = int(np.count_nonzero(repaid & approved))
    fp = int(np.count_nonzero((~repaid) & approved))
    fn = int(np.count_nonzero(repaid & (~approved)))
    tn = int(np.count_nonzero((~repaid) & (~approved)))
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def confusion_counts(y, decision) -> ConfusionCounts:
    """Compute TP, FP, FN, and TN counts for binary y and decision vectors."""
    y_arr = as_binary_vector(y, "y")
    decision_arr = as_binary_vector(decision, "decision")
    if decision_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("decision must have the same length as y.")
    return _counts_from_binary(y_arr, decision_arr)


def group_confusion_rates(y, a, decision) -> dict[int, GroupRates]:
    """Compute per-group confusion counts and approval, TPR, FPR, and PPV rates.

    Any rate whose denominator is zero evaluates to ``math.nan``.
    """
    y_arr = as_binary_vector(y, "y")
    a_arr = as_binary_vector(a, "a")
    decision_arr = as_binary_vector(decision, "decision")
    if a_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("a must have the same length as y.")
    if decision_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("decision must have the same length as y.")

    rates: dict[int, GroupRates] = {}
    for group in (0, 1):
        mask = a_arr == group
        counts = _counts_from_binary(y_arr[mask], decision_arr[mask])
        rates[group] = GroupRates(
            counts=counts,
            approval_rate=_safe_ratio(counts.tp + counts.fp, counts.n),
            tpr=_safe_ratio(counts.tp, counts.tp + counts.fn),
            fpr=_safe_ratio(counts.fp, counts.fp + counts.tn),
            ppv=_safe_ratio(counts.tp, counts.tp + counts.fp),
        )
    return rates


def _validate_profit_parameters(c: float, p0: float, p1: float) -> tuple[float, float]:
    for name, value in (("c", c), ("p0", p0), ("p1", p1)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be a finite real number.")
    c_val, p0_val, p1_val = float(c), float(p0), float(p1)
    if not math.isfinite(c_val) or c_val < 0.0:
        raise ValueError("c must be finite and nonnegative.")
    if not math.isfinite(p0_val) or not 0.0 <= p0_val <= 1.0:
        raise ValueError("p0 must be finite and in [0, 1].")
    if not math.isfinite(p1_val) or not 0.0 <= p1_val <= 1.0:
        raise ValueError("p1 must be finite and in [0, 1].")
    if p0_val + p1_val > 1.0 + 1e-12:
        raise ValueError("p0 + p1 must not exceed 1.")
    p2_val = max(0.0, 1.0 - p0_val - p1_val)
    b_expected = p1_val + p2_val / 2.0
    return c_val, b_expected


def compute_profit_eq16(
    y,
    decision,
    *,
    c: float = DEFAULT_PROFIT_C,
    p0: float = DEFAULT_PROFIT_P0,
    p1: float = DEFAULT_PROFIT_P1,
) -> float:
    """Compute unit-loan per-application profit from Eq. (16) of Kozodoi et al.

    Returns ``(c * (TP - FN) - B_expected * FP) / N`` where
    ``B_expected = p1 + (1 - p0 - p1) / 2``.
    """
    c_val, b_expected = _validate_profit_parameters(c, p0, p1)
    counts = confusion_counts(y, decision)
    return float((c_val * (counts.tp - counts.fn) - b_expected * counts.fp) / counts.n)


def auc(y, a, score, decision) -> float:
    """Compute ROC AUC from continuous repayment scores; NaN if a class is absent."""
    y_arr, _, score_arr, _ = _validate_metric_inputs(y, a, score, decision)
    if not (np.any(y_arr == 0) and np.any(y_arr == 1)):
        return math.nan
    return float(roc_auc_score(y_arr, score_arr))


def balanced_accuracy(y, a, score, decision) -> float:
    """Compute balanced accuracy 0.5 * (TPR + TNR); NaN if a class is absent."""
    y_arr, _, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
    counts = _counts_from_binary(y_arr, decision_arr)
    tpr = _safe_ratio(counts.tp, counts.tp + counts.fn)
    tnr = _safe_ratio(counts.tn, counts.tn + counts.fp)
    if math.isnan(tpr) or math.isnan(tnr):
        return math.nan
    return 0.5 * (tpr + tnr)


def approval_rate(y, a, score, decision) -> float:
    """Compute fraction of approved applications (decision == 1)."""
    _, _, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
    return float(np.mean(decision_arr))


def profit_eq16(y, a, score, decision) -> float:
    """Metric hook wrapper for Eq. (16) per-application profit with default parameters."""
    y_arr, _, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
    return compute_profit_eq16(y_arr, decision_arr)


def ind(y, a, score, decision) -> float:
    """Independence gap: ``|P(decision=1 | a=1) - P(decision=1 | a=0)|``."""
    y_arr, a_arr, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
    rates = group_confusion_rates(y_arr, a_arr, decision_arr)
    r0, r1 = rates[0].approval_rate, rates[1].approval_rate
    if math.isnan(r0) or math.isnan(r1):
        return math.nan
    return abs(r1 - r0)


def sp_paper(y, a, score, decision) -> float:
    """Paper separation metric: ``0.5 * |(FPR_1 - FPR_0) + (TPR_1 - TPR_0)|``."""
    y_arr, a_arr, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
    rates = group_confusion_rates(y_arr, a_arr, decision_arr)
    values = (rates[0].fpr, rates[1].fpr, rates[0].tpr, rates[1].tpr)
    if any(math.isnan(v) for v in values):
        return math.nan
    return 0.5 * abs((rates[1].fpr - rates[0].fpr) + (rates[1].tpr - rates[0].tpr))


def sf(y, a, score, decision) -> float:
    """Sufficiency gap: ``|PPV_1 - PPV_0|``."""
    y_arr, a_arr, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
    rates = group_confusion_rates(y_arr, a_arr, decision_arr)
    p0, p1 = rates[0].ppv, rates[1].ppv
    if math.isnan(p0) or math.isnan(p1):
        return math.nan
    return abs(p1 - p0)


def eo_gap(y, a, score, decision) -> float:
    """Equalized odds max gap: ``max(|FPR_1 - FPR_0|, |TPR_1 - TPR_0|)``."""
    y_arr, a_arr, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
    rates = group_confusion_rates(y_arr, a_arr, decision_arr)
    values = (rates[0].fpr, rates[1].fpr, rates[0].tpr, rates[1].tpr)
    if any(math.isnan(v) for v in values):
        return math.nan
    return max(abs(rates[1].fpr - rates[0].fpr), abs(rates[1].tpr - rates[0].tpr))


def _group_rate_metric(group: int, attribute: str):
    def metric(y, a, score, decision) -> float:
        y_arr, a_arr, _, decision_arr = _validate_metric_inputs(y, a, score, decision)
        rates = group_confusion_rates(y_arr, a_arr, decision_arr)
        return float(getattr(rates[group], attribute))

    metric.__name__ = f"{attribute}_{group}"
    return metric


METRICS = {
    "auc": auc,
    "balanced_accuracy": balanced_accuracy,
    "approval_rate": approval_rate,
    "profit_eq16": profit_eq16,
    "ind": ind,
    "sp_paper": sp_paper,
    "sf": sf,
    "eo_gap": eo_gap,
    "approval_rate_0": _group_rate_metric(0, "approval_rate"),
    "approval_rate_1": _group_rate_metric(1, "approval_rate"),
    "tpr_0": _group_rate_metric(0, "tpr"),
    "tpr_1": _group_rate_metric(1, "tpr"),
    "fpr_0": _group_rate_metric(0, "fpr"),
    "fpr_1": _group_rate_metric(1, "fpr"),
    "ppv_0": _group_rate_metric(0, "ppv"),
    "ppv_1": _group_rate_metric(1, "ppv"),
}


def evaluate_metrics(
    y, a, score, decision, names: tuple[str, ...] | list[str] | None = None
) -> dict[str, float]:
    """Evaluate a suite of built-in metrics (all built-ins by default)."""
    selected = tuple(METRICS) if names is None else tuple(names)
    results: dict[str, float] = {}
    for name in selected:
        if name not in METRICS:
            raise ValueError(
                f"Unknown metric {name!r}. Available: {', '.join(sorted(METRICS))}."
            )
        results[name] = float(METRICS[name](y, a, score, decision))
    return results
