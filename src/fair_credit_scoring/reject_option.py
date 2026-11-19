"""Reject Option Classification post-processor for fairness-aware credit scoring."""

import math
from collections.abc import Sequence

import numpy as np

from ._validation import _as_score_vector, as_binary_vector
from .metrics import (
    DEFAULT_PROFIT_C,
    DEFAULT_PROFIT_P0,
    DEFAULT_PROFIT_P1,
    METRICS,
    auc,
)

FAIRNESS_METRICS = frozenset({"ind", "sp_paper", "eo_gap"})


def _validate_nonbool_float(
    value, name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise ValueError(f"{name} must be a finite real number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and scalar < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return scalar


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be >= 1.")
    return integer


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _evaluate_from_counts(
    name: str,
    tp0: np.ndarray,
    fp0: np.ndarray,
    tp1: np.ndarray,
    fp1: np.ndarray,
    *,
    n01: int,
    n00: int,
    n11: int,
    n10: int,
    n0: int,
    n1: int,
    n_total: int,
    auc_value: float | None = None,
) -> np.ndarray:
    """Vectorized evaluation of any built-in metric from group confusion counts."""
    if name == "profit_eq16":
        tp = tp0 + tp1
        fp = fp0 + fp1
        fn = (n01 + n11) - tp
        p2_val = max(0.0, 1.0 - DEFAULT_PROFIT_P0 - DEFAULT_PROFIT_P1)
        b_expected = DEFAULT_PROFIT_P1 + p2_val / 2.0
        return (DEFAULT_PROFIT_C * (tp - fn) - b_expected * fp) / n_total

    if name == "balanced_accuracy":
        n_pos = n01 + n11
        n_neg = n00 + n10
        if n_pos == 0 or n_neg == 0:
            return np.full(tp0.shape, np.nan, dtype=np.float64)
        tp = tp0 + tp1
        tn = n_neg - (fp0 + fp1)
        return 0.5 * (tp / n_pos + tn / n_neg)

    if name == "approval_rate":
        return ((tp0 + tp1) + (fp0 + fp1)) / n_total

    if name == "auc":
        return np.full(tp0.shape, auc_value, dtype=np.float64)

    if name == "ind":
        if n0 == 0 or n1 == 0:
            return np.full(tp0.shape, np.nan, dtype=np.float64)
        return np.abs((tp1 + fp1) / n1 - (tp0 + fp0) / n0)

    if name in ("sp_paper", "eo_gap"):
        if n00 == 0 or n10 == 0 or n01 == 0 or n11 == 0:
            return np.full(tp0.shape, np.nan, dtype=np.float64)
        fpr_gap = (fp1 / n10) - (fp0 / n00)
        tpr_gap = (tp1 / n11) - (tp0 / n01)
        if name == "sp_paper":
            return 0.5 * np.abs(fpr_gap + tpr_gap)
        return np.maximum(np.abs(fpr_gap), np.abs(tpr_gap))

    if name == "sf":
        pos0 = tp0 + fp0
        pos1 = tp1 + fp1
        valid = (pos0 > 0) & (pos1 > 0)
        out = np.full(tp0.shape, np.nan, dtype=np.float64)
        if np.any(valid):
            out[valid] = np.abs(tp1[valid] / pos1[valid] - tp0[valid] / pos0[valid])
        return out

    if name == "approval_rate_0":
        return (
            (tp0 + fp0) / n0 if n0 > 0 else np.full(tp0.shape, np.nan, dtype=np.float64)
        )
    if name == "approval_rate_1":
        return (
            (tp1 + fp1) / n1 if n1 > 0 else np.full(tp0.shape, np.nan, dtype=np.float64)
        )
    if name == "tpr_0":
        return tp0 / n01 if n01 > 0 else np.full(tp0.shape, np.nan, dtype=np.float64)
    if name == "tpr_1":
        return tp1 / n11 if n11 > 0 else np.full(tp0.shape, np.nan, dtype=np.float64)
    if name == "fpr_0":
        return fp0 / n00 if n00 > 0 else np.full(tp0.shape, np.nan, dtype=np.float64)
    if name == "fpr_1":
        return fp1 / n10 if n10 > 0 else np.full(tp0.shape, np.nan, dtype=np.float64)

    if name in ("ppv_0", "ppv_1"):
        tp_g, fp_g = (tp0, fp0) if name == "ppv_0" else (tp1, fp1)
        pos_g = tp_g + fp_g
        valid = pos_g > 0
        out = np.full(tp0.shape, np.nan, dtype=np.float64)
        if np.any(valid):
            out[valid] = tp_g[valid] / pos_g[valid]
        return out

    raise ValueError(f"Unsupported metric {name!r}.")


class RejectOptionClassifier:
    """Reject Option Classification post-processor generalized to arbitrary thresholds.

    Group convention: ``a == 0`` (unprivileged, age <= 25) receives ``1`` (approved)
    inside the critical region ``|score - threshold_| <= margin_`` (when ``margin_ > 0``),
    while ``a == 1`` (privileged, age > 25) receives ``0`` (rejected). Outside the
    critical region, decisions follow ``score >= threshold_``. When ``margin_ == 0``,
    the critical region is empty and ``predict`` equals ``score >= threshold_``.
    """

    def __init__(
        self,
        *,
        fairness_metric: str = "sp_paper",
        fairness_bound: float = 0.05,
        utility_metric: str = "profit_eq16",
        tune_threshold: bool = True,
        thresholds: Sequence[float] | np.ndarray | None = None,
        n_thresholds: int = 101,
        n_margins: int = 50,
        max_margin: float = 0.5,
    ) -> None:
        if (
            not isinstance(fairness_metric, str)
            or fairness_metric not in FAIRNESS_METRICS
        ):
            raise ValueError(
                f"fairness_metric must be one of {sorted(FAIRNESS_METRICS)}."
            )
        if not isinstance(utility_metric, str) or utility_metric not in METRICS:
            raise ValueError(
                f"utility_metric must be a built-in metric in {sorted(METRICS)}."
            )
        if not isinstance(tune_threshold, (bool, np.bool_)):
            raise ValueError("tune_threshold must be a boolean.")

        self.fairness_metric = fairness_metric
        self.fairness_bound = _validate_nonbool_float(
            fairness_bound, "fairness_bound", minimum=0.0
        )
        self.utility_metric = utility_metric
        self.tune_threshold = bool(tune_threshold)
        self.n_thresholds = _validate_positive_int(n_thresholds, "n_thresholds")
        self.n_margins = _validate_positive_int(n_margins, "n_margins")
        self.max_margin = _validate_nonbool_float(
            max_margin, "max_margin", minimum=0.0, maximum=1.0
        )

        if thresholds is None:
            self.thresholds: tuple[float, ...] | None = None
        else:
            if isinstance(thresholds, (str, bytes, dict)):
                raise ValueError(
                    "thresholds must be a nonempty sequence of finite floats in [0, 1]."
                )
            try:
                items = list(thresholds)
            except TypeError as exc:
                raise ValueError(
                    "thresholds must be a nonempty sequence of finite floats in [0, 1]."
                ) from exc
            if not items:
                raise ValueError("thresholds must be nonempty.")
            self.thresholds = tuple(
                _validate_nonbool_float(t, "threshold", minimum=0.0, maximum=1.0)
                for t in items
            )

        self.base_threshold_: float | None = None
        self.threshold_: float | None = None
        self.margin_: float | None = None
        self.feasible_: bool | None = None
        self.validation_utility_: float | None = None
        self.validation_fairness_: float | None = None
        self.base_validation_utility_: float | None = None
        self.base_validation_fairness_: float | None = None

    def fit(self, score, y, a, threshold: float = 0.5) -> "RejectOptionClassifier":
        """Tune threshold and critical-region margin on validation scores."""
        y_arr = as_binary_vector(y, "y")
        a_arr = as_binary_vector(a, "a")
        if a_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("a must have the same length as y.")
        score_arr = _as_score_vector(score, y_arr.shape[0])
        base_threshold = _validate_nonbool_float(
            threshold, "threshold", minimum=0.0, maximum=1.0
        )
        self.base_threshold_ = base_threshold

        if not self.tune_threshold:
            candidate_thresholds = np.array([base_threshold], dtype=np.float64)
        elif self.thresholds is not None:
            candidate_thresholds = np.unique(
                np.asarray(self.thresholds, dtype=np.float64)
            )
        else:
            grid = np.linspace(0.0, 1.0, self.n_thresholds, dtype=np.float64)
            candidate_thresholds = np.unique(np.append(grid, base_threshold))

        candidate_margins = np.unique(
            np.linspace(0.0, self.max_margin, self.n_margins, dtype=np.float64)
        )
        pos_margins = candidate_margins[1:]
        n_pos_margins = pos_margins.shape[0]

        repaid = y_arr == 1
        defaulted = ~repaid
        g0 = a_arr == 0
        g1 = ~g0
        g0_y1 = g0 & repaid
        g0_y0 = g0 & defaulted
        g1_y1 = g1 & repaid
        g1_y0 = g1 & defaulted

        counts_kwargs = {
            "n01": int(np.count_nonzero(g0_y1)),
            "n00": int(np.count_nonzero(g0_y0)),
            "n11": int(np.count_nonzero(g1_y1)),
            "n10": int(np.count_nonzero(g1_y0)),
            "n0": int(np.count_nonzero(g0)),
            "n1": int(np.count_nonzero(g1)),
            "n_total": y_arr.shape[0],
            "auc_value": (
                float(auc(y_arr, a_arr, score_arr, y_arr))
                if self.utility_metric == "auc"
                else None
            ),
        }

        n_t = candidate_thresholds.shape[0]
        n_m = candidate_margins.shape[0]
        utilities = np.empty((n_t, n_m), dtype=np.float64)
        fairnesses = np.empty((n_t, n_m), dtype=np.float64)

        for t_idx, tau in enumerate(candidate_thresholds):
            base = score_arr >= tau
            tp0_base = int(np.count_nonzero(g0_y1 & base))
            fp0_base = int(np.count_nonzero(g0_y0 & base))
            tp1_base = int(np.count_nonzero(g1_y1 & base))
            fp1_base = int(np.count_nonzero(g1_y0 & base))

            if n_pos_margins > 0:
                dist = np.abs(score_arr - tau)
                idx = np.searchsorted(pos_margins, dist, side="left")
                minlength = n_pos_margins + 1

                tp0_add = np.cumsum(
                    np.bincount(idx[g0_y1 & (~base)], minlength=minlength)[
                        :n_pos_margins
                    ]
                )
                fp0_add = np.cumsum(
                    np.bincount(idx[g0_y0 & (~base)], minlength=minlength)[
                        :n_pos_margins
                    ]
                )
                tp1_sub = np.cumsum(
                    np.bincount(idx[g1_y1 & base], minlength=minlength)[:n_pos_margins]
                )
                fp1_sub = np.cumsum(
                    np.bincount(idx[g1_y0 & base], minlength=minlength)[:n_pos_margins]
                )

                tp0 = np.empty(n_m, dtype=np.int64)
                fp0 = np.empty(n_m, dtype=np.int64)
                tp1 = np.empty(n_m, dtype=np.int64)
                fp1 = np.empty(n_m, dtype=np.int64)
                tp0[0], tp0[1:] = tp0_base, tp0_base + tp0_add
                fp0[0], fp0[1:] = fp0_base, fp0_base + fp0_add
                tp1[0], tp1[1:] = tp1_base, tp1_base - tp1_sub
                fp1[0], fp1[1:] = fp1_base, fp1_base - fp1_sub
            else:
                tp0 = np.array([tp0_base], dtype=np.int64)
                fp0 = np.array([fp0_base], dtype=np.int64)
                tp1 = np.array([tp1_base], dtype=np.int64)
                fp1 = np.array([fp1_base], dtype=np.int64)

            utilities[t_idx] = _evaluate_from_counts(
                self.utility_metric, tp0, fp0, tp1, fp1, **counts_kwargs
            )
            fairnesses[t_idx] = _evaluate_from_counts(
                self.fairness_metric, tp0, fp0, tp1, fp1, **counts_kwargs
            )

        finite_mask = np.isfinite(utilities) & np.isfinite(fairnesses)
        if not np.any(finite_mask):
            raise ValueError(
                "No candidate (threshold, margin) produced finite utility and fairness."
            )

        taus_grid, margins_grid = np.broadcast_arrays(
            candidate_thresholds[:, None], candidate_margins[None, :]
        )
        feasible_mask = finite_mask & (fairnesses <= self.fairness_bound)
        self.feasible_ = bool(np.any(feasible_mask))
        mask = feasible_mask if self.feasible_ else finite_mask
        u = utilities[mask]
        f = fairnesses[mask]
        m = margins_grid[mask]
        t = taus_grid[mask]
        d = np.abs(t - base_threshold)
        keys = (t, d, m, f, -u) if self.feasible_ else (t, d, m, -u, f)
        order = np.lexsort(keys)

        best = int(order[0])
        self.threshold_ = float(t[best])
        self.margin_ = float(m[best])
        self.validation_utility_ = float(u[best])
        self.validation_fairness_ = float(f[best])

        base_decision = (score_arr >= base_threshold).astype(np.int8)
        self.base_validation_utility_ = float(
            METRICS[self.utility_metric](y_arr, a_arr, score_arr, base_decision)
        )
        self.base_validation_fairness_ = float(
            METRICS[self.fairness_metric](y_arr, a_arr, score_arr, base_decision)
        )
        return self

    def _check_fitted(self) -> None:
        if (
            self.threshold_ is None
            or self.margin_ is None
            or self.base_threshold_ is None
        ):
            raise RuntimeError(
                "RejectOptionClassifier must be fitted before prediction or metadata."
            )

    def _validate_predict_inputs(self, score, a) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        a_arr = as_binary_vector(a, "a")
        score_arr = _as_score_vector(score, a_arr.shape[0], length_name="a")
        return score_arr, a_arr

    def _apply_policy(self, score_arr: np.ndarray, a_arr: np.ndarray) -> np.ndarray:
        base = score_arr >= self.threshold_
        if self.margin_ == 0.0:
            return base.astype(np.int8)
        in_critical = np.abs(score_arr - self.threshold_) <= self.margin_
        return np.where(in_critical, a_arr == 0, base).astype(np.int8)

    def predict(self, score, a) -> np.ndarray:
        """Predict binary approval decisions using the fitted threshold and margin."""
        score_arr, a_arr = self._validate_predict_inputs(score, a)
        return self._apply_policy(score_arr, a_arr)

    def predict_before_and_after(self, score, a) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(decision_before, decision_after)`` as binary ``int8`` arrays."""
        score_arr, a_arr = self._validate_predict_inputs(score, a)
        before = (score_arr >= self.base_threshold_).astype(np.int8)
        after = self._apply_policy(score_arr, a_arr)
        return before, after

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable fitted parameters and validation metrics."""
        self._check_fitted()
        return {
            "fairness_metric": self.fairness_metric,
            "fairness_bound": self.fairness_bound,
            "utility_metric": self.utility_metric,
            "tune_threshold": self.tune_threshold,
            "base_threshold": self.base_threshold_,
            "threshold": self.threshold_,
            "margin": self.margin_,
            "feasible": self.feasible_,
            "validation_utility": _finite_or_none(self.validation_utility_),
            "validation_fairness": _finite_or_none(self.validation_fairness_),
            "base_validation_utility": _finite_or_none(self.base_validation_utility_),
            "base_validation_fairness": _finite_or_none(self.base_validation_fairness_),
        }
