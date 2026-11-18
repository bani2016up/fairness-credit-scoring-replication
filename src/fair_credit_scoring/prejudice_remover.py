"""Independent NumPy/SciPy implementation of Kamishima's Prejudice Remover."""

import math

import numpy as np
import scipy.sparse
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from ._validation import as_1d_array, as_binary_vector

# Clamp probabilities only near float64 underflow in log/logit.
_PROB_EPS = 1e-300


def _is_numeric_dtype(dtype) -> bool:
    return (
        dtype == np.bool_
        or np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def _validate_nonbool_float(value, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    if allow_zero:
        if scalar < 0.0:
            raise ValueError(f"{name} must be nonnegative.")
    elif scalar <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return scalar


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _validate_features(x):
    if isinstance(x, (str, bytes)):
        raise ValueError("x must be a 2-D numeric matrix.")
    if scipy.sparse.issparse(x):
        if x.ndim != 2:
            raise ValueError("x must be a 2-D matrix.")
        n_samples, n_features = x.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("x must have at least one sample and one feature.")
        if not _is_numeric_dtype(x.dtype):
            raise ValueError("x must contain only numeric or boolean values.")
        matrix = x.tocsr(copy=False).astype(np.float64, copy=False)
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("x must contain only finite values.")
        return matrix

    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError("x must be a 2-D matrix.")
    n_samples, n_features = array.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError("x must have at least one sample and one feature.")
    if not _is_numeric_dtype(array.dtype):
        raise ValueError("x must contain only numeric or boolean values.")
    dense = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(dense)):
        raise ValueError("x must contain only finite values.")
    return dense


def _validate_weights(sample_weight, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    array = as_1d_array(sample_weight, "sample_weight")
    if array.shape[0] != n_samples:
        raise ValueError("sample_weight must have the same length as x.")
    if not _is_numeric_dtype(array.dtype):
        raise ValueError("sample_weight must be a finite nonnegative numeric vector.")
    weights = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(weights) & (weights >= 0.0)):
        raise ValueError("sample_weight must be finite and nonnegative.")
    return weights


def _split_training_groups(x, y, a, sample_weight):
    features = _validate_features(x)
    n_samples = features.shape[0]
    y_arr = as_binary_vector(y, "y")
    a_arr = as_binary_vector(a, "a")
    if y_arr.shape[0] != n_samples:
        raise ValueError("y must have the same number of rows as x.")
    if a_arr.shape[0] != n_samples:
        raise ValueError("a must have the same number of rows as x.")
    weights = _validate_weights(sample_weight, n_samples)

    mask0 = a_arr == 0
    mask1 = a_arr == 1
    if not np.any(mask0) or not np.any(mask1):
        raise ValueError(
            "PrejudiceRemover requires training samples from both groups a=0 and a=1."
        )
    v0 = weights[mask0]
    v1 = weights[mask1]
    w0 = float(v0.sum())
    w1 = float(v1.sum())
    if w0 <= 0.0 or w1 <= 0.0:
        raise ValueError(
            "PrejudiceRemover requires positive training weight mass in both groups."
        )

    return (
        features.shape[1],
        features[mask0],
        y_arr[mask0].astype(np.float64),
        v0,
        w0,
        features[mask1],
        y_arr[mask1].astype(np.float64),
        v1,
        w1,
    )


def _group_loss_and_grad(
    theta: np.ndarray,
    x0,
    y0: np.ndarray,
    v0: np.ndarray,
    w0: float,
    x1,
    y1: np.ndarray,
    v1: np.ndarray,
    w1: float,
    eta: float,
    l2_strength: float,
) -> tuple[float, np.ndarray]:
    n_features = x0.shape[1]
    params = theta.reshape(2, n_features + 1)
    beta0, b0 = params[0, :-1], float(params[0, -1])
    beta1, b1 = params[1, :-1], float(params[1, -1])

    z0 = x0 @ beta0 + b0
    z1 = x1 @ beta1 + b1
    p0 = expit(z0)
    one_minus_p0 = expit(-z0)
    p1 = expit(z1)
    one_minus_p1 = expit(-z1)

    bce = float(
        np.dot(v0, np.logaddexp(0.0, z0) - y0 * z0)
        + np.dot(v1, np.logaddexp(0.0, z1) - y1 * z1)
    )
    l2_term = 0.5 * l2_strength * float(np.dot(theta, theta))

    w_total = w0 + w1
    if eta == 0.0:
        fair_term = 0.0
        fair_diff_0 = 0.0
        fair_diff_1 = 0.0
    else:
        q0 = float(np.dot(v0, p0) / w0)
        one_minus_q0 = float(np.dot(v0, one_minus_p0) / w0)
        q1 = float(np.dot(v1, p1) / w1)
        one_minus_q1 = float(np.dot(v1, one_minus_p1) / w1)

        pi0 = w0 / w_total
        pi1 = w1 / w_total
        q_pos = max(pi0 * q0 + pi1 * q1, _PROB_EPS)
        q_neg = max(pi0 * one_minus_q0 + pi1 * one_minus_q1, _PROB_EPS)
        q0_pos, q0_neg = max(q0, _PROB_EPS), max(one_minus_q0, _PROB_EPS)
        q1_pos, q1_neg = max(q1, _PROB_EPS), max(one_minus_q1, _PROB_EPS)

        log_q_pos, log_q_neg = math.log(q_pos), math.log(q_neg)
        log_q0_pos, log_q0_neg = math.log(q0_pos), math.log(q0_neg)
        log_q1_pos, log_q1_neg = math.log(q1_pos), math.log(q1_neg)

        mutual_info = pi0 * (
            q0 * (log_q0_pos - log_q_pos) + one_minus_q0 * (log_q0_neg - log_q_neg)
        ) + pi1 * (
            q1 * (log_q1_pos - log_q_pos) + one_minus_q1 * (log_q1_neg - log_q_neg)
        )
        fair_term = eta * w_total * mutual_info

        # Cross-group dI/dq terms cancel identically: sum_g W_g * (-q_g/q + (1-q_g)/(1-q)) = 0.
        logit_q = log_q_pos - log_q_neg
        fair_diff_0 = eta * ((log_q0_pos - log_q0_neg) - logit_q)
        fair_diff_1 = eta * ((log_q1_pos - log_q1_neg) - logit_q)

    res0 = v0 * ((p0 - y0) + fair_diff_0 * p0 * one_minus_p0)
    res1 = v1 * ((p1 - y1) + fair_diff_1 * p1 * one_minus_p1)

    grad = np.empty_like(params)
    grad[0, :-1] = x0.T @ res0 + l2_strength * beta0
    grad[0, -1] = float(res0.sum()) + l2_strength * b0
    grad[1, :-1] = x1.T @ res1 + l2_strength * beta1
    grad[1, -1] = float(res1.sum()) + l2_strength * b1
    return bce + fair_term + l2_term, grad.ravel()


def objective_and_gradient(
    theta: np.ndarray,
    x,
    y,
    a,
    sample_weight=None,
    *,
    eta: float = 1.0,
    l2_strength: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Evaluate the unscaled Prejudice Remover objective J and its gradient."""
    eta_val = _validate_nonbool_float(eta, "eta", allow_zero=True)
    l2_val = _validate_nonbool_float(l2_strength, "l2_strength", allow_zero=False)
    n_features, x0, y0, v0, w0, x1, y1, v1, w1 = _split_training_groups(
        x, y, a, sample_weight
    )
    theta_arr = np.asarray(theta, dtype=np.float64)
    expected_shape = (2 * (n_features + 1),)
    if theta_arr.shape != expected_shape or not np.all(np.isfinite(theta_arr)):
        raise ValueError(
            f"theta must be a finite 1-D vector of length {expected_shape[0]}."
        )
    return _group_loss_and_grad(
        theta_arr, x0, y0, v0, w0, x1, y1, v1, w1, eta_val, l2_val
    )


class PrejudiceRemover:
    """Group-parameterized logistic regression with mutual-information fairness penalty.

    Fits separate parameter vectors ``(beta_g, b_g)`` for ``a=0`` and ``a=1`` under the
    summed objective:

        J = sum_i v_i * BCE(y_i, p_i) + eta * W * I(p; a)
            + (l2_strength / 2) * sum_g (||beta_g||^2 + b_g^2)

    where L2 regularization intentionally applies to both feature weights and intercepts
    to match Kamishima's reference formulation.
    """

    classes_ = np.array([0, 1], dtype=np.int8)

    def __init__(
        self,
        *,
        eta: float = 1.0,
        l2_strength: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        seed: int | None = None,
        threads: int = 1,
    ) -> None:
        self.eta = _validate_nonbool_float(eta, "eta", allow_zero=True)
        self.l2_strength = _validate_nonbool_float(
            l2_strength, "l2_strength", allow_zero=False
        )
        self.max_iter = _validate_positive_int(max_iter, "max_iter")
        self.tol = _validate_nonbool_float(tol, "tol", allow_zero=False)
        if seed is not None:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, (int, np.integer))
                or not 0 <= int(seed) < 2**32
            ):
                raise ValueError("seed must be an integer in [0, 2**32) or None.")
            seed = int(seed)
        self.seed = seed
        self.threads = _validate_positive_int(threads, "threads")

        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_iter_: int | None = None
        self.loss_: float | None = None
        self.normalized_loss_: float | None = None
        self.converged_: bool = False
        self.optimizer_status_: int | None = None
        self.optimizer_message_: str | None = None

    def fit(self, x, y, *, a, sample_weight=None):
        """Fit group-specific logistic coefficients using L-BFGS-B."""
        n_features, x0, y0, v0, w0, x1, y1, v1, w1 = _split_training_groups(
            x, y, a, sample_weight
        )
        w_total = w0 + w1
        theta0 = np.zeros(2 * (n_features + 1), dtype=np.float64)

        def scaled_objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            loss, grad = _group_loss_and_grad(
                theta,
                x0,
                y0,
                v0,
                w0,
                x1,
                y1,
                v1,
                w1,
                self.eta,
                self.l2_strength,
            )
            return loss / w_total, grad / w_total

        with threadpool_limits(limits=self.threads):
            result = minimize(
                scaled_objective,
                theta0,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": self.max_iter,
                    "maxfun": max(15000, 20 * self.max_iter),
                    "ftol": min(1e-12, self.tol * 1e-4),
                    "gtol": self.tol,
                },
            )

        self.n_features_in_ = n_features
        self.n_iter_ = int(result.nit)
        self.normalized_loss_ = float(result.fun)
        self.loss_ = float(result.fun * w_total)
        self.converged_ = bool(result.success)
        self.optimizer_status_ = int(result.status)
        self.optimizer_message_ = str(result.message)

        if not result.success:
            raise RuntimeError(
                f"PrejudiceRemover failed to converge (status={result.status}, "
                f"nit={result.nit}): {result.message}"
            )

        params = np.asarray(result.x, dtype=np.float64).reshape(2, n_features + 1)
        self.coef_ = params[:, :-1].copy()
        self.intercept_ = params[:, -1].copy()
        return self

    def predict_proba(self, x, *, a) -> np.ndarray:
        """Predict [P(y=0 | x, a), P(y=1 | x, a)] using group-specific weights."""
        if self.coef_ is None or self.intercept_ is None or self.n_features_in_ is None:
            raise RuntimeError("PrejudiceRemover must be fitted before predict_proba.")
        features = _validate_features(x)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        a_arr = as_binary_vector(a, "a")
        if a_arr.shape[0] != features.shape[0]:
            raise ValueError("a must have the same number of rows as x.")

        logits = np.empty(features.shape[0], dtype=np.float64)
        for group in (0, 1):
            mask = a_arr == group
            if np.any(mask):
                logits[mask] = (
                    features[mask] @ self.coef_[group] + self.intercept_[group]
                )
        positive = expit(logits)
        return np.column_stack((1.0 - positive, positive))


def prejudice_remover_factory(
    params: dict, seed: int, threads: int
) -> PrejudiceRemover:
    """Create a PrejudiceRemover instance for the pipeline Estimator contract."""
    if "seed" in params or "threads" in params:
        raise ValueError("seed and threads are controlled by the runner config.")
    return PrejudiceRemover(**params, seed=seed, threads=threads)
