"""Shared input validation helpers for metrics and estimators."""

import numpy as np


def as_1d_array(values, name: str) -> np.ndarray:
    """Convert values to a nonempty 1-D NumPy array without mutating inputs."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector.")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    return array


def as_binary_vector(values, name: str) -> np.ndarray:
    """Validate a nonempty 1-D binary {0, 1} vector and return an int8 view/copy."""
    array = as_1d_array(values, name)
    if array.dtype == np.bool_:
        return array.astype(np.int8)
    if np.issubdtype(array.dtype, np.integer):
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} must contain only binary 0/1 values.")
        return array.astype(np.int8, copy=False)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array) & ((array == 0.0) | (array == 1.0))):
            raise ValueError(f"{name} must contain only binary 0/1 values.")
        return array.astype(np.int8)
    raise ValueError(f"{name} must be a numeric or boolean binary 0/1 vector.")


def _as_score_vector(
    values, expected_length: int, length_name: str = "y"
) -> np.ndarray:
    """Validate a finite numeric 1-D score vector of expected length as float64."""
    array = as_1d_array(values, "score")
    if array.shape[0] != expected_length:
        raise ValueError(f"score must have the same length as {length_name}.")
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
