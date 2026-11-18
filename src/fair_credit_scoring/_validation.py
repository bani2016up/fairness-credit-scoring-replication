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
