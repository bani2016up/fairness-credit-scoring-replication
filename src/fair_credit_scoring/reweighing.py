"""Train-only Reweighing for binary groups and outcomes."""

import numpy as np


def reweighing(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Return the expected/observed joint-frequency ratio for each train row."""
    y, a = np.asarray(y), np.asarray(a)
    if y.ndim != 1 or y.shape != a.shape or not len(y):
        raise ValueError("Reweighing needs two equally sized, nonempty vectors.")
    if not (np.isin(y, [0, 1]).all() and np.isin(a, [0, 1]).all()):
        raise ValueError("Reweighing needs binary 0/1 labels and groups.")
    cells = 2 * a.astype(int) + y.astype(int)
    joint = np.bincount(cells, minlength=4).reshape(2, 2)
    missing = np.argwhere(joint == 0)
    if len(missing):
        combinations = ", ".join(f"(a={g}, y={c})" for g, c in missing)
        raise ValueError(
            f"Reweighing cannot estimate weights for empty cells: {combinations}."
        )
    # Under independence the expected count is the product of the marginals / n.
    expected = np.outer(joint.sum(axis=1, dtype=float), joint.sum(axis=0)) / len(y)
    return (expected / joint).ravel()[cells]
