"""Prepared data and shared, joint-stratified split assignments."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import DatasetConfig

N_FOLDS = 5


@dataclass(frozen=True, slots=True)
class Dataset:
    row_id: np.ndarray
    y: np.ndarray
    a: np.ndarray
    features: pd.DataFrame


@dataclass(frozen=True, slots=True)
class Splits:
    split_id: str
    assignment: np.ndarray
    development: np.ndarray
    test: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def load_csv(config: DatasetConfig) -> pd.DataFrame:
    return pd.read_csv(
        config.path,
        usecols=["row_id", "y", "a", *config.numeric, *config.categorical],
        dtype={column: "string" for column in ("row_id", *config.categorical)},
        keep_default_na=False,
        na_values=[""],
        float_precision="round_trip",
    )


def prepare_dataset(frame: pd.DataFrame, config: DatasetConfig) -> Dataset:
    columns = ["row_id", "y", "a", *config.numeric, *config.categorical]
    if not frame.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")
    if missing := set(columns).difference(frame.columns):
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}.")
    row_ids = frame["row_id"].astype("string")
    if (
        row_ids.isna().any()
        or row_ids.str.strip().eq("").any()
        or not row_ids.is_unique
    ):
        raise ValueError("row_id must contain unique, nonblank values.")
    order = row_ids.argsort().to_numpy()
    labels = []
    for column in ("y", "a"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be binary (0/1) with no missing values.")
        labels.append(frame[column].to_numpy(dtype=np.int8)[order])
    features = pd.DataFrame(index=pd.Index(row_ids.iloc[order], name="row_id"))
    for column in config.numeric:
        values = pd.to_numeric(frame[column]).to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if np.isinf(values).any():
            raise ValueError(f"Numeric feature {column!r} contains infinity.")
        features[column] = values[order]
    for column in config.categorical:
        values = frame[column].astype("string").to_numpy(dtype=object, na_value=None)
        features[column] = pd.Series(values[order], index=features.index, dtype=object)
    return Dataset(row_ids.to_numpy(dtype=object)[order], *labels, features)


def _test_counts(counts: np.ndarray) -> np.ndarray:
    # Round to 40% overall, reserving a test row and five dev rows per stratum.
    for index, count in enumerate(counts):
        if count < N_FOLDS + 1:
            raise ValueError(
                f"Stratum (a={index // 2}, y={index % 2}) has {count} rows and needs "
                f"at least {N_FOLDS + 1} for five development folds and a test set."
            )
    total = (2 * counts.sum() + 4) // 5
    maximum = counts - N_FOLDS
    if total > maximum.sum() or total < len(counts):
        raise ValueError(
            "The 60/40 split must leave at least five development rows per stratum."
        )
    quota = 0.4 * counts
    test_counts = np.clip(np.rint(quota).astype(int), 1, maximum)
    difference = total - test_counts.sum()
    step = 1 if difference > 0 else -1
    for _ in range(abs(difference)):
        eligible = test_counts < maximum if step == 1 else test_counts > 1
        penalty = np.where(eligible, step * (test_counts - quota), np.inf)
        test_counts[penalty.argmin()] += step
    return test_counts


def make_splits(data: Dataset, name: str, seed: int) -> Splits:
    """Return positional indices in stable row_id order and a content-derived ID."""
    strata = 2 * data.a + data.y
    test_counts = _test_counts(np.bincount(strata, minlength=4))
    rng = np.random.default_rng(seed)
    assignment = np.full(len(strata), -1, dtype=np.int8)
    for stratum, count in enumerate(test_counts):
        sample = rng.permutation(np.flatnonzero(strata == stratum))
        assignment[sample[count:]] = 0
    development = np.flatnonzero(assignment >= 0)
    test = np.flatnonzero(assignment < 0)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for fold, (train, validation) in enumerate(
        splitter.split(development, strata[development])
    ):
        train, validation = development[train], development[validation]
        assignment[validation] = fold
        folds.append((train, validation))
    identity = json.dumps(
        [name, seed, data.row_id.tolist()], ensure_ascii=True, separators=(",", ":")
    )
    digest = hashlib.sha256(identity.encode())
    digest.update(strata.tobytes())
    digest.update(assignment.tobytes())
    return Splits(digest.hexdigest(), assignment, development, test, tuple(folds))
