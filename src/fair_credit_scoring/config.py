"""Configuration shared by the CLI and Python API."""

import math
import re
import tomllib
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    path: Path
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    loader: str = "csv"

    def __post_init__(self):
        columns = (*self.numeric, *self.categorical)
        if not self.name:
            raise ValueError("Dataset name must be nonempty.")
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Declare at least one feature and list each feature once.")
        if {"row_id", "y", "a"}.intersection(columns):
            raise ValueError("row_id, y and a are reserved and cannot be features.")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    estimator: str
    grid: dict = field(default_factory=dict)

    def __post_init__(self):
        if any(
            not isinstance(values, (list, tuple)) or not values
            for values in self.grid.values()
        ):
            raise ValueError(
                f"Each grid parameter for {self.name!r} needs a nonempty list."
            )

    def candidates(self):
        return [
            dict(zip(self.grid, values, strict=True))
            for values in product(*self.grid.values())
        ]


@dataclass(frozen=True, slots=True)
class RunConfig:
    dataset: DatasetConfig
    models: tuple[ModelConfig, ...]
    run_id: str
    output_dir: Path = Path("outputs")
    seed: int = 42
    threads: int = 1
    thresholds: tuple[float, ...] = (0.4, 0.5, 0.6)
    selection_metric: str = "balanced_accuracy"
    metrics: tuple[str, ...] = ("auc", "approval_rate")
    method: str = "baseline"
    weight: str | None = None
    postprocessor: str | None = None

    def __post_init__(self):
        if not _IDENTIFIER.fullmatch(self.run_id):
            raise ValueError(
                "run_id must start with a letter or digit and use only A-Z, 0-9, _, ., -."
            )
        if (
            not self.models
            or not all(m.name for m in self.models)
            or len({m.name for m in self.models}) != len(self.models)
        ):
            raise ValueError("Declare at least one model with a unique, nonempty name.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError("threads must be a positive integer.")
        if not self.thresholds or any(
            not math.isfinite(t) or not 0 <= t <= 1 for t in self.thresholds
        ):
            raise ValueError("Declare at least one finite threshold in [0, 1].")


def load_config(path: Path) -> RunConfig:
    """Load a TOML file, interpreting both paths relative to the file."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    dataset = raw.pop("dataset")
    models = raw.pop("models")
    dataset["path"] = path.parent / dataset["path"]
    for key in ("numeric", "categorical"):
        if key in dataset:
            dataset[key] = tuple(dataset[key])
    raw["output_dir"] = path.parent / raw.get("output_dir", "../outputs")
    for key in ("thresholds", "metrics"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return RunConfig(
        dataset=DatasetConfig(**dataset),
        models=tuple(ModelConfig(**model) for model in models),
        **raw,
    )
