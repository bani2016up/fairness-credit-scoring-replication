"""Fit, select and evaluate models against shared partitions."""

import json
import math
import os
import platform
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import RunConfig
from .data import load_csv, make_splits, prepare_dataset
from .hooks import BUILTIN_ESTIMATORS, BUILTIN_METRICS, BUILTIN_WEIGHTS, Hooks
from .preprocessing import make_preprocessor, prepare_fold


def _lookup(registry, name, kind):
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown {kind} {name!r}. Available: {', '.join(sorted(registry))}."
        ) from None


def _probability(estimator, features, a):
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or set(classes) != {0, 1}:
        raise ValueError("Estimator classes_ must contain both binary labels.")
    probabilities = np.asarray(estimator.predict_proba(features, a=a))
    if probabilities.shape != (features.shape[0], 2):
        raise ValueError(
            "Estimator predict_proba must return one row per input and two class columns."
        )
    score = np.asarray(
        probabilities[:, np.flatnonzero(classes == 1).item()], dtype=float
    )
    if not np.all(np.isfinite(score) & (score >= 0) & (score <= 1)):
        raise ValueError("Repayment probabilities must be finite and in [0, 1].")
    return score


def _fit_score(
    factory, params, config, x_train, y, a, weights, x_evaluation, a_evaluation
):
    estimator = factory(params, config.seed, config.threads)
    estimator.fit(x_train, y, a=a, sample_weight=weights)
    return _probability(estimator, x_evaluation, a_evaluation)


def _weights(weight, y, a):
    if weight is None:
        return None
    values = np.asarray(weight(y, a), dtype=float)
    if (
        values.shape != y.shape
        or not np.all(np.isfinite(values) & (values >= 0))
        or not np.any(values > 0)
    ):
        raise ValueError(
            "Training weights must be a finite nonnegative vector with positive mass."
        )
    return values


def _training_data(config, weight, x_train, y, a, split_index):
    weights = _weights(weight, y, a)
    if config.weight_mode == "sample_weight":
        return x_train, y, a, weights

    # One reproducible draw per training partition, shared by all models and grids.
    rng = np.random.default_rng([config.seed, split_index])
    probabilities = weights / weights.max()
    rows = rng.choice(
        len(y), size=len(y), replace=True, p=probabilities / probabilities.sum()
    )
    sampled_y = y[rows]
    if not np.any(sampled_y == 0) or not np.any(sampled_y == 1):
        raise ValueError("Weighted resampling produced only one training class.")
    return x_train[rows], sampled_y, a[rows], None


def _select(candidates, scores, thresholds, objective, y, a):
    best = None
    best_value = -math.inf
    for params, score in zip(candidates, scores, strict=True):
        for threshold in thresholds:
            value = float(objective(y, a, score, score >= threshold))
            if not math.isfinite(value):
                raise ValueError(
                    "The selection metric must be finite for every candidate."
                )
            if value > best_value:
                best = (params, score, threshold, value)
                best_value = value
    return best


def _decisions(score, a, threshold, postprocessor):
    decision = (
        score >= threshold
        if postprocessor is None
        else np.asarray(postprocessor.predict(score, a))
    )
    if decision.shape != score.shape or not np.all(np.isin(decision, [0, 1])):
        raise ValueError(
            "Postprocessor must return one binary approval decision per row."
        )
    return decision.astype(np.int8)


def _predictions(data, rows, scores, decisions, config, model, splits, partition):
    return pd.DataFrame(
        {
            "row_id": data.row_id[rows],
            "y": data.y[rows],
            "a": data.a[rows],
            "score": scores,
            "decision": decisions,
            "run_id": config.run_id,
            "dataset": config.dataset.name,
            "model": model.name,
            "split_id": splits.split_id,
            "partition": partition,
            "validation_fold": splits.assignment[rows]
            if partition == "validation"
            else "",
        }
    )


def _evaluate(metrics, y, a, score, decision):
    values = {
        name: float(metric(y, a, score, decision)) for name, metric in metrics.items()
    }
    return {
        name: value if math.isfinite(value) else None for name, value in values.items()
    }


def _forest_scores(factory, grid, config, x_train, y, a, weights, x_val, val_a):
    # Match the forest's native layouts once per fold instead of once per fit/predict.
    if hasattr(x_train, "tocsc"):
        x_train = x_train.astype(np.float32, copy=False).tocsc()
        x_val = x_val.astype(np.float32, copy=False).tocsr()
    else:
        x_train = np.asarray(x_train, dtype=np.float32)
        x_val = np.asarray(x_val, dtype=np.float32)

    groups = {}
    for index, params in enumerate(grid):
        count = params.get("n_estimators")
        if (
            config.threads != 1
            or type(count) is not int
            or count < 1
            or "warm_start" in params
            or params.get("oob_score")
            or params.get("class_weight") == "balanced"
        ):
            yield (
                index,
                _fit_score(
                    factory, params, config, x_train, y, a, weights, x_val, val_a
                ),
            )
            continue
        structure = {
            key: value for key, value in params.items() if key != "n_estimators"
        }
        key = json.dumps(structure, sort_keys=True)
        groups.setdefault(key, []).append((count, index, params))

    for candidates in groups.values():
        if len(candidates) == 1:
            _, index, params = candidates[0]
            yield (
                index,
                _fit_score(
                    factory, params, config, x_train, y, a, weights, x_val, val_a
                ),
            )
            continue
        candidates.sort()
        estimator = factory(
            candidates[0][2] | {"warm_start": True}, config.seed, config.threads
        )
        previous_count = 0
        for count, index, _ in candidates:
            if count != previous_count:
                # With a fixed seed, a larger forest starts with the same tree sequence.
                estimator.classifier.set_params(n_estimators=count)
                estimator.fit(x_train, y, a=a, sample_weight=weights)
                score = _probability(estimator, x_val, val_a)
                previous_count = count
            yield index, score


def _oof_scores(config, data, splits, factories, weight, candidates):
    dev = splits.development
    dataset = config.dataset
    scores = [np.empty((len(grid), len(dev))) for grid in candidates]
    for fold, (train, validation) in enumerate(splits.folds):
        preprocessor = make_preprocessor(
            dataset.numeric, dataset.categorical, scaling=dataset.scaling
        )
        x_train, x_val = prepare_fold(preprocessor, data.features, train, validation)
        y, a, val_a = data.y[train], data.a[train], data.a[validation]
        x_train, y, a, weights = _training_data(config, weight, x_train, y, a, fold)
        val_positions = np.searchsorted(dev, validation)
        for factory, grid, predictions in zip(
            factories, candidates, scores, strict=True
        ):
            if factory is BUILTIN_ESTIMATORS["random_forest"] and len(grid) > 1:
                for index, score in _forest_scores(
                    factory, grid, config, x_train, y, a, weights, x_val, val_a
                ):
                    predictions[index, val_positions] = score
                continue
            for index, params in enumerate(grid):
                predictions[index, val_positions] = _fit_score(
                    factory, params, config, x_train, y, a, weights, x_val, val_a
                )
        del x_train, x_val
    return scores


def _run_models(config, data, splits, factories, weight, post_factory, metrics, output):
    dev, test = splits.development, splits.test
    dev_y, dev_a = data.y[dev], data.a[dev]
    objective = metrics[config.selection_metric]
    dataset = config.dataset
    selection = {}
    policies = []
    candidates = [model.candidates() for model in config.models]
    scores = _oof_scores(config, data, splits, factories, weight, candidates)

    for i, model in enumerate(config.models):
        params, val_score, threshold, objective_value = _select(
            candidates[i], scores[i], config.thresholds, objective, dev_y, dev_a
        )
        postprocessor = None if post_factory is None else post_factory()
        if postprocessor is not None:
            postprocessor.fit(val_score, dev_y, dev_a, threshold)
        policies.append(postprocessor)
        val_decision = _decisions(val_score, dev_a, threshold, postprocessor)
        selection[model.name] = {
            "params": params,
            "threshold": threshold,
            "selection_value": objective_value,
            "validation_metrics": _evaluate(
                metrics, dev_y, dev_a, val_score, val_decision
            ),
        }
        _predictions(
            data, dev, val_score, val_decision, config, model, splits, "validation"
        ).to_csv(
            output / "validation_predictions.csv", index=False, mode="a", header=i == 0
        )
    del scores, val_score, val_decision

    preprocessor = make_preprocessor(
        dataset.numeric, dataset.categorical, scaling=dataset.scaling
    )
    x_train, x_test = prepare_fold(preprocessor, data.features, dev, test)
    x_train, train_y, train_a, weights = _training_data(
        config, weight, x_train, dev_y, dev_a, len(splits.folds)
    )
    test_y, test_a = data.y[test], data.a[test]
    for i, (model, factory, postprocessor) in enumerate(
        zip(config.models, factories, policies, strict=True)
    ):
        settings = selection[model.name]
        test_score = _fit_score(
            factory,
            settings["params"],
            config,
            x_train,
            train_y,
            train_a,
            weights,
            x_test,
            test_a,
        )
        test_decision = _decisions(
            test_score, test_a, settings["threshold"], postprocessor
        )
        settings["test_metrics"] = _evaluate(
            metrics, test_y, test_a, test_score, test_decision
        )
        _predictions(
            data, test, test_score, test_decision, config, model, splits, "test"
        ).to_csv(output / "test_predictions.csv", index=False, mode="a", header=i == 0)
    return selection


def run(config: RunConfig, hooks: Hooks | None = None) -> Path:
    """Execute a run and return the path of the published run directory."""
    hooks = Hooks() if hooks is None else hooks
    target = config.output_dir / config.run_id
    if target.exists():
        raise FileExistsError(f"Run directory already exists: {target}")
    loader = _lookup(
        {"csv": load_csv} | hooks.loaders, config.dataset.loader, "dataset loader"
    )
    factories = [
        _lookup(BUILTIN_ESTIMATORS | hooks.estimators, model.estimator, "estimator")
        for model in config.models
    ]
    weight = (
        None
        if config.weight is None
        else _lookup(BUILTIN_WEIGHTS | hooks.weights, config.weight, "weight hook")
    )
    post_factory = (
        None
        if config.postprocessor is None
        else _lookup(hooks.postprocessors, config.postprocessor, "postprocessor")
    )
    metric_registry = BUILTIN_METRICS | hooks.metrics
    metrics = {
        name: _lookup(metric_registry, name, "metric")
        for name in dict.fromkeys((config.selection_metric, *config.metrics))
    }
    data = prepare_dataset(loader(config.dataset), config.dataset)
    splits = make_splits(data, config.dataset.name, config.seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        output = Path(temporary)
        with threadpool_limits(limits=config.threads):
            selection = _run_models(
                config, data, splits, factories, weight, post_factory, metrics, output
            )
        folds = pd.array(splits.assignment, dtype="Int8")
        folds[splits.test] = pd.NA
        pd.DataFrame(
            {
                "row_id": data.row_id,
                "partition": np.where(splits.assignment < 0, "test", "development"),
                "validation_fold": folds,
                "split_id": splits.split_id,
            }
        ).to_csv(output / "splits.csv", index=False)
        metadata = {
            "config": asdict(config),
            "split_id": splits.split_id,
            "selected": selection,
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "threadpoolctl")
                },
            },
        }
        (output / "run.json").write_text(
            json.dumps(metadata, indent=2, default=os.fspath, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output.rename(target)
    return target
