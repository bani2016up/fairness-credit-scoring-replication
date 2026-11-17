import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_credit_scoring import DatasetConfig, Hooks, ModelConfig, RunConfig, run
from fair_credit_scoring.data import make_splits, prepare_dataset
from fair_credit_scoring.preprocessing import make_preprocessor

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.csv"


@pytest.fixture
def dataset():
    return DatasetConfig(
        name="tiny",
        path=FIXTURE,
        numeric=("income", "balance", "tenure"),
        categorical=("channel", "region"),
    )


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_joint_splits_are_seeded_shared_and_input_order_independent(dataset):
    raw = pd.read_csv(FIXTURE)
    data = prepare_dataset(raw, dataset)
    splits = make_splits(data, dataset.name, seed=37)
    shuffled = make_splits(
        prepare_dataset(raw.sample(frac=1, random_state=3), dataset),
        dataset.name,
        seed=37,
    )
    changed = make_splits(data, dataset.name, seed=38)
    np.testing.assert_array_equal(splits.assignment, shuffled.assignment)
    assert splits.split_id == shuffled.split_id
    assert splits.split_id != changed.split_id
    assert not np.array_equal(splits.assignment, changed.assignment)
    assert len(splits.development) == 48
    assert len(splits.test) == 32
    assert len(splits.folds) == 5
    for rows in (splits.test, *[val for _, val in splits.folds]):
        assert set(zip(data.a[rows], data.y[rows], strict=True)) == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
    for train, val in splits.folds:
        assert not set(train) & set(val)
        assert set(train) | set(val) == set(splits.development)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([val for _, val in splits.folds])), splits.development
    )


@pytest.mark.parametrize("count", [0, 5])
def test_impossible_joint_split_raises(dataset, count):
    raw = pd.read_csv(FIXTURE)
    rows = raw.index[(raw.a == 1) & (raw.y == 0)]
    data = prepare_dataset(raw.drop(rows[count:]), dataset)
    with pytest.raises(ValueError, match=r"\(a=1, y=0\).*at least 6"):
        make_splits(data, dataset.name, seed=37)


def test_feasible_imbalance_keeps_six_member_stratum(dataset):
    raw = pd.read_csv(FIXTURE)
    rows = raw.index[(raw.a == 1) & (raw.y == 0)]
    data = prepare_dataset(raw.drop(rows[6:]), dataset)
    splits = make_splits(data, dataset.name, seed=37)
    for partition in (splits.test, *[val for _, val in splits.folds]):
        assert (
            np.count_nonzero((data.a[partition] == 1) & (data.y[partition] == 0)) == 1
        )


def test_preprocessing_is_train_fitted_and_handles_missing_and_unseen_values():
    train = pd.DataFrame(
        {
            "amount": [1.0, 3.0, np.nan],
            "channel": pd.Series(["branch", "web", None], dtype=object),
        }
    )
    validation = pd.DataFrame(
        {
            "amount": [99.0, np.nan],
            "channel": pd.Series(["heldout", None], dtype=object),
        }
    )
    preprocessor = make_preprocessor(("amount",), ("channel",)).fit(train)
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    assert numeric["impute"].statistics_.tolist() == [2.0]
    assert numeric["scale"].mean_.tolist() == [2.0]
    assert set(categorical["encode"].categories_[0]) == {"branch", "web"}
    transformed = preprocessor.transform(validation)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    np.testing.assert_array_equal(transformed[0, 1:], [0, 0])
    np.testing.assert_array_equal(transformed[1], [0, 1, 0])


def test_runner_fits_preprocessing_on_training_rows_only(
    dataset, tmp_path, monkeypatch
):
    from fair_credit_scoring import runner

    data = prepare_dataset(pd.read_csv(FIXTURE), dataset)
    splits = make_splits(data, dataset.name, seed=37)
    training = (splits.development, *[train for train, _ in splits.folds])
    allowed = {tuple(data.row_id[rows]) for rows in training}
    original = runner.make_preprocessor
    fitted = set()

    def recording_preprocessor(numeric, categorical):
        preprocessor = original(numeric, categorical)
        fit_transform = preprocessor.fit_transform

        def record(features):
            ids = tuple(features.index)
            assert ids in allowed
            fitted.add(ids)
            return fit_transform(features)

        preprocessor.fit_transform = record
        return preprocessor

    monkeypatch.setattr(runner, "make_preprocessor", recording_preprocessor)
    run(
        RunConfig(
            dataset,
            (ModelConfig("lr", "logistic_regression"),),
            "train",
            output_dir=tmp_path,
            seed=37,
        )
    )
    assert fitted == allowed


def test_validation_only_selection_and_train_only_hooks(dataset, tmp_path):
    raw = pd.read_csv(FIXTURE)
    data = prepare_dataset(raw, dataset)
    splits = make_splits(data, dataset.name, seed=37)
    expected = {
        tuple(data.y[rows]) + tuple(data.a[rows])
        for rows in (splits.development, *[train for train, _ in splits.folds])
    }
    seen_train = set()
    fits = []
    metric_calls = []
    postprocess_fits = []

    class ConstantEstimator:
        classes_ = [1, 0]

        def __init__(self, score):
            self.score = score

        def fit(self, x, y, *, a, sample_weight):
            assert tuple(y) + tuple(a) in expected
            np.testing.assert_array_equal(sample_weight, np.ones(len(y)))
            fits.append(self.score)

        def predict_proba(self, x, *, a):
            assert len(a) == x.shape[0]
            return np.tile([self.score, 1 - self.score], (x.shape[0], 1))

    class Postprocessor:
        def fit(self, score, y, a, threshold):
            np.testing.assert_array_equal(y, data.y[splits.development])
            np.testing.assert_array_equal(a, data.a[splits.development])
            np.testing.assert_array_equal(score, np.full(len(y), 0.8))
            assert threshold == 0.7
            postprocess_fits.append(len(y))
            self.threshold = threshold

        def predict(self, score, a):
            return score >= self.threshold

    def weights(y, a):
        seen_train.add(tuple(y) + tuple(a))
        return np.ones(len(y))

    def objective(y, a, score, decision):
        metric_calls.append((len(y), float(score[0]), bool(decision[0])))
        if len(y) == len(splits.test):
            assert len(postprocess_fits) == 1
            return float(score[0] == 0.2 and np.all(decision))
        np.testing.assert_array_equal(y, data.y[splits.development])
        np.testing.assert_array_equal(a, data.a[splits.development])
        return float(score[0] == 0.8 and np.all(decision))

    config = RunConfig(
        dataset=dataset,
        seed=37,
        run_id="hook-test",
        method="custom",
        models=(ModelConfig("constant", "constant", {"score": [0.2, 0.8]}),),
        thresholds=(0.7, 0.9),
        selection_metric="objective",
        metrics=(),
        weight="weights",
        postprocessor="postprocessor",
        output_dir=tmp_path,
    )
    hooks = Hooks(
        loaders={"csv": lambda spec: raw},
        estimators={
            "constant": lambda params, seed, threads: ConstantEstimator(params["score"])
        },
        weights={"weights": weights},
        metrics={"objective": objective},
        postprocessors={"postprocessor": Postprocessor},
    )
    result = run(config, hooks)
    assert seen_train == expected
    assert sorted(fits) == [0.2] * 5 + [0.8] * 6
    assert postprocess_fits == [48]
    assert metric_calls[:4] == [
        (48, 0.2, False),
        (48, 0.2, False),
        (48, 0.8, True),
        (48, 0.8, False),
    ]
    assert metric_calls.count((32, 0.8, True)) == 1
    saved = json.loads((result / "run.json").read_text())
    assert saved["selected"]["constant"]["params"] == {"score": 0.8}
    assert saved["selected"]["constant"]["threshold"] == 0.7


def test_baseline_prediction_schema_and_shared_assignments(dataset, tmp_path):
    config = RunConfig(
        dataset=dataset,
        seed=37,
        run_id="schema-test",
        output_dir=tmp_path,
        models=(
            ModelConfig("lr", "logistic_regression", {"C": [0.5, 1.0]}),
            ModelConfig(
                "rf", "random_forest", {"n_estimators": [7, 4], "max_depth": [3, 5]}
            ),
        ),
    )
    result = run(config)
    validation = read_csv(result / "validation_predictions.csv")
    test = read_csv(result / "test_predictions.csv")
    partitions = {row["row_id"]: row for row in read_csv(result / "splits.csv")}
    assert len(partitions) == 80
    assert len(validation) == 2 * 48
    assert len(test) == 2 * 32
    for partition, records in (("validation", validation), ("test", test)):
        assert list(records[0]) == [
            "row_id",
            "y",
            "a",
            "score",
            "decision",
            "run_id",
            "dataset",
            "model",
            "split_id",
            "partition",
            "validation_fold",
        ]
        assert len({(r["model"], r["row_id"]) for r in records}) == len(records)
        for record in records:
            assignment = partitions[record["row_id"]]
            assert {record["y"], record["a"], record["decision"]} <= {"0", "1"}
            assert 0 <= float(record["score"]) <= 1
            assert record["dataset"] == "tiny"
            assert record["run_id"] == "schema-test"
            assert record["partition"] == partition
            assert record["split_id"] == assignment["split_id"]
            assert record["validation_fold"] == assignment["validation_fold"]
            assert assignment["partition"] == (
                "development" if partition == "validation" else "test"
            )
    saved = json.loads((result / "run.json").read_text())
    assert saved["config"]["seed"] == 37
    assert saved["split_id"] == test[0]["split_id"]
    assert set(saved["selected"]) == {"lr", "rf"}
    with pytest.raises(FileExistsError):
        run(config)


def test_forest_tree_reuse_matches_independent_fits(dataset, tmp_path):
    from dataclasses import replace

    from fair_credit_scoring.hooks import BUILTIN_ESTIMATORS

    config = RunConfig(
        dataset=dataset,
        models=(
            ModelConfig(
                "rf", "random_forest", {"n_estimators": [7, 3, 7], "max_depth": [2, 4]}
            ),
        ),
        run_id="forest",
        output_dir=tmp_path / "reused",
        seed=37,
    )

    def independent_estimator(params, seed, threads):
        return BUILTIN_ESTIMATORS["random_forest"](params, seed, threads)

    reused = run(config)
    independent = run(
        replace(config, output_dir=tmp_path / "independent"),
        Hooks(estimators={"random_forest": independent_estimator}),
    )
    for name in ("validation_predictions.csv", "test_predictions.csv"):
        assert (reused / name).read_bytes() == (independent / name).read_bytes()


def test_joint_split_rejects_overall_ratio_with_insufficient_capacity(dataset):
    raw = pd.read_csv(FIXTURE).groupby(["a", "y"]).head(6)
    data = prepare_dataset(raw, dataset)
    with pytest.raises(ValueError, match="60/40"):
        make_splits(data, dataset.name, seed=37)
