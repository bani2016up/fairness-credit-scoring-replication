import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_credit_scoring import Hooks, ModelConfig, load_config, run
from fair_credit_scoring.data import load_csv, make_splits, prepare_dataset
from fair_credit_scoring.prepare_credit_data import (
    PAY_COLUMNS,
    prepare_credit_data,
)
from fair_credit_scoring.reweighing import reweighing

ROOT = Path(__file__).resolve().parents[1]


def test_german_labels_ids_and_age_boundary():
    prepared, audit = prepare_credit_data(
        "german", b",BAD,age,amount\n001,GOOD,25,50\n002,BAD,26,100\n"
    )
    assert prepared.to_dict("list") == {
        "row_id": ["001", "002"],
        "y": [1, 0],
        "a": [0, 1],
        "AGE": [0, 1],
        "CREDIT_AMNT": [50, 100],
    }
    assert audit["excluded_rows"] == 0
    assert audit["before"] == audit["after"]
    assert audit["after"]["age_25"] == 1


def test_taiwan_filters_nonpositive_amount_and_recodes_categories():
    raw = pd.DataFrame(
        {
            "ID": ["08", "09", "10", "11"],
            "AGE": [25, 35, 26, 28],
            "default.payment.next.month": [0, 1, 1, 0],
            "BILL_AMT1": [20, 30, 40, 50],
            "PAY_AMT1": [1, 2, 40, 51],
            "SEX": [1, 2, 1, 2],
            "EDUCATION": [0, 2, 3, 4],
            "MARRIAGE": [0, 2, 1, 3],
            **dict.fromkeys(PAY_COLUMNS, [-2, -1, 0, 1]),
        }
    )
    with io.StringIO() as stream:
        raw.to_csv(stream, index=False)
        prepared, audit = prepare_credit_data("taiwan", stream.getvalue().encode())
    assert prepared["row_id"].tolist() == ["08", "09"]
    assert prepared["y"].tolist() == [1, 0]
    assert prepared["AGE"].tolist() == [0, 1]
    assert prepared["CREDIT_AMNT"].tolist() == [19, 28]
    assert prepared["EDUCATION"].tolist() == ["others", "university"]
    assert prepared["MARRIAGE"].tolist() == ["others", "single"]
    assert prepared["PAY_0"].tolist() == [0, 0]
    assert audit["before"]["rows"] == 4
    assert audit["after"]["rows"] == 2
    assert audit["excluded"]["nonpositive_credit_amount"] == 2


def test_reweighing_matches_hand_calculation_and_preserves_marginals():
    a = np.repeat([0, 1], [4, 6])
    y = np.repeat([0, 1, 0, 1], [1, 3, 4, 2])
    weights = reweighing(y, a)
    np.testing.assert_allclose(
        weights, np.repeat([2, 2 / 3, 3 / 4, 3 / 2], [1, 3, 4, 2])
    )
    np.testing.assert_allclose(np.bincount(2 * a + y, weights=weights), [2, 2, 3, 3])
    with pytest.raises(ValueError, match=r"empty cells: \(a=0, y=0\)"):
        reweighing(y[1:], a[1:])
    with pytest.raises(ValueError, match="binary"):
        reweighing([2, 1], [0, 1])


def test_resampling_is_seeded_and_evaluates_the_original_rows(tmp_path):
    config = load_config(ROOT / "configs/tiny.toml")
    config = replace(
        config,
        run_id="resampled",
        output_dir=tmp_path,
        models=(ModelConfig("recording", "recording"),),
        weight="reweighing",
        weight_mode="resample",
    )
    data = prepare_dataset(load_csv(config.dataset), config.dataset)
    splits = make_splits(data, config.dataset.name, config.seed)
    train_ids = [train for train, _ in splits.folds] + [splits.development]
    fitted = []

    class RecordingEstimator:
        classes_ = np.array([0, 1])

        def fit(self, x, y, *, a, sample_weight):
            assert sample_weight is None
            i = len(fitted)
            rows = train_ids[i]
            weights = reweighing(data.y[rows], data.a[rows])
            rng = np.random.default_rng([config.seed, i])
            expected = rng.choice(
                rows, len(rows), replace=True, p=weights / weights.sum()
            )
            assert len(x.shape) == 2 and x.shape[0] == len(rows)
            np.testing.assert_array_equal(y, data.y[expected])
            np.testing.assert_array_equal(a, data.a[expected])
            assert len(np.unique(expected)) < len(rows)
            fitted.append(i)

        def predict_proba(self, x, *, a):
            return np.full((x.shape[0], 2), 0.5)

    result = run(
        config,
        Hooks(estimators={"recording": lambda *_: RecordingEstimator()}),
    )
    assert len(fitted) == 6
    for part, rows in (("validation", splits.development), ("test", splits.test)):
        observed = pd.read_csv(
            result / f"{part}_predictions.csv", dtype={"row_id": str}
        )
        assert observed["row_id"].tolist() == data.row_id[rows].tolist()
        np.testing.assert_array_equal(observed["y"], data.y[rows])
        np.testing.assert_array_equal(observed["a"], data.a[rows])
