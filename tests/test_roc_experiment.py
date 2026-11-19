import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_credit_scoring import DatasetConfig, ModelConfig, RunConfig, load_config, run
from fair_credit_scoring.handoff import load_predictions
from fair_credit_scoring.metrics import METRICS, compute_profit_eq16
from fair_credit_scoring.roc_experiment import (
    DEFAULT_FAIRNESS_BOUNDS,
    DEFAULT_FAIRNESS_METRICS,
    DEFAULT_PRIMARY_BOUND,
    PRIMARY_BOUNDS,
    SUMMARY_COLUMNS,
    main,
    run_roc_experiment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
FIXTURE = Path(__file__).parent / "fixtures" / "tiny.csv"


def _make_synthetic_baseline_run(
    tmp_path: Path,
    *,
    selection_metric: str = "profit_eq16",
    dataset_name: str = "synthetic-tiny",
    dataset_path: Path = FIXTURE,
    postprocessor: str | None = None,
    postprocessor_params: dict | None = None,
    run_id: str = "tiny-baseline-profit",
) -> Path:
    cfg = RunConfig(
        dataset=DatasetConfig(
            name=dataset_name,
            path=dataset_path,
            numeric=("income", "balance", "tenure"),
            categorical=("channel", "region"),
        ),
        models=(
            ModelConfig("lr", "logistic_regression", {"C": [0.5, 1.0]}),
            ModelConfig(
                "rf", "random_forest", {"n_estimators": [16], "max_depth": [4]}
            ),
        ),
        run_id=run_id,
        output_dir=tmp_path / "outputs",
        seed=37,
        thresholds=(0.0, 0.25, 0.5, 0.75, 1.0),
        selection_metric=selection_metric,
        postprocessor=postprocessor,
        postprocessor_params={}
        if postprocessor_params is None
        else postprocessor_params,
        metrics=tuple(METRICS.keys()),
    )
    return run(cfg)


def test_roc_toml_configs_match_baselines():
    tiny_base = load_config(CONFIGS_DIR / "tiny.toml")
    tiny_roc = load_config(CONFIGS_DIR / "tiny-roc.toml")
    assert tiny_roc.run_id == "tiny-roc"
    assert tiny_roc.method == "reject_option"
    assert tiny_roc.postprocessor == "reject_option"
    assert tiny_roc.selection_metric == "profit_eq16"
    assert tiny_roc.seed == tiny_base.seed == 37
    assert tiny_roc.dataset == tiny_base.dataset
    assert tiny_roc.models == tiny_base.models
    assert tiny_roc.thresholds == tuple(round(0.1 * i, 1) for i in range(11))
    assert tiny_roc.postprocessor_params == {
        "fairness_metric": "sp_paper",
        "fairness_bound": 0.05,
        "utility_metric": "profit_eq16",
    }

    for prefix, expected_bound in (("german", 0.01), ("taiwan", 0.005)):
        base = load_config(CONFIGS_DIR / f"{prefix}-baseline-profit.toml")
        roc = load_config(CONFIGS_DIR / f"{prefix}-roc-profit.toml")
        assert roc.run_id == f"{prefix}-roc-profit"
        assert roc.method == "reject_option"
        assert roc.postprocessor == "reject_option"
        assert roc.selection_metric == "profit_eq16"
        assert roc.dataset == base.dataset
        assert roc.seed == base.seed
        assert roc.thresholds == base.thresholds
        assert roc.metrics == base.metrics
        assert roc.models == base.models
        assert roc.postprocessor_params == {
            "fairness_metric": "sp_paper",
            "fairness_bound": expected_bound,
            "utility_metric": "profit_eq16",
            "tune_threshold": True,
            "n_thresholds": 101,
            "n_margins": 50,
            "max_margin": 0.5,
        }
        assert tiny_roc.metrics == base.metrics


def test_roc_experiment_on_synthetic_baseline_run(tmp_path, monkeypatch):
    from fair_credit_scoring import roc_experiment as roc_mod

    baseline_dir = _make_synthetic_baseline_run(tmp_path)
    meta = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    expected_split_id = meta["split_id"]

    val_base = pd.read_csv(
        baseline_dir / "validation_predictions.csv",
        keep_default_na=False,
        dtype={"row_id": str},
        float_precision="round_trip",
    )
    test_base = pd.read_csv(
        baseline_dir / "test_predictions.csv",
        keep_default_na=False,
        dtype={"row_id": str},
        float_precision="round_trip",
    )

    val_scores_by_model = {
        m: val_base[val_base["model"] == m]["score"].to_numpy()
        for m in meta["selected"]
    }
    fit_lengths: list[int] = []
    original_fit = roc_mod.RejectOptionClassifier.fit

    def spy_fit(self, score, y, a, threshold=0.5):
        fit_lengths.append(len(y))
        assert len(y) == 48
        assert any(
            np.array_equal(np.asarray(score), s) for s in val_scores_by_model.values()
        )
        return original_fit(self, score, y, a, threshold=threshold)

    monkeypatch.setattr(roc_mod.RejectOptionClassifier, "fit", spy_fit)

    summary_csv = tmp_path / "roc-sweep-summary.csv"
    preds_dir = tmp_path / "roc-predictions"
    main(
        [
            "--run-dirs",
            str(baseline_dir),
            "--output-csv",
            str(summary_csv),
            "--predictions-dir",
            str(preds_dir),
        ]
    )

    expected_fits = (
        len(meta["selected"])
        * len(DEFAULT_FAIRNESS_METRICS)
        * len(DEFAULT_FAIRNESS_BOUNDS)
    )
    assert len(fit_lengths) == expected_fits

    assert summary_csv.is_file()
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == SUMMARY_COLUMNS
        rows = list(reader)

    assert len(rows) == expected_fits * 2
    assert {r["split_id"] for r in rows} == {expected_split_id}
    assert {r["feasible"] for r in rows} <= {"True", "False"}

    for row in rows:
        model_name = row["model"]
        assert float(row["base_threshold"]) == meta["selected"][model_name]["threshold"]
        for col in (
            "selected_threshold",
            "margin",
            "delta_profit_vs_baseline",
            *METRICS,
        ):
            val = row[col]
            assert val != ""
            if val != "NaN":
                assert math.isfinite(float(val))

        part_df = val_base if row["partition"] == "validation" else test_base
        sub = part_df[part_df["model"] == model_name]
        base_profit = compute_profit_eq16(
            sub["y"].to_numpy(), sub["decision"].to_numpy()
        )
        assert float(row["delta_profit_vs_baseline"]) == pytest.approx(
            float(row["profit_eq16"]) - base_profit, abs=1e-12
        )

    for partition, base_df in (("validation", val_base), ("test", test_base)):
        exported = load_predictions(
            preds_dir / "synthetic-tiny" / f"{partition}_predictions.csv"
        )
        assert list(exported["row_id"]) == list(base_df["row_id"])
        assert set(exported["split_id"]) == {expected_split_id}
        assert set(exported["fairness_metric"]) == {"sp_paper"}
        assert set(exported["fairness_bound"]) == {DEFAULT_PRIMARY_BOUND}
        np.testing.assert_array_equal(
            exported["decision_before"].to_numpy(), base_df["decision"].to_numpy()
        )
        np.testing.assert_array_equal(
            exported["decision"].to_numpy(), exported["decision_after"].to_numpy()
        )


def test_roc_experiment_rejects_non_profit_selection_metric(tmp_path):
    bad_dir = _make_synthetic_baseline_run(
        tmp_path, selection_metric="balanced_accuracy"
    )
    with pytest.raises(ValueError, match="selection_metric='profit_eq16'"):
        run_roc_experiment([bad_dir], tmp_path / "summary.csv")


def test_roc_experiment_rejects_postprocessed_run(tmp_path):
    roc_dir = _make_synthetic_baseline_run(
        tmp_path,
        run_id="tiny-roc",
        postprocessor="reject_option",
        postprocessor_params={
            "fairness_metric": "sp_paper",
            "fairness_bound": 0.05,
            "utility_metric": "profit_eq16",
        },
    )
    with pytest.raises(ValueError, match="postprocessor=None"):
        run_roc_experiment([roc_dir], tmp_path / "summary.csv")


def test_roc_experiment_rejects_mismatched_and_non_integer_decisions(tmp_path):
    baseline_dir = _make_synthetic_baseline_run(tmp_path / "flipped")
    test_csv = baseline_dir / "test_predictions.csv"
    df = load_predictions(test_csv)
    df.loc[0, "decision"] = 1 - int(df.loc[0, "decision"])
    df.to_csv(test_csv, index=False)

    with pytest.raises(ValueError, match="does not match threshold"):
        run_roc_experiment([baseline_dir], tmp_path / "summary.csv")

    bad_dtype_dir = _make_synthetic_baseline_run(tmp_path / "nonint")
    bad_csv = bad_dtype_dir / "validation_predictions.csv"
    bad_df = load_predictions(bad_csv)
    bad_df["decision"] = bad_df["decision"].astype(str)
    bad_df.loc[0, "decision"] = "1.0"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="decision column"):
        run_roc_experiment([bad_dtype_dir], tmp_path / "summary2.csv")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fairness_bounds": [True, 0.05]},
        {"fairness_bounds": [np.bool_(False), 0.05]},
        {"fairness_bounds": [0.05, 0.05]},
        {"fairness_bounds": []},
        {"fairness_bounds": [-0.01]},
        {"fairness_bounds": [float("nan")]},
        {"fairness_bounds": [float("inf")]},
        {"fairness_metrics": "ind"},
        {"fairness_metrics": ["ind", "ind"]},
        {"fairness_metrics": []},
        {"fairness_metrics": ["unknown_metric"]},
    ],
)
def test_roc_experiment_rejects_invalid_sweep_grids(tmp_path, kwargs):
    baseline_dir = _make_synthetic_baseline_run(tmp_path)
    with pytest.raises(ValueError):
        run_roc_experiment([baseline_dir], tmp_path / "summary.csv", **kwargs)


def test_roc_experiment_preserves_leading_zero_and_na_row_ids(tmp_path):
    raw_df = pd.read_csv(FIXTURE, keep_default_na=False, dtype={"row_id": str})
    expected_ids = ["NA", *(f"{idx:03d}" for idx in range(1, len(raw_df)))]
    raw_df["row_id"] = expected_ids
    custom_fixture = tmp_path / "tiny_custom_ids.csv"
    raw_df.to_csv(custom_fixture, index=False)

    baseline_dir = _make_synthetic_baseline_run(
        tmp_path, dataset_name="german", dataset_path=custom_fixture
    )
    preds_dir = tmp_path / "preds"
    run_roc_experiment(
        [baseline_dir],
        tmp_path / "summary.csv",
        predictions_dir=preds_dir,
        fairness_metrics=("sp_paper",),
        fairness_bounds=(0.05, PRIMARY_BOUNDS["german"]),
    )

    seen_ids: set[str] = set()
    for partition in ("validation", "test"):
        base_part = load_predictions(baseline_dir / f"{partition}_predictions.csv")
        out_path = preds_dir / "german" / f"{partition}_predictions.csv"
        exported = load_predictions(out_path)
        assert list(exported["row_id"]) == list(base_part["row_id"])
        assert set(exported["fairness_metric"]) == {"sp_paper"}
        assert set(exported["fairness_bound"]) == {PRIMARY_BOUNDS["german"]}
        with out_path.open(newline="", encoding="utf-8") as stream:
            raw_ids = [row["row_id"] for row in csv.DictReader(stream)]
        assert raw_ids == list(base_part["row_id"])
        seen_ids.update(raw_ids)
    assert seen_ids == set(expected_ids)
    assert {"NA", "001", "002"} <= seen_ids


def test_roc_experiment_outputs_explicit_nan_for_zero_group_approvals(tmp_path):
    run_dir = tmp_path / "zero_approval_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    split_id = "synthetic-split-001"
    base_threshold = 0.9

    rows = []
    specs = [
        ("001", 0, 0, 0.25),
        ("002", 0, 0, 0.20),
        ("003", 0, 0, 0.15),
        ("004", 1, 0, 0.10),
        ("005", 0, 1, 0.91),
        ("006", 1, 1, 0.95),
        ("007", 1, 1, 0.96),
        ("008", 1, 1, 0.98),
    ]
    for row_id, y_val, a_val, score_val in specs:
        rows.append(
            {
                "row_id": row_id,
                "y": y_val,
                "a": a_val,
                "score": score_val,
                "decision": int(score_val >= base_threshold),
                "run_id": "zero-app",
                "dataset": "synthetic-zero",
                "model": "lr",
                "split_id": split_id,
                "partition": "validation",
                "validation_fold": "0",
            }
        )
    val_df = pd.DataFrame(rows)
    test_df = val_df.copy()
    test_df["row_id"] = [f"t{r['row_id']}" for r in rows]
    test_df["partition"] = "test"
    test_df["validation_fold"] = ""

    val_df.to_csv(run_dir / "validation_predictions.csv", index=False)
    test_df.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "split_id": split_id,
                "config": {
                    "selection_metric": "profit_eq16",
                    "postprocessor": None,
                    "dataset": {"name": "synthetic-zero"},
                },
                "selected": {"lr": {"threshold": base_threshold}},
            }
        ),
        encoding="utf-8",
    )

    summary_csv = tmp_path / "nan-summary.csv"
    run_roc_experiment(
        [run_dir],
        summary_csv,
        fairness_metrics=("sp_paper",),
        fairness_bounds=(1.0,),
    )
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        out_rows = list(csv.DictReader(stream))
    assert len(out_rows) == 2
    for out_row in out_rows:
        assert out_row["approval_rate_0"] == "0.0"
        assert out_row["ppv_0"] == "NaN"
        assert out_row["sf"] == "NaN"
        assert math.isfinite(float(out_row["ppv_1"]))


def test_roc_experiment_error_paths_and_cli_exit_codes(tmp_path):
    missing_dir = tmp_path / "missing-run"
    with pytest.raises(FileNotFoundError):
        run_roc_experiment([missing_dir], tmp_path / "out.csv")
    with pytest.raises(SystemExit) as exc:
        main(
            ["--run-dirs", str(missing_dir), "--output-csv", str(tmp_path / "out.csv")]
        )
    assert exc.value.code == 1

    baseline_dir = _make_synthetic_baseline_run(tmp_path / "split_err")
    test_csv = baseline_dir / "test_predictions.csv"
    df = load_predictions(test_csv)
    df["split_id"] = "wrong-split-id"
    df.to_csv(test_csv, index=False)

    with pytest.raises(ValueError, match="split_id"):
        run_roc_experiment([baseline_dir], tmp_path / "out.csv")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--run-dirs",
                str(baseline_dir),
                "--output-csv",
                str(tmp_path / "out.csv"),
            ]
        )
    assert exc.value.code == 1

    valid_dir = _make_synthetic_baseline_run(tmp_path / "valid")
    with pytest.raises(ValueError, match="predictions_metric"):
        run_roc_experiment(
            [valid_dir],
            tmp_path / "out.csv",
            predictions_dir=tmp_path / "preds",
            predictions_metric="ind",
            fairness_metrics=("sp_paper",),
        )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--run-dirs",
                str(valid_dir),
                "--output-csv",
                str(tmp_path / "out.csv"),
                "--predictions-dir",
                str(tmp_path / "preds"),
                "--predictions-metric",
                "invalid_metric",
            ]
        )
    assert exc.value.code == 2

    for bad_bound in ("0.99", "nan"):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "--run-dirs",
                    str(valid_dir),
                    "--output-csv",
                    str(tmp_path / "out.csv"),
                    "--predictions-dir",
                    str(tmp_path / "preds"),
                    "--predictions-bound",
                    bad_bound,
                ]
            )
        assert exc.value.code == 1
