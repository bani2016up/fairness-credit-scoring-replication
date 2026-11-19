"""Stage B Reject Option Classification fairness-profit sweep experiment."""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._validation import _as_score_vector
from .handoff import load_predictions, write_csv
from .metrics import METRICS, compute_profit_eq16, evaluate_metrics
from .reject_option import (
    FAIRNESS_METRICS,
    RejectOptionClassifier,
    _validate_nonbool_float,
)

DEFAULT_RUN_DIRS = (
    Path("outputs/german-baseline-profit"),
    Path("outputs/taiwan-baseline-profit"),
)
DEFAULT_OUTPUT_CSV = Path("outputs/roc-sweep-summary.csv")

DEFAULT_FAIRNESS_METRICS = ("sp_paper", "ind", "eo_gap")
DEFAULT_FAIRNESS_BOUNDS = (0.10, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0)
PRIMARY_BOUNDS = {"german": 0.01, "taiwan": 0.005}
DEFAULT_PRIMARY_BOUND = 0.05

SUMMARY_COLUMNS = (
    "dataset",
    "model",
    "fairness_metric",
    "fairness_bound",
    "partition",
    "split_id",
    "base_threshold",
    "selected_threshold",
    "margin",
    "feasible",
    "delta_profit_vs_baseline",
    *METRICS.keys(),
)

REQUIRED_PREDICTION_COLUMNS = frozenset(
    {
        "row_id",
        "y",
        "a",
        "score",
        "decision",
        "model",
        "split_id",
        "partition",
        "validation_fold",
    }
)


@dataclass(frozen=True, slots=True)
class _BaselineModelData:
    dataset: str
    model: str
    split_id: str
    base_threshold: float
    validation_df: pd.DataFrame
    test_df: pd.DataFrame


def _parse_int_decisions(series: pd.Series, *, context: str) -> np.ndarray:
    raw = series.to_numpy()
    if raw.dtype == np.bool_ or np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"{context} must contain integer 0/1 values.")
    if np.issubdtype(raw.dtype, np.integer):
        decisions = raw
    else:
        parsed: list[int] = []
        for item in raw:
            if isinstance(item, (bool, np.bool_, float, np.floating)):
                raise ValueError(f"{context} must contain integer 0/1 values.")
            if isinstance(item, (int, np.integer)):
                parsed.append(int(item))
            elif isinstance(item, str):
                text = item.strip()
                if not text:
                    raise ValueError(f"{context} contains empty value.")
                try:
                    parsed.append(int(text))
                except ValueError as exc:
                    raise ValueError(
                        f"{context} contains non-integer value {item!r}."
                    ) from exc
            else:
                raise ValueError(f"{context} contains invalid value {item!r}.")
        decisions = np.asarray(parsed, dtype=np.int64)
    if not np.all((decisions == 0) | (decisions == 1)):
        raise ValueError(f"{context} must contain only binary 0/1 values.")
    return decisions.astype(np.int8, copy=False)


def _validate_partition_frame(
    df: pd.DataFrame,
    *,
    model_name: str,
    dataset_name: str,
    split_id: str,
    partition: str,
    base_threshold: float,
    run_dir: Path,
) -> pd.DataFrame:
    sub = df[df["model"] == model_name].reset_index(drop=True)
    if sub.empty:
        raise ValueError(
            f"{run_dir}: no {partition} predictions found for model {model_name!r}."
        )
    if set(sub["split_id"]) != {split_id}:
        raise ValueError(
            f"{run_dir}: {partition} split_id does not match run.json ({split_id!r})."
        )
    if set(sub["partition"]) != {partition}:
        raise ValueError(
            f"{run_dir}: expected partition={partition!r} in {partition}_predictions.csv."
        )
    if "dataset" in sub.columns and set(sub["dataset"]) != {dataset_name}:
        raise ValueError(
            f"{run_dir}: {partition} dataset column does not match {dataset_name!r}."
        )
    if not sub["row_id"].is_unique:
        raise ValueError(
            f"{run_dir}: duplicate row_id in {partition} predictions for {model_name!r}."
        )
    raw_score = sub["score"].to_numpy()
    if raw_score.dtype == np.bool_:
        raise ValueError(
            f"{run_dir}: {partition} score column for {model_name!r} must be numeric."
        )
    score = _as_score_vector(raw_score, len(sub), length_name="decision")
    decisions = _parse_int_decisions(
        sub["decision"],
        context=f"{run_dir}: {partition} decision column for {model_name!r}",
    )
    expected_decisions = (score >= base_threshold).astype(np.int8)
    if not np.array_equal(decisions, expected_decisions):
        raise ValueError(
            f"{run_dir}: {partition} decision column for {model_name!r} does not "
            f"match threshold {base_threshold!r}."
        )
    return sub


def _load_baseline_run(run_dir: Path) -> list[_BaselineModelData]:
    resolved = Path(run_dir).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved}")

    run_json_path = resolved / "run.json"
    val_csv_path = resolved / "validation_predictions.csv"
    test_csv_path = resolved / "test_predictions.csv"
    for path in (run_json_path, val_csv_path, test_csv_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required baseline artifact: {path}")

    meta = json.loads(run_json_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{resolved}: run.json must decode to a JSON object.")

    config = meta.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{resolved}: run.json missing 'config' object.")
    if config.get("selection_metric") != "profit_eq16":
        raise ValueError(
            f"{resolved}: expected selection_metric='profit_eq16', "
            f"got {config.get('selection_metric')!r}."
        )
    if config.get("postprocessor") is not None:
        raise ValueError(
            f"{resolved}: expected baseline run with postprocessor=None, "
            f"got {config.get('postprocessor')!r}."
        )

    split_id = meta.get("split_id")
    if not isinstance(split_id, str) or not split_id:
        raise ValueError(f"{resolved}: run.json missing nonempty 'split_id'.")

    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, dict) or not isinstance(
        dataset_cfg.get("name"), str
    ):
        raise ValueError(f"{resolved}: run.json missing dataset name.")
    dataset_name = dataset_cfg["name"]
    if not dataset_name:
        raise ValueError(f"{resolved}: dataset name must be nonempty.")

    selected = meta.get("selected")
    if not isinstance(selected, dict) or not selected:
        raise ValueError(f"{resolved}: run.json missing nonempty 'selected' mapping.")

    val_df = load_predictions(val_csv_path)
    test_df = load_predictions(test_csv_path)
    for label, frame in (("validation", val_df), ("test", test_df)):
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"{resolved}: {label}_predictions.csv missing columns {sorted(missing)}."
            )

    models: list[_BaselineModelData] = []
    for model_name, selection in selected.items():
        if not isinstance(selection, dict) or "threshold" not in selection:
            raise ValueError(
                f"{resolved}: selected[{model_name!r}] missing 'threshold'."
            )
        base_threshold = _validate_nonbool_float(
            selection["threshold"],
            f"selected[{model_name!r}]['threshold']",
            minimum=0.0,
            maximum=1.0,
        )
        val_model = _validate_partition_frame(
            val_df,
            model_name=model_name,
            dataset_name=dataset_name,
            split_id=split_id,
            partition="validation",
            base_threshold=base_threshold,
            run_dir=resolved,
        )
        test_model = _validate_partition_frame(
            test_df,
            model_name=model_name,
            dataset_name=dataset_name,
            split_id=split_id,
            partition="test",
            base_threshold=base_threshold,
            run_dir=resolved,
        )
        if not set(val_model["row_id"]).isdisjoint(set(test_model["row_id"])):
            raise ValueError(
                f"{resolved}: validation and test row_id sets overlap for {model_name!r}."
            )
        models.append(
            _BaselineModelData(
                dataset=dataset_name,
                model=model_name,
                split_id=split_id,
                base_threshold=base_threshold,
                validation_df=val_model,
                test_df=test_model,
            )
        )
    return models


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if not math.isfinite(value):
        raise ValueError(f"Expected finite float or NaN, got {value!r}.")
    return repr(float(value))


def _evaluate_partition(
    part_df: pd.DataFrame,
    *,
    clf: RejectOptionClassifier,
    dataset: str,
    model: str,
    fairness_metric: str,
    fairness_bound: float,
    partition: str,
    split_id: str,
) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    y = part_df["y"].to_numpy()
    a = part_df["a"].to_numpy()
    score = part_df["score"].to_numpy()

    decision_before, decision_after = clf.predict_before_and_after(score, a)
    metrics_after = evaluate_metrics(y, a, score, decision_after)
    profit_before = compute_profit_eq16(y, decision_before)
    delta_profit = metrics_after["profit_eq16"] - profit_before

    assert clf.base_threshold_ is not None
    assert clf.threshold_ is not None
    assert clf.margin_ is not None
    assert clf.feasible_ is not None

    row = {
        "dataset": dataset,
        "model": model,
        "fairness_metric": fairness_metric,
        "fairness_bound": _format_float(fairness_bound),
        "partition": partition,
        "split_id": split_id,
        "base_threshold": _format_float(clf.base_threshold_),
        "selected_threshold": _format_float(clf.threshold_),
        "margin": _format_float(clf.margin_),
        "feasible": str(clf.feasible_),
        "delta_profit_vs_baseline": _format_float(delta_profit),
    }
    for metric_name, metric_value in metrics_after.items():
        row[metric_name] = _format_float(metric_value)
    return row, decision_before, decision_after


def _build_prediction_frame(
    part_df: pd.DataFrame,
    *,
    decision_before: np.ndarray,
    decision_after: np.ndarray,
    dataset: str,
    fairness_metric: str,
    fairness_bound: float,
    clf: RejectOptionClassifier,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": part_df["row_id"],
            "y": part_df["y"],
            "a": part_df["a"],
            "score": part_df["score"],
            "decision": decision_after,
            "decision_before": decision_before,
            "decision_after": decision_after,
            "run_id": f"{dataset}-roc",
            "dataset": dataset,
            "model": part_df["model"],
            "split_id": part_df["split_id"],
            "partition": part_df["partition"],
            "validation_fold": part_df["validation_fold"],
            "fairness_metric": fairness_metric,
            "fairness_bound": fairness_bound,
            "base_threshold": clf.base_threshold_,
            "selected_threshold": clf.threshold_,
            "margin": clf.margin_,
        }
    )


def _target_bound_for_dataset(
    dataset: str, bounds: tuple[float, ...], requested_bound: float | None
) -> float:
    target = (
        PRIMARY_BOUNDS.get(dataset, DEFAULT_PRIMARY_BOUND)
        if requested_bound is None
        else _validate_nonbool_float(requested_bound, "predictions_bound", minimum=0.0)
    )
    for candidate in bounds:
        if math.isclose(candidate, target, rel_tol=0.0, abs_tol=1e-12):
            return candidate
    raise ValueError(
        f"Export fairness_bound {target} for dataset {dataset!r} is not in "
        f"the sweep grid {bounds}."
    )


def run_roc_experiment(
    run_dirs: Sequence[Path],
    output_csv: Path,
    *,
    predictions_dir: Path | None = None,
    predictions_metric: str = "sp_paper",
    predictions_bound: float | None = None,
    fairness_metrics: Sequence[str] = DEFAULT_FAIRNESS_METRICS,
    fairness_bounds: Sequence[float] = DEFAULT_FAIRNESS_BOUNDS,
) -> list[dict[str, str]]:
    """Run Stage B ROC fairness-profit sweeps over completed baseline runs."""
    if not run_dirs:
        raise ValueError("Provide at least one baseline run directory.")

    if isinstance(fairness_metrics, (str, bytes)):
        raise ValueError(
            "fairness_metrics must be a sequence of metric names, not a string."
        )
    metrics_grid = tuple(fairness_metrics)
    if not metrics_grid or any(
        not isinstance(m, str) or m not in FAIRNESS_METRICS for m in metrics_grid
    ):
        raise ValueError(
            f"fairness_metrics must be a nonempty subset of {sorted(FAIRNESS_METRICS)}."
        )
    if len(set(metrics_grid)) != len(metrics_grid):
        raise ValueError("fairness_metrics must not contain duplicate entries.")

    if isinstance(fairness_bounds, (str, bytes)):
        raise ValueError("fairness_bounds must be a sequence of finite floats >= 0.")
    bounds_grid = tuple(
        _validate_nonbool_float(b, "fairness_bound", minimum=0.0)
        for b in fairness_bounds
    )
    if not bounds_grid:
        raise ValueError("fairness_bounds must be nonempty.")
    if len(set(bounds_grid)) != len(bounds_grid):
        raise ValueError("fairness_bounds must not contain duplicate values.")

    model_entries: list[_BaselineModelData] = []
    seen_pairs: set[tuple[str, str]] = set()
    for run_dir in run_dirs:
        for entry in _load_baseline_run(run_dir):
            key = (entry.dataset, entry.model)
            if key in seen_pairs:
                raise ValueError(
                    f"Duplicate baseline entry for (dataset, model)={key}."
                )
            seen_pairs.add(key)
            model_entries.append(entry)

    export_targets: dict[str, float] = {}
    if predictions_dir is not None:
        if (
            not isinstance(predictions_metric, str)
            or predictions_metric not in metrics_grid
        ):
            raise ValueError(
                f"predictions_metric {predictions_metric!r} is not in {metrics_grid}."
            )
        for dataset in dict.fromkeys(entry.dataset for entry in model_entries):
            export_targets[dataset] = _target_bound_for_dataset(
                dataset, bounds_grid, predictions_bound
            )

    summary_rows: list[dict[str, str]] = []
    exported_frames: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for entry in model_entries:
        val_score = entry.validation_df["score"].to_numpy()
        val_y = entry.validation_df["y"].to_numpy()
        val_a = entry.validation_df["a"].to_numpy()

        for fairness_metric in metrics_grid:
            for fairness_bound in bounds_grid:
                clf = RejectOptionClassifier(
                    fairness_metric=fairness_metric,
                    fairness_bound=fairness_bound,
                    utility_metric="profit_eq16",
                    tune_threshold=True,
                    n_thresholds=101,
                    n_margins=50,
                    max_margin=0.5,
                ).fit(val_score, val_y, val_a, threshold=entry.base_threshold)

                export_match = (
                    predictions_dir is not None
                    and fairness_metric == predictions_metric
                    and math.isclose(
                        fairness_bound,
                        export_targets[entry.dataset],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )

                for partition, part_df in (
                    ("validation", entry.validation_df),
                    ("test", entry.test_df),
                ):
                    row, dec_before, dec_after = _evaluate_partition(
                        part_df,
                        clf=clf,
                        dataset=entry.dataset,
                        model=entry.model,
                        fairness_metric=fairness_metric,
                        fairness_bound=fairness_bound,
                        partition=partition,
                        split_id=entry.split_id,
                    )
                    summary_rows.append(row)

                    if export_match:
                        frame = _build_prediction_frame(
                            part_df,
                            decision_before=dec_before,
                            decision_after=dec_after,
                            dataset=entry.dataset,
                            fairness_metric=fairness_metric,
                            fairness_bound=fairness_bound,
                            clf=clf,
                        )
                        exported_frames.setdefault(
                            (entry.dataset, partition), []
                        ).append(frame)

    write_csv([Path(output_csv)], SUMMARY_COLUMNS, summary_rows)

    if predictions_dir is not None:
        base_dir = Path(predictions_dir)
        for (dataset, partition), frames in exported_frames.items():
            out_path = base_dir / dataset / f"{partition}_predictions.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)

    return summary_rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_RUN_DIRS),
        help="completed baseline run directories",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="output path for the ROC sweep summary CSV",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="optional directory to export row-level decision_before/decision_after CSVs",
    )
    parser.add_argument(
        "--predictions-metric",
        default="sp_paper",
        choices=sorted(FAIRNESS_METRICS),
        help="fairness metric variant to export when --predictions-dir is set",
    )
    parser.add_argument(
        "--predictions-bound",
        type=float,
        default=None,
        help="fairness bound to export when --predictions-dir is set (defaults per dataset)",
    )
    args = parser.parse_args(argv)
    try:
        rows = run_roc_experiment(
            args.run_dirs,
            args.output_csv,
            predictions_dir=args.predictions_dir,
            predictions_metric=args.predictions_metric,
            predictions_bound=args.predictions_bound,
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Wrote {len(rows)} ROC sweep rows to {args.output_csv}.")


if __name__ == "__main__":
    main()
