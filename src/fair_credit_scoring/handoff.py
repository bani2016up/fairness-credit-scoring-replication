"""Verify cross-method alignment and build the handoff manifest CSV."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import evaluate_metrics

EXPECTED_ROW_COUNTS = {
    "german": (600, 400),
    "taiwan": (14118, 9413),
}

MANIFEST_COLUMNS = (
    "dataset",
    "method",
    "model",
    "eta",
    "seed",
    "split_id",
    "run_id",
    "run_dir",
    "validation_predictions",
    "test_predictions",
    "splits_csv",
    "run_json",
    "selected_threshold",
    "selection_value",
)

CONTROL_METRICS = ("profit_eq16", "approval_rate", "ind", "sp_paper", "eo_gap", "sf")

CONTROL_TABLE_COLUMNS = (
    "dataset",
    "partition",
    "method",
    "model",
    "eta",
    "selected_threshold",
    "profit_eq16",
    "delta_profit_vs_approve_all",
    "approval_rate",
    "ind",
    "sp_paper",
    "eo_gap",
    "sf",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        keep_default_na=False,
        dtype={
            "row_id": str,
            "run_id": str,
            "dataset": str,
            "model": str,
            "split_id": str,
            "partition": str,
            "validation_fold": str,
        },
        float_precision="round_trip",
    )


def _group_runs_by_dataset(
    run_dirs: list[Path],
) -> dict[str, list[tuple[Path, dict]]]:
    by_dataset: dict[str, list[tuple[Path, dict]]] = {}
    for run_dir in run_dirs:
        resolved = Path(run_dir).resolve()
        meta = json.loads((resolved / "run.json").read_text(encoding="utf-8"))
        dataset_name = meta["config"]["dataset"]["name"]
        by_dataset.setdefault(dataset_name, []).append((resolved, meta))
    return by_dataset


def _write_csv(
    paths: list[Path], fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    for target in paths:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def _assert_aligned_group(runs: list[tuple[Path, dict]], dataset_name: str) -> None:
    """Check cross-run consistency against the current prepared CSVs and splits."""
    if not runs:
        raise ValueError(f"No runs provided for dataset {dataset_name!r}.")

    first_dir, first_meta = runs[0]
    first_config = first_meta["config"]
    first_dataset = first_config["dataset"]
    reference_data_path = Path(first_dataset["path"]).resolve()
    if not reference_data_path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {reference_data_path}")
    reference_data_sha = _sha256(reference_data_path)
    reference_splits_bytes = (first_dir / "splits.csv").read_bytes()
    reference_split_id = first_meta["split_id"]
    reference_seed = first_config["seed"]
    reference_thresholds = first_config["thresholds"]
    reference_categorical = first_dataset["categorical"]
    reference_numeric_no_age = [c for c in first_dataset["numeric"] if c != "AGE"]

    expected_counts = EXPECTED_ROW_COUNTS.get(dataset_name)
    reference_val_keys: pd.DataFrame | None = None
    reference_test_keys: pd.DataFrame | None = None

    for run_dir, meta in runs:
        config = meta["config"]
        dataset = config["dataset"]
        if dataset["name"] != dataset_name:
            raise ValueError(
                f"{run_dir}: expected dataset {dataset_name!r}, got {dataset['name']!r}."
            )
        data_path = Path(dataset["path"]).resolve()
        if not data_path.is_file() or _sha256(data_path) != reference_data_sha:
            raise ValueError(
                f"{run_dir}: prepared dataset bytes differ from {reference_data_path}."
            )
        if config["seed"] != reference_seed:
            raise ValueError(
                f"{run_dir}: seed mismatch ({config['seed']} != {reference_seed})."
            )
        if config["selection_metric"] != "profit_eq16":
            raise ValueError(
                f"{run_dir}: expected selection_metric='profit_eq16', got {config['selection_metric']!r}."
            )
        if config["thresholds"] != reference_thresholds:
            raise ValueError(f"{run_dir}: thresholds grid differs from reference run.")
        if dataset["scaling"] != "max_abs":
            raise ValueError(
                f"{run_dir}: expected max_abs scaling, got {dataset['scaling']!r}."
            )
        if meta["split_id"] != reference_split_id:
            raise ValueError(f"{run_dir}: split_id mismatch.")
        if (run_dir / "splits.csv").read_bytes() != reference_splits_bytes:
            raise ValueError(f"{run_dir}: splits.csv bytes differ.")

        numeric = dataset["numeric"]
        if dataset["categorical"] != reference_categorical:
            raise ValueError(f"{run_dir}: categorical columns differ.")
        if config["method"] == "prejudice_remover":
            if "AGE" in numeric or numeric != reference_numeric_no_age:
                raise ValueError(
                    f"{run_dir}: PR must exclude AGE and keep identical remaining numeric covariates."
                )
        elif [c for c in numeric if c != "AGE"] != reference_numeric_no_age:
            raise ValueError(f"{run_dir}: numeric covariates differ.")

        val_df = _load_predictions(run_dir / "validation_predictions.csv")
        test_df = _load_predictions(run_dir / "test_predictions.csv")
        for model_name in meta["selected"]:
            val_model = val_df[val_df["model"] == model_name].reset_index(drop=True)
            test_model = test_df[test_df["model"] == model_name].reset_index(drop=True)
            if expected_counts is not None:
                exp_val, exp_test = expected_counts
                if len(val_model) != exp_val or len(test_model) != exp_test:
                    raise ValueError(
                        f"{run_dir} model {model_name!r}: expected ({exp_val}, {exp_test}) "
                        f"rows, got ({len(val_model)}, {len(test_model)})."
                    )
            val_keys = val_model[
                ["row_id", "y", "a", "partition", "validation_fold", "split_id"]
            ]
            test_keys = test_model[
                ["row_id", "y", "a", "partition", "validation_fold", "split_id"]
            ]
            if reference_val_keys is None:
                reference_val_keys = val_keys
                reference_test_keys = test_keys
            else:
                pd.testing.assert_frame_equal(
                    val_keys, reference_val_keys, check_exact=True
                )
                pd.testing.assert_frame_equal(
                    test_keys, reference_test_keys, check_exact=True
                )


def build_handoff_manifest(
    run_dirs: list[Path],
    output_paths: list[Path],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, str]]:
    """Verify cross-method alignment per dataset and write manifest CSVs."""
    root = Path.cwd().resolve() if repo_root is None else Path(repo_root).resolve()
    by_dataset = _group_runs_by_dataset(run_dirs)
    rows: list[dict[str, str]] = []

    for dataset_name, group in by_dataset.items():
        _assert_aligned_group(group, dataset_name)
        for resolved, meta in group:
            try:
                rel_dir = resolved.relative_to(root)
            except ValueError:
                rel_dir = resolved

            for model_name, selection in meta["selected"].items():
                eta = selection["params"].get("eta", "")
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": meta["config"]["method"],
                        "model": model_name,
                        "eta": "" if eta == "" else str(eta),
                        "seed": str(meta["config"]["seed"]),
                        "split_id": meta["split_id"],
                        "run_id": meta["config"]["run_id"],
                        "run_dir": rel_dir.as_posix(),
                        "validation_predictions": (
                            rel_dir / "validation_predictions.csv"
                        ).as_posix(),
                        "test_predictions": (
                            rel_dir / "test_predictions.csv"
                        ).as_posix(),
                        "splits_csv": (rel_dir / "splits.csv").as_posix(),
                        "run_json": (rel_dir / "run.json").as_posix(),
                        "selected_threshold": f"{selection['threshold']:.2f}",
                        "selection_value": f"{selection['selection_value']:.12g}",
                    }
                )

    _write_csv(output_paths, MANIFEST_COLUMNS, rows)
    return rows


def _fmt_metric(value: float, digits: int = 6) -> str:
    return "NaN" if math.isnan(value) else f"{value:.{digits}f}"


def _control_row(
    *,
    dataset: str,
    partition: str,
    method: str,
    model: str,
    eta: str,
    threshold: str,
    metrics: dict[str, float],
    approve_all_profit: float,
) -> dict[str, str]:
    delta = metrics["profit_eq16"] - approve_all_profit
    return {
        "dataset": dataset,
        "partition": partition,
        "method": method,
        "model": model,
        "eta": eta,
        "selected_threshold": threshold,
        "profit_eq16": _fmt_metric(metrics["profit_eq16"], 6),
        "delta_profit_vs_approve_all": _fmt_metric(delta, 6),
        "approval_rate": _fmt_metric(metrics["approval_rate"], 4),
        "ind": _fmt_metric(metrics["ind"], 4),
        "sp_paper": _fmt_metric(metrics["sp_paper"], 4),
        "eo_gap": _fmt_metric(metrics["eo_gap"], 4),
        "sf": _fmt_metric(metrics["sf"], 4),
    }


def build_control_table(
    run_dirs: list[Path],
    csv_outputs: list[Path],
) -> list[dict[str, str]]:
    """Compute approve-all, reject-all, and saved model metrics without retraining."""
    by_dataset = _group_runs_by_dataset(run_dirs)
    rows: list[dict[str, str]] = []
    for dataset_name, group in by_dataset.items():
        _assert_aligned_group(group, dataset_name)
        first_dir, first_meta = group[0]
        first_model = next(iter(first_meta["selected"]))

        for partition in ("validation", "test"):
            first_df = _load_predictions(first_dir / f"{partition}_predictions.csv")
            ref = first_df[first_df["model"] == first_model]
            y_ref = ref["y"].to_numpy()
            a_ref = ref["a"].to_numpy()
            score_ref = ref["score"].to_numpy()

            approve_all = evaluate_metrics(
                y_ref, a_ref, score_ref, np.ones_like(y_ref), CONTROL_METRICS
            )
            reject_all = evaluate_metrics(
                y_ref, a_ref, score_ref, np.zeros_like(y_ref), CONTROL_METRICS
            )
            base_profit = approve_all["profit_eq16"]

            rows.append(
                _control_row(
                    dataset=dataset_name,
                    partition=partition,
                    method="constant_policy",
                    model="approve-all",
                    eta="",
                    threshold="",
                    metrics=approve_all,
                    approve_all_profit=base_profit,
                )
            )
            rows.append(
                _control_row(
                    dataset=dataset_name,
                    partition=partition,
                    method="constant_policy",
                    model="reject-all",
                    eta="",
                    threshold="",
                    metrics=reject_all,
                    approve_all_profit=base_profit,
                )
            )

            for run_dir, meta in group:
                pred_df = _load_predictions(run_dir / f"{partition}_predictions.csv")
                method = meta["config"]["method"]
                for model_name, sel in meta["selected"].items():
                    sub = pred_df[pred_df["model"] == model_name]
                    res = evaluate_metrics(
                        sub["y"].to_numpy(),
                        sub["a"].to_numpy(),
                        sub["score"].to_numpy(),
                        sub["decision"].to_numpy(),
                        CONTROL_METRICS,
                    )
                    saved = sel[f"{partition}_metrics"]
                    for metric_name in CONTROL_METRICS:
                        if metric_name not in saved:
                            continue
                        if math.isnan(res[metric_name]):
                            if saved[metric_name] is not None:
                                raise ValueError(
                                    f"{run_dir} {model_name} {partition}.{metric_name} mismatch."
                                )
                        elif (
                            saved[metric_name] is None
                            or abs(res[metric_name] - saved[metric_name]) > 1e-12
                        ):
                            raise ValueError(
                                f"{run_dir} {model_name} {partition}.{metric_name} mismatch."
                            )
                    eta = sel["params"].get("eta", "")
                    rows.append(
                        _control_row(
                            dataset=dataset_name,
                            partition=partition,
                            method=method,
                            model=model_name,
                            eta="" if eta == "" else str(eta),
                            threshold=f"{sel['threshold']:.2f}",
                            metrics=res,
                            approve_all_profit=base_profit,
                        )
                    )

    _write_csv(csv_outputs, CONTROL_TABLE_COLUMNS, rows)
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="completed run directories to verify and include in the manifest",
    )
    parser.add_argument(
        "--output",
        action="append",
        dest="outputs",
        type=Path,
        help="manifest CSV output path (can be specified multiple times)",
    )
    parser.add_argument(
        "--control-csv",
        action="append",
        dest="control_csvs",
        type=Path,
        help="optional control table CSV output path (can be specified multiple times)",
    )
    args = parser.parse_args(argv)
    outputs = args.outputs or [Path("outputs/timur-handoff-manifest.csv")]
    rows = build_handoff_manifest(args.run_dirs, outputs)
    if args.control_csvs:
        build_control_table(args.run_dirs, args.control_csvs)
    print(f"Verified {len(rows)} model entries across {len(args.run_dirs)} runs.")


if __name__ == "__main__":
    main()
