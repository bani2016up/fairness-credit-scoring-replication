"""Stage C cross-method analysis, F4 separation stability, and visualizations."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .handoff import _assert_aligned_group, _group_runs_by_dataset, load_predictions
from .metrics import METRICS, evaluate_metrics, group_confusion_rates

DEFAULT_RUN_DIRS: tuple[Path, ...] = (
    Path("outputs/german-baseline-profit"),
    Path("outputs/german-reweighing-profit"),
    Path("outputs/german-pr-sweep"),
    Path("outputs/german-roc-profit"),
    Path("outputs/taiwan-baseline-profit"),
    Path("outputs/taiwan-reweighing-profit"),
    Path("outputs/taiwan-pr-sweep"),
    Path("outputs/taiwan-roc-profit"),
)
DEFAULT_ROC_SWEEP_CSV = Path("outputs/roc-sweep-summary.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis")

PARTITIONS: tuple[str, str] = ("validation", "test")
BUILTIN_METRIC_NAMES: tuple[str, ...] = tuple(METRICS.keys())
CORRELATION_METRICS: tuple[str, ...] = (
    "auc",
    "profit_eq16",
    "ind",
    "sp_paper",
    "eo_gap",
    "sf",
)
GAIN_METRICS: tuple[str, ...] = CORRELATION_METRICS
HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({"auc", "profit_eq16"})
LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset({"ind", "sp_paper", "eo_gap", "sf"})
PARETO_FAIRNESS_METRICS: tuple[str, ...] = ("sp_paper", "eo_gap", "ind")

FAMILY_BY_METHOD: dict[str, str] = {
    "baseline": "Baseline",
    "reweighing": "Reweighing",
    "prejudice_remover": "Prejudice Remover",
    "reject_option": "Reject Option",
    "roc_sweep": "Reject Option",
    "constant_policy": "Constant Policy",
}
FAMILY_ORDER: tuple[str, ...] = (
    "Baseline",
    "Reweighing",
    "Prejudice Remover",
    "Reject Option",
)
FAMILY_COLORS: dict[str, str] = {
    "Baseline": "#1f77b4",
    "Reweighing": "#2ca02c",
    "Prejudice Remover": "#ff7f0e",
    "Reject Option": "#d62728",
}
PRIMARY_VARIANT_ORDER: tuple[str, ...] = (
    "B1",
    "B2",
    "P1",
    "P2",
    "I1",
    "O1",
    "O2",
    "CONST_APPROVE_ALL",
)
PRIMARY_7_ORDER: tuple[str, ...] = ("B1", "B2", "P1", "P2", "I1", "O1", "O2")

REQUIRED_ROC_SWEEP_COLUMNS: frozenset[str] = frozenset(
    {
        "dataset",
        "model",
        "fairness_metric",
        "fairness_bound",
        "partition",
        "split_id",
        "selected_threshold",
        "margin",
        "feasible",
        *BUILTIN_METRIC_NAMES,
    }
)

MASTER_SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "partition",
    "variant_id",
    "family",
    "method",
    "model",
    "eta",
    "split_id",
    "run_id",
    "base_threshold",
    "selected_threshold",
    "roc_margin",
    "selection_value",
    "delta_profit_vs_own_baseline",
    "delta_profit_vs_approve_all",
    *BUILTIN_METRIC_NAMES,
    "delta_fpr",
    "delta_tpr",
    "abs_delta_fpr",
    "abs_delta_tpr",
)
PRIMARY_7_COLUMNS: tuple[str, ...] = ("primary_variant", *MASTER_SUMMARY_COLUMNS)
RELATIVE_GAINS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "partition",
    "scope",
    "comparison",
    "variant_id",
    "baseline_variant_id",
    "family",
    "method",
    "model",
    "eta",
    *[
        f"{metric}_{suffix}"
        for metric in GAIN_METRICS
        for suffix in ("method", "baseline", "abs_change", "gain_pct")
    ],
)
SPEARMAN_CORRELATIONS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "partition",
    "pool",
    "sign_convention",
    "metric_x",
    "metric_y",
    "n_obs",
    "rho",
    "p_value",
)
SEPARATION_STABILITY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "partition",
    "variant_id",
    "family",
    "method",
    "model",
    "eta",
    "is_i1_best_val",
    "selected_threshold",
    "profit_eq16",
    "sp_paper",
    "eo_gap",
    "delta_fpr",
    "delta_tpr",
    "abs_delta_fpr",
    "abs_delta_tpr",
    "opposite_sign_cancellation",
    "cancellation_ratio",
    "rank_sp_paper",
    "rank_eo_gap",
    "rank_shift",
)
SEPARATION_RANK_SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "partition",
    "n_models",
    "spearman_rho",
    "spearman_p_value",
    "kendall_tau",
    "kendall_p_value",
    "n_opposite_sign_cancellation",
    "n_rank_shift_nonzero",
    "max_abs_rank_shift",
)
POOLED_TEST_COLUMNS: tuple[str, ...] = (
    "dataset",
    "variant_id",
    "source_pool",
    "family",
    "method",
    "model",
    "eta",
    "fairness_metric",
    "fairness_bound",
    "selected_threshold",
    "margin",
    "profit_eq16",
    "sp_paper",
    "eo_gap",
    "ind",
    "auc",
    "sf",
)
PARETO_FRONTIERS_COLUMNS: tuple[str, ...] = (
    *POOLED_TEST_COLUMNS,
    *(f"is_pareto_{metric}" for metric in PARETO_FAIRNESS_METRICS),
)


def _ensure_agg_backend() -> None:
    """Switch Matplotlib backend to non-interactive Agg when plotting."""
    if plt.get_backend().lower() != "agg":
        plt.switch_backend("Agg")


def _format_eta_label(eta_value: Any) -> str:
    if eta_value is None or eta_value == "":
        return ""
    numeric = float(eta_value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _resolve_eta_value(method: str, model_name: str, raw_eta: Any) -> float:
    """Parse numeric eta from selected params or pr_eta_<val> model name."""
    if raw_eta is not None and raw_eta != "":
        return float(raw_eta)
    if method == "prejudice_remover" and model_name.startswith("pr_eta_"):
        suffix = model_name.removeprefix("pr_eta_")
        if suffix:
            try:
                return float(suffix)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse numeric eta from prejudice_remover model {model_name!r}."
                ) from exc
    return math.nan


def _map_variant_id(method: str, model: str, eta_value: Any = "") -> str:
    direct_map = {
        ("baseline", "lr"): "B1",
        ("baseline", "rf"): "B2",
        ("reweighing", "lr"): "P1",
        ("reweighing", "rf"): "P2",
        ("reject_option", "lr"): "O1",
        ("reject_option", "rf"): "O2",
        ("constant_policy", "approve-all"): "CONST_APPROVE_ALL",
        ("constant_policy", "reject-all"): "CONST_REJECT_ALL",
    }
    key = (method, model)
    if key in direct_map:
        return direct_map[key]
    if method == "prejudice_remover":
        eta_label = _format_eta_label(eta_value)
        if not eta_label and model.startswith("pr_eta_"):
            eta_label = model.removeprefix("pr_eta_")
        if not eta_label:
            raise ValueError(
                f"Cannot determine eta for prejudice_remover model {model!r}."
            )
        return f"I1_eta_{eta_label}"
    raise ValueError(f"Unsupported (method, model) pair: ({method!r}, {model!r}).")


def _compute_confusion_deltas(
    y: np.ndarray, a: np.ndarray, decision: np.ndarray
) -> dict[str, float]:
    rates = group_confusion_rates(y, a, decision)
    delta_fpr = float(rates[1].fpr - rates[0].fpr)
    delta_tpr = float(rates[1].tpr - rates[0].tpr)
    return {
        "delta_fpr": delta_fpr,
        "delta_tpr": delta_tpr,
        "abs_delta_fpr": float(abs(delta_fpr)),
        "abs_delta_tpr": float(abs(delta_tpr)),
    }


def _verify_saved_metrics(
    run_dir: Path,
    model_name: str,
    partition: str,
    computed: dict[str, float],
    saved: dict[str, Any],
) -> None:
    for metric_name, computed_val in computed.items():
        if metric_name not in saved:
            continue
        saved_val = saved[metric_name]
        if math.isnan(computed_val):
            if saved_val is not None:
                raise ValueError(
                    f"{run_dir} {model_name} {partition}.{metric_name} mismatch: "
                    f"computed NaN vs saved {saved_val!r}."
                )
        elif saved_val is None or abs(computed_val - float(saved_val)) > 1e-12:
            raise ValueError(
                f"{run_dir} {model_name} {partition}.{metric_name} mismatch: "
                f"computed {computed_val!r} vs saved {saved_val!r}."
            )


def _evaluate_all_runs(
    by_dataset: dict[str, list[tuple[Path, dict[str, Any]]]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compute row-level metrics for all trained models and constant policies."""
    raw_rows: list[dict[str, Any]] = []
    split_ids: dict[str, str] = {}
    seen_run_dirs: set[Path] = set()
    seen_keys: set[tuple[str, str, str]] = set()

    for dataset_name, group in by_dataset.items():
        for run_dir, _ in group:
            resolved_dir = Path(run_dir).resolve()
            if resolved_dir in seen_run_dirs:
                raise ValueError(f"Duplicate run directory provided: {resolved_dir}.")
            seen_run_dirs.add(resolved_dir)

        _assert_aligned_group(group, dataset_name)
        first_dir, first_meta = group[0]
        split_ids[dataset_name] = str(first_meta["split_id"])
        first_model = next(iter(first_meta["selected"]))

        for partition in PARTITIONS:
            first_df = load_predictions(first_dir / f"{partition}_predictions.csv")
            ref = first_df[first_df["model"] == first_model]
            y_ref = ref["y"].to_numpy()
            a_ref = ref["a"].to_numpy()

            for policy_model, decision_vec, score_vec in (
                (
                    "approve-all",
                    np.ones_like(y_ref, dtype=np.int8),
                    np.ones_like(y_ref, dtype=np.float64),
                ),
                (
                    "reject-all",
                    np.zeros_like(y_ref, dtype=np.int8),
                    np.zeros_like(y_ref, dtype=np.float64),
                ),
            ):
                variant_id = _map_variant_id("constant_policy", policy_model)
                key = (dataset_name, partition, variant_id)
                if key in seen_keys:
                    raise ValueError(
                        f"Duplicate (dataset, partition, variant_id) entry: {key!r}."
                    )
                seen_keys.add(key)
                metrics = evaluate_metrics(y_ref, a_ref, score_vec, decision_vec)
                deltas = _compute_confusion_deltas(y_ref, a_ref, decision_vec)
                raw_rows.append(
                    {
                        "dataset": dataset_name,
                        "partition": partition,
                        "variant_id": variant_id,
                        "family": FAMILY_BY_METHOD["constant_policy"],
                        "method": "constant_policy",
                        "model": policy_model,
                        "eta": math.nan,
                        "split_id": split_ids[dataset_name],
                        "run_id": "",
                        "base_threshold": math.nan,
                        "selected_threshold": math.nan,
                        "roc_margin": math.nan,
                        "selection_value": math.nan,
                        **metrics,
                        **deltas,
                    }
                )

            for run_dir, meta in group:
                pred_df = load_predictions(run_dir / f"{partition}_predictions.csv")
                method = str(meta["config"]["method"])
                run_id = str(meta["config"]["run_id"])
                for model_name, sel in meta["selected"].items():
                    sub = pred_df[pred_df["model"] == model_name]
                    y = sub["y"].to_numpy()
                    a = sub["a"].to_numpy()
                    score = sub["score"].to_numpy()
                    decision = sub["decision"].to_numpy()

                    metrics = evaluate_metrics(y, a, score, decision)
                    _verify_saved_metrics(
                        run_dir,
                        model_name,
                        partition,
                        metrics,
                        sel.get(f"{partition}_metrics", {}),
                    )
                    deltas = _compute_confusion_deltas(y, a, decision)
                    raw_eta = sel.get("params", {}).get("eta", None)
                    eta_float = _resolve_eta_value(method, model_name, raw_eta)
                    variant_id = _map_variant_id(
                        method,
                        model_name,
                        None if math.isnan(eta_float) else eta_float,
                    )
                    key = (dataset_name, partition, variant_id)
                    if key in seen_keys:
                        raise ValueError(
                            f"Duplicate (dataset, partition, variant_id) entry: {key!r} "
                            f"(from {run_dir})."
                        )
                    seen_keys.add(key)

                    postproc = sel.get("postprocessor")
                    postproc_dict = postproc if isinstance(postproc, dict) else {}
                    base_threshold = float(sel["threshold"])
                    raw_post_thresh = postproc_dict.get("threshold")
                    selected_threshold = (
                        base_threshold
                        if raw_post_thresh is None
                        else float(raw_post_thresh)
                    )
                    raw_margin = postproc_dict.get("margin", math.nan)
                    roc_margin = math.nan if raw_margin is None else float(raw_margin)

                    raw_rows.append(
                        {
                            "dataset": dataset_name,
                            "partition": partition,
                            "variant_id": variant_id,
                            "family": FAMILY_BY_METHOD[method],
                            "method": method,
                            "model": model_name,
                            "eta": eta_float,
                            "split_id": split_ids[dataset_name],
                            "run_id": run_id,
                            "base_threshold": base_threshold,
                            "selected_threshold": selected_threshold,
                            "roc_margin": roc_margin,
                            "selection_value": float(sel["selection_value"]),
                            **metrics,
                            **deltas,
                        }
                    )

    df = pd.DataFrame(raw_rows)
    df = _attach_profit_deltas(df)
    return df, split_ids


def _attach_profit_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute delta_profit_vs_own_baseline and delta_profit_vs_approve_all."""
    if df.empty:
        return pd.DataFrame(columns=MASTER_SUMMARY_COLUMNS)
    out = df.copy()
    baseline_lookup: dict[tuple[str, str, str], float] = {}
    approve_all_lookup: dict[tuple[str, str], float] = {}

    for row in out.itertuples(index=False):
        if row.variant_id in {"B1", "B2"}:
            baseline_lookup[(row.dataset, row.partition, row.variant_id)] = float(
                row.profit_eq16
            )
        elif row.variant_id == "CONST_APPROVE_ALL":
            approve_all_lookup[(row.dataset, row.partition)] = float(row.profit_eq16)

    delta_own: list[float] = []
    delta_approve: list[float] = []
    for row in out.itertuples(index=False):
        profit = float(row.profit_eq16)
        approve_profit = approve_all_lookup.get((row.dataset, row.partition), math.nan)
        delta_approve.append(profit - approve_profit)

        if row.method == "constant_policy":
            delta_own.append(math.nan)
        else:
            base_variant = "B2" if row.model == "rf" else "B1"
            base_profit = baseline_lookup.get(
                (row.dataset, row.partition, base_variant), math.nan
            )
            delta_own.append(profit - base_profit)

    out["delta_profit_vs_own_baseline"] = delta_own
    out["delta_profit_vs_approve_all"] = delta_approve
    return out[list(MASTER_SUMMARY_COLUMNS)]


def _load_and_verify_roc_sweep(
    roc_sweep_csv: Path | None,
    expected_split_ids: dict[str, str],
    *,
    require_test_rows: bool = False,
) -> pd.DataFrame | None:
    """Load ROC sweep CSV and verify split_id matches aligned runs per dataset."""
    if roc_sweep_csv is None:
        return None
    if not roc_sweep_csv.is_file():
        raise FileNotFoundError(f"ROC sweep CSV not found: {roc_sweep_csv}")

    roc_df = pd.read_csv(
        roc_sweep_csv,
        dtype={
            "dataset": str,
            "model": str,
            "fairness_metric": str,
            "partition": str,
            "split_id": str,
        },
        float_precision="round_trip",
    )
    missing = REQUIRED_ROC_SWEEP_COLUMNS - set(roc_df.columns)
    if missing:
        raise ValueError(
            f"{roc_sweep_csv}: missing required columns {sorted(missing)}."
        )

    roc_df = roc_df[roc_df["dataset"].isin(expected_split_ids)].copy()
    for dataset_name, expected_split in expected_split_ids.items():
        sub = roc_df[roc_df["dataset"] == dataset_name]
        if sub.empty:
            continue
        found_splits = set(sub["split_id"].astype(str))
        if found_splits != {expected_split}:
            raise ValueError(
                f"{roc_sweep_csv}: split_id mismatch for dataset {dataset_name!r}: "
                f"expected {expected_split!r}, got {sorted(found_splits)!r}."
            )

    test_roc = roc_df[roc_df["partition"] == "test"].copy()
    if test_roc.empty:
        if require_test_rows:
            raise ValueError(
                f"{roc_sweep_csv}: no 'test' rows found for analyzed datasets "
                f"{sorted(expected_split_ids)!r}."
            )
        return test_roc

    test_roc["variant_id"] = [
        f"ROC_{row.model}_{row.fairness_metric}_{float(row.fairness_bound):g}"
        for row in test_roc.itertuples(index=False)
    ]
    test_roc["family"] = FAMILY_BY_METHOD["roc_sweep"]
    test_roc["method"] = "roc_sweep"
    test_roc["source_pool"] = "roc_sweep"
    test_roc["eta"] = math.nan
    test_roc["delta_fpr"] = test_roc["fpr_1"].astype(float) - test_roc["fpr_0"].astype(
        float
    )
    test_roc["delta_tpr"] = test_roc["tpr_1"].astype(float) - test_roc["tpr_0"].astype(
        float
    )
    test_roc["abs_delta_fpr"] = test_roc["delta_fpr"].abs()
    test_roc["abs_delta_tpr"] = test_roc["delta_tpr"].abs()
    return test_roc


def _select_i1_best_val(master_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Select I1_best_val per dataset by max validation profit_eq16, tie-break smaller eta."""
    selections: dict[str, dict[str, Any]] = {}
    pr_val = master_df[
        (master_df["method"] == "prejudice_remover")
        & (master_df["partition"] == "validation")
    ]
    for dataset_name, group in pr_val.groupby("dataset", sort=False):
        finite = group[np.isfinite(group["profit_eq16"])].copy()
        if finite.empty:
            raise ValueError(
                f"No finite validation profit_eq16 values for prejudice_remover in {dataset_name!r}."
            )
        max_profit = float(finite["profit_eq16"].max())
        tied = finite[finite["profit_eq16"] == max_profit].sort_values(
            "eta", ascending=True
        )
        winner = tied.iloc[0]

        test_match = master_df[
            (master_df["dataset"] == dataset_name)
            & (master_df["partition"] == "test")
            & (master_df["variant_id"] == winner["variant_id"])
        ]
        test_profit = (
            float(test_match.iloc[0]["profit_eq16"])
            if not test_match.empty
            else math.nan
        )

        selections[str(dataset_name)] = {
            "variant_id": str(winner["variant_id"]),
            "model": str(winner["model"]),
            "eta": float(winner["eta"]),
            "selected_threshold": float(winner["selected_threshold"]),
            "validation_profit_eq16": max_profit,
            "test_profit_eq16": test_profit,
            "tie_broken_by_smaller_eta": len(tied) > 1,
            "tied_models": [str(m) for m in tied["model"].tolist()],
        }
    return selections


def build_primary_7_table(
    master_df: pd.DataFrame,
    i1_selections: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build table of B1, B2, P1, P2, I1_best_val, O1, O2, and approve-all."""
    rows: list[dict[str, Any]] = []
    fixed_map = {
        "B1": "B1",
        "B2": "B2",
        "P1": "P1",
        "P2": "P2",
        "O1": "O1",
        "O2": "O2",
        "CONST_APPROVE_ALL": "CONST_APPROVE_ALL",
    }
    order_rank = {label: idx for idx, label in enumerate(PRIMARY_VARIANT_ORDER)}

    for dataset_name, ds_df in master_df.groupby("dataset", sort=False):
        best_i1_variant = i1_selections.get(str(dataset_name), {}).get("variant_id")
        for partition in PARTITIONS:
            part_df = ds_df[ds_df["partition"] == partition]
            for row in part_df.to_dict(orient="records"):
                vid = row["variant_id"]
                if vid in fixed_map:
                    primary_label = fixed_map[vid]
                elif best_i1_variant is not None and vid == best_i1_variant:
                    primary_label = "I1"
                else:
                    continue
                rows.append({"primary_variant": primary_label, **row})

    out = pd.DataFrame(rows, columns=PRIMARY_7_COLUMNS)
    if out.empty:
        return out
    out["_order"] = out["primary_variant"].map(order_rank)
    out["_part_order"] = out["partition"].map(
        {p: idx for idx, p in enumerate(PARTITIONS)}
    )
    out = out.sort_values(["dataset", "_part_order", "_order"]).drop(
        columns=["_order", "_part_order"]
    )
    return out.reset_index(drop=True)


def _paper_gain_pct(metric: str, method_val: float, base_val: float) -> float:
    if not (math.isfinite(method_val) and math.isfinite(base_val)) or base_val == 0.0:
        return math.nan
    if metric in HIGHER_IS_BETTER_METRICS:
        return float((method_val - base_val) / abs(base_val) * 100.0)
    if metric in LOWER_IS_BETTER_METRICS:
        return float((base_val - method_val) / base_val * 100.0)
    raise ValueError(f"Unsupported gain metric {metric!r}.")


def build_relative_gains_table(
    master_df: pd.DataFrame,
    i1_selections: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compute absolute and paper percentage gains vs corresponding baselines."""
    rows: list[dict[str, Any]] = []

    for dataset_name, ds_df in master_df.groupby("dataset", sort=False):
        best_i1_variant = i1_selections.get(str(dataset_name), {}).get("variant_id")
        for partition in PARTITIONS:
            part_df = ds_df[ds_df["partition"] == partition]
            by_vid = {r["variant_id"]: r for r in part_df.to_dict(orient="records")}

            def _emit_comparison(
                scope: str,
                comparison: str,
                method_row: dict[str, Any],
                base_row: dict[str, Any],
            ) -> None:
                entry: dict[str, Any] = {
                    "dataset": dataset_name,
                    "partition": partition,
                    "scope": scope,
                    "comparison": comparison,
                    "variant_id": method_row["variant_id"],
                    "baseline_variant_id": base_row["variant_id"],
                    "family": method_row["family"],
                    "method": method_row["method"],
                    "model": method_row["model"],
                    "eta": method_row["eta"],
                }
                for metric in GAIN_METRICS:
                    m_val = float(method_row[metric])
                    b_val = float(base_row[metric])
                    entry[f"{metric}_method"] = m_val
                    entry[f"{metric}_baseline"] = b_val
                    entry[f"{metric}_abs_change"] = m_val - b_val
                    entry[f"{metric}_gain_pct"] = _paper_gain_pct(metric, m_val, b_val)
                rows.append(entry)

            primary_pairs: list[tuple[str, str, str]] = [
                ("P1 vs B1", "P1", "B1"),
                ("P2 vs B2", "P2", "B2"),
            ]
            if best_i1_variant is not None:
                primary_pairs.append(("I1_best_val vs B1", best_i1_variant, "B1"))
            primary_pairs.extend([("O1 vs B1", "O1", "B1"), ("O2 vs B2", "O2", "B2")])

            for label, method_vid, base_vid in primary_pairs:
                if method_vid in by_vid and base_vid in by_vid:
                    _emit_comparison(
                        "primary_7", label, by_vid[method_vid], by_vid[base_vid]
                    )

            if "B1" in by_vid:
                b1_row = by_vid["B1"]
                pr_rows = sorted(
                    (
                        r
                        for r in part_df.to_dict(orient="records")
                        if r["method"] == "prejudice_remover"
                    ),
                    key=lambda r: float(r["eta"]),
                )
                for pr_row in pr_rows:
                    _emit_comparison(
                        "pr_sweep",
                        f"{pr_row['variant_id']} vs B1",
                        pr_row,
                        b1_row,
                    )

    return pd.DataFrame(rows, columns=RELATIVE_GAINS_COLUMNS)


def _safe_rank_correlation(
    x: np.ndarray,
    y: np.ndarray,
    fn: Any,
) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n_obs = int(np.count_nonzero(mask))
    if n_obs < 2:
        return math.nan, math.nan, n_obs
    xf, yf = x[mask], y[mask]
    if np.unique(xf).size < 2 or np.unique(yf).size < 2:
        return math.nan, math.nan, n_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fn(xf, yf)
    return float(res.statistic), float(res.pvalue), n_obs


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    return _safe_rank_correlation(x, y, stats.spearmanr)


def _safe_kendall(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    tau, p_val, _ = _safe_rank_correlation(x, y, stats.kendalltau)
    return tau, p_val


def build_spearman_correlations_table(
    master_df: pd.DataFrame,
    roc_test_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute Spearman rank correlations on test across trained_15 and trained_plus_roc_sweep."""
    trained_test = master_df[
        (master_df["partition"] == "test") & (master_df["method"] != "constant_policy")
    ]
    rows: list[dict[str, Any]] = []

    for dataset_name, ds_trained in trained_test.groupby("dataset", sort=False):
        pools: list[tuple[str, pd.DataFrame]] = [("trained_15", ds_trained)]
        if roc_test_df is not None:
            ds_roc = roc_test_df[roc_test_df["dataset"] == dataset_name]
            if not ds_roc.empty:
                combined = pd.concat(
                    [
                        ds_trained[list(CORRELATION_METRICS)],
                        ds_roc[list(CORRELATION_METRICS)],
                    ],
                    ignore_index=True,
                )
                pools.append(("trained_plus_roc_sweep", combined))

        for pool_name, pool_df in pools:
            for sign_convention in ("raw", "paper_higher_is_better"):
                work = pool_df[list(CORRELATION_METRICS)].astype(float).copy()
                if sign_convention == "paper_higher_is_better":
                    for metric in LOWER_IS_BETTER_METRICS:
                        work[metric] = -work[metric]
                for mx in CORRELATION_METRICS:
                    for my in CORRELATION_METRICS:
                        rho, p_val, n_obs = _safe_spearman(
                            work[mx].to_numpy(), work[my].to_numpy()
                        )
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "partition": "test",
                                "pool": pool_name,
                                "sign_convention": sign_convention,
                                "metric_x": mx,
                                "metric_y": my,
                                "n_obs": n_obs,
                                "rho": rho,
                                "p_value": p_val,
                            }
                        )

    return pd.DataFrame(rows, columns=SPEARMAN_CORRELATIONS_COLUMNS)


def _rank_finite(values: np.ndarray) -> list[int | None]:
    """Rank only finite entries using min-rank; non-finite entries map to None."""
    finite = np.isfinite(values)
    ranks: list[int | None] = [None] * len(values)
    if np.any(finite):
        computed = stats.rankdata(values[finite], method="min").astype(int)
        for idx, rank in zip(np.flatnonzero(finite), computed, strict=True):
            ranks[int(idx)] = int(rank)
    return ranks


def build_separation_stability_tables(
    master_df: pd.DataFrame,
    i1_selections: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build F4 separation_stability.csv and separation_rank_summary.csv."""
    trained = master_df[master_df["method"] != "constant_policy"].copy()
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset_name, ds_df in trained.groupby("dataset", sort=False):
        best_i1_variant = i1_selections.get(str(dataset_name), {}).get("variant_id")
        for partition in PARTITIONS:
            part_df = ds_df[ds_df["partition"] == partition].reset_index(drop=True)
            if part_df.empty:
                continue

            sp_vals = part_df["sp_paper"].to_numpy(dtype=float)
            eo_vals = part_df["eo_gap"].to_numpy(dtype=float)
            rank_sp = _rank_finite(sp_vals)
            rank_eo = _rank_finite(eo_vals)
            rank_shift = [
                (e - s) if (s is not None and e is not None) else None
                for s, e in zip(rank_sp, rank_eo, strict=True)
            ]

            dfpr = part_df["delta_fpr"].to_numpy(dtype=float)
            dtpr = part_df["delta_tpr"].to_numpy(dtype=float)
            opp_sign = np.isfinite(dfpr) & np.isfinite(dtpr) & ((dfpr * dtpr) < 0.0)

            for idx, row in enumerate(part_df.to_dict(orient="records")):
                eo_v = float(row["eo_gap"])
                sp_v = float(row["sp_paper"])
                ratio = (
                    sp_v / eo_v if (math.isfinite(eo_v) and eo_v > 0.0) else math.nan
                )
                detail_rows.append(
                    {
                        "dataset": dataset_name,
                        "partition": partition,
                        "variant_id": row["variant_id"],
                        "family": row["family"],
                        "method": row["method"],
                        "model": row["model"],
                        "eta": row["eta"],
                        "is_i1_best_val": bool(
                            best_i1_variant is not None
                            and row["variant_id"] == best_i1_variant
                        ),
                        "selected_threshold": row["selected_threshold"],
                        "profit_eq16": row["profit_eq16"],
                        "sp_paper": sp_v,
                        "eo_gap": eo_v,
                        "delta_fpr": float(row["delta_fpr"]),
                        "delta_tpr": float(row["delta_tpr"]),
                        "abs_delta_fpr": float(row["abs_delta_fpr"]),
                        "abs_delta_tpr": float(row["abs_delta_tpr"]),
                        "opposite_sign_cancellation": bool(opp_sign[idx]),
                        "cancellation_ratio": ratio,
                        "rank_sp_paper": rank_sp[idx],
                        "rank_eo_gap": rank_eo[idx],
                        "rank_shift": rank_shift[idx],
                    }
                )

            rho, rho_p, _ = _safe_spearman(sp_vals, eo_vals)
            tau, tau_p = _safe_kendall(sp_vals, eo_vals)
            finite_shifts = [rs for rs in rank_shift if rs is not None]
            n_rank_shift_nonzero = sum(1 for rs in finite_shifts if rs != 0)
            max_abs_rank_shift = max((abs(rs) for rs in finite_shifts), default=None)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "partition": partition,
                    "n_models": int(len(part_df)),
                    "spearman_rho": rho,
                    "spearman_p_value": rho_p,
                    "kendall_tau": tau,
                    "kendall_p_value": tau_p,
                    "n_opposite_sign_cancellation": int(np.count_nonzero(opp_sign)),
                    "n_rank_shift_nonzero": n_rank_shift_nonzero,
                    "max_abs_rank_shift": max_abs_rank_shift,
                }
            )

    detail_df = pd.DataFrame(detail_rows, columns=SEPARATION_STABILITY_COLUMNS)
    for col in ("rank_sp_paper", "rank_eo_gap", "rank_shift"):
        detail_df[col] = detail_df[col].astype(pd.Int64Dtype())

    summary_df = pd.DataFrame(summary_rows, columns=SEPARATION_RANK_SUMMARY_COLUMNS)
    summary_df["max_abs_rank_shift"] = summary_df["max_abs_rank_shift"].astype(
        pd.Int64Dtype()
    )
    return detail_df, summary_df


def _pareto_mask(profit: np.ndarray, fairness: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated points (maximize profit, minimize fairness)."""
    valid = np.isfinite(profit) & np.isfinite(fairness)
    mask = np.zeros(profit.shape[0], dtype=bool)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return mask

    p = profit[valid_idx]
    f = fairness[valid_idx]
    dominates = (
        (p[:, None] >= p[None, :])
        & (f[:, None] <= f[None, :])
        & ((p[:, None] > p[None, :]) | (f[:, None] < f[None, :]))
    )
    is_dominated = np.any(dominates, axis=0)
    mask[valid_idx[~is_dominated]] = True
    return mask


def _build_pooled_test_frame(
    master_df: pd.DataFrame,
    roc_test_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine trained_15 and optional roc_sweep test rows with uniform schema."""
    trained_test = master_df[
        (master_df["partition"] == "test") & (master_df["method"] != "constant_policy")
    ].copy()
    trained_test["source_pool"] = "trained_15"
    trained_test["fairness_metric"] = ""
    trained_test["fairness_bound"] = math.nan
    trained_test["margin"] = (
        trained_test["roc_margin"] if "roc_margin" in trained_test.columns else math.nan
    )

    common_cols = list(POOLED_TEST_COLUMNS)
    frames = [trained_test[common_cols]]
    if roc_test_df is not None and not roc_test_df.empty:
        frames.append(roc_test_df[common_cols])
    return pd.concat(frames, ignore_index=True)


def build_pareto_frontiers_table(
    master_df: pd.DataFrame,
    roc_test_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute non-dominated (profit_eq16, fairness) points on test across full pool."""
    pooled = _build_pooled_test_frame(master_df, roc_test_df)
    frontier_parts: list[pd.DataFrame] = []

    for _, ds_df in pooled.groupby("dataset", sort=False):
        ds_df = ds_df.copy().reset_index(drop=True)
        profit = ds_df["profit_eq16"].to_numpy(dtype=float)
        any_pareto = np.zeros(len(ds_df), dtype=bool)
        for fairness_metric in PARETO_FAIRNESS_METRICS:
            flag_col = f"is_pareto_{fairness_metric}"
            mask = _pareto_mask(profit, ds_df[fairness_metric].to_numpy(dtype=float))
            ds_df[flag_col] = mask
            any_pareto |= mask

        selected = ds_df[any_pareto].sort_values(
            by=["profit_eq16", "sp_paper", "eo_gap", "ind"],
            ascending=[False, True, True, True],
        )
        frontier_parts.append(selected)

    if not frontier_parts:
        return pd.DataFrame(columns=PARETO_FRONTIERS_COLUMNS)
    return pd.concat(frontier_parts, ignore_index=True)[list(PARETO_FRONTIERS_COLUMNS)]


def _primary_label_map(
    ds_primary_test: pd.DataFrame,
) -> dict[str, str]:
    return {
        str(row["variant_id"]): str(row["primary_variant"])
        for row in ds_primary_test.to_dict(orient="records")
        if row["primary_variant"] in PRIMARY_7_ORDER
    }


def plot_profit_vs_fairness_per_dataset(
    pooled_test: pd.DataFrame,
    pareto_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate 1x3 profit vs fairness (SP_paper, EO_gap, IND) PNG per dataset."""
    _ensure_agg_backend()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    metric_labels = (
        ("sp_paper", r"$SP_{\mathrm{paper}}$ (lower is fairer)"),
        ("eo_gap", r"$EO_{\mathrm{gap}}$ (lower is fairer)"),
        ("ind", r"$IND$ (lower is fairer)"),
    )

    for dataset_name, ds_pool in pooled_test.groupby("dataset", sort=False):
        ds_pareto = pareto_df[pareto_df["dataset"] == dataset_name]
        ds_primary_test = primary_df[
            (primary_df["dataset"] == dataset_name)
            & (primary_df["partition"] == "test")
        ]
        approve_rows = ds_primary_test[
            ds_primary_test["primary_variant"] == "CONST_APPROVE_ALL"
        ]
        approve_profit = (
            float(approve_rows.iloc[0]["profit_eq16"])
            if not approve_rows.empty
            else math.nan
        )
        label_by_vid = _primary_label_map(ds_primary_test)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
        try:
            for ax, (metric_col, x_label) in zip(axes, metric_labels, strict=True):
                # Plot ROC sweep points first (semi-transparent) so primary/trained points stand out
                roc_pts = ds_pool[ds_pool["source_pool"] == "roc_sweep"]
                roc_finite = roc_pts[
                    np.isfinite(roc_pts[metric_col])
                    & np.isfinite(roc_pts["profit_eq16"])
                ]
                if not roc_finite.empty:
                    ax.scatter(
                        roc_finite[metric_col],
                        roc_finite["profit_eq16"],
                        c=FAMILY_COLORS["Reject Option"],
                        alpha=0.45,
                        s=30,
                        marker="o",
                        label="Reject Option (ROC sweep)",
                        zorder=2,
                    )

                trained_pts = ds_pool[ds_pool["source_pool"] == "trained_15"]
                for family in FAMILY_ORDER:
                    fam_pts = trained_pts[trained_pts["family"] == family]
                    fam_finite = fam_pts[
                        np.isfinite(fam_pts[metric_col])
                        & np.isfinite(fam_pts["profit_eq16"])
                    ]
                    if fam_finite.empty:
                        continue
                    ax.scatter(
                        fam_finite[metric_col],
                        fam_finite["profit_eq16"],
                        c=FAMILY_COLORS[family],
                        s=62,
                        edgecolors="black",
                        linewidths=0.6,
                        label=family,
                        zorder=4,
                    )

                grouped_labels: dict[tuple[float, float], list[str]] = {}
                for row in trained_pts.to_dict(orient="records"):
                    label = label_by_vid.get(str(row["variant_id"]))
                    if not label:
                        continue
                    x_val = float(row[metric_col])
                    y_val = float(row["profit_eq16"])
                    if not (math.isfinite(x_val) and math.isfinite(y_val)):
                        continue
                    coord_key = (round(x_val, 6), round(y_val, 6))
                    grouped_labels.setdefault(coord_key, []).append(label)

                for (x_val, y_val), lbls in grouped_labels.items():
                    ax.annotate(
                        " / ".join(lbls),
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                        fontweight="bold",
                        zorder=5,
                    )

                flag_col = f"is_pareto_{metric_col}"
                if not ds_pareto.empty and flag_col in ds_pareto.columns:
                    frontier = (
                        ds_pareto[ds_pareto[flag_col]]
                        .drop_duplicates(subset=[metric_col, "profit_eq16"])
                        .sort_values(
                            [metric_col, "profit_eq16"], ascending=[True, True]
                        )
                    )
                    frontier = frontier[
                        np.isfinite(frontier[metric_col])
                        & np.isfinite(frontier["profit_eq16"])
                    ]
                    if not frontier.empty:
                        ax.plot(
                            frontier[metric_col],
                            frontier["profit_eq16"],
                            color="black",
                            linestyle="-",
                            linewidth=1.5,
                            label="Pareto frontier",
                            zorder=3,
                        )

                if math.isfinite(approve_profit):
                    ax.axhline(
                        approve_profit,
                        color="#444444",
                        linestyle="--",
                        linewidth=1.2,
                        label="Approve-all",
                        zorder=1,
                    )

                ax.set_xlabel(x_label, fontsize=10)
                ax.grid(True, linestyle=":", alpha=0.5)

            axes[0].set_ylabel("Profit (Eq. 16)", fontsize=10)
            handles, labels = axes[0].get_legend_handles_labels()
            if labels:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.03),
                    ncol=max(min(len(labels), 4), 1),
                    frameon=True,
                    fontsize=8.5,
                )
            fig.suptitle(
                f"{str(dataset_name).title()} Dataset — Test Profit vs Fairness",
                y=1.10,
                fontsize=12,
                fontweight="bold",
            )
            fig.tight_layout()
            out_path = figures_dir / f"profit_vs_fairness_{dataset_name}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            saved_paths.append(out_path)
        finally:
            plt.close(fig)

    return saved_paths


def plot_separation_stability(
    sep_df: pd.DataFrame,
    sep_summary_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot SP_paper vs EO_gap and Rank(SP_paper) vs Rank(EO_gap) on test per dataset."""
    _ensure_agg_backend()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_sep = sep_df[sep_df["partition"] == "test"]
    test_sum = sep_summary_df[sep_summary_df["partition"] == "test"]
    datasets = list(dict.fromkeys(test_sep["dataset"].tolist()))

    fig, axes = plt.subplots(
        max(len(datasets), 1),
        2,
        figsize=(11.5, 4.6 * max(len(datasets), 1)),
        squeeze=False,
    )
    try:
        for row_idx, dataset_name in enumerate(datasets):
            ds_df = test_sep[test_sep["dataset"] == dataset_name]
            ds_sum = test_sum[test_sum["dataset"] == dataset_name].iloc[0]

            ax_val = axes[row_idx, 0]
            ax_rank = axes[row_idx, 1]

            finite_val_df = ds_df[
                np.isfinite(ds_df["sp_paper"]) & np.isfinite(ds_df["eo_gap"])
            ]
            for family in FAMILY_ORDER:
                fam_df = finite_val_df[finite_val_df["family"] == family]
                if fam_df.empty:
                    continue
                ax_val.scatter(
                    fam_df["sp_paper"],
                    fam_df["eo_gap"],
                    c=FAMILY_COLORS[family],
                    s=64,
                    edgecolors="black",
                    linewidths=0.6,
                    label=family,
                    zorder=3,
                )

            opp = finite_val_df[finite_val_df["opposite_sign_cancellation"]]
            if not opp.empty:
                ax_val.scatter(
                    opp["sp_paper"],
                    opp["eo_gap"],
                    facecolors="none",
                    edgecolors="black",
                    s=140,
                    linewidths=1.4,
                    label="Opposite-sign cancellation",
                    zorder=4,
                )

            sp_arr = ds_df["sp_paper"].to_numpy(dtype=float)
            eo_arr = ds_df["eo_gap"].to_numpy(dtype=float)
            finite_gaps = np.concatenate(
                [sp_arr[np.isfinite(sp_arr)], eo_arr[np.isfinite(eo_arr)]]
            )
            max_gap = (
                max(float(np.max(finite_gaps)), 0.01) if finite_gaps.size > 0 else 0.01
            )
            pad = max_gap * 0.08
            ax_val.plot(
                [0.0, max_gap + pad],
                [0.0, max_gap + pad],
                linestyle="--",
                color="#555555",
                linewidth=1.1,
                label="y = x",
                zorder=1,
            )
            ax_val.set_xlim(-0.002, max_gap + pad)
            ax_val.set_ylim(-0.002, max_gap + pad)
            ax_val.set_xlabel(r"$SP_{\mathrm{paper}}$ (test)", fontsize=10)
            ax_val.set_ylabel(r"$EO_{\mathrm{gap}}$ (test)", fontsize=10)
            ax_val.set_title(
                f"{dataset_name.title()} — Gap Magnitude "
                f"(opp-sign={int(ds_sum['n_opposite_sign_cancellation'])}/{int(ds_sum['n_models'])})",
                fontsize=10.5,
            )
            ax_val.grid(True, linestyle=":", alpha=0.5)
            if row_idx == 0:
                ax_val.legend(loc="lower right", fontsize=8, frameon=True)

            finite_rank_df = ds_df[
                ds_df["rank_sp_paper"].notna()
                & ds_df["rank_eo_gap"].notna()
                & ds_df["rank_shift"].notna()
            ]
            unshifted = finite_rank_df[finite_rank_df["rank_shift"] == 0]
            shifted = finite_rank_df[finite_rank_df["rank_shift"] != 0]
            if not unshifted.empty:
                ax_rank.scatter(
                    unshifted["rank_sp_paper"],
                    unshifted["rank_eo_gap"],
                    c="#4c72b0",
                    s=64,
                    edgecolors="black",
                    linewidths=0.6,
                    label="No rank shift",
                    zorder=3,
                )

            rank_sp_valid = ds_df["rank_sp_paper"].dropna().to_numpy(dtype=int)
            rank_eo_valid = ds_df["rank_eo_gap"].dropna().to_numpy(dtype=int)
            valid_ranks = np.concatenate([rank_sp_valid, rank_eo_valid])
            max_rank = max(int(np.max(valid_ranks)), 1) if valid_ranks.size > 0 else 1

            if not shifted.empty:
                ax_rank.scatter(
                    shifted["rank_sp_paper"],
                    shifted["rank_eo_gap"],
                    c="#d62728",
                    marker="D",
                    s=72,
                    edgecolors="black",
                    linewidths=0.7,
                    label="Rank reversal (shift != 0)",
                    zorder=4,
                )
                coord_counts: dict[tuple[int, int], int] = {}
                for row in shifted.to_dict(orient="records"):
                    x_r = int(row["rank_sp_paper"])
                    y_r = int(row["rank_eo_gap"])
                    count = coord_counts.get((x_r, y_r), 0)
                    coord_counts[(x_r, y_r)] = count + 1
                    offset_cycle = (
                        ((-46, 6), (-46, -10), (-46, 16), (-46, -20))
                        if x_r >= max_rank - 1
                        else ((5, 4), (5, -10), (5, 14), (5, -20))
                    )
                    offset = offset_cycle[count % len(offset_cycle)]
                    ax_rank.annotate(
                        f"{row['variant_id']} ({int(row['rank_shift']):+d})",
                        (float(x_r), float(y_r)),
                        textcoords="offset points",
                        xytext=offset,
                        fontsize=7.5,
                        zorder=5,
                    )

            ax_rank.plot(
                [0.5, max_rank + 0.5],
                [0.5, max_rank + 0.5],
                linestyle="--",
                color="#555555",
                linewidth=1.1,
                label="y = x",
                zorder=1,
            )
            ax_rank.set_xlim(0.4, max_rank + 1.4)
            ax_rank.set_ylim(0.4, max_rank + 1.0)
            ax_rank.set_xlabel(
                r"$\mathrm{Rank}(SP_{\mathrm{paper}})$ (1 = fairest)", fontsize=10
            )
            ax_rank.set_ylabel(
                r"$\mathrm{Rank}(EO_{\mathrm{gap}})$ (1 = fairest)", fontsize=10
            )
            max_abs_shift = ds_sum["max_abs_rank_shift"]
            max_shift_str = "NA" if pd.isna(max_abs_shift) else str(int(max_abs_shift))
            ax_rank.set_title(
                f"{dataset_name.title()} — Rank Stability "
                f"(rho={ds_sum['spearman_rho']:.3f}, tau={ds_sum['kendall_tau']:.3f}, "
                f"max|shift|={max_shift_str})",
                fontsize=10.5,
            )
            ax_rank.grid(True, linestyle=":", alpha=0.5)
            if row_idx == 0:
                ax_rank.legend(loc="lower right", fontsize=8, frameon=True)

        fig.suptitle(
            "F4 Separation Stability — SP_paper vs EO_gap Across 15 Trained Models (Test)",
            y=1.01,
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        out_path = figures_dir / "separation_stability.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        return out_path
    finally:
        plt.close(fig)


def plot_method_comparison_7_variants(
    primary_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot grouped bar comparison of B1, B2, P1, P2, I1, O1, O2 on test across datasets."""
    _ensure_agg_backend()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_df = primary_df[primary_df["partition"] == "test"]
    datasets = list(dict.fromkeys(test_df["dataset"].tolist()))

    metrics_spec = (
        ("profit_eq16", "Profit (Eq. 16)"),
        ("sp_paper", r"$SP_{\mathrm{paper}}$"),
        ("eo_gap", r"$EO_{\mathrm{gap}}$"),
        ("ind", r"$IND$"),
    )
    variant_colors = {
        "B1": "#1f77b4",
        "B2": "#6baed6",
        "P1": "#2ca02c",
        "P2": "#74c476",
        "I1": "#ff7f0e",
        "O1": "#d62728",
        "O2": "#fb6a4a",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    try:
        x_base = np.arange(len(datasets), dtype=float)
        n_bars = len(PRIMARY_7_ORDER)
        bar_width = 0.11
        offsets = (np.arange(n_bars, dtype=float) - (n_bars - 1) / 2.0) * bar_width

        for ax, (metric_col, title) in zip(axes.flat, metrics_spec, strict=True):
            plotted_values: list[float] = []
            for idx, variant in enumerate(PRIMARY_7_ORDER):
                vals: list[float] = []
                for ds in datasets:
                    match = test_df[
                        (test_df["dataset"] == ds)
                        & (test_df["primary_variant"] == variant)
                    ]
                    val = (
                        float(match.iloc[0][metric_col])
                        if not match.empty
                        else math.nan
                    )
                    vals.append(val)
                    if math.isfinite(val):
                        plotted_values.append(val)
                ax.bar(
                    x_base + offsets[idx],
                    vals,
                    width=bar_width * 0.92,
                    color=variant_colors[variant],
                    edgecolor="black",
                    linewidth=0.4,
                    label=variant,
                    zorder=3,
                )

            if metric_col == "profit_eq16":
                half_span = (n_bars * bar_width) / 2.0 + 0.03
                for ds_idx, ds in enumerate(datasets):
                    approve_match = test_df[
                        (test_df["dataset"] == ds)
                        & (test_df["primary_variant"] == "CONST_APPROVE_ALL")
                    ]
                    if approve_match.empty:
                        continue
                    approve_val = float(approve_match.iloc[0]["profit_eq16"])
                    if math.isfinite(approve_val):
                        plotted_values.append(approve_val)
                        ax.hlines(
                            approve_val,
                            x_base[ds_idx] - half_span,
                            x_base[ds_idx] + half_span,
                            colors="#222222",
                            linestyles="--",
                            linewidth=1.5,
                            label="Approve-all" if ds_idx == 0 else None,
                            zorder=4,
                        )
                if plotted_values:
                    lo = min(plotted_values)
                    hi = max(plotted_values)
                    margin = max((hi - lo) * 0.25, 0.002)
                    ax.set_ylim(lo - margin, hi + margin)

            ax.set_xticks(x_base)
            ax.set_xticklabels([ds.title() for ds in datasets], fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if labels:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=max(len(labels), 1),
                frameon=True,
                fontsize=9,
            )
        fig.suptitle(
            "Primary 7 Variants Comparison on Test Set",
            y=1.04,
            fontsize=12.5,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_path = figures_dir / "method_comparison_7_variants.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        return out_path
    finally:
        plt.close(fig)


def _to_json_safe(value: Any) -> Any:
    """Recursively convert pandas/NumPy values and non-finite floats for JSON."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return value


def build_analysis_report(
    *,
    split_ids: dict[str, str],
    i1_selections: dict[str, dict[str, Any]],
    primary_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    sep_df: pd.DataFrame,
    sep_summary_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    pooled_test: pd.DataFrame,
    csv_paths: dict[str, Path],
    figure_paths: dict[str, Path],
    report_path: Path,
) -> dict[str, Any]:
    """Assemble JSON-safe analysis_report.json payload."""
    datasets_report: dict[str, Any] = {}

    for dataset_name, split_id in split_ids.items():
        ds_primary = primary_df[primary_df["dataset"] == dataset_name]
        primary_by_partition: dict[str, list[dict[str, Any]]] = {}
        for partition in PARTITIONS:
            part_rows = ds_primary[ds_primary["partition"] == partition]
            primary_by_partition[partition] = [
                {
                    "primary_variant": r["primary_variant"],
                    "variant_id": r["variant_id"],
                    "method": r["method"],
                    "model": r["model"],
                    "eta": r["eta"],
                    "base_threshold": r["base_threshold"],
                    "selected_threshold": r["selected_threshold"],
                    "roc_margin": r["roc_margin"],
                    "profit_eq16": r["profit_eq16"],
                    "delta_profit_vs_own_baseline": r["delta_profit_vs_own_baseline"],
                    "delta_profit_vs_approve_all": r["delta_profit_vs_approve_all"],
                    "auc": r["auc"],
                    "approval_rate": r["approval_rate"],
                    "ind": r["ind"],
                    "sp_paper": r["sp_paper"],
                    "eo_gap": r["eo_gap"],
                    "sf": r["sf"],
                }
                for r in part_rows.to_dict(orient="records")
            ]

        ds_spearman = spearman_df[spearman_df["dataset"] == dataset_name]
        corr_highlights: dict[str, Any] = {}
        highlight_pairs = (
            ("profit_eq16", "auc"),
            ("profit_eq16", "sp_paper"),
            ("profit_eq16", "eo_gap"),
            ("profit_eq16", "ind"),
            ("profit_eq16", "sf"),
            ("sp_paper", "eo_gap"),
            ("sp_paper", "ind"),
            ("sp_paper", "sf"),
        )
        for pool_name in ds_spearman["pool"].unique().tolist():
            pool_block: dict[str, Any] = {}
            for conv in ("paper_higher_is_better", "raw"):
                sub = ds_spearman[
                    (ds_spearman["pool"] == pool_name)
                    & (ds_spearman["sign_convention"] == conv)
                ]
                pair_map: dict[str, Any] = {}
                for mx, my in highlight_pairs:
                    match = sub[(sub["metric_x"] == mx) & (sub["metric_y"] == my)]
                    if not match.empty:
                        rec = match.iloc[0]
                        pair_map[f"{mx}__vs__{my}"] = {
                            "n_obs": int(rec["n_obs"]),
                            "rho": rec["rho"],
                            "p_value": rec["p_value"],
                        }
                pool_block[conv] = pair_map
            corr_highlights[pool_name] = pool_block

        sep_report: dict[str, Any] = {}
        for partition in PARTITIONS:
            sum_match = sep_summary_df[
                (sep_summary_df["dataset"] == dataset_name)
                & (sep_summary_df["partition"] == partition)
            ]
            if sum_match.empty:
                continue
            sum_rec = sum_match.iloc[0].to_dict()
            det_match = sep_df[
                (sep_df["dataset"] == dataset_name)
                & (sep_df["partition"] == partition)
                & (sep_df["rank_shift"].ne(0).fillna(False))
            ].copy()
            det_match["abs_shift"] = det_match["rank_shift"].abs()
            det_match = det_match.sort_values(
                ["abs_shift", "variant_id"], ascending=[False, True]
            ).head(5)
            examples = [
                {
                    "variant_id": r["variant_id"],
                    "model": r["model"],
                    "sp_paper": r["sp_paper"],
                    "eo_gap": r["eo_gap"],
                    "delta_fpr": r["delta_fpr"],
                    "delta_tpr": r["delta_tpr"],
                    "opposite_sign_cancellation": r["opposite_sign_cancellation"],
                    "cancellation_ratio": r["cancellation_ratio"],
                    "rank_sp_paper": r["rank_sp_paper"],
                    "rank_eo_gap": r["rank_eo_gap"],
                    "rank_shift": r["rank_shift"],
                }
                for r in det_match.to_dict(orient="records")
            ]
            sep_report[partition] = {
                "n_models": sum_rec["n_models"],
                "spearman_rho": sum_rec["spearman_rho"],
                "spearman_p_value": sum_rec["spearman_p_value"],
                "kendall_tau": sum_rec["kendall_tau"],
                "kendall_p_value": sum_rec["kendall_p_value"],
                "n_opposite_sign_cancellation": sum_rec["n_opposite_sign_cancellation"],
                "n_rank_shift_nonzero": sum_rec["n_rank_shift_nonzero"],
                "max_abs_rank_shift": sum_rec["max_abs_rank_shift"],
                "rank_reversal_examples": examples,
            }

        ds_pool = pooled_test[pooled_test["dataset"] == dataset_name]
        ds_pareto = pareto_df[pareto_df["dataset"] == dataset_name]
        pareto_counts = {
            metric: int(ds_pareto[f"is_pareto_{metric}"].sum())
            if not ds_pareto.empty
            else 0
            for metric in PARETO_FAIRNESS_METRICS
        }

        datasets_report[dataset_name] = {
            "split_id": split_id,
            "n_trained_models": int((ds_pool["source_pool"] == "trained_15").sum()),
            "n_roc_sweep_test": int((ds_pool["source_pool"] == "roc_sweep").sum()),
            "i1_best_val": i1_selections.get(dataset_name),
            "primary_7_variants": primary_by_partition,
            "correlation_highlights": corr_highlights,
            "separation_stability": sep_report,
            "pareto_counts": pareto_counts,
        }

    payload = _to_json_safe(
        {
            "datasets": datasets_report,
            "generated_files": {
                "csvs": {k: p.as_posix() for k, p in csv_paths.items()},
                "figures": {k: p.as_posix() for k, p in figure_paths.items()},
                "report_json": report_path.as_posix(),
            },
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_dataframe_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep="NaN")
    return path


def run_analysis(
    run_dirs: list[Path] | tuple[Path, ...] = DEFAULT_RUN_DIRS,
    *,
    roc_sweep_csv: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    require_roc_sweep_rows: bool | None = None,
) -> dict[str, Any]:
    """Execute Stage C analysis, F4 stability tables, figures, and JSON report."""
    if not run_dirs:
        raise ValueError("At least one run directory must be provided.")

    by_dataset = _group_runs_by_dataset([Path(p) for p in run_dirs])
    master_df, split_ids = _evaluate_all_runs(by_dataset)
    explicit_roc = (
        roc_sweep_csv is not None
        if require_roc_sweep_rows is None
        else require_roc_sweep_rows
    )
    roc_test_df = _load_and_verify_roc_sweep(
        roc_sweep_csv,
        split_ids,
        require_test_rows=explicit_roc,
    )
    i1_selections = _select_i1_best_val(master_df)

    primary_df = build_primary_7_table(master_df, i1_selections)
    gains_df = build_relative_gains_table(master_df, i1_selections)
    spearman_df = build_spearman_correlations_table(master_df, roc_test_df)
    sep_df, sep_summary_df = build_separation_stability_tables(master_df, i1_selections)
    pareto_df = build_pareto_frontiers_table(master_df, roc_test_df)
    pooled_test = _build_pooled_test_frame(master_df, roc_test_df)

    out_dir = Path(output_dir)
    csv_paths = {
        "master_summary": _write_dataframe_csv(
            master_df, out_dir / "master_summary.csv"
        ),
        "primary_7_variants": _write_dataframe_csv(
            primary_df, out_dir / "primary_7_variants.csv"
        ),
        "relative_gains_vs_baseline": _write_dataframe_csv(
            gains_df, out_dir / "relative_gains_vs_baseline.csv"
        ),
        "spearman_correlations": _write_dataframe_csv(
            spearman_df, out_dir / "spearman_correlations.csv"
        ),
        "separation_stability": _write_dataframe_csv(
            sep_df, out_dir / "separation_stability.csv"
        ),
        "separation_rank_summary": _write_dataframe_csv(
            sep_summary_df, out_dir / "separation_rank_summary.csv"
        ),
        "pareto_frontiers": _write_dataframe_csv(
            pareto_df, out_dir / "pareto_frontiers.csv"
        ),
    }

    pvf_paths = plot_profit_vs_fairness_per_dataset(
        pooled_test, pareto_df, primary_df, out_dir
    )
    sep_fig_path = plot_separation_stability(sep_df, sep_summary_df, out_dir)
    cmp_fig_path = plot_method_comparison_7_variants(primary_df, out_dir)

    figure_paths: dict[str, Path] = {p.stem: p for p in pvf_paths}
    figure_paths[sep_fig_path.stem] = sep_fig_path
    figure_paths[cmp_fig_path.stem] = cmp_fig_path

    report_path = out_dir / "analysis_report.json"
    return build_analysis_report(
        split_ids=split_ids,
        i1_selections=i1_selections,
        primary_df=primary_df,
        spearman_df=spearman_df,
        sep_df=sep_df,
        sep_summary_df=sep_summary_df,
        pareto_df=pareto_df,
        pooled_test=pooled_test,
        csv_paths=csv_paths,
        figure_paths=figure_paths,
        report_path=report_path,
    )


def _resolve_roc_sweep_argument(
    roc_sweep_csv: Path | None,
    no_roc_sweep: bool,
) -> tuple[Path | None, bool]:
    if no_roc_sweep:
        return None, False
    if roc_sweep_csv is not None:
        if not roc_sweep_csv.is_file():
            raise FileNotFoundError(f"ROC sweep CSV not found: {roc_sweep_csv}")
        return roc_sweep_csv, True
    if DEFAULT_ROC_SWEEP_CSV.is_file():
        return DEFAULT_ROC_SWEEP_CSV, False
    return None, False


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for Stage C analysis and F4 separation stability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_RUN_DIRS),
        help="Aligned run directories to analyze (default: all 8 German & Taiwan runs).",
    )
    roc_group = parser.add_mutually_exclusive_group()
    roc_group.add_argument(
        "--roc-sweep-csv",
        type=Path,
        default=None,
        help="Optional ROC sweep summary CSV (default: outputs/roc-sweep-summary.csv if present).",
    )
    roc_group.add_argument(
        "--no-roc-sweep",
        action="store_true",
        help="Disable loading outputs/roc-sweep-summary.csv even if it exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSVs, figures/, and analysis_report.json (default: outputs/analysis).",
    )
    args = parser.parse_args(argv)
    roc_csv, explicit_roc = _resolve_roc_sweep_argument(
        args.roc_sweep_csv, args.no_roc_sweep
    )
    report = run_analysis(
        args.run_dirs,
        roc_sweep_csv=roc_csv,
        output_dir=args.output_dir,
        require_roc_sweep_rows=explicit_roc,
    )
    datasets = ", ".join(sorted(report["datasets"].keys()))
    print(f"Stage C analysis complete for datasets [{datasets}] -> {args.output_dir}")


if __name__ == "__main__":
    main()
