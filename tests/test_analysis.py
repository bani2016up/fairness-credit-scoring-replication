import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fair_credit_scoring import DatasetConfig, ModelConfig, RunConfig, run
from fair_credit_scoring.analysis import (
    GAIN_METRICS,
    HIGHER_IS_BETTER_METRICS,
    LOWER_IS_BETTER_METRICS,
    MASTER_SUMMARY_COLUMNS,
    PARETO_FRONTIERS_COLUMNS,
    PRIMARY_7_COLUMNS,
    RELATIVE_GAINS_COLUMNS,
    SEPARATION_RANK_SUMMARY_COLUMNS,
    SEPARATION_STABILITY_COLUMNS,
    SPEARMAN_CORRELATIONS_COLUMNS,
    _evaluate_all_runs,
    _map_variant_id,
    _paper_gain_pct,
    _pareto_mask,
    _safe_kendall,
    _safe_spearman,
    _select_i1_best_val,
    _verify_saved_metrics,
    build_analysis_report,
    build_pareto_frontiers_table,
    build_primary_7_table,
    build_relative_gains_table,
    build_separation_stability_tables,
    build_spearman_correlations_table,
    main,
    plot_method_comparison_7_variants,
    plot_profit_vs_fairness_per_dataset,
    plot_separation_stability,
    run_analysis,
)
from fair_credit_scoring.handoff import _group_runs_by_dataset
from fair_credit_scoring.metrics import METRICS
from fair_credit_scoring.roc_experiment import run_roc_experiment

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.csv"


def _build_dataset_suite(
    output_root: Path,
    dataset_name: str = "synthetic-tiny",
    seed: int = 37,
) -> list[Path]:
    """Create aligned baseline, reweighing, PR sweep, and ROC runs for one dataset."""
    base_ds = DatasetConfig(
        name=dataset_name,
        path=FIXTURE,
        numeric=("income", "balance", "tenure"),
        categorical=("channel", "region"),
        scaling="max_abs",
    )
    lr_rf = (
        ModelConfig("lr", "logistic_regression", {"C": [0.5, 1.0]}),
        ModelConfig("rf", "random_forest", {"n_estimators": [12], "max_depth": [4]}),
    )
    thresholds = (0.0, 0.25, 0.5, 0.75, 1.0)
    all_metrics = tuple(METRICS.keys())

    baseline_dir = run(
        RunConfig(
            dataset=base_ds,
            models=lr_rf,
            run_id=f"{dataset_name}-baseline-profit",
            output_dir=output_root,
            seed=seed,
            thresholds=thresholds,
            selection_metric="profit_eq16",
            metrics=all_metrics,
            method="baseline",
        )
    )
    reweighing_dir = run(
        RunConfig(
            dataset=base_ds,
            models=lr_rf,
            run_id=f"{dataset_name}-reweighing-profit",
            output_dir=output_root,
            seed=seed,
            thresholds=thresholds,
            selection_metric="profit_eq16",
            metrics=all_metrics,
            method="reweighing",
            weight="reweighing",
        )
    )
    pr_dir = run(
        RunConfig(
            dataset=base_ds,
            models=(
                ModelConfig("pr_eta_0", "prejudice_remover", {"eta": [0.0]}),
                ModelConfig("pr_eta_5", "prejudice_remover", {"eta": [5.0]}),
                ModelConfig("pr_eta_30", "prejudice_remover", {"eta": [30.0]}),
            ),
            run_id=f"{dataset_name}-pr-sweep",
            output_dir=output_root,
            seed=seed,
            thresholds=thresholds,
            selection_metric="profit_eq16",
            metrics=all_metrics,
            method="prejudice_remover",
        )
    )
    roc_dir = run(
        RunConfig(
            dataset=base_ds,
            models=lr_rf,
            run_id=f"{dataset_name}-roc-profit",
            output_dir=output_root,
            seed=seed,
            thresholds=thresholds,
            selection_metric="profit_eq16",
            metrics=all_metrics,
            method="reject_option",
            postprocessor="reject_option",
            postprocessor_params={
                "fairness_metric": "sp_paper",
                "fairness_bound": 0.05,
                "utility_metric": "profit_eq16",
            },
        )
    )
    return [baseline_dir, reweighing_dir, pr_dir, roc_dir]


def _build_synthetic_suite(tmp_path: Path) -> tuple[list[Path], Path]:
    """Create aligned baseline, reweighing, PR sweep, ROC runs, and ROC sweep CSV."""
    output_root = tmp_path / "outputs"
    run_dirs = _build_dataset_suite(output_root, dataset_name="synthetic-tiny", seed=37)
    roc_sweep_csv = output_root / "roc-sweep-summary.csv"
    run_roc_experiment(
        [run_dirs[0]],
        roc_sweep_csv,
        fairness_metrics=("sp_paper", "ind"),
        fairness_bounds=(0.1, 0.05, 0.01),
    )
    return run_dirs, roc_sweep_csv


def test_map_variant_id_and_helpers():
    assert _map_variant_id("baseline", "lr") == "B1"
    assert _map_variant_id("baseline", "rf") == "B2"
    assert _map_variant_id("reweighing", "lr") == "P1"
    assert _map_variant_id("reweighing", "rf") == "P2"
    assert _map_variant_id("reject_option", "lr") == "O1"
    assert _map_variant_id("reject_option", "rf") == "O2"
    assert _map_variant_id("prejudice_remover", "pr_eta_5", 5.0) == "I1_eta_5"
    assert _map_variant_id("prejudice_remover", "pr_eta_0.5", "") == "I1_eta_0.5"
    assert _map_variant_id("constant_policy", "approve-all") == "CONST_APPROVE_ALL"
    assert _map_variant_id("constant_policy", "reject-all") == "CONST_REJECT_ALL"
    with pytest.raises(ValueError, match="Unsupported"):
        _map_variant_id("baseline", "xgb")


def test_paper_gain_pct_conventions():
    assert _paper_gain_pct("profit_eq16", 0.12, 0.10) == pytest.approx(20.0)
    # Negative profit baseline uses |base| in denominator
    assert _paper_gain_pct("profit_eq16", -0.05, -0.10) == pytest.approx(50.0)
    assert _paper_gain_pct("sp_paper", 0.02, 0.05) == pytest.approx(60.0)
    assert math.isnan(_paper_gain_pct("sp_paper", 0.02, 0.0))
    assert math.isnan(_paper_gain_pct("auc", math.nan, 0.7))


def test_pareto_mask_ties_dominance_and_nan():
    profit = np.array([0.10, 0.10, 0.10, 0.12, math.nan])
    fairness = np.array([0.02, 0.02, 0.03, 0.04, 0.00])
    mask = _pareto_mask(profit, fairness)
    assert mask.tolist() == [True, True, False, True, False]


def test_run_analysis_end_to_end_with_synthetic_suite(tmp_path):
    run_dirs, roc_sweep_csv = _build_synthetic_suite(tmp_path)
    out_dir = tmp_path / "analysis_out"

    report = run_analysis(
        run_dirs,
        roc_sweep_csv=roc_sweep_csv,
        output_dir=out_dir,
    )

    expected_csvs = (
        "master_summary.csv",
        "primary_7_variants.csv",
        "relative_gains_vs_baseline.csv",
        "spearman_correlations.csv",
        "separation_stability.csv",
        "separation_rank_summary.csv",
        "pareto_frontiers.csv",
    )
    for name in expected_csvs:
        assert (out_dir / name).is_file(), f"Missing {name}"

    expected_figs = (
        out_dir / "figures" / "profit_vs_fairness_synthetic-tiny.png",
        out_dir / "figures" / "separation_stability.png",
        out_dir / "figures" / "method_comparison_7_variants.png",
    )
    for fig_path in expected_figs:
        assert fig_path.is_file(), f"Missing {fig_path}"
        assert fig_path.read_bytes()[:4] == b"\x89PNG"

    report_path = out_dir / "analysis_report.json"
    assert report_path.is_file()
    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    assert "synthetic-tiny" in parsed["datasets"]
    assert parsed == report

    master_df = pd.read_csv(out_dir / "master_summary.csv")
    # 9 trained models (2 baseline + 2 reweighing + 3 PR + 2 ROC) + 2 constant policies = 11 per partition
    assert len(master_df) == 22
    assert set(master_df["partition"]) == {"validation", "test"}

    # Verify delta_fpr / delta_tpr consistency
    np.testing.assert_allclose(
        master_df["delta_fpr"],
        master_df["fpr_1"] - master_df["fpr_0"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        master_df["abs_delta_tpr"],
        np.abs(master_df["tpr_1"] - master_df["tpr_0"]),
        equal_nan=True,
    )

    # Verify I1_best_val matches max validation profit_eq16 with smaller eta tie-break
    pr_val = master_df[
        (master_df["method"] == "prejudice_remover")
        & (master_df["partition"] == "validation")
    ]
    max_val_profit = pr_val["profit_eq16"].max()
    expected_i1 = (
        pr_val[pr_val["profit_eq16"] == max_val_profit]
        .sort_values("eta", ascending=True)
        .iloc[0]["variant_id"]
    )
    assert (
        parsed["datasets"]["synthetic-tiny"]["i1_best_val"]["variant_id"] == expected_i1
    )

    primary_df = pd.read_csv(out_dir / "primary_7_variants.csv")
    assert len(primary_df) == 16  # (7 + approve-all) * 2 partitions
    assert set(primary_df["primary_variant"]) == {
        "B1",
        "B2",
        "P1",
        "P2",
        "I1",
        "O1",
        "O2",
        "CONST_APPROVE_ALL",
    }

    gains_df = pd.read_csv(out_dir / "relative_gains_vs_baseline.csv")
    assert set(gains_df["scope"]) == {"primary_7", "pr_sweep"}
    # 5 primary comparisons + 3 PR sweep comparisons per partition = 16
    assert len(gains_df) == 16

    # AUC must be unchanged between baseline (B1/B2) and ROC (O1/O2 and roc_sweep)
    for partition in ("validation", "test"):
        part_master = master_df[master_df["partition"] == partition].set_index(
            "variant_id"
        )
        assert part_master.loc["O1", "auc"] == pytest.approx(
            part_master.loc["B1", "auc"]
        )
        assert part_master.loc["O2", "auc"] == pytest.approx(
            part_master.loc["B2", "auc"]
        )
    test_master = master_df[master_df["partition"] == "test"].set_index("variant_id")
    roc_test = pd.read_csv(roc_sweep_csv)
    roc_test = roc_test[roc_test["partition"] == "test"]
    assert not roc_test.empty
    for model_name, base_vid in (("lr", "B1"), ("rf", "B2")):
        sub_auc = roc_test.loc[roc_test["model"] == model_name, "auc"]
        assert not sub_auc.empty
        np.testing.assert_allclose(sub_auc, float(test_master.loc[base_vid, "auc"]))

    # Numerical assertions on relative_gains_vs_baseline.csv
    for metric in GAIN_METRICS:
        m_vals = gains_df[f"{metric}_method"].to_numpy(dtype=float)
        b_vals = gains_df[f"{metric}_baseline"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            gains_df[f"{metric}_abs_change"],
            m_vals - b_vals,
            atol=1e-12,
            equal_nan=True,
        )
        expected_pct = np.array(
            [_paper_gain_pct(metric, m, b) for m, b in zip(m_vals, b_vals, strict=True)]
        )
        np.testing.assert_allclose(
            gains_df[f"{metric}_gain_pct"], expected_pct, atol=1e-12, equal_nan=True
        )
    roc_gains = gains_df[gains_df["comparison"].isin(["O1 vs B1", "O2 vs B2"])]
    assert len(roc_gains) == 4
    np.testing.assert_allclose(roc_gains["auc_abs_change"], 0.0, atol=1e-12)
    np.testing.assert_allclose(roc_gains["auc_gain_pct"], 0.0, atol=1e-12)

    spearman_df = pd.read_csv(out_dir / "spearman_correlations.csv")
    assert set(spearman_df["pool"]) == {"trained_15", "trained_plus_roc_sweep"}
    assert set(spearman_df["sign_convention"]) == {"raw", "paper_higher_is_better"}

    sep_df = pd.read_csv(out_dir / "separation_stability.csv")
    assert len(sep_df) == 18  # 9 trained models * 2 partitions
    assert (
        sep_df["rank_shift"] == sep_df["rank_eo_gap"] - sep_df["rank_sp_paper"]
    ).all()

    sep_sum_df = pd.read_csv(out_dir / "separation_rank_summary.csv")
    assert len(sep_sum_df) == 2

    pareto_df = pd.read_csv(out_dir / "pareto_frontiers.csv")
    assert not pareto_df.empty
    assert (
        pareto_df["is_pareto_sp_paper"]
        | pareto_df["is_pareto_eo_gap"]
        | pareto_df["is_pareto_ind"]
    ).all()


def test_cli_no_roc_sweep_and_mismatch_guards(tmp_path):
    run_dirs, roc_sweep_csv = _build_synthetic_suite(tmp_path)
    out_dir = tmp_path / "analysis_no_roc"

    main(
        [
            "--run-dirs",
            *[str(p) for p in run_dirs],
            "--no-roc-sweep",
            "--output-dir",
            str(out_dir),
        ]
    )
    spearman_df = pd.read_csv(out_dir / "spearman_correlations.csv")
    assert set(spearman_df["pool"]) == {"trained_15"}
    report = json.loads((out_dir / "analysis_report.json").read_text(encoding="utf-8"))
    assert report["datasets"]["synthetic-tiny"]["n_roc_sweep_test"] == 0

    # Tamper split_id in ROC sweep CSV -> must fail loudly
    bad_roc_csv = tmp_path / "bad_roc.csv"
    roc_df = pd.read_csv(roc_sweep_csv)
    roc_df["split_id"] = "wrong-split-id"
    roc_df.to_csv(bad_roc_csv, index=False)
    with pytest.raises(ValueError, match="split_id mismatch"):
        run_analysis(
            run_dirs, roc_sweep_csv=bad_roc_csv, output_dir=tmp_path / "bad_out"
        )

    # Missing explicit ROC CSV -> FileNotFoundError
    with pytest.raises(FileNotFoundError, match="ROC sweep CSV not found"):
        main(
            [
                "--run-dirs",
                *[str(p) for p in run_dirs],
                "--roc-sweep-csv",
                str(tmp_path / "does_not_exist.csv"),
                "--output-dir",
                str(tmp_path / "unused"),
            ]
        )


def test_unaligned_runs_raise_value_error(tmp_path):
    run_dirs, _ = _build_synthetic_suite(tmp_path)
    other_dir = run(
        RunConfig(
            dataset=DatasetConfig(
                name="synthetic-tiny",
                path=FIXTURE,
                numeric=("income", "balance", "tenure"),
                categorical=("channel", "region"),
                scaling="max_abs",
            ),
            models=(ModelConfig("lr", "logistic_regression", {"C": [1.0]}),),
            run_id="tiny-other-seed",
            output_dir=tmp_path / "other_outputs",
            seed=99,
            thresholds=(0.0, 0.5, 1.0),
            selection_metric="profit_eq16",
            metrics=tuple(METRICS.keys()),
            method="baseline",
        )
    )
    with pytest.raises(ValueError, match="seed mismatch"):
        run_analysis([run_dirs[0], other_dir], output_dir=tmp_path / "unaligned_out")


def test_o1_o2_threshold_and_margin_from_postprocessor(tmp_path):
    run_dirs, _ = _build_synthetic_suite(tmp_path)
    roc_dir = run_dirs[3]
    roc_meta = json.loads((roc_dir / "run.json").read_text(encoding="utf-8"))
    out_dir = tmp_path / "analysis_o_thresh"

    run_analysis(run_dirs, output_dir=out_dir)
    master_df = pd.read_csv(out_dir / "master_summary.csv")

    for variant_id, model_name in (("O1", "lr"), ("O2", "rf")):
        sel = roc_meta["selected"][model_name]
        postproc = sel["postprocessor"]
        row = master_df[
            (master_df["partition"] == "test") & (master_df["variant_id"] == variant_id)
        ].iloc[0]
        assert row["base_threshold"] == pytest.approx(float(sel["threshold"]))
        assert row["selected_threshold"] == pytest.approx(float(postproc["threshold"]))
        assert row["roc_margin"] == pytest.approx(float(postproc["margin"]))


def test_duplicate_run_dir_and_duplicate_variant_raise(tmp_path):
    run_dirs, _ = _build_synthetic_suite(tmp_path)

    # Duplicate resolved run_dir path
    with pytest.raises(ValueError, match="Duplicate run directory"):
        run_analysis([run_dirs[0], run_dirs[0]], output_dir=tmp_path / "dup_dir_out")

    # Distinct run_dir with identical dataset + variant_id (e.g. duplicate baseline B1/B2)
    dup_baseline = run(
        RunConfig(
            dataset=DatasetConfig(
                name="synthetic-tiny",
                path=FIXTURE,
                numeric=("income", "balance", "tenure"),
                categorical=("channel", "region"),
                scaling="max_abs",
            ),
            models=(
                ModelConfig("lr", "logistic_regression", {"C": [0.5, 1.0]}),
                ModelConfig(
                    "rf", "random_forest", {"n_estimators": [12], "max_depth": [4]}
                ),
            ),
            run_id="tiny-baseline-duplicate",
            output_dir=tmp_path / "dup_outputs",
            seed=37,
            thresholds=(0.0, 0.25, 0.5, 0.75, 1.0),
            selection_metric="profit_eq16",
            metrics=tuple(METRICS.keys()),
            method="baseline",
        )
    )
    with pytest.raises(
        ValueError, match="Duplicate \\(dataset, partition, variant_id\\)"
    ):
        run_analysis([run_dirs[0], dup_baseline], output_dir=tmp_path / "dup_vid_out")


def test_eta_float_fallback_from_pr_model_name(tmp_path):
    run_dirs, _ = _build_synthetic_suite(tmp_path)
    pr_dir = run_dirs[2]
    run_json_path = pr_dir / "run.json"
    meta = json.loads(run_json_path.read_text(encoding="utf-8"))
    # Strip eta from params of pr_eta_5 to force suffix fallback
    meta["selected"]["pr_eta_5"]["params"].pop("eta", None)
    run_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    by_ds = _group_runs_by_dataset(run_dirs)
    master_df, _ = _evaluate_all_runs(by_ds)
    pr5 = master_df[master_df["variant_id"] == "I1_eta_5"]
    assert not pr5.empty
    assert (pr5["eta"] == 5.0).all()


def test_explicit_roc_sweep_zero_test_rows_raises(tmp_path):
    run_dirs, roc_sweep_csv = _build_synthetic_suite(tmp_path)
    roc_df = pd.read_csv(roc_sweep_csv)
    val_only_csv = tmp_path / "roc_val_only.csv"
    roc_df[roc_df["partition"] == "validation"].to_csv(val_only_csv, index=False)

    # Explicit CLI --roc-sweep-csv with 0 matching test rows -> ValueError
    with pytest.raises(ValueError, match="no 'test' rows found"):
        main(
            [
                "--run-dirs",
                *[str(p) for p in run_dirs],
                "--roc-sweep-csv",
                str(val_only_csv),
                "--output-dir",
                str(tmp_path / "explicit_empty_roc"),
            ]
        )

    # Direct run_analysis with explicit roc_sweep_csv -> ValueError
    with pytest.raises(ValueError, match="no 'test' rows found"):
        run_analysis(
            run_dirs,
            roc_sweep_csv=val_only_csv,
            output_dir=tmp_path / "explicit_empty_roc_api",
        )

    # Auto-discovered (require_roc_sweep_rows=False) with 0 test rows -> succeeds with 0 ROC rows
    report = run_analysis(
        run_dirs,
        roc_sweep_csv=val_only_csv,
        output_dir=tmp_path / "auto_empty_roc",
        require_roc_sweep_rows=False,
    )
    assert report["datasets"]["synthetic-tiny"]["n_roc_sweep_test"] == 0


def test_separation_stability_and_plots_handle_nans_and_staggering(tmp_path):
    # Partial NaNs + ties + overlapping shifted points
    master_partial = pd.DataFrame(
        [
            {
                "dataset": "ds1",
                "partition": "test",
                "variant_id": "B1",
                "family": "Baseline",
                "method": "baseline",
                "model": "lr",
                "eta": math.nan,
                "base_threshold": 0.5,
                "selected_threshold": 0.5,
                "roc_margin": math.nan,
                "profit_eq16": 0.12,
                "sp_paper": 0.08,
                "eo_gap": 0.01,
                "ind": 0.02,
                "auc": 0.75,
                "sf": 0.03,
                "delta_fpr": 0.01,
                "delta_tpr": -0.01,
                "abs_delta_fpr": 0.01,
                "abs_delta_tpr": 0.01,
            },
            {
                "dataset": "ds1",
                "partition": "test",
                "variant_id": "P1",
                "family": "Reweighing",
                "method": "reweighing",
                "model": "lr",
                "eta": math.nan,
                "base_threshold": 0.5,
                "selected_threshold": 0.5,
                "roc_margin": math.nan,
                "profit_eq16": 0.11,
                "sp_paper": 0.08,
                "eo_gap": 0.01,
                "ind": 0.02,
                "auc": 0.74,
                "sf": 0.03,
                "delta_fpr": 0.01,
                "delta_tpr": 0.01,
                "abs_delta_fpr": 0.01,
                "abs_delta_tpr": 0.01,
            },
            {
                "dataset": "ds1",
                "partition": "test",
                "variant_id": "O1",
                "family": "Reject Option",
                "method": "reject_option",
                "model": "lr",
                "eta": math.nan,
                "base_threshold": 0.5,
                "selected_threshold": 0.52,
                "roc_margin": 0.02,
                "profit_eq16": 0.10,
                "sp_paper": 0.02,
                "eo_gap": 0.05,
                "ind": 0.04,
                "auc": 0.72,
                "sf": 0.04,
                "delta_fpr": 0.02,
                "delta_tpr": 0.05,
                "abs_delta_fpr": 0.02,
                "abs_delta_tpr": 0.05,
            },
            {
                "dataset": "ds1",
                "partition": "test",
                "variant_id": "B2",
                "family": "Baseline",
                "method": "baseline",
                "model": "rf",
                "eta": math.nan,
                "base_threshold": 0.5,
                "selected_threshold": 0.5,
                "roc_margin": math.nan,
                "profit_eq16": math.nan,
                "sp_paper": math.nan,
                "eo_gap": math.nan,
                "ind": math.nan,
                "auc": math.nan,
                "sf": math.nan,
                "delta_fpr": math.nan,
                "delta_tpr": math.nan,
                "abs_delta_fpr": math.nan,
                "abs_delta_tpr": math.nan,
            },
        ]
    )
    i1_sel: dict[str, dict] = {}
    sep_df, sep_sum_df = build_separation_stability_tables(master_partial, i1_sel)

    assert sep_df["rank_sp_paper"].dtype == pd.Int64Dtype()
    assert sep_df["rank_eo_gap"].dtype == pd.Int64Dtype()
    assert sep_df["rank_shift"].dtype == pd.Int64Dtype()
    assert sep_sum_df["max_abs_rank_shift"].dtype == pd.Int64Dtype()

    nan_row = sep_df[sep_df["variant_id"] == "B2"].iloc[0]
    assert pd.isna(nan_row["rank_sp_paper"])
    assert pd.isna(nan_row["rank_eo_gap"])
    assert pd.isna(nan_row["rank_shift"])

    # B1 and P1 tie at sp=0.08 (rank 2) and eo=0.01 (rank 1) -> shift = -1 for both
    b1_row = sep_df[sep_df["variant_id"] == "B1"].iloc[0]
    p1_row = sep_df[sep_df["variant_id"] == "P1"].iloc[0]
    assert b1_row["rank_sp_paper"] == 2
    assert b1_row["rank_eo_gap"] == 1
    assert b1_row["rank_shift"] == -1
    assert p1_row["rank_shift"] == -1

    sum_row = sep_sum_df.iloc[0]
    assert sum_row["n_rank_shift_nonzero"] == 3
    assert sum_row["max_abs_rank_shift"] == 2

    # Plotting with NaNs & overlapping annotations succeeds cleanly
    primary_df = build_primary_7_table(master_partial, i1_sel)
    pareto_df = build_pareto_frontiers_table(master_partial, None)
    pooled_test = master_partial.copy()
    pooled_test["source_pool"] = "trained_15"
    pooled_test["fairness_metric"] = ""
    pooled_test["fairness_bound"] = math.nan
    pooled_test["margin"] = math.nan

    out_dir = tmp_path / "nan_plots"
    pvf_paths = plot_profit_vs_fairness_per_dataset(
        pooled_test, pareto_df, primary_df, out_dir
    )
    sep_fig = plot_separation_stability(sep_df, sep_sum_df, out_dir)
    cmp_fig = plot_method_comparison_7_variants(primary_df, out_dir)
    assert all(p.is_file() and p.read_bytes()[:4] == b"\x89PNG" for p in pvf_paths)
    assert sep_fig.is_file() and sep_fig.read_bytes()[:4] == b"\x89PNG"
    assert cmp_fig.is_file() and cmp_fig.read_bytes()[:4] == b"\x89PNG"

    # All-NaN case: n_rank_shift_nonzero == 0, max_abs_rank_shift is pd.NA, JSON report valid
    master_all_nan = master_partial[master_partial["variant_id"] == "B2"].copy()
    sep_nan_df, sep_nan_sum = build_separation_stability_tables(master_all_nan, {})
    assert sep_nan_sum.iloc[0]["n_rank_shift_nonzero"] == 0
    assert pd.isna(sep_nan_sum.iloc[0]["max_abs_rank_shift"])

    sep_nan_fig = plot_separation_stability(
        sep_nan_df, sep_nan_sum, tmp_path / "all_nan_plots"
    )
    assert sep_nan_fig.is_file() and sep_nan_fig.read_bytes()[:4] == b"\x89PNG"

    report = build_analysis_report(
        split_ids={"ds1": "split1"},
        i1_selections={},
        primary_df=build_primary_7_table(master_all_nan, {}),
        spearman_df=build_spearman_correlations_table(master_all_nan, None),
        sep_df=sep_nan_df,
        sep_summary_df=sep_nan_sum,
        pareto_df=build_pareto_frontiers_table(master_all_nan, None),
        pooled_test=pooled_test[pooled_test["variant_id"] == "B2"],
        csv_paths={},
        figure_paths={"separation_stability": sep_nan_fig},
        report_path=tmp_path / "all_nan_report.json",
    )
    assert (
        report["datasets"]["ds1"]["separation_stability"]["test"]["max_abs_rank_shift"]
        is None
    )


def test_empty_table_schemas_and_rank_correlation_helpers():
    empty_master = pd.DataFrame(columns=MASTER_SUMMARY_COLUMNS)
    assert tuple(build_primary_7_table(empty_master, {}).columns) == PRIMARY_7_COLUMNS
    assert (
        tuple(build_relative_gains_table(empty_master, {}).columns)
        == RELATIVE_GAINS_COLUMNS
    )
    assert (
        tuple(build_spearman_correlations_table(empty_master, None).columns)
        == SPEARMAN_CORRELATIONS_COLUMNS
    )
    sep_df, sep_sum_df = build_separation_stability_tables(empty_master, {})
    assert tuple(sep_df.columns) == SEPARATION_STABILITY_COLUMNS
    assert tuple(sep_sum_df.columns) == SEPARATION_RANK_SUMMARY_COLUMNS
    assert (
        tuple(build_pareto_frontiers_table(empty_master, None).columns)
        == PARETO_FRONTIERS_COLUMNS
    )

    x = np.array([1.0, 2.0, math.nan, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    rho, p_rho, n_obs = _safe_spearman(x, y)
    tau, p_tau = _safe_kendall(x, y)
    assert n_obs == 3
    assert rho == pytest.approx(1.0)
    assert tau == pytest.approx(1.0)
    assert math.isfinite(p_rho) and math.isfinite(p_tau)

    const = np.array([1.0, 1.0, 1.0])
    assert math.isnan(_safe_spearman(const, np.array([1.0, 2.0, 3.0]))[0])
    assert math.isnan(_safe_kendall(const, np.array([1.0, 2.0, 3.0]))[0])


def _master_row(**overrides: object) -> dict[str, object]:
    """Return a minimal valid master_summary row with sensible defaults."""
    base: dict[str, object] = {
        "dataset": "ds1",
        "partition": "test",
        "variant_id": "B1",
        "family": "Baseline",
        "method": "baseline",
        "model": "lr",
        "eta": math.nan,
        "split_id": "split-1",
        "run_id": "run-1",
        "base_threshold": 0.5,
        "selected_threshold": 0.5,
        "roc_margin": math.nan,
        "selection_value": 0.1,
        "delta_profit_vs_own_baseline": 0.0,
        "delta_profit_vs_approve_all": 0.0,
        "auc": 0.75,
        "balanced_accuracy": 0.70,
        "approval_rate": 0.60,
        "profit_eq16": 0.10,
        "ind": 0.02,
        "sp_paper": 0.04,
        "eo_gap": 0.02,
        "sf": 0.02,
        "delta_fpr": 0.01,
        "delta_tpr": 0.02,
        "abs_delta_fpr": 0.01,
        "abs_delta_tpr": 0.02,
    }
    base.update(overrides)
    return base


def test_separation_stability_math_ties_cancellation_and_partial_nans(tmp_path):
    # Mix of ties (method="min" skips next rank), opposite-sign cancellation,
    # eo_gap == 0.0, partial NaN (sp finite, eo NaN), and full NaN.
    df = pd.DataFrame(
        [
            _master_row(
                variant_id="B1",
                family="Baseline",
                method="baseline",
                model="lr",
                sp_paper=0.02,
                eo_gap=0.05,
                delta_fpr=0.05,
                delta_tpr=-0.03,
            ),
            _master_row(
                variant_id="P1",
                family="Reweighing",
                method="reweighing",
                model="lr",
                sp_paper=0.02,
                eo_gap=0.01,
                delta_fpr=0.01,
                delta_tpr=0.01,
            ),
            _master_row(
                variant_id="O1",
                family="Reject Option",
                method="reject_option",
                model="lr",
                sp_paper=0.08,
                eo_gap=0.01,
                delta_fpr=-0.01,
                delta_tpr=0.00,
            ),
            _master_row(
                variant_id="I1_eta_5",
                family="Prejudice Remover",
                method="prejudice_remover",
                model="pr_eta_5",
                eta=5.0,
                sp_paper=0.04,
                eo_gap=0.0,
                delta_fpr=0.0,
                delta_tpr=0.0,
            ),
            _master_row(
                variant_id="B2",
                family="Baseline",
                method="baseline",
                model="rf",
                sp_paper=0.05,
                eo_gap=math.nan,
                delta_fpr=0.01,
                delta_tpr=math.nan,
            ),
            _master_row(
                variant_id="P2",
                family="Reweighing",
                method="reweighing",
                model="rf",
                sp_paper=math.nan,
                eo_gap=math.nan,
                delta_fpr=math.nan,
                delta_tpr=math.nan,
            ),
        ]
    )
    sep_df, sep_sum_df = build_separation_stability_tables(df, {})
    by_vid = sep_df.set_index("variant_id")

    # sp_paper finite values: B1=0.02, P1=0.02, I1=0.04, B2=0.05, O1=0.08
    # min-rank skips rank 2 after tie at rank 1 -> [1, 1, 3, 4, 5], P2=NA
    assert by_vid.loc["B1", "rank_sp_paper"] == 1
    assert by_vid.loc["P1", "rank_sp_paper"] == 1
    assert by_vid.loc["I1_eta_5", "rank_sp_paper"] == 3
    assert by_vid.loc["B2", "rank_sp_paper"] == 4
    assert by_vid.loc["O1", "rank_sp_paper"] == 5
    assert pd.isna(by_vid.loc["P2", "rank_sp_paper"])

    # eo_gap finite values: I1=0.0 (1), P1=0.01 (2), O1=0.01 (2), B1=0.05 (4); B2/P2=NA
    assert by_vid.loc["I1_eta_5", "rank_eo_gap"] == 1
    assert by_vid.loc["P1", "rank_eo_gap"] == 2
    assert by_vid.loc["O1", "rank_eo_gap"] == 2
    assert by_vid.loc["B1", "rank_eo_gap"] == 4
    assert pd.isna(by_vid.loc["B2", "rank_eo_gap"])
    assert pd.isna(by_vid.loc["P2", "rank_eo_gap"])

    # rank_shift = rank_eo_gap - rank_sp_paper (NA if either rank is NA)
    assert by_vid.loc["B1", "rank_shift"] == 3
    assert by_vid.loc["P1", "rank_shift"] == 1
    assert by_vid.loc["O1", "rank_shift"] == -3
    assert by_vid.loc["I1_eta_5", "rank_shift"] == -2
    assert pd.isna(by_vid.loc["B2", "rank_shift"])
    assert pd.isna(by_vid.loc["P2", "rank_shift"])

    # opposite_sign_cancellation & cancellation_ratio
    assert bool(by_vid.loc["B1", "opposite_sign_cancellation"]) is True
    assert bool(by_vid.loc["P1", "opposite_sign_cancellation"]) is False
    assert bool(by_vid.loc["O1", "opposite_sign_cancellation"]) is False
    assert bool(by_vid.loc["I1_eta_5", "opposite_sign_cancellation"]) is False
    assert bool(by_vid.loc["B2", "opposite_sign_cancellation"]) is False
    assert by_vid.loc["B1", "cancellation_ratio"] == pytest.approx(0.02 / 0.05)
    assert by_vid.loc["P1", "cancellation_ratio"] == pytest.approx(0.02 / 0.01)
    assert by_vid.loc["O1", "cancellation_ratio"] == pytest.approx(0.08 / 0.01)
    assert math.isnan(float(by_vid.loc["I1_eta_5", "cancellation_ratio"]))
    assert math.isnan(float(by_vid.loc["B2", "cancellation_ratio"]))

    # Summary correlations & counts match scipy on the 4 mutually finite pairs
    sum_row = sep_sum_df.iloc[0]
    expected_sp = np.array([0.02, 0.02, 0.08, 0.04])
    expected_eo = np.array([0.05, 0.01, 0.01, 0.0])
    exp_rho = stats.spearmanr(expected_sp, expected_eo)
    exp_tau = stats.kendalltau(expected_sp, expected_eo)
    assert sum_row["n_models"] == 6
    assert sum_row["spearman_rho"] == pytest.approx(float(exp_rho.statistic))
    assert sum_row["spearman_p_value"] == pytest.approx(float(exp_rho.pvalue))
    assert sum_row["kendall_tau"] == pytest.approx(float(exp_tau.statistic))
    assert sum_row["kendall_p_value"] == pytest.approx(float(exp_tau.pvalue))
    assert sum_row["n_opposite_sign_cancellation"] == 1
    assert sum_row["n_rank_shift_nonzero"] == 4
    assert sum_row["max_abs_rank_shift"] == 3

    # Plots must succeed with partial NaNs and ties
    primary_df = build_primary_7_table(df, {})
    pareto_df = build_pareto_frontiers_table(df, None)
    pooled_test = df.copy()
    pooled_test["source_pool"] = "trained_15"
    pooled_test["fairness_metric"] = ""
    pooled_test["fairness_bound"] = math.nan
    pooled_test["margin"] = math.nan
    for fig in (
        *plot_profit_vs_fairness_per_dataset(
            pooled_test, pareto_df, primary_df, tmp_path
        ),
        plot_separation_stability(sep_df, sep_sum_df, tmp_path),
        plot_method_comparison_7_variants(primary_df, tmp_path),
    ):
        assert fig.is_file() and fig.read_bytes()[:4] == b"\x89PNG"


def test_spearman_correlations_sign_flip_edge_cases_and_json_null(tmp_path):
    # profit_eq16: 3 distinct values [0.10, 0.20, 0.30]
    # auc: 0 finite values (< 2 obs)
    # ind: 1 finite value (< 2 obs)
    # sp_paper: 2 finite values [0.08, 0.02] (n_obs == 2 -> rho = -1 vs profit, p_value = NaN)
    # eo_gap: 3 finite values [0.06, 0.04, 0.01] (with sp_paper: n_obs == 2 -> p_value = NaN)
    # sf: constant [0.05, 0.05, 0.05] -> rho = NaN, p_value = NaN, n_obs == 3
    df = pd.DataFrame(
        [
            _master_row(
                variant_id="B1",
                profit_eq16=0.10,
                auc=math.nan,
                ind=0.03,
                sp_paper=0.08,
                eo_gap=0.06,
                sf=0.05,
            ),
            _master_row(
                variant_id="P1",
                method="reweighing",
                profit_eq16=0.20,
                auc=math.nan,
                ind=math.nan,
                sp_paper=0.02,
                eo_gap=0.04,
                sf=0.05,
            ),
            _master_row(
                variant_id="O1",
                method="reject_option",
                profit_eq16=0.30,
                auc=math.nan,
                ind=math.nan,
                sp_paper=math.nan,
                eo_gap=0.01,
                sf=0.05,
            ),
        ]
    )
    sp_df = build_spearman_correlations_table(df, None)

    def _get(conv: str, mx: str, my: str) -> pd.Series:
        return sp_df[
            (sp_df["sign_convention"] == conv)
            & (sp_df["metric_x"] == mx)
            & (sp_df["metric_y"] == my)
        ].iloc[0]

    # Sign-flip: profit_eq16 (higher=better) vs eo_gap (lower=better) flips sign;
    # sp_paper vs eo_gap (both lower=better) preserves sign.
    raw_pe = _get("raw", "profit_eq16", "eo_gap")
    paper_pe = _get("paper_higher_is_better", "profit_eq16", "eo_gap")
    assert raw_pe["n_obs"] == 3
    assert raw_pe["rho"] == pytest.approx(-1.0)
    assert paper_pe["rho"] == pytest.approx(1.0)
    assert paper_pe["p_value"] == pytest.approx(raw_pe["p_value"])

    raw_se = _get("raw", "sp_paper", "eo_gap")
    paper_se = _get("paper_higher_is_better", "sp_paper", "eo_gap")
    assert raw_se["n_obs"] == 2
    assert raw_se["rho"] == pytest.approx(1.0)
    assert paper_se["rho"] == pytest.approx(1.0)
    assert math.isnan(float(raw_se["p_value"]))
    assert math.isnan(float(paper_se["p_value"]))

    # Constant input -> rho=NaN, p_value=NaN, n_obs=3
    const_row = _get("raw", "profit_eq16", "sf")
    assert const_row["n_obs"] == 3
    assert math.isnan(float(const_row["rho"]))
    assert math.isnan(float(const_row["p_value"]))

    # < 2 finite observations -> rho=NaN, p_value=NaN, n_obs in {0, 1}
    one_obs = _get("raw", "profit_eq16", "ind")
    zero_obs = _get("raw", "profit_eq16", "auc")
    assert one_obs["n_obs"] == 1 and math.isnan(float(one_obs["rho"]))
    assert zero_obs["n_obs"] == 0 and math.isnan(float(zero_obs["rho"]))

    # Verify p_value == NaN serializes to JSON null in analysis_report.json
    sep_df, sep_sum_df = build_separation_stability_tables(df, {})
    pooled_test = df.copy()
    pooled_test["source_pool"] = "trained_15"
    pooled_test["fairness_metric"] = ""
    pooled_test["fairness_bound"] = math.nan
    pooled_test["margin"] = math.nan
    report_path = tmp_path / "spearman_null_report.json"
    build_analysis_report(
        split_ids={"ds1": "split-1"},
        i1_selections={},
        primary_df=build_primary_7_table(df, {}),
        spearman_df=sp_df,
        sep_df=sep_df,
        sep_summary_df=sep_sum_df,
        pareto_df=build_pareto_frontiers_table(df, None),
        pooled_test=pooled_test,
        csv_paths={},
        figure_paths={},
        report_path=report_path,
    )
    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    hl = parsed["datasets"]["ds1"]["correlation_highlights"]["trained_15"]["raw"][
        "sp_paper__vs__eo_gap"
    ]
    assert hl["n_obs"] == 2
    assert hl["rho"] == pytest.approx(1.0)
    assert hl["p_value"] is None


def test_select_i1_best_val_tie_break_and_no_test_leakage():
    df = pd.DataFrame(
        [
            # ds_tie: eta=10 appears BEFORE eta=2 with equal validation profit (0.15)
            # and higher test profit (0.90 vs 0.05); smaller eta (2.0) must win.
            _master_row(
                dataset="ds_tie",
                partition="validation",
                variant_id="I1_eta_10",
                method="prejudice_remover",
                model="pr_eta_10",
                eta=10.0,
                profit_eq16=0.15,
            ),
            _master_row(
                dataset="ds_tie",
                partition="validation",
                variant_id="I1_eta_2",
                method="prejudice_remover",
                model="pr_eta_2",
                eta=2.0,
                profit_eq16=0.15,
            ),
            _master_row(
                dataset="ds_tie",
                partition="validation",
                variant_id="I1_eta_5",
                method="prejudice_remover",
                model="pr_eta_5",
                eta=5.0,
                profit_eq16=0.10,
            ),
            _master_row(
                dataset="ds_tie",
                partition="test",
                variant_id="I1_eta_10",
                method="prejudice_remover",
                model="pr_eta_10",
                eta=10.0,
                profit_eq16=0.90,
            ),
            _master_row(
                dataset="ds_tie",
                partition="test",
                variant_id="I1_eta_2",
                method="prejudice_remover",
                model="pr_eta_2",
                eta=2.0,
                profit_eq16=0.05,
            ),
            # ds_no_leak: eta=1 wins validation (0.25 > 0.12) even though eta=30 has
            # much higher test profit (0.99 vs 0.02).
            _master_row(
                dataset="ds_no_leak",
                partition="validation",
                variant_id="I1_eta_1",
                method="prejudice_remover",
                model="pr_eta_1",
                eta=1.0,
                profit_eq16=0.25,
            ),
            _master_row(
                dataset="ds_no_leak",
                partition="validation",
                variant_id="I1_eta_30",
                method="prejudice_remover",
                model="pr_eta_30",
                eta=30.0,
                profit_eq16=0.12,
            ),
            _master_row(
                dataset="ds_no_leak",
                partition="test",
                variant_id="I1_eta_1",
                method="prejudice_remover",
                model="pr_eta_1",
                eta=1.0,
                profit_eq16=0.02,
            ),
            _master_row(
                dataset="ds_no_leak",
                partition="test",
                variant_id="I1_eta_30",
                method="prejudice_remover",
                model="pr_eta_30",
                eta=30.0,
                profit_eq16=0.99,
            ),
        ]
    )
    sel = _select_i1_best_val(df)
    assert sel["ds_tie"]["variant_id"] == "I1_eta_2"
    assert sel["ds_tie"]["eta"] == 2.0
    assert sel["ds_tie"]["tie_broken_by_smaller_eta"] is True
    assert sel["ds_tie"]["tied_models"] == ["pr_eta_2", "pr_eta_10"]
    assert sel["ds_tie"]["validation_profit_eq16"] == pytest.approx(0.15)
    assert sel["ds_tie"]["test_profit_eq16"] == pytest.approx(0.05)

    assert sel["ds_no_leak"]["variant_id"] == "I1_eta_1"
    assert sel["ds_no_leak"]["tie_broken_by_smaller_eta"] is False
    assert sel["ds_no_leak"]["validation_profit_eq16"] == pytest.approx(0.25)
    assert sel["ds_no_leak"]["test_profit_eq16"] == pytest.approx(0.02)

    all_nan = pd.DataFrame(
        [
            _master_row(
                dataset="ds_nan",
                partition="validation",
                variant_id="I1_eta_0",
                method="prejudice_remover",
                model="pr_eta_0",
                eta=0.0,
                profit_eq16=math.nan,
            )
        ]
    )
    with pytest.raises(ValueError, match="No finite validation profit_eq16"):
        _select_i1_best_val(all_nan)


def test_relative_gains_hand_calculated_and_edge_cases():
    df = pd.DataFrame(
        [
            _master_row(
                partition="test",
                variant_id="B1",
                auc=0.80,
                profit_eq16=-0.10,
                ind=0.05,
                sp_paper=0.10,
                eo_gap=0.0,
                sf=0.04,
            ),
            _master_row(
                partition="test",
                variant_id="O1",
                family="Reject Option",
                method="reject_option",
                auc=0.80,
                profit_eq16=-0.05,
                ind=0.02,
                sp_paper=0.04,
                eo_gap=0.02,
                sf=0.06,
            ),
        ]
    )
    gains = build_relative_gains_table(df, {})
    assert len(gains) == 1
    row = gains.iloc[0]
    assert row["comparison"] == "O1 vs B1"
    # Higher-is-better: profit (-0.05 - (-0.10)) / |-0.10| * 100 = +50%; auc unchanged -> 0%
    assert row["profit_eq16_abs_change"] == pytest.approx(0.05)
    assert row["profit_eq16_gain_pct"] == pytest.approx(50.0)
    assert row["auc_abs_change"] == pytest.approx(0.0)
    assert row["auc_gain_pct"] == pytest.approx(0.0)
    # Lower-is-better: (base - method) / base * 100; baseline=0 -> NaN gain_pct
    assert row["sp_paper_abs_change"] == pytest.approx(-0.06)
    assert row["sp_paper_gain_pct"] == pytest.approx(60.0)
    assert row["ind_gain_pct"] == pytest.approx(60.0)
    assert row["sf_gain_pct"] == pytest.approx(-50.0)
    assert row["eo_gap_abs_change"] == pytest.approx(0.02)
    assert math.isnan(float(row["eo_gap_gain_pct"]))
    for m in GAIN_METRICS:
        assert (m in HIGHER_IS_BETTER_METRICS) or (m in LOWER_IS_BETTER_METRICS)


def test_invalid_cli_args_and_saved_metrics_verification(tmp_path):
    run_dirs, roc_sweep_csv = _build_synthetic_suite(tmp_path)

    # 1. Mutual exclusion: --roc-sweep-csv + --no-roc-sweep -> argparse SystemExit(2)
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--run-dirs",
                *[str(p) for p in run_dirs],
                "--roc-sweep-csv",
                str(roc_sweep_csv),
                "--no-roc-sweep",
            ]
        )
    assert exc_info.value.code == 2

    # 2. Missing required columns in ROC sweep CSV -> ValueError
    missing_col_csv = tmp_path / "roc_missing_col.csv"
    roc_df = pd.read_csv(roc_sweep_csv)
    roc_df.drop(columns=["margin", "sp_paper"]).to_csv(missing_col_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        main(
            [
                "--run-dirs",
                *[str(p) for p in run_dirs],
                "--roc-sweep-csv",
                str(missing_col_csv),
                "--output-dir",
                str(tmp_path / "unused_missing_col"),
            ]
        )

    # 3. Duplicate --run-dirs via CLI -> ValueError
    with pytest.raises(ValueError, match="Duplicate run directory"):
        main(
            [
                "--run-dirs",
                str(run_dirs[0]),
                str(run_dirs[0]),
                "--no-roc-sweep",
                "--output-dir",
                str(tmp_path / "unused_dup"),
            ]
        )

    # 4. Explicit --roc-sweep-csv with test rows only for another dataset -> 0 matching rows
    other_ds_csv = tmp_path / "roc_other_dataset.csv"
    other_df = roc_df.copy()
    other_df["dataset"] = "other-dataset"
    other_df.to_csv(other_ds_csv, index=False)
    with pytest.raises(ValueError, match="no 'test' rows found"):
        main(
            [
                "--run-dirs",
                *[str(p) for p in run_dirs],
                "--roc-sweep-csv",
                str(other_ds_csv),
                "--output-dir",
                str(tmp_path / "unused_other_ds"),
            ]
        )

    # 5. _verify_saved_metrics unit checks & run.json tampering guard
    with pytest.raises(ValueError, match="mismatch"):
        _verify_saved_metrics(
            Path("dummy"), "lr", "test", {"auc": math.nan}, {"auc": 0.5}
        )
    with pytest.raises(ValueError, match="mismatch"):
        _verify_saved_metrics(Path("dummy"), "lr", "test", {"auc": 0.75}, {"auc": None})

    run_json_path = run_dirs[0] / "run.json"
    meta = json.loads(run_json_path.read_text(encoding="utf-8"))
    meta["selected"]["lr"]["test_metrics"]["profit_eq16"] += 0.01
    run_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        run_analysis(run_dirs, output_dir=tmp_path / "tampered_out")


def test_multi_dataset_analysis_and_figures(tmp_path):
    output_root = tmp_path / "multi_outputs"
    alpha_runs = _build_dataset_suite(
        output_root, dataset_name="synthetic-alpha", seed=37
    )
    beta_runs = _build_dataset_suite(
        output_root, dataset_name="synthetic-beta", seed=41
    )
    roc_csv = output_root / "roc-sweep-summary.csv"
    run_roc_experiment(
        [alpha_runs[0], beta_runs[0]],
        roc_csv,
        fairness_metrics=("sp_paper", "ind"),
        fairness_bounds=(0.1, 0.05),
    )

    single_out = tmp_path / "single_out"
    run_analysis(alpha_runs, roc_sweep_csv=roc_csv, output_dir=single_out)
    single_sep_img = mpimg.imread(single_out / "figures" / "separation_stability.png")

    multi_out = tmp_path / "multi_out"
    report = run_analysis(
        alpha_runs + beta_runs,
        roc_sweep_csv=roc_csv,
        output_dir=multi_out,
    )

    assert set(report["datasets"].keys()) == {"synthetic-alpha", "synthetic-beta"}
    for ds in ("synthetic-alpha", "synthetic-beta"):
        pvf_fig = multi_out / "figures" / f"profit_vs_fairness_{ds}.png"
        assert pvf_fig.is_file() and pvf_fig.read_bytes()[:4] == b"\x89PNG"
        assert f"profit_vs_fairness_{ds}" in report["generated_files"]["figures"]

    sep_fig = multi_out / "figures" / "separation_stability.png"
    cmp_fig = multi_out / "figures" / "method_comparison_7_variants.png"
    assert sep_fig.is_file() and sep_fig.read_bytes()[:4] == b"\x89PNG"
    assert cmp_fig.is_file() and cmp_fig.read_bytes()[:4] == b"\x89PNG"

    # Two-row separation_stability.png must be substantially taller than single-dataset figure
    multi_sep_img = mpimg.imread(sep_fig)
    assert multi_sep_img.shape[0] > 1.5 * single_sep_img.shape[0]

    primary_df = pd.read_csv(multi_out / "primary_7_variants.csv")
    assert set(primary_df["dataset"]) == {"synthetic-alpha", "synthetic-beta"}
    assert len(primary_df) == 32  # 8 primary variants * 2 partitions * 2 datasets

    for table_name in (
        "relative_gains_vs_baseline.csv",
        "spearman_correlations.csv",
        "separation_stability.csv",
        "separation_rank_summary.csv",
        "pareto_frontiers.csv",
    ):
        tbl = pd.read_csv(multi_out / table_name)
        assert set(tbl["dataset"]) == {"synthetic-alpha", "synthetic-beta"}
