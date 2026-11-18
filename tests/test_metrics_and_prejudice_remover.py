import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.linear_model import LogisticRegression

from fair_credit_scoring import (
    PrejudiceRemover,
    compute_profit_eq16,
    confusion_counts,
    evaluate_metrics,
    group_confusion_rates,
    load_config,
    run,
)
from fair_credit_scoring.__main__ import main as cli_main
from fair_credit_scoring.data import make_splits, prepare_dataset
from fair_credit_scoring.handoff import build_control_table, build_handoff_manifest
from fair_credit_scoring.hooks import BUILTIN_ESTIMATORS, BUILTIN_METRICS
from fair_credit_scoring.metrics import (
    auc,
    balanced_accuracy,
    eo_gap,
    ind,
    profit_eq16,
    sf,
    sp_paper,
)
from fair_credit_scoring.prejudice_remover import objective_and_gradient

ROOT = Path(__file__).resolve().parents[1]


def test_confusion_counts_and_profit_eq16_hand_calculation():
    # One TP, one FP, two FN, one TN (N = 5)
    y = np.array([1, 0, 1, 1, 0])
    decision = np.array([1, 1, 0, 0, 0])
    counts = confusion_counts(y, decision)
    assert (counts.tp, counts.fp, counts.fn, counts.tn, counts.n) == (1, 1, 2, 1, 5)

    expected = (0.2664 * (1 - 2) - 0.275 * 1) / 5
    assert compute_profit_eq16(y, decision) == pytest.approx(expected)
    assert profit_eq16(y, np.array([0, 0, 1, 1, 0]), np.full(5, 0.5), decision) == (
        pytest.approx(expected)
    )

    # All-approve and all-reject edge cases
    all_approve = compute_profit_eq16(y, np.ones(5, dtype=int))
    assert all_approve == pytest.approx((0.2664 * 3 - 0.275 * 2) / 5)
    all_reject = compute_profit_eq16(y, np.zeros(5, dtype=int))
    assert all_reject == pytest.approx((-0.2664 * 3) / 5)
    assert all_reject < 0.0


def test_sp_paper_counterexample_and_eo_gap():
    # Group 0: FPR=0.1, TPR=0.8 (10 neg with 1 FP, 10 pos with 8 TP)
    # Group 1: FPR=0.3, TPR=0.6 (10 neg with 3 FP, 10 pos with 6 TP)
    a = np.array([0] * 20 + [1] * 20)
    y = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
    decision = np.array(
        [1] * 1 + [0] * 9 + [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7 + [1] * 6 + [0] * 4
    )
    score = np.linspace(0.1, 0.9, len(y))
    rates = group_confusion_rates(y, a, decision)
    assert rates[0].fpr == pytest.approx(0.1)
    assert rates[0].tpr == pytest.approx(0.8)
    assert rates[1].fpr == pytest.approx(0.3)
    assert rates[1].tpr == pytest.approx(0.6)

    assert sp_paper(y, a, score, decision) == pytest.approx(0.0)
    assert eo_gap(y, a, score, decision) == pytest.approx(0.2)
    assert ind(y, a, score, decision) == pytest.approx(abs(9 / 20 - 9 / 20))
    assert sf(y, a, score, decision) == pytest.approx(abs(6 / 9 - 8 / 9))


def test_equal_group_rates_produce_zero_fairness_gaps():
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    decision = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    score = np.full(8, 0.5)
    results = evaluate_metrics(y, a, score, decision)
    assert results["ind"] == 0.0
    assert results["sp_paper"] == 0.0
    assert results["sf"] == 0.0
    assert results["eo_gap"] == 0.0


def test_undefined_metrics_return_nan_and_auc_uses_scores():
    # No approvals in group 1 -> PPV_1 and sf are NaN; ind, sp_paper, eo_gap stay finite.
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    decision = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.6, 0.3, 0.2, 0.8, 0.7, 0.1, 0.4])
    res = evaluate_metrics(y, a, score, decision)
    assert math.isnan(res["ppv_1"])
    assert math.isnan(res["sf"])
    assert math.isfinite(res["ind"])
    assert math.isfinite(res["sp_paper"])
    assert math.isfinite(res["eo_gap"])
    assert res["auc"] == pytest.approx(auc(y, a, score, np.zeros_like(decision)))
    assert res["auc"] > 0.8

    # Missing y=0 in group 1 -> FPR_1 is NaN -> sp_paper and eo_gap must both be NaN.
    y_no_neg_g1 = np.array([1, 0, 1, 0, 1, 1])
    a_g = np.array([0, 0, 0, 0, 1, 1])
    d_g = np.array([1, 1, 0, 0, 1, 0])
    s_g = np.linspace(0.2, 0.8, 6)
    assert math.isnan(sp_paper(y_no_neg_g1, a_g, s_g, d_g))
    assert math.isnan(eo_gap(y_no_neg_g1, a_g, s_g, d_g))

    # Missing y=0 in group 0 -> test first argument NaN as well.
    y_no_neg_g0 = np.array([1, 1, 1, 1, 1, 0])
    assert math.isnan(sp_paper(y_no_neg_g0, a_g, s_g, d_g))
    assert math.isnan(eo_gap(y_no_neg_g0, a_g, s_g, d_g))

    # Single-class y -> auc and balanced_accuracy return NaN instead of raising.
    y_single = np.ones(4, dtype=int)
    a_single = np.array([0, 0, 1, 1])
    d_single = np.array([1, 0, 1, 0])
    s_single = np.array([0.2, 0.4, 0.6, 0.8])
    assert math.isnan(auc(y_single, a_single, s_single, d_single))
    assert math.isnan(balanced_accuracy(y_single, a_single, s_single, d_single))

    # Missing group a=0 -> all group rates and gaps return NaN.
    assert math.isnan(ind(y[:4], np.ones(4, dtype=int), score[:4], decision[:4]))


def test_metrics_validate_shapes_and_encodings():
    with pytest.raises(ValueError, match="1-D"):
        confusion_counts(np.ones((2, 2)), np.ones(4))
    with pytest.raises(ValueError, match="nonempty"):
        confusion_counts([], [])
    with pytest.raises(ValueError, match="same length"):
        confusion_counts([0, 1], [0])
    with pytest.raises(ValueError, match="binary"):
        confusion_counts([0, 2], [0, 1])
    with pytest.raises(ValueError, match="binary"):
        ind([0, 1], [0, 2], [0.2, 0.8], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        auc([0, 1], [0, 1], [0.5, np.nan], [0, 1])
    with pytest.raises(ValueError, match="p0 \\+ p1"):
        compute_profit_eq16([0, 1], [0, 1], p0=0.8, p1=0.3)


@pytest.mark.parametrize("use_sparse", [False, True])
def test_prejudice_remover_analytical_gradient_matches_finite_differences(use_sparse):
    rng = np.random.default_rng(123)
    n0, n1, d = 17, 29, 5
    x = rng.normal(size=(n0 + n1, d))
    a = np.array([0] * n0 + [1] * n1)
    y = rng.integers(0, 2, size=n0 + n1)
    weights = rng.uniform(0.2, 2.5, size=n0 + n1)
    weights[0] = 0.0
    theta = rng.normal(scale=0.6, size=2 * (d + 1))
    x_in = scipy.sparse.csr_matrix(x) if use_sparse else x

    _, analytic = objective_and_gradient(
        theta, x_in, y, a, weights, eta=25.0, l2_strength=1.75
    )

    eps = 1e-6
    numeric = np.empty_like(theta)
    for idx in range(len(theta)):
        step = np.zeros_like(theta)
        step[idx] = eps
        f_plus, _ = objective_and_gradient(
            theta + step, x_in, y, a, weights, eta=25.0, l2_strength=1.75
        )
        f_minus, _ = objective_and_gradient(
            theta - step, x_in, y, a, weights, eta=25.0, l2_strength=1.75
        )
        numeric[idx] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_prejudice_remover_eta_zero_matches_independent_oracle(use_sparse, weighted):
    rng = np.random.default_rng(42)
    n, d = 120, 6
    x = rng.normal(size=(n, d))
    a = np.array([0] * 45 + [1] * 75)
    logits = np.where(a == 0, x[:, 0] - 0.5 * x[:, 1] + 0.3, -0.8 * x[:, 2] - 0.4)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    weights = rng.uniform(0.5, 2.0, size=n) if weighted else None
    l2_strength = 2.5

    x_in = scipy.sparse.csr_matrix(x) if use_sparse else x
    model = PrejudiceRemover(eta=0.0, l2_strength=l2_strength, tol=1e-10).fit(
        x_in, y, a=a, sample_weight=weights
    )
    assert model.converged_
    proba = model.predict_proba(x_in, a=a)

    for group in (0, 1):
        mask = a == group
        x_aug = np.column_stack([x[mask], np.ones(np.count_nonzero(mask))])
        oracle = LogisticRegression(
            C=1.0 / l2_strength, fit_intercept=False, tol=1e-12, max_iter=2000
        )
        oracle.fit(
            x_aug, y[mask], sample_weight=None if weights is None else weights[mask]
        )
        expected_theta = oracle.coef_.ravel()
        np.testing.assert_allclose(model.coef_[group], expected_theta[:-1], atol=1e-5)
        np.testing.assert_allclose(
            model.intercept_[group], expected_theta[-1], atol=1e-5
        )
        np.testing.assert_allclose(proba[mask], oracle.predict_proba(x_aug), atol=1e-5)


def test_prejudice_remover_dense_sparse_reproducibility_and_immutability():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 4))
    a = np.tile([0, 1], 40)
    y = rng.integers(0, 2, size=80)
    w = rng.uniform(0.5, 1.5, size=80)
    for arr in (x, a, y, w):
        arr.setflags(write=False)

    dense_model = PrejudiceRemover(eta=15.0, l2_strength=1.0, seed=1, threads=1).fit(
        x, y, a=a, sample_weight=w
    )
    sparse_model = PrejudiceRemover(eta=15.0, l2_strength=1.0, seed=1, threads=1).fit(
        scipy.sparse.csc_matrix(x), y, a=a, sample_weight=w
    )
    repeat_model = PrejudiceRemover(eta=15.0, l2_strength=1.0, seed=1, threads=1).fit(
        x, y, a=a, sample_weight=w
    )

    np.testing.assert_array_equal(dense_model.classes_, [0, 1])
    np.testing.assert_allclose(dense_model.coef_, sparse_model.coef_, atol=1e-10)
    np.testing.assert_allclose(
        dense_model.intercept_, sparse_model.intercept_, atol=1e-10
    )
    np.testing.assert_allclose(dense_model.loss_, sparse_model.loss_, atol=1e-10)
    np.testing.assert_array_equal(dense_model.coef_, repeat_model.coef_)

    p_dense = dense_model.predict_proba(x, a=a)
    p_sparse = sparse_model.predict_proba(scipy.sparse.csr_matrix(x), a=a)
    np.testing.assert_allclose(p_dense, p_sparse, atol=1e-10)
    assert p_dense.shape == (80, 2)
    assert np.all(np.isfinite(p_dense))
    assert np.all((p_dense >= 0.0) & (p_dense <= 1.0))
    np.testing.assert_allclose(p_dense.sum(axis=1), 1.0)

    # Single-group prediction slices must work cleanly after two-group fit.
    p_g0 = dense_model.predict_proba(x[:5], a=np.zeros(5, dtype=int))
    p_g1 = dense_model.predict_proba(x[:5], a=np.ones(5, dtype=int))
    np.testing.assert_allclose(p_g0[[0, 2, 4]], p_dense[[0, 2, 4]])
    np.testing.assert_allclose(p_g1[[1, 3]], p_dense[[1, 3]])


def test_prejudice_remover_invalid_inputs_and_nonconvergence():
    with pytest.raises(ValueError, match="eta"):
        PrejudiceRemover(eta=-1.0)
    with pytest.raises(ValueError, match="l2_strength"):
        PrejudiceRemover(l2_strength=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        PrejudiceRemover(max_iter=0)

    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 1, 0, 1])
    a = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="both groups"):
        PrejudiceRemover().fit(x, y, a=np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="positive training weight mass"):
        PrejudiceRemover().fit(x, y, a=a, sample_weight=[1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        PrejudiceRemover().fit(
            np.array([[np.nan, 1.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), y, a=a
        )
    with pytest.raises(RuntimeError, match="fitted"):
        PrejudiceRemover().predict_proba(x, a=a)

    # max_iter=1 triggers non-convergence error instead of silently returning bad weights.
    rng = np.random.default_rng(99)
    big_x = rng.normal(size=(100, 12))
    big_a = np.tile([0, 1], 50)
    big_y = rng.integers(0, 2, size=100)
    failing = PrejudiceRemover(eta=50.0, l2_strength=0.1, max_iter=1, tol=1e-12)
    with pytest.raises(RuntimeError, match="failed to converge"):
        failing.fit(big_x, big_y, a=big_a)
    assert failing.converged_ is False


def test_cli_pr_smoke_and_undefined_report_metrics_become_json_null(tmp_path):
    assert "profit_eq16" in BUILTIN_METRICS
    assert "prejudice_remover" in BUILTIN_ESTIMATORS

    cli_main(
        [
            str(ROOT / "configs/tiny-pr.toml"),
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "cli-smoke",
        ]
    )
    run_dir = tmp_path / "cli-smoke"
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert set(meta["selected"]) == {"pr_eta_0", "pr_eta_10"}
    assert meta["config"]["selection_metric"] == "profit_eq16"
    assert (run_dir / "validation_predictions.csv").is_file()
    assert (run_dir / "test_predictions.csv").is_file()
    assert (run_dir / "splits.csv").is_file()

    # Force threshold=1.0 so no application is approved -> sf, ppv_0, ppv_1 become null.
    config = replace(
        load_config(ROOT / "configs/tiny-pr.toml"),
        run_id="null-metrics",
        output_dir=tmp_path,
        thresholds=(1.0,),
    )
    null_dir = run(config)
    null_meta = json.loads((null_dir / "run.json").read_text(encoding="utf-8"))
    for model_name in ("pr_eta_0", "pr_eta_10"):
        for part in ("validation_metrics", "test_metrics"):
            metrics = null_meta["selected"][model_name][part]
            assert metrics["profit_eq16"] is not None
            assert metrics["sf"] is None
            assert metrics["ppv_0"] is None
            assert metrics["ppv_1"] is None


def test_pr_selection_does_not_leak_test_outcomes_and_manifest_aligns(
    tmp_path, monkeypatch
):
    from fair_credit_scoring import runner

    base_cfg = load_config(ROOT / "configs/tiny-pr.toml")
    raw = pd.read_csv(base_cfg.dataset.path, dtype={"row_id": str})
    data = prepare_dataset(raw, base_cfg.dataset)
    splits = make_splits(data, base_cfg.dataset.name, base_cfg.seed)

    # Verify preprocessing only fits on training partitions during PR sweep.
    allowed_train = {
        tuple(data.row_id[rows])
        for rows in (splits.development, *[train for train, _ in splits.folds])
    }
    fitted_train = set()
    original_make_preprocessor = runner.make_preprocessor

    def recording_preprocessor(numeric, categorical, *, scaling):
        preprocessor = original_make_preprocessor(numeric, categorical, scaling=scaling)
        fit_transform = preprocessor.fit_transform

        def record(features):
            ids = tuple(features.index)
            assert ids in allowed_train
            fitted_train.add(ids)
            return fit_transform(features)

        preprocessor.fit_transform = record
        return preprocessor

    monkeypatch.setattr(runner, "make_preprocessor", recording_preprocessor)
    run_a = run(replace(base_cfg, run_id="pr-leak", output_dir=tmp_path / "a"))
    assert fitted_train == allowed_train
    monkeypatch.undo()

    # Flip y on the fixed test rows only; OOF selection, scores, and decisions must not change.
    flipped_y = data.y.copy()
    flipped_y[splits.test] = 1 - flipped_y[splits.test]
    flipped_data = replace(data, y=flipped_y)
    monkeypatch.setattr(runner, "prepare_dataset", lambda *_: flipped_data)
    monkeypatch.setattr(runner, "make_splits", lambda *_: splits)
    run_b = run(replace(base_cfg, run_id="pr-leak", output_dir=tmp_path / "b"))
    monkeypatch.undo()

    meta_a = json.loads((run_a / "run.json").read_text(encoding="utf-8"))
    meta_b = json.loads((run_b / "run.json").read_text(encoding="utf-8"))
    assert meta_a["split_id"] == meta_b["split_id"]
    for model_name in ("pr_eta_0", "pr_eta_10"):
        assert (
            meta_a["selected"][model_name]["params"]
            == meta_b["selected"][model_name]["params"]
        )
        assert (
            meta_a["selected"][model_name]["threshold"]
            == meta_b["selected"][model_name]["threshold"]
        )
        assert (
            meta_a["selected"][model_name]["selection_value"]
            == meta_b["selected"][model_name]["selection_value"]
        )
    assert (run_a / "validation_predictions.csv").read_bytes() == (
        run_b / "validation_predictions.csv"
    ).read_bytes()
    test_a = pd.read_csv(run_a / "test_predictions.csv")
    test_b = pd.read_csv(run_b / "test_predictions.csv")
    pd.testing.assert_frame_equal(
        test_a.drop(columns="y"), test_b.drop(columns="y"), check_exact=True
    )

    # Check manifest builder on aligned synthetic baseline + PR runs (scaling="max_abs").
    baseline_cfg = replace(
        load_config(ROOT / "configs/tiny.toml"),
        run_id="tiny-base-profit",
        output_dir=tmp_path,
        selection_metric="profit_eq16",
        thresholds=(0.0, 0.5, 1.0),
        dataset=replace(base_cfg.dataset, scaling="max_abs"),
    )
    pr_maxabs_cfg = replace(
        base_cfg,
        run_id="tiny-pr-profit",
        output_dir=tmp_path,
        thresholds=(0.0, 0.5, 1.0),
        dataset=replace(base_cfg.dataset, scaling="max_abs"),
    )
    base_dir = run(baseline_cfg)
    pr_dir = run(pr_maxabs_cfg)
    manifest_path = tmp_path / "manifest.csv"
    rows = build_handoff_manifest(
        [base_dir, pr_dir], [manifest_path], repo_root=tmp_path
    )
    assert len(rows) == 4
    assert manifest_path.is_file()

    control_path = tmp_path / "control.csv"
    control_rows = build_control_table([base_dir, pr_dir], [control_path])
    # 2 partitions * (2 constant policies + 4 models) = 12 rows
    assert len(control_rows) == 12
    assert control_path.is_file()
    control_df = pd.read_csv(control_path, keep_default_na=False)
    reject_rows = control_df[control_df["model"] == "reject-all"]
    assert (reject_rows["sf"] == "NaN").all()
    approve_rows = control_df[control_df["model"] == "approve-all"]
    assert (approve_rows["delta_profit_vs_approve_all"].astype(float) == 0.0).all()

    # Manifest verifier must reject mismatched selection metrics.
    bad_cfg = replace(
        baseline_cfg, run_id="tiny-bacc", selection_metric="balanced_accuracy"
    )
    bad_dir = run(bad_cfg)
    with pytest.raises(ValueError, match="profit_eq16"):
        build_handoff_manifest([base_dir, bad_dir], [manifest_path], repo_root=tmp_path)
