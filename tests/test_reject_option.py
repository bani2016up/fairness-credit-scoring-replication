import csv
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import fair_credit_scoring
from fair_credit_scoring import (
    DatasetConfig,
    ModelConfig,
    RejectOptionClassifier,
    RunConfig,
    evaluate_metrics,
    load_config,
    run,
)
from fair_credit_scoring.handoff import build_control_table, build_handoff_manifest
from fair_credit_scoring.hooks import BUILTIN_POSTPROCESSORS
from fair_credit_scoring.metrics import METRICS

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.csv"


def _brute_force_select(
    score: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    base_threshold: float,
    *,
    fairness_metric: str,
    fairness_bound: float,
    utility_metric: str,
    candidate_thresholds: np.ndarray,
    candidate_margins: np.ndarray,
):
    best_feasible = None
    best_infeasible = None

    for tau in candidate_thresholds:
        for margin in candidate_margins:
            base = score >= tau
            if margin == 0.0:
                decision = base.astype(np.int8)
            else:
                in_crit = np.abs(score - tau) <= margin
                decision = np.where(in_crit, a == 0, base).astype(np.int8)

            u = float(METRICS[utility_metric](y, a, score, decision))
            f = float(METRICS[fairness_metric](y, a, score, decision))
            if not (math.isfinite(u) and math.isfinite(f)):
                continue

            dist_tau = abs(float(tau) - float(base_threshold))
            if f <= fairness_bound:
                key = (-u, f, float(margin), dist_tau, float(tau))
                if best_feasible is None or key < best_feasible[0]:
                    best_feasible = (key, float(tau), float(margin), u, f)
            elif best_feasible is None:
                key = (f, -u, float(margin), dist_tau, float(tau))
                if best_infeasible is None or key < best_infeasible[0]:
                    best_infeasible = (key, float(tau), float(margin), u, f)

    if best_feasible is not None:
        _, tau, margin, u, f = best_feasible
        return True, tau, margin, u, f
    assert best_infeasible is not None
    _, tau, margin, u, f = best_infeasible
    return False, tau, margin, u, f


def test_init_validation():
    with pytest.raises(ValueError, match="fairness_metric"):
        RejectOptionClassifier(fairness_metric="sf")
    with pytest.raises(ValueError, match="fairness_metric"):
        RejectOptionClassifier(fairness_metric=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="utility_metric"):
        RejectOptionClassifier(utility_metric="nonexistent")
    with pytest.raises(ValueError, match="fairness_bound"):
        RejectOptionClassifier(fairness_bound=-0.01)
    with pytest.raises(ValueError, match="fairness_bound"):
        RejectOptionClassifier(fairness_bound=math.nan)
    with pytest.raises(ValueError, match="fairness_bound"):
        RejectOptionClassifier(fairness_bound=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tune_threshold"):
        RejectOptionClassifier(tune_threshold=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_thresholds"):
        RejectOptionClassifier(n_thresholds=0)
    with pytest.raises(ValueError, match="n_thresholds"):
        RejectOptionClassifier(n_thresholds=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_margins"):
        RejectOptionClassifier(n_margins=0)
    with pytest.raises(ValueError, match="max_margin"):
        RejectOptionClassifier(max_margin=-0.1)
    with pytest.raises(ValueError, match="max_margin"):
        RejectOptionClassifier(max_margin=1.1)
    with pytest.raises(ValueError, match="thresholds"):
        RejectOptionClassifier(thresholds=[])
    with pytest.raises(ValueError, match="thresholds"):
        RejectOptionClassifier(thresholds="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="thresholds"):
        RejectOptionClassifier(thresholds={0.5: 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="threshold"):
        RejectOptionClassifier(thresholds=[0.5, 1.2])
    with pytest.raises(ValueError, match="threshold"):
        RejectOptionClassifier(thresholds=[True])  # type: ignore[list-item]


def test_fit_and_predict_validation():
    clf = RejectOptionClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([0.5, 0.6], [0, 1])
    with pytest.raises(RuntimeError):
        clf.predict_before_and_after([0.5, 0.6], [0, 1])
    with pytest.raises(RuntimeError):
        clf.metadata()

    with pytest.raises(ValueError, match="same length"):
        clf.fit([0.5, 0.6], [0, 1], [0])
    with pytest.raises(ValueError, match="same length"):
        clf.fit([0.5], [0, 1], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        clf.fit([0.5, np.nan], [0, 1], [0, 1])
    with pytest.raises(ValueError, match="threshold"):
        clf.fit([0.2, 0.8], [0, 1], [0, 1], threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        clf.fit([0.2, 0.8], [0, 1], [0, 1], threshold=False)  # type: ignore[arg-type]

    # Single group makes sp_paper / eo_gap / ind NaN for every candidate -> ValueError.
    with pytest.raises(ValueError, match="No candidate"):
        clf.fit([0.2, 0.8, 0.4, 0.6], [0, 1, 0, 1], [0, 0, 0, 0])


def test_critical_region_group_direction_and_zero_margin():
    clf = RejectOptionClassifier(
        tune_threshold=False,
        n_margins=1,
        max_margin=0.0,
        fairness_bound=1.0,
    )
    scores = np.array([0.3, 0.5, 0.7, 0.3, 0.5, 0.7])
    y = np.array([0, 1, 1, 0, 0, 1])
    a = np.array([0, 0, 0, 1, 1, 1])
    clf.fit(scores, y, a, threshold=0.5)
    assert clf.margin_ == 0.0
    assert clf.threshold_ == 0.5

    # margin == 0 -> empty critical region, so score == 0.5 for a == 1 stays approved (1).
    pred = clf.predict(scores, a)
    assert pred.dtype == np.int8
    np.testing.assert_array_equal(pred, np.array([0, 1, 1, 0, 1, 1], dtype=np.int8))

    # Nonzero margin: |score - 0.5| <= 0.125 -> a==0 gets 1, a==1 gets 0.
    clf.margin_ = 0.125
    clf.threshold_ = 0.5
    clf.base_threshold_ = 0.4
    probe_scores = np.array([0.2, 0.375, 0.5, 0.625, 0.8, 0.2, 0.375, 0.5, 0.625, 0.8])
    probe_a = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    before, after = clf.predict_before_and_after(probe_scores, probe_a)
    assert after.dtype == np.int8
    assert before.dtype == np.int8
    np.testing.assert_array_equal(
        after, np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=np.int8)
    )
    np.testing.assert_array_equal(before, (probe_scores >= 0.4).astype(np.int8))
    np.testing.assert_array_equal(after, clf.predict(probe_scores, probe_a))


@pytest.mark.parametrize("fairness_metric", ["ind", "sp_paper", "eo_gap"])
@pytest.mark.parametrize(
    "utility_metric",
    ["profit_eq16", "balanced_accuracy", "approval_rate", "sf", "auc"],
)
def test_vectorized_search_matches_bruteforce_metrics(fairness_metric, utility_metric):
    rng = np.random.default_rng(12345)
    n = 250
    a = rng.integers(0, 2, size=n)
    score = np.clip(rng.normal(loc=0.55 + 0.1 * a, scale=0.2, size=n), 0.0, 1.0)
    y = rng.binomial(1, np.clip(score, 0.05, 0.95))

    clf = RejectOptionClassifier(
        fairness_metric=fairness_metric,
        fairness_bound=0.06,
        utility_metric=utility_metric,
        n_thresholds=21,
        n_margins=15,
        max_margin=0.35,
    )
    returned = clf.fit(score, y, a, threshold=0.53)
    assert returned is clf

    candidate_thresholds = np.unique(np.append(np.linspace(0.0, 1.0, 21), 0.53))
    candidate_margins = np.unique(np.linspace(0.0, 0.35, 15))
    feasible, tau, margin, u, f = _brute_force_select(
        score,
        y,
        a,
        0.53,
        fairness_metric=fairness_metric,
        fairness_bound=0.06,
        utility_metric=utility_metric,
        candidate_thresholds=candidate_thresholds,
        candidate_margins=candidate_margins,
    )

    assert clf.feasible_ is feasible
    assert clf.threshold_ == tau
    assert clf.margin_ == margin
    assert clf.validation_utility_ == pytest.approx(u, abs=1e-12)
    assert clf.validation_fairness_ == pytest.approx(f, abs=1e-12)

    base_decision = (score >= 0.53).astype(np.int8)
    expected_base_u = float(METRICS[utility_metric](y, a, score, base_decision))
    expected_base_f = float(METRICS[fairness_metric](y, a, score, base_decision))
    assert clf.base_validation_utility_ == pytest.approx(expected_base_u, abs=1e-12)
    assert clf.base_validation_fairness_ == pytest.approx(expected_base_f, abs=1e-12)

    after_decision = clf.predict(score, a)
    assert float(METRICS[utility_metric](y, a, score, after_decision)) == pytest.approx(
        clf.validation_utility_, abs=1e-12
    )
    assert float(
        METRICS[fairness_metric](y, a, score, after_decision)
    ) == pytest.approx(clf.validation_fairness_, abs=1e-12)


def test_infeasible_fallback_and_tie_breaking():
    # Construct data where ind cannot reach 0.0 for threshold=0.8, margin=0 only.
    score = np.array([0.2, 0.2, 0.9, 0.2, 0.9, 0.9, 0.9])
    y = np.array([0, 1, 1, 0, 1, 1, 0])
    a = np.array([0, 0, 0, 1, 1, 1, 1])

    clf = RejectOptionClassifier(
        fairness_metric="ind",
        fairness_bound=0.0,
        utility_metric="approval_rate",
        tune_threshold=False,
        n_margins=3,
        max_margin=0.0,
    )
    clf.fit(score, y, a, threshold=0.8)
    assert clf.feasible_ is False
    assert clf.threshold_ == 0.8
    assert clf.margin_ == 0.0
    assert clf.validation_fairness_ == pytest.approx(abs(0.75 - 1.0 / 3.0))

    # Explicit thresholds and tie-breaking: two thresholds produce identical decisions.
    score_tie = np.array([0.1, 0.9, 0.1, 0.9])
    y_tie = np.array([0, 1, 0, 1])
    a_tie = np.array([0, 0, 1, 1])
    tie_clf = RejectOptionClassifier(
        fairness_metric="ind",
        fairness_bound=0.1,
        utility_metric="profit_eq16",
        thresholds=(0.2, 0.45, 0.8),
        n_margins=3,
        max_margin=0.2,
    )
    tie_clf.fit(score_tie, y_tie, a_tie, threshold=0.4)
    # Closest to base_threshold_=0.4 among (0.2, 0.45, 0.8) is 0.45, with margin=0.0.
    assert tie_clf.feasible_ is True
    assert tie_clf.margin_ == 0.0
    assert tie_clf.threshold_ == 0.45


def test_performance_15k_rows_under_one_second():
    rng = np.random.default_rng(99)
    n = 15_000
    a = rng.integers(0, 2, size=n)
    score = rng.random(size=n)
    y = rng.binomial(1, np.clip(0.3 + 0.5 * score, 0.0, 1.0))

    clf = RejectOptionClassifier(
        fairness_metric="sp_paper",
        fairness_bound=0.05,
        utility_metric="profit_eq16",
        n_thresholds=101,
        n_margins=50,
        max_margin=0.5,
    )
    start = time.perf_counter()
    clf.fit(score, y, a, threshold=0.5)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"RejectOptionClassifier.fit took {elapsed:.3f}s"


def test_pipeline_integration_with_builtin_reject_option(tmp_path):
    assert "reject_option" in BUILTIN_POSTPROCESSORS
    assert BUILTIN_POSTPROCESSORS["reject_option"] is RejectOptionClassifier
    assert "RejectOptionClassifier" in fair_credit_scoring.__all__

    with pytest.raises(ValueError, match="postprocessor_params"):
        RunConfig(
            dataset=DatasetConfig(
                name="tiny",
                path=FIXTURE,
                numeric=("income", "balance", "tenure"),
                categorical=("channel", "region"),
            ),
            models=(ModelConfig("lr", "logistic_regression"),),
            run_id="bad-params",
            postprocessor_params=["invalid"],  # type: ignore[arg-type]
        )
    with pytest.raises(
        ValueError, match="postprocessor_params requires a postprocessor"
    ):
        RunConfig(
            dataset=DatasetConfig(
                name="tiny",
                path=FIXTURE,
                numeric=("income", "balance", "tenure"),
                categorical=("channel", "region"),
            ),
            models=(ModelConfig("lr", "logistic_regression"),),
            run_id="orphan-params",
            postprocessor_params={"fairness_bound": 0.05},
        )

    config = RunConfig(
        dataset=DatasetConfig(
            name="tiny",
            path=FIXTURE,
            numeric=("income", "balance", "tenure"),
            categorical=("channel", "region"),
        ),
        models=(ModelConfig("lr", "logistic_regression"),),
        run_id="roc-integration",
        output_dir=tmp_path,
        seed=37,
        thresholds=(0.3, 0.5, 0.7),
        selection_metric="profit_eq16",
        metrics=("auc", "profit_eq16", "approval_rate", "sp_paper", "ind", "eo_gap"),
        postprocessor="reject_option",
        postprocessor_params={
            "fairness_metric": "sp_paper",
            "fairness_bound": 0.05,
            "utility_metric": "profit_eq16",
            "n_thresholds": 21,
            "n_margins": 11,
            "max_margin": 0.3,
        },
    )
    result = run(config)
    saved = json.loads((result / "run.json").read_text(encoding="utf-8"))
    lr_selected = saved["selected"]["lr"]
    assert "postprocessor" in lr_selected
    pp_meta = lr_selected["postprocessor"]
    assert pp_meta["fairness_metric"] == "sp_paper"
    assert pp_meta["base_threshold"] == lr_selected["threshold"]
    assert set(pp_meta) == {
        "fairness_metric",
        "fairness_bound",
        "utility_metric",
        "tune_threshold",
        "base_threshold",
        "threshold",
        "margin",
        "feasible",
        "validation_utility",
        "validation_fairness",
        "base_validation_utility",
        "base_validation_fairness",
    }

    for split_name in ("validation", "test"):
        with (result / f"{split_name}_predictions.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert set(rows[0]) == {
            "row_id",
            "y",
            "a",
            "score",
            "decision",
            "decision_before",
            "decision_after",
            "run_id",
            "dataset",
            "model",
            "split_id",
            "partition",
            "validation_fold",
        }
        y_vec = np.array([int(r["y"]) for r in rows])
        a_vec = np.array([int(r["a"]) for r in rows])
        score_vec = np.array([float(r["score"]) for r in rows])
        dec_vec = np.array([int(r["decision"]) for r in rows])
        before_vec = np.array([int(r["decision_before"]) for r in rows])
        after_vec = np.array([int(r["decision_after"]) for r in rows])

        np.testing.assert_array_equal(dec_vec, after_vec)
        np.testing.assert_array_equal(
            before_vec, (score_vec >= lr_selected["threshold"]).astype(np.int8)
        )
        recomputed = evaluate_metrics(
            y_vec,
            a_vec,
            score_vec,
            dec_vec,
            ("auc", "profit_eq16", "approval_rate", "sp_paper", "ind", "eo_gap"),
        )
        saved_metrics = lr_selected[f"{split_name}_metrics"]
        for metric_name, value in recomputed.items():
            if math.isnan(value):
                assert saved_metrics[metric_name] is None
            else:
                assert saved_metrics[metric_name] == pytest.approx(value, abs=1e-12)


def test_natural_nonzero_margin_flips_and_boundary_scores():
    # Natural fit() (no manual attribute overrides) must select margin_ > 0 and flip
    # both a=0 (0 -> 1) and a=1 (1 -> 0), while respecting exact 0.0 / 1.0 scores.
    scores = np.array([0.0, 0.1, 0.4, 1.0, 0.0, 0.55, 0.8, 1.0])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    clf = RejectOptionClassifier(
        fairness_metric="ind",
        fairness_bound=0.05,
        utility_metric="approval_rate",
        tune_threshold=False,
        n_margins=5,
        max_margin=0.2,
    )
    clf.fit(scores, y, a, threshold=0.5)
    assert clf.feasible_ is True
    assert clf.threshold_ == 0.5
    assert clf.margin_ == pytest.approx(0.1)
    assert clf.validation_fairness_ == pytest.approx(0.0)

    before, after = clf.predict_before_and_after(scores, a)
    np.testing.assert_array_equal(
        before, np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int8)
    )
    np.testing.assert_array_equal(
        after, np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    )
    # Row 2 (a=0, score=0.4) flips 0 -> 1; row 5 (a=1, score=0.55) flips 1 -> 0.
    assert before[2] == 0 and after[2] == 1
    assert before[5] == 1 and after[5] == 0
    # Boundary scores 0.0 and 1.0 stay outside the critical region around 0.5.
    assert after[0] == 0 and after[4] == 0
    assert after[3] == 1 and after[7] == 1

    # Boundary thresholds tau == 0.0 and tau == 1.0 with boundary scores 0.0 and 1.0.
    clf_zero = RejectOptionClassifier(
        fairness_metric="ind",
        fairness_bound=1.0,
        utility_metric="balanced_accuracy",
        tune_threshold=False,
        n_margins=3,
        max_margin=0.2,
    )
    scores_zero = np.array([0.0, 0.1, 0.8, 1.0, 0.0, 0.1, 0.8, 1.0])
    y_zero = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    a_zero = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    clf_zero.fit(scores_zero, y_zero, a_zero, threshold=0.0)
    assert clf_zero.threshold_ == 0.0
    assert clf_zero.margin_ > 0.0
    before_z, after_z = clf_zero.predict_before_and_after(scores_zero, a_zero)
    np.testing.assert_array_equal(before_z, np.ones(8, dtype=np.int8))
    # At tau=0.0, score==0.0 for a=0 stays 1, score==0.0 for a=1 flips 1 -> 0, score==1.0 stays 1.
    assert after_z[0] == 1 and after_z[3] == 1
    assert after_z[4] == 0 and after_z[7] == 1

    clf_one = RejectOptionClassifier(
        fairness_metric="ind",
        fairness_bound=1.0,
        utility_metric="balanced_accuracy",
        tune_threshold=False,
        n_margins=3,
        max_margin=0.2,
    )
    scores_one = np.array([0.0, 0.9, 1.0, 0.0, 0.2, 1.0])
    y_one = np.array([0, 1, 1, 0, 1, 0])
    a_one = np.array([0, 0, 0, 1, 1, 1])
    clf_one.fit(scores_one, y_one, a_one, threshold=1.0)
    assert clf_one.threshold_ == 1.0
    assert clf_one.margin_ == pytest.approx(0.1)
    before_o, after_o = clf_one.predict_before_and_after(scores_one, a_one)
    np.testing.assert_array_equal(before_o, np.array([0, 0, 1, 0, 0, 1], dtype=np.int8))
    np.testing.assert_array_equal(after_o, np.array([0, 1, 1, 0, 0, 0], dtype=np.int8))


def test_missing_stratum_nans_and_metadata_json_serialization(tmp_path):
    # n00 == 0 (all a=0 rows have y=1) makes sp_paper and eo_gap NaN for every candidate.
    scores = np.array([0.2, 0.8, 0.3, 0.7])
    y_no_neg_a0 = np.array([1, 1, 0, 1])
    a_both = np.array([0, 0, 1, 1])
    for fairness_metric in ("sp_paper", "eo_gap"):
        clf = RejectOptionClassifier(fairness_metric=fairness_metric)
        with pytest.raises(ValueError, match="No candidate"):
            clf.fit(scores, y_no_neg_a0, a_both)

    # balanced_accuracy is NaN when n_pos == 0 or n_neg == 0, even if ind is finite.
    for y_single_class in (np.zeros(4, dtype=int), np.ones(4, dtype=int)):
        clf = RejectOptionClassifier(
            fairness_metric="ind", utility_metric="balanced_accuracy"
        )
        with pytest.raises(ValueError, match="No candidate"):
            clf.fit(scores, y_single_class, a_both)

    # base_validation_utility_ is NaN at base_threshold=1.0 (sf undefined with 0 approvals),
    # while lower tau yields finite sf; metadata() must emit None and serialize with allow_nan=False.
    sf_scores = np.array([0.2, 0.4, 0.6, 0.8])
    sf_y = np.array([0, 1, 0, 1])
    sf_clf = RejectOptionClassifier(
        fairness_metric="ind",
        fairness_bound=0.5,
        utility_metric="sf",
        thresholds=(0.5, 1.0),
        n_margins=3,
        max_margin=0.2,
    )
    sf_clf.fit(sf_scores, sf_y, a_both, threshold=1.0)
    assert math.isnan(sf_clf.base_validation_utility_)
    assert math.isfinite(sf_clf.base_validation_fairness_)
    assert math.isfinite(sf_clf.validation_utility_)
    meta = sf_clf.metadata()
    assert meta["base_validation_utility"] is None
    assert meta["validation_utility"] is not None
    json.dumps(meta, allow_nan=False)

    # End-to-end run() with base threshold 1.0 and utility_metric="sf" writes valid run.json.
    cfg = RunConfig(
        dataset=DatasetConfig(
            name="tiny",
            path=FIXTURE,
            numeric=("income", "balance", "tenure"),
            categorical=("channel", "region"),
        ),
        models=(ModelConfig("lr", "logistic_regression"),),
        run_id="roc-nan-base-utility",
        output_dir=tmp_path,
        seed=37,
        thresholds=(1.0,),
        selection_metric="profit_eq16",
        metrics=("auc", "profit_eq16", "sf", "ind"),
        postprocessor="reject_option",
        postprocessor_params={
            "fairness_metric": "ind",
            "fairness_bound": 0.05,
            "utility_metric": "sf",
            "thresholds": [0.4, 0.5, 0.6],
            "n_margins": 5,
            "max_margin": 0.2,
        },
    )
    result = run(cfg)
    raw_json = (result / "run.json").read_text(encoding="utf-8")
    saved = json.loads(raw_json)
    pp_meta = saved["selected"]["lr"]["postprocessor"]
    assert pp_meta["base_threshold"] == 1.0
    assert pp_meta["base_validation_utility"] is None
    assert pp_meta["validation_utility"] is not None
    json.dumps(saved, allow_nan=False)


def test_toml_config_predict_before_and_after_and_handoff(tmp_path):
    rel_fixture = os.path.relpath(FIXTURE, tmp_path)
    toml_path = tmp_path / "roc.toml"
    toml_path.write_text(
        f"""
run_id = "tiny-roc"
output_dir = "outputs"
seed = 37
threads = 1
thresholds = [0.0, 0.5, 1.0]
selection_metric = "profit_eq16"
metrics = ["auc", "profit_eq16", "approval_rate", "sp_paper", "ind", "eo_gap", "sf"]
method = "reject_option"
postprocessor = "reject_option"

[postprocessor_params]
fairness_metric = "sp_paper"
fairness_bound = 0.05
utility_metric = "profit_eq16"
thresholds = [0.3, 0.5, 0.7]
n_margins = 11
max_margin = 0.3

[dataset]
name = "synthetic-tiny"
path = "{rel_fixture}"
scaling = "max_abs"
numeric = ["income", "balance", "tenure"]
categorical = ["channel", "region"]

[[models]]
name = "lr"
estimator = "logistic_regression"
[models.grid]
C = [0.5, 1.0]
""".lstrip(),
        encoding="utf-8",
    )

    roc_cfg = load_config(toml_path)
    assert roc_cfg.postprocessor == "reject_option"
    assert roc_cfg.postprocessor_params == {
        "fairness_metric": "sp_paper",
        "fairness_bound": 0.05,
        "utility_metric": "profit_eq16",
        "thresholds": [0.3, 0.5, 0.7],
        "n_margins": 11,
        "max_margin": 0.3,
    }

    roc_dir = run(roc_cfg)
    base_cfg = replace(
        roc_cfg,
        run_id="tiny-base",
        method="baseline",
        postprocessor=None,
        postprocessor_params={},
    )
    base_dir = run(base_cfg)

    saved_roc = json.loads((roc_dir / "run.json").read_text(encoding="utf-8"))
    lr_selected = saved_roc["selected"]["lr"]
    pp_meta = lr_selected["postprocessor"]
    assert pp_meta["fairness_metric"] == "sp_paper"
    assert pp_meta["fairness_bound"] == 0.05
    assert pp_meta["threshold"] in {0.3, 0.5, 0.7}

    # Reconstruct fitted RejectOptionClassifier from validation predictions and verify
    # decision_before, decision_after, and decision in both validation & test CSVs.
    def _read_preds(split_name: str):
        with (roc_dir / f"{split_name}_predictions.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        return {
            "y": np.array([int(r["y"]) for r in rows]),
            "a": np.array([int(r["a"]) for r in rows]),
            "score": np.array([float(r["score"]) for r in rows]),
            "decision": np.array([int(r["decision"]) for r in rows], dtype=np.int8),
            "decision_before": np.array(
                [int(r["decision_before"]) for r in rows], dtype=np.int8
            ),
            "decision_after": np.array(
                [int(r["decision_after"]) for r in rows], dtype=np.int8
            ),
        }

    val = _read_preds("validation")
    clf = RejectOptionClassifier(**roc_cfg.postprocessor_params).fit(
        val["score"], val["y"], val["a"], threshold=lr_selected["threshold"]
    )
    assert clf.threshold_ == pp_meta["threshold"]
    assert clf.margin_ == pp_meta["margin"]
    assert clf.base_threshold_ == pp_meta["base_threshold"]

    for split_name in ("validation", "test"):
        split_data = _read_preds(split_name)
        expected_before, expected_after = clf.predict_before_and_after(
            split_data["score"], split_data["a"]
        )
        np.testing.assert_array_equal(split_data["decision_before"], expected_before)
        np.testing.assert_array_equal(split_data["decision_after"], expected_after)
        np.testing.assert_array_equal(split_data["decision"], expected_after)

    manifest_path = tmp_path / "manifest.csv"
    manifest_rows = build_handoff_manifest(
        [base_dir, roc_dir], [manifest_path], repo_root=tmp_path
    )
    assert len(manifest_rows) == 2
    assert {row["method"] for row in manifest_rows} == {"baseline", "reject_option"}
    assert manifest_path.is_file()

    control_path = tmp_path / "control.csv"
    control_rows = build_control_table([base_dir, roc_dir], [control_path])
    # 2 partitions * (2 constant policies + 2 runs) = 8 rows
    assert len(control_rows) == 8
    assert {row["method"] for row in control_rows} == {
        "constant_policy",
        "baseline",
        "reject_option",
    }
    assert control_path.is_file()
