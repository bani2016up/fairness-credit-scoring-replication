# Fairness in Credit Scoring Replication

A partial replication of [Fairness in Credit Scoring: Assessment, Implementation and Profit Implications](https://arxiv.org/abs/2103.01907). The experiment runner fits Logistic Regression, Random Forest, and Prejudice Remover, supports built-in Reject Option Classification post-processing, and exposes hooks for the dataset, metrics, and fairness methods. See [the experiment protocol](docs/protocol.md), [the data and Reweighing notes](docs/replication-data.md), and [the final analysis](docs/final-analysis.md) for methodology, data preparation, results, and differences from the original study.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these commands from the repository root.

```sh
uv python install 3.13
uv sync --locked
uv run --locked python -m fair_credit_scoring configs/tiny.toml
uv run --locked python -m fair_credit_scoring configs/tiny-pr.toml
uv run --locked python -m fair_credit_scoring configs/tiny-roc.toml
```

The synthetic smoke commands evaluate the baseline, Prejudice Remover, and Reject Option Classification configs on `tests/fixtures/tiny.csv` and write to `outputs/tiny-baseline/`, `outputs/tiny-pr/`, and `outputs/tiny-roc/`. Use `--run-id another-run` for a repeat or `--output-dir path/to/outputs` to change the parent directory. Each run publishes a new directory after all files are ready.

Prepare the real datasets and run the full 8-config economic comparison suite (`profit_eq16`, `seed = 1`) plus the Stage B ROC sweep and Stage C analysis from the repository root (`german-reweighing.toml` and `taiwan-reweighing.toml` remain available for the earlier `balanced_accuracy` runs):

```sh
uv run --locked python -m fair_credit_scoring.prepare_credit_data

uv run --locked python -m fair_credit_scoring configs/german-baseline-profit.toml
uv run --locked python -m fair_credit_scoring configs/german-reweighing-profit.toml
uv run --locked python -m fair_credit_scoring configs/german-pr-sweep.toml
uv run --locked python -m fair_credit_scoring configs/german-roc-profit.toml

uv run --locked python -m fair_credit_scoring configs/taiwan-baseline-profit.toml
uv run --locked python -m fair_credit_scoring configs/taiwan-reweighing-profit.toml
uv run --locked python -m fair_credit_scoring configs/taiwan-pr-sweep.toml
uv run --locked python -m fair_credit_scoring configs/taiwan-roc-profit.toml

# Verify cross-run alignment across all 8 runs and build outputs/ manifest & control table
uv run --locked python -m fair_credit_scoring.handoff \
  outputs/german-baseline-profit \
  outputs/german-reweighing-profit \
  outputs/german-pr-sweep \
  outputs/german-roc-profit \
  outputs/taiwan-baseline-profit \
  outputs/taiwan-reweighing-profit \
  outputs/taiwan-pr-sweep \
  outputs/taiwan-roc-profit \
  --output outputs/fedor-handoff-manifest.csv \
  --control-csv outputs/fedor-control-table.csv

# Run the Stage B ROC fairness-profit sweep and Stage C cross-method analysis
uv run --locked python -m fair_credit_scoring.roc_experiment
uv run --locked python -m fair_credit_scoring.analysis
```

The `handoff` command must be run from the repository root and requires `data/prepared/{german,taiwan}.csv` (run `prepare_credit_data` first on a fresh checkout). Saved runs write to `outputs/<run_id>/`, the 8-run inventory and control tables go to `outputs/fedor-handoff-manifest.csv` and `outputs/fedor-control-table.csv` (the earlier 6-run reference control table comparing constant `approve-all` / `reject-all` policies with the 26 pre/in-processing/baseline models remains committed at [docs/timur-control-table.csv](docs/timur-control-table.csv)), the ROC sweep writes `outputs/roc-sweep-summary.csv` (and optional `outputs/roc-sweep-predictions/` with `--predictions-dir`), and `fair_credit_scoring.analysis` writes summary CSVs, PNG figures, and `analysis_report.json` under `outputs/analysis/`.

## Handoff notes for post-processing (Reject Option Classification)

- **Group direction**: `a = 0` means age $\le 25$, `a = 1` means age $> 25$ (reversed relative to the paper text notation).
- **No test leakage**: `validation_predictions.csv` and `validation_metrics` were used to select hyperparameters and discrete `profit_eq16` thresholds. Built-in `reject_option` (`RejectOptionClassifier`) calibrates `(threshold, margin)` strictly on pooled OOF validation scores (`sp_paper <= 0.01` on German in `configs/german-roc-profit.toml`, `sp_paper <= 0.005` on Taiwan in `configs/taiwan-roc-profit.toml`) and evaluates once with frozen parameters on `test`.
- **Shared metrics**: evaluate post-processed decisions directly via `fair_credit_scoring.evaluate_metrics(y, a, score, decision)` or configure `postprocessor = "reject_option"` and `[postprocessor_params]` (or custom `Hooks(postprocessors=...)`) with `selection_metric = "profit_eq16"`. Undefined rates return `math.nan` (`null` in JSON).
- **Covariates**: baseline, Reweighing, and Reject Option configs include binary `AGE` (`= a`) in `dataset.numeric`; Prejudice Remover configs exclude `AGE` and route coefficients directly via `a`.
- **Findings**: see [the final analysis](docs/final-analysis.md).

## Prepared data and configuration

| Input | Meaning |
| --- | --- |
| `row_id` | Stable, unique, nonblank string ID. Leading zeros are preserved. |
| `y` | Binary outcome. `1` means repaid and `0` means default. |
| `a` | Binary age group. German and Taiwan use `0` for age ≤ 25 and `1` for age > 25. |
| Numeric features | Columns declared in `dataset.numeric`. Blank cells are missing. |
| Categorical features | Columns declared in `dataset.categorical`. Blank cells are missing. |

The built-in loader reads prepared CSVs. TOML paths are relative to the configuration file. Keep the dataset name and seed identical across methods on the same prepared dataset. For German and Taiwan, baseline and Reweighing include binary `AGE` (= `a`) in `dataset.numeric`, whereas `prejudice_remover` excludes `AGE` from `dataset.numeric` and uses `a` directly to route group-specific coefficients.

The runner sorts rows by ID, jointly stratifies on `(a, y)`, and reserves `ceil(0.4 × n)` rows for testing. It requires at least one test row and five development rows per stratum. Every development row is held out once across five shared folds. Median imputation, scaling, most-frequent categorical imputation, and one-hot encoding are learned on each training partition. Unseen categories produce an all-zero encoding for that feature.

For each model, the runner evaluates the candidate grid and thresholds on pooled out-of-fold predictions. It maximizes `selection_metric` and breaks ties in configuration order. `configs/tiny.toml` and `configs/german-reweighing.toml` / `configs/taiwan-reweighing.toml` use `balanced_accuracy`; the economic suite (`*-profit.toml` and `*-pr-sweep.toml`) uses `profit_eq16`. Once settings and any post-processor are selected, the runner refits on all development rows and evaluates on the held-out test set. See [the experiment protocol](docs/protocol.md).

## Python interface

```python
from dataclasses import replace
from pathlib import Path

from fair_credit_scoring import Hooks, load_config, run


def accuracy(y, a, score, decision):
    return (y == decision).mean()


config = load_config(Path("configs/tiny.toml"))
config = replace(
    config,
    run_id="tiny-accuracy",
    selection_metric="accuracy",
)
output = run(config, Hooks(metrics={"accuracy": accuracy}))
print(output)
```

Register custom components by name using `Hooks`, then select them in `RunConfig` or TOML. Hooks receive pipeline-owned data and should treat it as read-only.

| Hook registry | Contract |
| --- | --- |
| `loaders` | `loader(dataset_config) -> pandas.DataFrame` with the prepared columns above. Select via `dataset.loader`. |
| `weights` | `weight(y, a) -> weights` in training-row order. Called once per training fold and once on all development rows. Select via `weight`. Built-in `reweighing` supports direct weighting or training-row resampling via `weight_mode`. |
| `estimators` | `factory(params, seed, threads) -> estimator`. Implement `fit(X, y, *, a, sample_weight)`, `predict_proba(X, *, a)` and `classes_`. Matrices can be dense or sparse. Built-in options: `logistic_regression`, `random_forest`, and `prejudice_remover`. See `SklearnEstimator` for adapting scikit-learn classifiers. Select via `models.estimator`. |
| `metrics` | `metric(y, a, score, decision) -> float`. Built-ins: `auc`, `balanced_accuracy`, `approval_rate`, `profit_eq16`, `ind`, `sp_paper`, `sf`, `eo_gap`, and group rates (`approval_rate_{0,1}`, `tpr_{0,1}`, `fpr_{0,1}`, `ppv_{0,1}`). Use `selection_metric` to maximize it during tuning or `metrics` to add it to the report. |
| `postprocessors` | A factory `factory(**postprocessor_params)` whose instance implements `fit(score, y, a, threshold)` and `predict(score, a) -> decisions`. An instance fits on the selected out-of-fold predictions for one model. Built-in option: `reject_option` (`RejectOptionClassifier`). Select via `postprocessor` and configure via `[postprocessor_params]` (`fairness_metric`, `fairness_bound`, `utility_metric`, `tune_threshold`, `thresholds`, `n_thresholds`, `n_margins`, `max_margin`). |

The CLI supports the built-in components. Pass a `Hooks` registry to the library entry point for project-specific methods. Custom components should use the supplied seed and thread budget and return finite scores and binary decisions.

## Outputs

| File | Contents |
| --- | --- |
| `splits.csv` | One stable ID per row with `partition`, `validation_fold` (0–4 on development) and `split_id`. |
| `validation_predictions.csv` | Out-of-fold predictions from the selected model settings. |
| `test_predictions.csv` | Held-out predictions from models refit on the development set. |
| `run.json` | Resolved configuration, seed, split ID, dependency versions, selected parameters, thresholds, any `postprocessor` metadata, and validation and test metrics. |

Prediction columns are `row_id`, `y`, `a`, `score` (repayment probability), `decision` (1 = approved), `run_id`, `dataset`, `model`, `split_id`, `partition`, and `validation_fold`. When a `postprocessor` is configured, prediction CSVs also include `decision_before` (from `score >= threshold`) and `decision_after` (`== decision`). The fold is blank for test rows. Validation metrics describe the predictions used to select or calibrate settings. The test metrics measure held-out performance.

## Benchmarks

Measured on macOS 27.2, Apple M2 Pro (10 cores), 16 GiB RAM, Python 3.13.18, NumPy 2.5.3, pandas 3.0.6, and scikit-learn 1.9.1. Peak RSS is the median of the fresh-process high-water readings.

| Workload | Runs per version | Before | After | Time reduction | Before peak RSS | After peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny (80 rows) | 5 | 4.62 s | 3.46 s | 25.1% | 199.4 MiB | 201.5 MiB |
| Large (24,000 rows) | 3 | 9.76 s | 5.24 s | 46.3% | 218.4 MiB | 218.9 MiB |

Sharing train-fitted preprocessing and writing model outputs incrementally brought median runtimes to 3.70 s and 6.93 s. On the large profile, preprocessing calls dropped from 42 to 6. Reusing float32 forest inputs and extending seeded forests across compatible tree counts reduced tree builds from 864 to 384 on that workload. Shared folds retain one out-of-fold score per row and candidate until selection. The measured speedup came with small increases in peak RSS. The maximum observed peaks were 202.2 → 208.4 MiB on tiny and 225.8 → 226.6 MiB on large.

The split assignments and approval decisions matched exactly at each optimization stage. The maximum absolute score difference was zero on both workloads. [Raw samples and workload hashes](docs/benchmark-results.json).

Both synthetic inputs are made by `benchmarks/generate_data.py` with generator seed 1461. The tiny input has 80 rows and 4 possible region levels. The large input has 24,000 rows and 64 region levels. Both contain numeric and categorical features with missing values and all four age/outcome strata. The benchmark configurations use split and model seed 37, one thread, four grid candidates per model and three thresholds. The runs include CSV loading, splitting, fitting, tuning, and writing all outputs. The harness times each fresh Python child process and records its high-water RSS. Data generation, environment setup, and profiling are outside the timings.

Run these commands in a clean checkout of this version. The patch restores the measured reference runner for the first two commands and is reversed before measuring the optimized runner.

```sh
uv sync --locked
uv run --locked python benchmarks/generate_data.py tiny tests/fixtures/tiny.csv
uv run --locked python benchmarks/generate_data.py large data/benchmark-large.csv

git apply benchmarks/reference.patch
uv run --locked python benchmarks/measure.py configs/tiny.toml outputs/benchmarks/before-tiny --label before-tiny --repeats 5
uv run --locked python benchmarks/measure.py configs/large.toml outputs/benchmarks/before-large --label before-large --repeats 3
git apply -R benchmarks/reference.patch

uv run --locked python benchmarks/measure.py configs/tiny.toml outputs/benchmarks/after-tiny --label after-tiny --repeats 5
uv run --locked python benchmarks/measure.py configs/large.toml outputs/benchmarks/after-large --label after-large --repeats 3
uv run --locked python benchmarks/compare.py outputs/benchmarks/before-tiny/predictions outputs/benchmarks/after-tiny/predictions
uv run --locked python benchmarks/compare.py outputs/benchmarks/before-large/predictions outputs/benchmarks/after-large/predictions
```

Each measurement writes `report.json` and retains the last run under `predictions/`. Use new output directories to repeat a measurement. The comparison checks exact split files, metadata and decisions, plus scores and metrics within absolute tolerance `1e-12`.

To profile either runner on the large workload, activate the desired version and run the commands below. Use a new run ID and profile path for each version.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  uv run --locked python -m cProfile -o outputs/benchmarks/large.prof -m fair_credit_scoring configs/large.toml --run-id profile-large
uv run --locked python -m pstats outputs/benchmarks/large.prof
```

## Checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Keep raw datasets and generated runs in the ignored `data/` and `outputs/` directories. Update `uv.lock` when dependencies change.
