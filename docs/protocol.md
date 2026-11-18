# Experiment protocol

## Scope and hypotheses

This project is a **partial replication** of [Fairness in Credit Scoring: Assessment, Implementation and Profit Implications](https://arxiv.org/abs/2103.01907) (Kozodoi, Jacob, Lessmann; pinned author repository revision `5ebe84bdb9238b2c9890f3a5a50e3d7deb7d0a22`).

The goal is to evaluate fairness-profit trade-offs under a consistent, leak-free Python pipeline rather than reproduce every implementation detail of the original R/AIF360 stack. On prepared German Credit (`1,000` rows) and Taiwan Credit (`23,531` rows with `BILL_AMT1 - PAY_AMT1 > 0`), we compare Logistic Regression and Random Forest baselines against Reweighing (train-only resampling), Prejudice Remover (in-processing sweep over `eta`), and downstream Reject Option Classification (post-processing). Experiments are not constrained to confirm the authors' empirical rankings or numbers.

## Encodings and conventions

- `row_id`: Stable, unique string identifier preserved across preparation and predictions.
- `y`: Binary repayment outcome (`1` = repaid, `0` = default).
- `a`: Binary age group (`0` = age $\le 25$, `1` = age $> 25$). The paper text uses reversed notation, whereas the authors' data loader (`> 25`) defines this boundary; absolute fairness gaps are symmetric, and group-specific rates or post-processors follow `a=0` ($\le 25$) vs `a=1` ($> 25$).
- `score`: Continuous model output $P(y=1 \mid x, a)$.
- `decision`: Binary credit approval decision (`1` = approved, `0` = rejected), equal to `score >= threshold` before post-processing.

## Shared evaluation and leak prevention

1. Compare methods using the same prepared CSVs, `dataset.name`, and `seed = 1`. Baseline and Reweighing include binary `AGE` (`= a`) in `dataset.numeric`; Prejudice Remover excludes duplicate `AGE` because `a` routes group-specific coefficients directly.
2. Reserve `ceil(0.4 × n)` rows for testing (`400` on German, `9,413` on Taiwan) and split development (`600` on German, `14,118` on Taiwan) into five shared folds stratified jointly on `(a, y)`.
3. Fit imputation, `MaxAbs` scaling, one-hot encoding, and training weights strictly inside each training partition.
4. Select model hyperparameters and the discrete threshold (`0.00, 0.01, ..., 1.00`) strictly on **pooled out-of-fold (OOF) development predictions** maximizing `selection_metric = "profit_eq16"`.
5. Calibrate any post-processor only on the selected pooled OOF predictions.
6. Refit the selected configuration on all development rows and evaluate once on the held-out test set with settings frozen. Test outcomes never inform hyperparameters, `eta`, or thresholds.

## Economic objective (`profit_eq16`) and author-code differences

Economic selection and evaluation use Eq. (16) with unit notional and fixed decisions:

$$\text{profit\_eq16} = \frac{C \cdot (\text{TP} - \text{FN}) - B_{\text{expected}} \cdot \text{FP}}{N}$$

where $C = 0.2664$, $p_0 = 0.55$, $p_1 = 0.10$, $B_{\text{expected}} = p_1 + \frac{1 - p_0 - p_1}{2} = 0.275$, and $N = \text{TP} + \text{FP} + \text{FN} + \text{TN}$ is the count of all evaluated applications. Unit contributions are $+0.2664$ (`TP`), $-0.2664$ (`FN` opportunity cost), $-0.275$ (`FP`), and $0$ (`TN`); `reject-all` gives negative profit when `y=1` rows exist.

Differences from the pinned author R scripts (`functions/99_compute_profit.R` and `functions/94_evaluate.R`):

1. `profit_eq16` uses `C = 0.2664` on a unit loan, whereas `99_compute_profit.R` defaults to `r = 0.2644` and weights outcomes by individual loan amounts (`profitPerEUR`). `CREDIT_AMNT` stays in `data/prepared/*.csv` for optional sensitivity checks; scaled `X` features are never used as currency amounts.
2. Thresholds are tuned discretely on OOF `profit_eq16`, not via Expected Maximum Profit (`empCreditScoring`).
3. `94_evaluate.R` re-estimates an EMP acceptance cutoff directly on the evaluated test set before computing profit, whereas this pipeline freezes the OOF threshold before test evaluation.

## Classification and fairness metrics

- `auc`: ROC AUC computed strictly on continuous `score` (never `decision`).
- `approval_rate`: Mean of `decision`.
- `ind` (Independence): $|\text{approval\_rate}_1 - \text{approval\_rate}_0|$. Note that `ind` is non-negative and cannot change sign; when younger approval exceeds older approval, the **signed** difference `approval_rate_1 - approval_rate_0` changes sign.
- `sp_paper` (Paper Separation): $\frac{1}{2}|(\text{FPR}_1 - \text{FPR}_0) + (\text{TPR}_1 - \text{TPR}_0)|$.
- `eo_gap` (Equalized Odds Max Gap): $\max(|\text{FPR}_1 - \text{FPR}_0|, |\text{TPR}_1 - \text{TPR}_0|)$.
- `sf` (Sufficiency): $|\text{PPV}_1 - \text{PPV}_0|$.

Any rate whose denominator is zero (`approval_rate_g`, `tpr_g`, `fpr_g`, `ppv_g`), single-class `auc`/`balanced_accuracy`, and any fairness gap depending on an undefined rate return `NaN` (serialized as `null` in `run.json`).

**Cancellation in `sp_paper`**: Because `sp_paper` sums signed FPR and TPR gaps before taking `abs`, opposing errors cancel: e.g. `FPR_0=0.1, TPR_0=0.8` vs `FPR_1=0.3, TPR_1=0.6` yields `sp_paper = 0` while `eo_gap = 0.2`. On `Taiwan / reweighing / lr / test`, `sp_paper = 4.765998456185416e-05` (which rounds to `0.0000` at four decimals, not exact zero) alongside `eo_gap = 0.009885206975302041`. Both `sp_paper` and `eo_gap` are therefore reported together.

## Prejudice Remover formulation

[`PrejudiceRemover`](../src/fair_credit_scoring/prejudice_remover.py) replaces the legacy AIF360/Kamishima file-based subprocess wrapper with a direct NumPy/SciPy CPU optimizer (`L-BFGS-B`) using group-specific linear parameters $(\beta_g, b_g)$ for $g \in \{0, 1\}$:

$$p_i = \sigma(X_i \beta_{a_i} + b_{a_i}), \qquad J(\theta) = \sum_{i=1}^N v_i\,\text{BCE}(y_i, p_i) + \eta W I(p; a) + \frac{\text{l2\_strength}}{2}\sum_{g \in \{0,1\}} \left(\|\beta_g\|_2^2 + b_g^2\right)$$

where $v_i \ge 0$ (`1` when `sample_weight` is `None`), $W = \sum_i v_i$, $W_g = \sum_{a_i=g} v_i > 0$, $\pi_g = W_g / W$, $q_g = \sum_{a_i=g} (v_i p_i) / W_g$, $q = \sum_i (v_i p_i) / W$, and

$$I(p; a) = \sum_{g \in \{0,1\}} \pi_g \left[ q_g \log\frac{q_g}{q} + (1 - q_g)\log\frac{1 - q_g}{1 - q} \right].$$

- **L2 and intercept**: Matching Kamishima's `pr.py`, L2 regularization (`l2_strength > 0`) applies to both $\beta_g$ and $b_g$. At $\eta = 0$, the model is validated against independent per-group `sklearn.linear_model.LogisticRegression(C=1/l2_strength, fit_intercept=False)` on $[X_g, 1]$ within numerical tolerance (`1e-5`, not bitwise equality).
- **Scaling by $W$**: For optimization, L-BFGS-B minimizes $J / W$ and $\nabla J / W$ together. Dividing the entire objective by positive $W$ preserves the minimizer for fixed data and fixed weights $v_i$, **but does not make the solution invariant to rescaling `sample_weight`** ($v_i \mapsto c v_i$) under fixed `l2_strength`, because BCE and $\eta W I$ scale with $c$ while the L2 term does not.
- **Sweep retention**: `configs/german-pr-sweep.toml` and `configs/taiwan-pr-sweep.toml` define singleton `ModelConfig` entries (`pr_eta_0` .. `pr_eta_150` for $\eta \in \{0, 1, 5, 15, 30, 50, 70, 100, 150\}$, `l2_strength = 1.0`) so OOF thresholds and predictions are preserved for every $\eta$.

## Limitations and baseline controls

- **Single seed and non-convexity**: Economic runs use a single split seed (`seed = 1`). For $\eta > 0$, $J(\theta)$ is non-convex, so zero-initialized `L-BFGS-B` finds a deterministic stationary point without a global-optimum guarantee.
- **Constant-policy controls**: [`timur-control-table.csv`](timur-control-table.csv) evaluates `approve-all` and `reject-all` alongside all 26 trained models on the saved predictions. On German (`70%` repayment rate), `profit_eq16` leads models to `95.75%–99.50%` test approval (where low fairness gaps occur near full approval and 9 of 13 models lose to `approve-all` on test), whereas on Taiwan all 13 models outperform `approve-all` on test.
- **Handoff scope**: `fair_credit_scoring.handoff` checks alignment (`split_id`, `splits.csv`, `row_id`/`y`/`a`/`partition`, row counts, and SHA-256 of the prepared CSVs currently on disk); because input CSV hashes are not recorded inside `run.json`, it verifies current disk state rather than historical file provenance at training time.
