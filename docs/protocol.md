# Experiment protocol

## Comparison

The main experiments will use prepared German Credit and Taiwan Credit data.
For each dataset, compare Logistic Regression and Random Forest as baselines,
each with Reweighing and Reject Option Classification, plus a standalone
Prejudice Remover model.

Report ROC AUC on continuous repayment scores. Report profit, independence,
separation, and sufficiency on credit decisions. Compare the paper's separation
metric with the largest absolute group gap in FPR or TPR.

## Shared evaluation

Use the same prepared row IDs, binary repayment labels, binary age group,
dataset name, and seed for every method on a dataset. The dataset preparation
records the age cutoff, its boundary, and whether age is used as a feature.

Reserve 40% for testing with integer rounding up and use five shared
development folds, stratified jointly on age group and repayment outcome.
Fit preprocessing and any training weights on each fold's training rows.
Select model settings and approval thresholds on pooled out-of-fold
predictions, then calibrate a post-processor on the selected predictions
when configured. Refit using all development rows and evaluate on the
held-out test set with the selected settings fixed.

The bundled baseline uses balanced accuracy as a runnable selection
criterion. The financial experiments can select the project profit
metric through the same interface once its assumptions are specified.
The saved per-row scores and decisions support the downstream fairness
analysis and comparison with [Fairness in Credit Scoring: Assessment, Implementation and Profit Implications](https://arxiv.org/abs/2103.01907).
