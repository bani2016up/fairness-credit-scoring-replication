# Proposed experiment protocol

## Comparison

Use German Credit and Taiwan Credit. For each dataset, compare Logistic
Regression and Random Forest as baselines, each with Reweighing and Reject
Option Classification, plus a standalone Prejudice Remover model.

Report ROC AUC on continuous repayment scores. Report profit, independence,
separation, and sufficiency on credit decisions. Compare the paper's separation
metric with the largest absolute group gap in FPR or TPR.

## Shared data and evaluation

- `row_id` identifies each record across outputs.
- `y = 1` means repaid and `y = 0` means default.
- `a` is the age group. Record the cutoff, its boundary, and how age is used in
  the feature matrix when implementing the dataset loaders.
- `score` is the predicted probability of repayment.
- `decision = 1` means approved and `decision = 0` means declined.

Save a seeded 60/40 development/test split per dataset and reuse five folds
within development. Fit preprocessing on each training fold. Use development
results to choose model settings, thresholds, and fairness parameters. Evaluate
on the held-out test split after those settings are fixed.

Persist the dataset, method, model, seed, and split ID with validation and
test predictions so comparisons can be regenerated.
