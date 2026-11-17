# Fairness in Credit Scoring Replication

A course project replicating selected experiments from Kozodoi, Jacob, and
Lessmann's [Fairness in Credit Scoring](https://arxiv.org/abs/2103.01907).
We plan to compare Logistic Regression and Random Forest on German Credit
and Taiwan Credit, with Reweighing, Prejudice Remover, and Reject Option
Classification. The analysis will compare AUC, estimated profit, and age-group
fairness across the methods.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv python install 3.13
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

The initial repository contains the project scaffold and installation smoke
test. Experiment code and dataset preparation will land in follow-up PRs.

## Layout

- `src/fair_credit_scoring/` contains the experiment code.
- `tests/` contains repeatable checks.
- `docs/protocol.md` records the proposed experiment design.
- `.github/` contains CI and the PR template.

## Pull requests

Branch from `main` and open a PR for one component at a time. Include the
commands you ran and any experiment settings or output affected by the change.
Keep raw datasets and generated results in the ignored `data/` and `outputs/`
directories. Commit `uv.lock` whenever dependencies change.
