"""Build the German and Taiwan experiment inputs from the authors' CSVs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

SOURCE_REVISION = "5ebe84bdb9238b2c9890f3a5a50e3d7deb7d0a22"
SOURCE_ROOT = (
    f"https://raw.githubusercontent.com/kozodoi/Fair_Credit_Scoring/{SOURCE_REVISION}"
    "/data/raw"
)
PAY_COLUMNS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")


def _labels(raw_age: pd.Series, y: pd.Series) -> pd.DataFrame:
    age = pd.to_numeric(raw_age)
    if not (np.isfinite(age) & age.gt(0) & age.mod(1).eq(0)).all():
        raise ValueError("Source ages must be positive integers.")
    if y.isna().any():
        raise ValueError("Source target has missing or unexpected values.")
    # The authors' project-specific loader uses >25, so age 25 belongs to a=0.
    a = age.gt(25).astype("int8")
    return pd.DataFrame({"y": y.astype("int8"), "a": a, "AGE": a})


def _counts(labels: pd.DataFrame, age: pd.Series) -> dict:
    return {
        "rows": len(labels),
        "age_25": int(age.loc[labels.index].eq(25).sum()),
        "default_rate": float(labels["y"].eq(0).mean()) if len(labels) else None,
        "groups": {
            str(a): {
                "default": int((labels["a"].eq(a) & labels["y"].eq(0)).sum()),
                "repaid": int((labels["a"].eq(a) & labels["y"].eq(1)).sum()),
            }
            for a in (0, 1)
        },
    }


def _german(raw: pd.DataFrame):
    age = raw["age"]
    labels = _labels(age, raw["BAD"].map({"GOOD": 1, "BAD": 0}))
    features = raw.drop(columns=["Unnamed: 0", "BAD", "age"]).rename(
        columns={"amount": "CREDIT_AMNT"}
    )
    return raw["Unnamed: 0"], age, labels, features, {}


def _taiwan(raw: pd.DataFrame):
    age = raw["AGE"]
    target = "default.payment.next.month"
    labels = _labels(age, raw[target].map({0: 1, 1: 0}))
    features = raw.drop(columns=["ID", target, "AGE"])
    # The positive remaining balance is the loan exposure used by the paper.
    amount = features["BILL_AMT1"] - features["PAY_AMT1"]
    keep = np.isfinite(amount) & amount.gt(0)
    features = features.loc[keep].assign(CREDIT_AMNT=amount.loc[keep])
    features["SEX"] = features["SEX"].map({1: "Male", 2: "Female"}).fillna("NA")
    features["EDUCATION"] = (
        features["EDUCATION"]
        .map({1: "graduate_school", 2: "university", 3: "high_school"})
        .fillna("others")
    )
    features["MARRIAGE"] = (
        features["MARRIAGE"].map({1: "married", 2: "single"}).fillna("others")
    )
    # StandardDataset keeps all declared categorical columns, including PAY_0.
    for name in PAY_COLUMNS:
        features[name] = features[name].replace({-2: 0, -1: 0}).astype("Int64")
    excluded = {
        "nonpositive_credit_amount": int((np.isfinite(amount) & amount.le(0)).sum()),
        "missing_or_nonfinite_credit_amount": int((~np.isfinite(amount)).sum()),
    }
    return raw["ID"], age, labels, features, excluded


def prepare_credit_data(name: str, content: bytes) -> tuple[pd.DataFrame, dict]:
    """Return a stable prepared table and an audit of the input and filter."""
    converters = {"german": ("Unnamed: 0", _german), "taiwan": ("ID", _taiwan)}
    id_column, convert = converters[name]
    raw = pd.read_csv(io.BytesIO(content), dtype={id_column: "string"})
    row_id, age, labels, features, excluded = convert(raw)
    if row_id.isna().any() or row_id.str.strip().eq("").any() or not row_id.is_unique:
        raise ValueError("Source identifiers must be unique and nonblank.")

    # Keep a as metadata and expose the authors' binary AGE separately to the model.
    prepared = labels.join(features, how="inner")
    prepared.insert(0, "row_id", row_id.loc[prepared.index])
    missing = raw.isna().sum()
    audit = {
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "source_columns": raw.columns.tolist(),
        "source_missing_by_column": {
            column: int(count) for column, count in missing.items() if count
        },
        "before": _counts(labels, age),
        "after": _counts(prepared, age),
        "excluded": excluded,
        "excluded_rows": len(raw) - len(prepared),
    }
    return prepared, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="directory with author-format german.csv and taiwan.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    args = parser.parse_args(argv)

    try:
        datasets = {}
        audits = {}
        for name in ("german", "taiwan"):
            filename = f"{name}.csv"
            if args.raw_dir is None:
                with urlopen(f"{SOURCE_ROOT}/{filename}", timeout=60) as response:
                    content = response.read()
            else:
                content = (args.raw_dir / filename).read_bytes()
            datasets[name], audits[name] = prepare_credit_data(name, content)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in datasets.items():
            data.to_csv(args.output_dir / f"{name}.csv", index=False)
        (args.output_dir / "audit.json").write_text(
            json.dumps(audits, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
