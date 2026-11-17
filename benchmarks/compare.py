"""Verify equivalent split assignments, predictions and selection metadata."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ATOL = 1e-12


def compare(reference: Path, optimized: Path) -> None:
    assert (optimized / "splits.csv").read_bytes() == (
        reference / "splits.csv"
    ).read_bytes()
    for name in ("validation_predictions.csv", "test_predictions.csv"):
        options = {
            "keep_default_na": False,
            "dtype": {"row_id": str, "run_id": str, "model": str, "dataset": str},
            "float_precision": "round_trip",
        }
        before = pd.read_csv(reference / name, **options)
        after = pd.read_csv(optimized / name, **options)
        pd.testing.assert_frame_equal(
            before.drop(columns="score"), after.drop(columns="score"), check_exact=True
        )
        np.testing.assert_allclose(before.score, after.score, rtol=0, atol=ATOL)
        print(
            f"{name}: {len(before):,} rows, max score difference {abs(before.score - after.score).max():.3g}"
        )

    before, after = (
        json.loads((path / "run.json").read_text(encoding="utf-8"))
        for path in (reference, optimized)
    )
    assert before["split_id"] == after["split_id"]
    for key in ("output_dir", "run_id"):
        before["config"].pop(key)
        after["config"].pop(key)
    assert before["config"] == after["config"]
    assert before["versions"] == after["versions"]
    assert before["selected"].keys() == after["selected"].keys()
    for model, settings in before["selected"].items():
        actual = after["selected"][model]
        assert settings["params"] == actual["params"]
        assert settings["threshold"] == actual["threshold"]
        np.testing.assert_allclose(
            settings["selection_value"], actual["selection_value"], rtol=0, atol=ATOL
        )
        for group in ("validation_metrics", "test_metrics"):
            assert settings[group].keys() == actual[group].keys()
            for name, value in settings[group].items():
                if value is None:
                    assert actual[group][name] is None
                else:
                    np.testing.assert_allclose(
                        value, actual[group][name], rtol=0, atol=ATOL
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("optimized", type=Path)
    args = parser.parse_args()
    compare(args.reference, args.optimized)


if __name__ == "__main__":
    main()
