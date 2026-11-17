"""Measure pipeline wall time and process peak RSS in fresh processes."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

WORKER = """
import resource
import sys
from fair_credit_scoring.__main__ import main
main()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(rss if sys.platform == "darwin" else rss * 1024)
"""
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def measure(config: Path, output: Path, repeats: int, label: str) -> None:
    settings = tomllib.loads(config.read_text(encoding="utf-8"))
    data = config.parent / settings["dataset"]["path"]
    threads = settings.get("threads", 1)
    env = os.environ | dict.fromkeys(THREAD_VARIABLES, str(threads))
    with data.open("rb") as stream:
        data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    output.mkdir(parents=True)
    samples = []
    for repetition in range(repeats):
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            command = [
                sys.executable,
                "-c",
                WORKER,
                str(config),
                "--output-dir",
                temporary,
                "--run-id",
                "measurement",
            ]
            start = time.perf_counter()
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, check=False
            )
            elapsed = time.perf_counter() - start
            if result.returncode:
                raise RuntimeError(result.stderr)
            sample = {
                "seconds": elapsed,
                "peak_rss_bytes": int(result.stdout.splitlines()[-1]),
            }
            samples.append(sample)
            print(f"{label} {repetition + 1}/{repeats} {sample}", flush=True)
            if repetition == repeats - 1:
                shutil.move(Path(temporary) / "measurement", output / "predictions")

    report = {
        "label": label,
        "dataset": settings["dataset"]["name"],
        "dataset_sha256": data_hash,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "seed": settings.get("seed", 42),
        "threads": threads,
        "machine": {
            "os": platform.system(),
            "release": platform.mac_ver()[0] or platform.release(),
            "arch": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn")
            },
        },
        "samples": samples,
        "median_seconds": statistics.median(sample["seconds"] for sample in samples),
        "median_peak_rss_bytes": statistics.median(
            sample["peak_rss_bytes"] for sample in samples
        ),
        "max_peak_rss_bytes": max(sample["peak_rss_bytes"] for sample in samples),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", default="pipeline")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if sys.platform not in {"darwin", "linux"}:
        parser.error("peak RSS measurement is supported on macOS and Linux")
    measure(args.config, args.output, args.repeats, args.label)


if __name__ == "__main__":
    main()
