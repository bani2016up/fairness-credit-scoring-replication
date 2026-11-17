"""Generate deterministic prepared CSV inputs without external data."""

import argparse
import csv
import random
from pathlib import Path

FIELDS = ("row_id", "y", "a", "income", "balance", "tenure", "channel", "region")
SIZES = {"tiny": (80, 4), "large": (24_000, 64)}


def generate(path: Path, size: str) -> None:
    rows, regions = SIZES[size]
    rng = random.Random(1461)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        for i in range(rows):
            group, outcome = divmod(i % 4, 2)
            latent = 1.2 * outcome - 0.3 * group
            writer.writerow(
                (
                    f"credit-{i:06}",
                    outcome,
                    group,
                    "" if i % 13 == 0 else int(rng.gauss(4100 + 700 * latent, 1300)),
                    "" if i % 19 == 0 else int(rng.gauss(2900 - 500 * latent, 870)),
                    rng.randrange(2 + outcome, 48),
                    "" if i % 11 == 0 else f"channel-{rng.randrange(3 + outcome)}",
                    "" if i % 17 == 0 else f"region-{rng.randrange(regions):02}",
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=SIZES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output, args.size)


if __name__ == "__main__":
    main()
