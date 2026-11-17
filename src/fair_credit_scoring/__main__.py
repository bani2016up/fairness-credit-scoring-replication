"""Command-line entry point."""

import argparse
from dataclasses import replace
from pathlib import Path

from . import load_config, run


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and evaluate credit scoring models."
    )
    parser.add_argument("config", type=Path, help="experiment TOML file")
    parser.add_argument("--run-id", help="override the configured run ID")
    parser.add_argument(
        "--output-dir", type=Path, help="override the configured output directory"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.run_id is not None:
            config = replace(config, run_id=args.run_id)
        if args.output_dir is not None:
            config = replace(config, output_dir=args.output_dir)
        result = run(config)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
