"""Generate report-ready figures from a completed Stage 2 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from pcam_config import DEFAULT_SEED
from visualizations import generate_report_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    outputs = generate_report_figures(args.experiment_dir, seed=args.seed)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
