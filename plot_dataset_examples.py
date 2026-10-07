"""Create a PCam training-example figure without loading the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from pcam_config import DEFAULT_SEED
from pcam_data import load_train
from visualizations import plot_dataset_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset_examples.png"))
    parser.add_argument("--examples-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    output = plot_dataset_examples(
        load_train(),
        args.output,
        examples_per_class=args.examples_per_class,
        seed=args.seed,
    )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
