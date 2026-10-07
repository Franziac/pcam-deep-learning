"""Run the complete leakage-safe Stage 2 experiment.

Order is fixed: train CNN -> train MLP -> compare on validation -> evaluate only
selected method on test.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from compare_models import compare_runs
from evaluate_final import evaluate_selected
from pcam_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
)
from train import train_model
from visualizations import generate_report_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_id = datetime.now(timezone.utc).strftime("stage2-%Y%m%d-%H%M%S")
    experiment_dir = args.output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)

    common = {
        "output_root": experiment_dir / "runs",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
    }
    cnn_run = train_model("cnn", **common)
    mlp_run = train_model("mlp", **common)

    comparison_dir = experiment_dir / "comparison"
    selection_path = compare_runs(cnn_run, mlp_run, comparison_dir)

    # The official test split is first loaded here, after validation-only selection.
    evaluate_selected(
        selection_path,
        output_dir=experiment_dir / "final_test",
        batch_size=args.batch_size,
        seed=args.seed,
    )

    figures = generate_report_figures(experiment_dir, seed=args.seed)
    print("Generated report figures:")
    for name, path in figures.items():
        print(f"  {name}: {Path(path).resolve()}")

    print(f"Completed Stage 2 experiment: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()
