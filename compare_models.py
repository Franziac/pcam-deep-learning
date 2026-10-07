"""Compare CNN and MLP using validation data only and choose the final method."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from training_utils import read_json, write_json


REQUIRED_PROTOCOL_KEYS = (
    "dataset_revision",
    "training_split",
    "validation_split",
    "epochs",
    "batch_size",
    "seed",
    "learning_rate",
    "training_loss",
)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary = read_json(Path(run_dir) / "validation_summary.json")
    if summary.get("test_split_loaded") is not False:
        raise ValueError(f"Training summary does not prove test isolation: {run_dir}")
    return summary


def compare_runs(cnn_run: Path, mlp_run: Path, output_dir: Path) -> Path:
    cnn = _load_summary(cnn_run)
    mlp = _load_summary(mlp_run)

    if cnn.get("model") != "cnn" or mlp.get("model") != "mlp":
        raise ValueError("Expected one CNN run and one MLP run")

    mismatches = [
        key for key in REQUIRED_PROTOCOL_KEYS if cnn.get(key) != mlp.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Runs do not use the same comparison protocol: " + ", ".join(mismatches)
        )

    rows = [cnn, mlp]
    # Primary selection metric is validation binary cross-entropy.
    selected = min(
        rows,
        key=lambda row: (
            float(row["validation_bce"]),
            float(row["validation_zero_one_error"]),
            str(row["model"]),
        ),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "validation_comparison.csv"
    fields = [
        "model",
        "parameter_count",
        "best_epoch",
        "training_bce",
        "validation_bce",
        "training_accuracy",
        "validation_accuracy",
        "training_zero_one_error",
        "validation_zero_one_error",
        "training_precision",
        "validation_precision",
        "training_recall",
        "validation_recall",
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    run_by_model = {"cnn": Path(cnn_run), "mlp": Path(mlp_run)}
    selected_run = run_by_model[str(selected["model"])]
    selection = {
        "selection_basis": "validation data only",
        "selection_metric": "binary_cross_entropy",
        "selection_rule": "lowest validation BCE; tie-breaker: lowest validation 0/1 error",
        "selected_model": selected["model"],
        "selected_run": str(selected_run),
        "selected_checkpoint": str(selected_run / "best.weights.h5"),
        "runs": {"cnn": str(Path(cnn_run)), "mlp": str(Path(mlp_run))},
        "selected_best_epoch": selected["best_epoch"],
        "selected_validation_bce": selected["validation_bce"],
        "selected_validation_zero_one_error": selected[
            "validation_zero_one_error"
        ],
        "test_data_used_for_selection": False,
        "comparison_csv": str(comparison_path),
    }
    selection_path = output_dir / "selection.json"
    write_json(selection_path, selection)

    print(f"Selected model: {selection['selected_model']}")
    print(f"Selection file: {selection_path.resolve()}")
    return selection_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-run", type=Path, required=True)
    parser.add_argument("--mlp-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_runs(args.cnn_run, args.mlp_run, args.output_dir)


if __name__ == "__main__":
    main()
