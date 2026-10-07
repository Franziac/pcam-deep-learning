"""Evaluate only the validation-selected final model on the official test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from metrics import binary_metrics
from pcam_config import DEFAULT_BATCH_SIZE, DEFAULT_SEED, DEFAULT_THRESHOLD
from training_utils import read_json, write_json


def evaluate_selected(
    selection_path: Path,
    *,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = DEFAULT_SEED,
) -> Path:
    selection = read_json(selection_path)
    if selection.get("test_data_used_for_selection") is not False:
        raise ValueError("Selection metadata does not guarantee test isolation")

    selected_model = str(selection["selected_model"])
    checkpoint = Path(selection["selected_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint}")

    # TensorFlow and test data are imported only after model selection has been read.
    from pcam_data import load_test, to_tf_dataset
    from pcam_models import build_model

    test_data = load_test()
    test_set = to_tf_dataset(
        test_data, batch_size=batch_size, training=False, seed=seed
    )

    model = build_model(selected_model)
    model.load_weights(checkpoint)

    all_labels = []
    all_probabilities = []
    for images, labels in test_set:
        probabilities = np.asarray(model.predict_on_batch(images)).reshape(-1)
        all_probabilities.append(probabilities)
        all_labels.append(np.asarray(labels).reshape(-1))

    labels_array = np.concatenate(all_labels)
    probabilities_array = np.concatenate(all_probabilities)
    metrics = binary_metrics(
        labels_array,
        probabilities_array,
        threshold=threshold,
    )
    result = {
        "model": selected_model,
        "selection_file": str(Path(selection_path)),
        "checkpoint": str(checkpoint),
        "test_split": "test",
        "test_used_only_after_validation_selection": True,
        **metrics,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "test_predictions.npz"
    np.savez_compressed(
        predictions_path,
        labels=labels_array.astype(np.int8, copy=False),
        probabilities=probabilities_array.astype(np.float32, copy=False),
    )
    result["predictions_file"] = str(predictions_path)

    json_path = output_dir / "final_test_metrics.json"
    write_json(json_path, result)

    csv_path = output_dir / "final_test_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(f"Final model: {selected_model}")
    print(f"Test BCE: {metrics['binary_cross_entropy']:.6f}")
    print(f"Test 0/1 error: {metrics['zero_one_error']:.6f}")
    print(f"Test accuracy: {metrics['accuracy']:.6f}")
    print(f"Metrics file: {json_path.resolve()}")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_selected(
        args.selection,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
