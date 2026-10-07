"""Train one Stage 2 method using the official PCam train/validation splits."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

from pcam_config import (
    DATASET_NAME,
    DATASET_REVISION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    MODEL_NAMES,
)
from training_utils import (
    configure_reproducibility,
    create_run_directory,
    select_best_epoch,
    write_history_csv,
    write_json,
)


def train_model(
    model_name: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    configure_reproducibility(seed)

    import tensorflow as tf

    from pcam_data import load_train_validation, to_tf_dataset
    from pcam_models import build_model, model_description

    train_data, validation_data = load_train_validation()
    train_set = to_tf_dataset(
        train_data, batch_size=batch_size, training=True, seed=seed
    )
    validation_set = to_tf_dataset(
        validation_data, batch_size=batch_size, training=False, seed=seed
    )

    model = build_model(model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
    )

    run_dir = create_run_directory(output_root, model_name)
    checkpoint_path = run_dir / "best.weights.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    started = time.monotonic()
    history = model.fit(
        train_set,
        validation_data=validation_set,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.monotonic() - started

    write_history_csv(run_dir / "history.csv", history.history)
    best = select_best_epoch(history.history)

    metadata: dict[str, Any] = {
        "status": "complete",
        "model": model_name,
        "model_description": model_description(model_name),
        "parameter_count": int(model.count_params()),
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "training_split": "train",
        "validation_split": "valid",
        "test_split_loaded": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "training_loss": "binary_cross_entropy",
        "checkpoint_selection": "lowest validation BCE; exact ties keep the earliest epoch",
        "best_checkpoint": checkpoint_path.name,
        "elapsed_seconds": elapsed,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        **best,
    }
    write_json(run_dir / "validation_summary.json", metadata)

    print(f"Run directory: {run_dir.resolve()}")
    print(f"Best epoch: {best['best_epoch']}")
    print(f"Validation BCE: {best['validation_bce']:.6f}")
    print(f"Validation 0/1 error: {best['validation_zero_one_error']:.6f}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        args.model,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
