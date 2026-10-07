"""Shared training and model-selection utilities."""

from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def configure_reproducibility(seed: int) -> None:
    """Configure deterministic random seeds before TensorFlow model creation."""
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def create_run_directory(output_root: Path, model_name: str) -> Path:
    output_root = Path(output_root) / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = output_root / stamp
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{stamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_best_epoch(history: Mapping[str, Sequence[float]]) -> dict[str, float | int]:
    """Select by lowest validation BCE, matching ModelCheckpoint behavior."""
    val_loss = list(history.get("val_loss", []))
    val_accuracy = list(history.get("val_accuracy", []))
    if not val_loss or len(val_loss) != len(val_accuracy):
        raise ValueError("history must contain equal-length val_loss and val_accuracy")

    candidates = list(range(len(val_loss)))
    # ModelCheckpoint(save_best_only=True, mode="min") keeps the earliest epoch
    # when validation loss is exactly tied, because an equal value is not an
    # improvement. Use the same rule here so metadata and saved weights match.
    best_index = min(candidates, key=lambda i: (float(val_loss[i]), i))

    def value(key: str) -> float:
        values = history.get(key)
        if values is None or best_index >= len(values):
            return float("nan")
        return float(values[best_index])

    train_accuracy = value("accuracy")
    validation_accuracy = value("val_accuracy")

    return {
        "best_epoch": best_index + 1,
        "training_bce": value("loss"),
        "validation_bce": value("val_loss"),
        "training_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "training_zero_one_error": 1.0 - train_accuracy,
        "validation_zero_one_error": 1.0 - validation_accuracy,
        "training_precision": value("precision"),
        "validation_precision": value("val_precision"),
        "training_recall": value("recall"),
        "validation_recall": value("val_recall"),
    }


def write_history_csv(path: Path, history: Mapping[str, Sequence[float]]) -> None:
    keys = list(history.keys())
    if not keys:
        raise ValueError("history is empty")
    epoch_count = len(history[keys[0]])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for i in range(epoch_count):
            writer.writerow([i + 1, *[history[key][i] for key in keys]])
