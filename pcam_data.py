"""PCam data loading helpers.

Training code deliberately loads only the official train and validation splits.
The test split is loaded only by final evaluation code after model selection.
"""

from __future__ import annotations

from typing import Any

from pcam_config import DATASET_NAME, DATASET_REVISION


def _load_split(split: str, load_dataset_fn: Any = None):
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset
    return load_dataset_fn(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split=split,
    )


def load_train(load_dataset_fn: Any = None):
    """Load only the official training split."""
    return _load_split("train", load_dataset_fn)


def load_train_validation(load_dataset_fn: Any = None):
    """Load only the official training and validation splits."""
    train_data = _load_split("train", load_dataset_fn)
    validation_data = _load_split("valid", load_dataset_fn)
    return train_data, validation_data


def load_test(load_dataset_fn: Any = None):
    """Load the official test split for final evaluation only."""
    return _load_split("test", load_dataset_fn)


def to_tf_dataset(dataset: Any, *, batch_size: int, training: bool, seed: int):
    """Convert a Hugging Face split to a batched TensorFlow dataset.

    The training split is deterministically shuffled once with the project seed.
    Validation and test order is fixed. No examples are dropped.
    """
    if training:
        dataset = dataset.shuffle(seed=seed)

    return dataset.to_tf_dataset(
        columns="image",
        label_cols="label",
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=False,
    )
