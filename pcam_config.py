"""Shared constants for the PCam Stage 2 experiment."""

from pathlib import Path

DATASET_NAME = "1aurent/PatchCamelyon"
# Pin the dataset card/data revision used by this repository.
DATASET_REVISION = "4a1aaa1f54b23ae0769f4f3dd6e4b519edefd0d6"

IMAGE_SHAPE = (96, 96, 3)
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 15
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_THRESHOLD = 0.5

DEFAULT_OUTPUT_ROOT = Path("artifacts")
MODEL_NAMES = ("cnn", "mlp")
