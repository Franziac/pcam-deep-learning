"""Model definitions for the two Stage 2 machine-learning methods."""

from __future__ import annotations

from pcam_config import IMAGE_SHAPE, MODEL_NAMES


def build_model(model_name: str):
    """Build either the CNN or the spatially agnostic MLP baseline."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")

    import tensorflow as tf

    keras = tf.keras

    if model_name == "cnn":
        return keras.Sequential(
            [
                keras.Input(shape=IMAGE_SHAPE),
                keras.layers.Rescaling(1.0 / 255.0),
                keras.layers.Conv2D(32, (3, 3), activation="relu"),
                keras.layers.MaxPool2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPool2D((2, 2)),
                keras.layers.Conv2D(128, (3, 3), activation="relu"),
                keras.layers.MaxPool2D((2, 2)),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.35),
                keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="pcam_cnn",
        )

    # MLP: same 96x96 RGB pixels and scaling, but no convolution or spatial
    # weight sharing. This creates a clearly different hypothesis space.
    return keras.Sequential(
        [
            keras.Input(shape=IMAGE_SHAPE),
            keras.layers.Rescaling(1.0 / 255.0),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="pcam_mlp",
    )


def model_description(model_name: str) -> str:
    if model_name == "cnn":
        return (
            "Three convolution/max-pooling blocks followed by global average "
            "pooling, a 64-unit ReLU layer, dropout, and one sigmoid output."
        )
    if model_name == "mlp":
        return (
            "All 96x96x3 pixels are flattened, followed by a 64-unit ReLU "
            "layer, dropout, and one sigmoid output."
        )
    raise ValueError(f"Unknown model {model_name!r}")
