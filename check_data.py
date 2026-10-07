"""Verify PCam split sizes, class counts, image shape, and labels from the pinned mirror."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pcam_config import DATASET_NAME, DATASET_REVISION, IMAGE_SHAPE


def inspect_dataset(output: Path | None = None) -> dict:
    from datasets import load_dataset

    summary = {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "splits": {},
    }

    expected_sizes = {"train": 262_144, "valid": 32_768, "test": 32_768}
    for split, expected_n in expected_sizes.items():
        data = load_dataset(
            DATASET_NAME,
            revision=DATASET_REVISION,
            split=split,
        )
        counts = Counter(bool(label) for label in data["label"])
        n = len(data)
        if n != expected_n:
            raise ValueError(f"Unexpected {split} size: {n} != {expected_n}")

        summary["splits"][split] = {
            "examples": n,
            "negative": int(counts[False]),
            "positive": int(counts[True]),
            "positive_share": float(counts[True] / n),
        }

        example = data[0]
        image = example["image"]
        expected_image_size = IMAGE_SHAPE[:2]
        if image.size != expected_image_size:
            raise ValueError(
                f"Unexpected image size in {split}: {image.size} != {expected_image_size}"
            )
        if image.mode != "RGB":
            raise ValueError(f"Unexpected image mode in {split}: {image.mode}")

    print(json.dumps(summary, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_dataset(args.output)


if __name__ == "__main__":
    main()
