"""Create BCE and 0/1-error learning curves for one completed run.

For the main Stage 2 report, prefer plot_report_figures.py, which places CNN and
MLP learning curves in one comparable figure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path) -> list[dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def plot_run(run_dir: Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    rows = read_history(run_dir / "history.csv")
    epochs = [int(row["epoch"]) for row in rows]

    bce_path = run_dir / "bce_curve.png"
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(epochs, [row["loss"] for row in rows], label="training BCE")
    plt.plot(epochs, [row["val_loss"] for row in rows], label="validation BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(bce_path, dpi=180)
    plt.close()

    error_path = run_dir / "zero_one_error_curve.png"
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(
        epochs,
        [1.0 - row["accuracy"] for row in rows],
        label="training 0/1 error",
    )
    plt.plot(
        epochs,
        [1.0 - row["val_accuracy"] for row in rows],
        label="validation 0/1 error",
    )
    plt.xlabel("Epoch")
    plt.ylabel("0/1 error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(error_path, dpi=180)
    plt.close()
    return bce_path, error_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    bce_path, error_path = plot_run(args.run_dir)
    print(f"Wrote {bce_path}")
    print(f"Wrote {error_path}")


if __name__ == "__main__":
    main()
