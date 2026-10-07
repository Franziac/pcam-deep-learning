"""Report-ready visualizations for the PCam Stage 2 experiment.

The functions in this module intentionally separate validation-only figures
from final-test figures. Test visualizations are generated only after a model
has already been selected from validation data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from metrics import binary_curve_metrics
from training_utils import read_json, write_json


MODEL_LABELS = {"cnn": "CNN", "mlp": "MLP"}
CENTER_REGION_START = 32
CENTER_REGION_SIZE = 32


def _save_figure(fig, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _read_history(path: Path) -> list[dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _read_comparison(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _draw_center_region(ax) -> None:
    ax.add_patch(
        Rectangle(
            (CENTER_REGION_START, CENTER_REGION_START),
            CENTER_REGION_SIZE,
            CENTER_REGION_SIZE,
            fill=False,
            edgecolor="white",
            linewidth=1.4,
        )
    )


def plot_dataset_examples(
    dataset: Any,
    output_path: Path,
    *,
    examples_per_class: int = 4,
    seed: int = 42,
) -> Path:
    """Plot training examples from each class and mark the label-relevant region."""
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be at least 1")

    rng = np.random.default_rng(seed)
    chosen: dict[int, list[tuple[int, Any]]] = {0: [], 1: []}
    used: set[int] = set()
    max_attempts = max(1000, examples_per_class * 100)

    for _ in range(max_attempts):
        if all(len(chosen[label]) >= examples_per_class for label in (0, 1)):
            break
        index = int(rng.integers(0, len(dataset)))
        if index in used:
            continue
        used.add(index)
        row = dataset[index]
        label = int(bool(row["label"]))
        if len(chosen[label]) < examples_per_class:
            chosen[label].append((index, row["image"]))

    if any(len(chosen[label]) < examples_per_class for label in (0, 1)):
        raise ValueError("Could not sample enough examples from both classes")

    fig, axes = plt.subplots(
        2,
        examples_per_class,
        figsize=(2.25 * examples_per_class, 4.8),
        squeeze=False,
    )
    for row_index, label in enumerate((0, 1)):
        for column, (index, image) in enumerate(chosen[label]):
            ax = axes[row_index, column]
            ax.imshow(np.asarray(image))
            _draw_center_region(ax)
            ax.set_title(f"index {index}", fontsize=9)
            ax.axis("off")
    fig.text(0.01, 0.68, "Negative", rotation=90, va="center", fontsize=11)
    fig.text(0.01, 0.28, "Positive", rotation=90, va="center", fontsize=11)

    fig.suptitle(
        "PCam training examples — white box marks the center 32 x 32 label region",
        fontsize=12,
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_learning_curves(
    run_dirs: Mapping[str, Path],
    output_path: Path,
) -> Path:
    """Plot train/validation BCE and 0/1 error for CNN and MLP."""
    model_order = [name for name in ("cnn", "mlp") if name in run_dirs]
    if not model_order:
        raise ValueError("run_dirs must contain cnn and/or mlp")

    fig, axes = plt.subplots(2, len(model_order), figsize=(5.2 * len(model_order), 7.0))
    if len(model_order) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, model_name in enumerate(model_order):
        run_dir = Path(run_dirs[model_name])
        rows = _read_history(run_dir / "history.csv")
        summary = read_json(run_dir / "validation_summary.json")
        epochs = np.array([int(row["epoch"]) for row in rows])
        label = MODEL_LABELS.get(model_name, model_name.upper())
        best_epoch = int(summary["best_epoch"])

        bce_ax = axes[0, column]
        bce_ax.plot(epochs, [row["loss"] for row in rows], label="training")
        bce_ax.plot(epochs, [row["val_loss"] for row in rows], label="validation")
        bce_ax.axvline(best_epoch, linestyle=":", linewidth=1.2, label="selected epoch")
        bce_ax.set_title(f"{label}: binary cross-entropy")
        bce_ax.set_ylabel("BCE")
        bce_ax.grid(alpha=0.25)
        bce_ax.legend(fontsize=8)
        bce_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        error_ax = axes[1, column]
        error_ax.plot(
            epochs,
            [100.0 * (1.0 - row["accuracy"]) for row in rows],
            label="training",
        )
        error_ax.plot(
            epochs,
            [100.0 * (1.0 - row["val_accuracy"]) for row in rows],
            label="validation",
        )
        error_ax.axvline(best_epoch, linestyle=":", linewidth=1.2, label="selected epoch")
        error_ax.set_title(f"{label}: 0/1 error")
        error_ax.set_xlabel("Epoch")
        error_ax.set_ylabel("0/1 error (%)")
        error_ax.grid(alpha=0.25)
        error_ax.legend(fontsize=8)
        error_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Training and validation learning curves", fontsize=13)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_validation_comparison(
    comparison_csv: Path,
    selection_json: Path,
    output_path: Path,
) -> Path:
    """Compare training and validation BCE/error for both candidate methods."""
    rows = _read_comparison(comparison_csv)
    selection = read_json(selection_json)
    labels = [MODEL_LABELS.get(row["model"], row["model"].upper()) for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))

    train_bce = np.array([float(row["training_bce"]) for row in rows])
    val_bce = np.array([float(row["validation_bce"]) for row in rows])
    axes[0].bar(x - width / 2, train_bce, width, label="training")
    axes[0].bar(x + width / 2, val_bce, width, label="validation")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].set_title("BCE at each selected epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    train_error = np.array(
        [100.0 * float(row["training_zero_one_error"]) for row in rows]
    )
    val_error = np.array(
        [100.0 * float(row["validation_zero_one_error"]) for row in rows]
    )
    axes[1].bar(x - width / 2, train_error, width, label="training")
    axes[1].bar(x + width / 2, val_error, width, label="validation")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("0/1 error (%)")
    axes[1].set_title("0/1 error at each selected epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    selected = MODEL_LABELS.get(
        str(selection["selected_model"]), str(selection["selected_model"]).upper()
    )
    fig.suptitle(
        f"Validation-only method comparison — selected final method: {selected}",
        fontsize=12,
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _plot_confusion_matrix(ax, metrics: Mapping[str, Any]) -> None:
    matrix = np.array(
        [
            [int(metrics["true_negatives"]), int(metrics["false_positives"])],
            [int(metrics["false_negatives"]), int(metrics["true_positives"])],
        ]
    )
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred. negative", "Pred. positive"])
    ax.set_yticks([0, 1], ["Actual negative", "Actual positive"])
    ax.set_title("Confusion matrix")

    for row in range(2):
        row_total = matrix[row].sum()
        for column in range(2):
            value = int(matrix[row, column])
            share = 100.0 * value / row_total if row_total else 0.0
            text_color = "white" if value > matrix.max() / 2 else "black"
            ax.text(
                column,
                row,
                f"{value:,}\n{share:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )


def plot_final_test_diagnostics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    metrics: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Plot confusion matrix, ROC, and precision-recall for the final model."""
    labels_array = np.asarray(labels).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype=float).reshape(-1)
    curves = binary_curve_metrics(labels_array, probabilities_array)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    _plot_confusion_matrix(axes[0], metrics)

    axes[1].plot(
        curves["fpr"],
        curves["tpr"],
        label=f"AUC = {float(curves['roc_auc']):.3f}",
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="random")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[1].set_title("ROC curve")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(
        curves["recall_curve"],
        curves["precision_curve"],
        label=f"AP = {float(curves['average_precision']):.3f}",
    )
    positive_share = float(curves["positive_share"])
    axes[2].axhline(
        positive_share,
        linestyle="--",
        linewidth=1.0,
        label=f"positive share = {positive_share:.3f}",
    )
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-recall curve")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    model = MODEL_LABELS.get(str(metrics["model"]), str(metrics["model"]).upper())
    fig.suptitle(
        f"Final {model} test diagnostics — BCE {float(metrics['binary_cross_entropy']):.3f}, "
        f"0/1 error {100.0 * float(metrics['zero_one_error']):.1f}%",
        fontsize=12,
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _select_confident_errors(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    examples_per_type: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = probabilities >= threshold
    false_positive_indices = np.flatnonzero((labels == 0) & predictions)
    false_negative_indices = np.flatnonzero((labels == 1) & ~predictions)

    if false_positive_indices.size:
        false_positive_indices = false_positive_indices[
            np.argsort(-probabilities[false_positive_indices])
        ][:examples_per_type]
    if false_negative_indices.size:
        false_negative_indices = false_negative_indices[
            np.argsort(probabilities[false_negative_indices])
        ][:examples_per_type]
    return false_positive_indices, false_negative_indices


def plot_confident_errors(
    dataset: Any,
    labels: Iterable[int],
    probabilities: Iterable[float],
    output_path: Path,
    *,
    threshold: float = 0.5,
    examples_per_type: int = 4,
) -> Path:
    """Show the most confident false positives and false negatives."""
    labels_array = np.asarray(labels).astype(np.int8, copy=False).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(dataset) != labels_array.size or labels_array.size != probabilities_array.size:
        raise ValueError("dataset, labels, and probabilities must have equal length")

    fp_indices, fn_indices = _select_confident_errors(
        labels_array, probabilities_array, threshold, examples_per_type
    )
    fig, axes = plt.subplots(
        2,
        examples_per_type,
        figsize=(2.25 * examples_per_type, 4.8),
        squeeze=False,
    )

    rows = [
        (fp_indices, "False positive"),
        (fn_indices, "False negative"),
    ]
    for row_index, (indices, row_label) in enumerate(rows):
        for column in range(examples_per_type):
            ax = axes[row_index, column]
            if column >= len(indices):
                ax.axis("off")
                continue
            index = int(indices[column])
            image = dataset[index]["image"]
            ax.imshow(np.asarray(image))
            _draw_center_region(ax)
            ax.set_title(f"p(tumour)={probabilities_array[index]:.3f}", fontsize=9)
            ax.axis("off")
    fig.text(0.01, 0.68, "False positive", rotation=90, va="center", fontsize=10)
    fig.text(0.01, 0.28, "False negative", rotation=90, va="center", fontsize=10)

    fig.suptitle(
        "Most confident final-test errors — white box is the center label region",
        fontsize=12,
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def generate_report_figures(experiment_dir: Path, *, seed: int = 42) -> dict[str, str]:
    """Generate all report-ready figures from one completed Stage 2 experiment."""
    experiment_dir = Path(experiment_dir)
    selection_path = experiment_dir / "comparison" / "selection.json"
    comparison_path = experiment_dir / "comparison" / "validation_comparison.csv"
    test_metrics_path = experiment_dir / "final_test" / "final_test_metrics.json"
    predictions_path = experiment_dir / "final_test" / "test_predictions.npz"

    for required in (
        selection_path,
        comparison_path,
        test_metrics_path,
        predictions_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required experiment output not found: {required}")

    selection = read_json(selection_path)
    run_paths = selection.get("runs")
    if not isinstance(run_paths, dict) or not {"cnn", "mlp"}.issubset(run_paths):
        raise ValueError("selection.json must contain cnn and mlp run paths")
    run_dirs = {name: Path(run_paths[name]) for name in ("cnn", "mlp")}

    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # Training examples use only the training split. No test data is loaded here.
    from pcam_data import load_train, load_test

    train_data = load_train()
    outputs["dataset_examples"] = str(
        plot_dataset_examples(
            train_data,
            figures_dir / "dataset_examples.png",
            seed=seed,
        )
    )
    outputs["learning_curves"] = str(
        plot_learning_curves(run_dirs, figures_dir / "learning_curves.png")
    )
    outputs["validation_comparison"] = str(
        plot_validation_comparison(
            comparison_path,
            selection_path,
            figures_dir / "validation_comparison.png",
        )
    )

    # Final-test figures are allowed only after selection.json and final test
    # outputs already exist. This keeps test data outside model selection.
    test_metrics = read_json(test_metrics_path)
    if test_metrics.get("test_used_only_after_validation_selection") is not True:
        raise ValueError("Final test metadata does not prove post-selection evaluation")
    with np.load(predictions_path) as data:
        labels = np.asarray(data["labels"])
        probabilities = np.asarray(data["probabilities"])

    outputs["final_test_diagnostics"] = str(
        plot_final_test_diagnostics(
            labels,
            probabilities,
            test_metrics,
            figures_dir / "final_test_diagnostics.png",
        )
    )

    test_data = load_test()
    outputs["final_test_errors"] = str(
        plot_confident_errors(
            test_data,
            labels,
            probabilities,
            figures_dir / "final_test_errors.png",
            threshold=float(test_metrics["threshold"]),
        )
    )

    manifest = {
        "experiment_dir": str(experiment_dir),
        "figures": outputs,
        "recommended_main_report": [
            "learning_curves",
            "validation_comparison",
            "final_test_diagnostics",
        ],
        "recommended_appendix": ["dataset_examples", "final_test_errors"],
        "data_usage": {
            "dataset_examples": "training split only",
            "learning_curves": "training and validation histories only",
            "validation_comparison": "validation-only method selection outputs",
            "final_test_diagnostics": "final selected model on test after selection",
            "final_test_errors": "final selected model on test after selection",
        },
    }
    write_json(figures_dir / "manifest.json", manifest)
    return outputs
