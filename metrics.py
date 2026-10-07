"""Pure NumPy metrics for binary classification and diagnostic curves."""

from __future__ import annotations

import numpy as np


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_binary_inputs(labels, probabilities) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(np.int8, copy=False).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)

    if labels.size == 0:
        raise ValueError("No examples were provided")
    if labels.size != probabilities.size:
        raise ValueError("labels and probabilities must have the same length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    return labels, probabilities


def binary_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float | int]:
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = probabilities >= threshold
    positives = labels == 1

    tp = int(np.count_nonzero(predictions & positives))
    tn = int(np.count_nonzero(~predictions & ~positives))
    fp = int(np.count_nonzero(predictions & ~positives))
    fn = int(np.count_nonzero(~predictions & positives))
    n = int(labels.size)

    accuracy = _safe_ratio(tp + tn, n)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)

    # Binary cross-entropy, averaged over examples.
    eps = np.finfo(np.float64).eps
    p = np.clip(probabilities, eps, 1.0 - eps)
    bce = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    return {
        "examples": n,
        "threshold": float(threshold),
        "binary_cross_entropy": bce,
        "accuracy": accuracy,
        "zero_one_error": 1.0 - accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "negative_examples": tn + fp,
        "positive_examples": tp + fn,
    }


def binary_curve_metrics(labels, probabilities) -> dict[str, np.ndarray | float]:
    """Return ROC and precision-recall points plus AUC/AP.

    The implementation uses the same threshold convention as a standard binary
    classifier: an example is positive when its score is greater than or equal
    to the current threshold. No scikit-learn dependency is required.
    """
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(np.count_nonzero(labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC and PR curves require both binary classes")

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    sorted_labels = labels[order]

    # Evaluate only after the final example of each distinct score. This avoids
    # drawing artificial steps inside a group that has one shared threshold.
    distinct_end = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
    true_positives = np.cumsum(sorted_labels)[distinct_end].astype(np.float64)
    predicted_positives = (distinct_end + 1).astype(np.float64)
    false_positives = predicted_positives - true_positives

    tpr = np.r_[0.0, true_positives / positive_count]
    fpr = np.r_[0.0, false_positives / negative_count]
    roc_thresholds = np.r_[np.inf, sorted_scores[distinct_end]]
    roc_auc = float(np.trapezoid(tpr, fpr))

    precision_values = true_positives / predicted_positives
    recall_values = true_positives / positive_count
    precision = np.r_[1.0, precision_values]
    recall = np.r_[0.0, recall_values]
    pr_thresholds = sorted_scores[distinct_end]
    average_precision = float(
        np.sum((recall[1:] - recall[:-1]) * precision[1:])
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "roc_auc": roc_auc,
        "precision_curve": precision,
        "recall_curve": recall,
        "pr_thresholds": pr_thresholds,
        "average_precision": average_precision,
        "positive_share": float(positive_count / labels.size),
    }
