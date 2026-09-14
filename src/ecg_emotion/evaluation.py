"""Evaluation and artifact serialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Return stable scalar metrics and a per-class classification report."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "support": int(len(y_true)),
        "labels": [int(label) for label in labels],
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": {
            str(label): {
                key: float(value)
                for key, value in report[str(label)].items()
                if key != "support"
            }
            | {"support": int(report[str(label)]["support"])}
            for label in labels
        },
    }


def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[int] | None = None,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Measure multiclass probability quality and confidence-based coverage."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    if labels is None:
        labels = list(range(probabilities.shape[1]))
    if len(labels) != probabilities.shape[1]:
        raise ValueError("labels must match the probability columns")

    label_indexes = {label: index for index, label in enumerate(labels)}
    try:
        true_indexes = np.asarray([label_indexes[int(label)] for label in y_true])
    except KeyError as error:
        raise ValueError("y_true contains a label absent from labels") from error
    predicted_indexes = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted_indexes == true_indexes
    one_hot = np.eye(len(labels), dtype=np.float64)[true_indexes]

    expected_calibration_error = 0.0
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidence >= lower) & (
            (confidence < upper) if upper < 1.0 else (confidence <= upper)
        )
        if in_bin.any():
            expected_calibration_error += float(in_bin.mean()) * abs(
                float(confidence[in_bin].mean()) - float(correct[in_bin].mean())
            )

    result: dict[str, Any] = {
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(
            -np.mean(np.log(np.maximum(probabilities[np.arange(len(y_true)), true_indexes], 1e-12)))
        ),
        "expected_calibration_error": float(expected_calibration_error),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
    }
    for threshold in (0.5, 0.7, 0.8, 0.9):
        accepted = confidence >= threshold
        suffix = str(threshold).replace(".", "_")
        result[f"coverage_at_{suffix}"] = float(accepted.mean())
        result[f"accuracy_at_{suffix}"] = (
            float(correct[accepted].mean()) if accepted.any() else 0.0
        )
    return result


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply multiclass temperature scaling to a probability matrix."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.maximum(probabilities, 1e-12)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[int] | None = None,
) -> float:
    """Fit a temperature using a deterministic grid and calibration labels only."""

    if labels is None:
        labels = list(range(np.asarray(probabilities).shape[1]))
    label_indexes = {label: index for index, label in enumerate(labels)}
    true_indexes = np.asarray([label_indexes[int(label)] for label in np.asarray(y_true)])
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidate_temperatures = np.exp(np.linspace(np.log(0.05), np.log(10.0), 161))
    losses = []
    for temperature in candidate_temperatures:
        scaled = apply_temperature(probabilities, float(temperature))
        losses.append(
            -np.mean(np.log(np.maximum(scaled[np.arange(len(true_indexes)), true_indexes], 1e-12)))
        )
    return float(candidate_temperatures[int(np.argmin(losses))])


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write an experiment payload as human-readable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
