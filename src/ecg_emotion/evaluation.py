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


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write an experiment payload as human-readable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

