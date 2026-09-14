"""Reproducible training entry points for feature-based baselines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import load_npz, split_by_subject
from .evaluation import (
    apply_temperature,
    classification_metrics,
    fit_temperature,
    probability_metrics,
    save_json,
)
from .features import FEATURE_NAMES, extract_features
from .models import build_baseline


def train_feature_baseline(
    data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "logistic-regression",
    sample_rate: float = 256.0,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train a baseline using subject-level splits and save all metadata."""

    started = time.perf_counter()
    dataset = load_npz(data_path)
    split_indexes = split_by_subject(
        dataset,
        test_size=test_size,
        validation_size=validation_size,
        random_seed=random_seed,
    )
    feature_matrix = extract_features(dataset.signals, sample_rate=sample_rate)
    model = build_baseline(model_name, random_seed=random_seed)
    train_indexes = split_indexes["train"]
    model.fit(feature_matrix[train_indexes], dataset.labels[train_indexes])

    labels = sorted(np.unique(dataset.labels).astype(int).tolist())
    split_metrics: dict[str, Any] = {}
    split_subjects: dict[str, list[str]] = {}
    for split_name, indexes in split_indexes.items():
        predictions = model.predict(feature_matrix[indexes])
        split_metrics[split_name] = classification_metrics(
            dataset.labels[indexes], predictions, labels=labels
        )
        split_subjects[split_name] = sorted(
            {str(subject) for subject in dataset.subjects[indexes]}
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "sample_rate": sample_rate,
        "feature_names": FEATURE_NAMES,
        "labels": labels,
        "confidence_threshold": 0.8,
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    result = {
        "model_name": model_name,
        "data_path": str(Path(data_path)),
        "num_samples": dataset.num_samples,
        "signal_length": dataset.signal_length,
        "num_subjects": dataset.num_subjects,
        "random_seed": random_seed,
        "split_subjects": split_subjects,
        "metrics": split_metrics,
        "training_seconds": round(time.perf_counter() - started, 4),
    }
    save_json(result, output_dir / "metrics.json")
    return result


def cross_validate_feature_baseline(
    data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "logistic-regression",
    sample_rate: float = 256.0,
    num_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run deterministic, label-balanced subject-level cross-validation."""

    started = time.perf_counter()
    dataset = load_npz(data_path)
    if dataset.num_subjects < num_folds:
        raise ValueError("num_folds cannot exceed the number of subjects")

    feature_matrix = extract_features(dataset.signals, sample_rate=sample_rate)
    labels = sorted(np.unique(dataset.labels).astype(int).tolist())
    splitter = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_seed,
    )
    fold_results: list[dict[str, Any]] = []
    subject_results: list[dict[str, Any]] = []
    confidence_results: list[dict[str, Any]] = []
    for fold_index, (train_indexes, test_indexes) in enumerate(
        splitter.split(feature_matrix, dataset.labels, groups=dataset.subjects),
        start=1,
    ):
        model = build_baseline(model_name, random_seed=random_seed + fold_index)
        model.fit(feature_matrix[train_indexes], dataset.labels[train_indexes])
        predictions = model.predict(feature_matrix[test_indexes])
        probabilities = model.predict_proba(feature_matrix[test_indexes])
        calibration_labels, calibration_probabilities = _cross_fitted_calibration_data(
            feature_matrix,
            dataset.labels,
            dataset.subjects,
            train_indexes,
            model_name,
            random_seed + fold_index,
        )
        temperature = fit_temperature(
            calibration_labels,
            calibration_probabilities,
            labels=labels,
        )
        calibrated_probabilities = apply_temperature(probabilities, temperature)
        fold_results.append(
            {
                "fold": fold_index,
                "train_subjects": sorted(
                    {str(subject) for subject in dataset.subjects[train_indexes]}
                ),
                "test_subjects": sorted(
                    {str(subject) for subject in dataset.subjects[test_indexes]}
                ),
                "calibration_subjects": sorted(
                    {str(subject) for subject in dataset.subjects[train_indexes]}
                ),
                "calibration_strategy": "3-fold cross-fitted OOF probabilities",
                "metrics": classification_metrics(
                    dataset.labels[test_indexes],
                    predictions,
                    labels=labels,
                ),
                "confidence": probability_metrics(
                    dataset.labels[test_indexes], probabilities, labels=labels
                ),
                "calibrated_confidence": probability_metrics(
                    dataset.labels[test_indexes], calibrated_probabilities, labels=labels
                ),
                "temperature": temperature,
            }
        )
        confidence_results.append(
            {
                "raw": fold_results[-1]["confidence"],
                "calibrated": fold_results[-1]["calibrated_confidence"],
            }
        )
        for subject in sorted({str(value) for value in dataset.subjects[test_indexes]}):
            subject_indexes = test_indexes[dataset.subjects[test_indexes] == subject]
            subject_predictions = model.predict(feature_matrix[subject_indexes])
            subject_results.append(
                {
                    "fold": fold_index,
                    "subject": subject,
                    "metrics": classification_metrics(
                        dataset.labels[subject_indexes],
                        subject_predictions,
                        labels=labels,
                    ),
                }
            )

    scalar_metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    aggregate = {
        metric: {
            "mean": float(np.mean([fold["metrics"][metric] for fold in fold_results])),
            "std": float(np.std([fold["metrics"][metric] for fold in fold_results])),
        }
        for metric in scalar_metrics
    }
    confidence_aggregate = {
        "raw": _aggregate_probability_metrics(
            [result["raw"] for result in confidence_results]
        ),
        "calibrated": _aggregate_probability_metrics(
            [result["calibrated"] for result in confidence_results]
        ),
    }
    result = {
        "model_name": model_name,
        "data_path": str(Path(data_path)),
        "num_samples": dataset.num_samples,
        "num_subjects": dataset.num_subjects,
        "num_folds": num_folds,
        "split_strategy": "StratifiedGroupKFold",
        "sample_rate": sample_rate,
        "feature_names": list(FEATURE_NAMES),
        "folds": fold_results,
        "per_subject": subject_results,
        "aggregate": aggregate,
        "confidence_aggregate": confidence_aggregate,
        "evaluation_seconds": round(time.perf_counter() - started, 4),
    }
    output_dir = Path(output_dir)
    save_json(result, output_dir / "cross_validation.json")
    return result


def _cross_fitted_calibration_data(
    feature_matrix: np.ndarray,
    labels_array: np.ndarray,
    subjects: np.ndarray,
    indexes: np.ndarray,
    model_name: str,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training-only OOF probabilities for leakage-safe calibration."""

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_seed)
    probabilities: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []
    local_features = feature_matrix[indexes]
    local_labels = labels_array[indexes]
    local_subjects = subjects[indexes]
    for inner_fold, (inner_train, inner_validation) in enumerate(
        splitter.split(local_features, local_labels, groups=local_subjects),
        start=1,
    ):
        model = build_baseline(model_name, random_seed=random_seed + inner_fold)
        model.fit(local_features[inner_train], local_labels[inner_train])
        probabilities.append(model.predict_proba(local_features[inner_validation]))
        true_labels.append(local_labels[inner_validation])
    return np.concatenate(true_labels), np.vstack(probabilities)


def _aggregate_probability_metrics(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate scalar probability metrics across outer folds."""

    keys = results[0].keys()
    return {
        key: {
            "mean": float(np.mean([result[key] for result in results])),
            "std": float(np.std([result[key] for result in results])),
        }
        for key in keys
    }
