"""Subject-domain robustness and feature ablation experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import load_npz
from .evaluation import (
    classification_metrics,
    fit_confidence_threshold,
    probability_metrics,
    save_json,
    selective_metrics,
)
from .features import FEATURE_NAMES, extract_features
from .models import build_baseline

_RATE_HRV_FEATURES = frozenset(
    {
        "r_peak_count",
        "rr_valid_ratio",
        "heart_rate_mean",
        "heart_rate_std",
        "rr_mean",
        "rr_median",
        "rr_std",
        "rr_iqr",
        "rr_rmssd",
        "rr_pnn50",
        "hrv_lf_power",
        "hrv_hf_power",
        "hrv_lf_hf_ratio",
    }
)
_MORPHOLOGY_SPECTRAL_FEATURES = frozenset(
    {
        "signal_range",
        "signal_line_length",
        "signal_skewness",
        "signal_kurtosis",
        "r_amplitude_mean",
        "r_amplitude_std",
        "qrs_band_power",
        "high_band_power",
        "spectral_entropy",
    }
)

FEATURE_SETS = {
    "all_features": tuple(FEATURE_NAMES),
    "rate_hrv": tuple(name for name in FEATURE_NAMES if name in _RATE_HRV_FEATURES),
    "morphology_spectral": tuple(
        name for name in FEATURE_NAMES if name in _MORPHOLOGY_SPECTRAL_FEATURES
    ),
    "without_amplitude": tuple(
        name for name in FEATURE_NAMES if name not in {"r_amplitude_mean", "r_amplitude_std"}
    ),
}


def run_loso_audit(
    data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "shrinkage-lda",
    sample_rate: float = 256.0,
    bootstrap_samples: int = 2000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run LOSO evaluation, feature ablations, and subject-level bootstrap CIs."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    started = time.perf_counter()
    dataset = load_npz(data_path)
    feature_matrix = extract_features(dataset.signals, sample_rate=sample_rate)
    labels = sorted(np.unique(dataset.labels).astype(int).tolist())
    subjects = sorted(np.unique(dataset.subjects), key=str)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    feature_results: dict[str, Any] = {}

    for feature_set_name, feature_names in FEATURE_SETS.items():
        selected_indexes = np.asarray([feature_index[name] for name in feature_names])
        subject_results: list[dict[str, Any]] = []
        for subject_index, subject in enumerate(subjects):
            test_mask = dataset.subjects == subject
            test_indexes = np.flatnonzero(test_mask)
            train_indexes = np.flatnonzero(~test_mask)
            model = build_baseline(model_name, random_seed=random_seed + subject_index)
            model.fit(
                feature_matrix[train_indexes][:, selected_indexes], dataset.labels[train_indexes]
            )
            train_predictions = model.predict(feature_matrix[train_indexes][:, selected_indexes])
            test_features = feature_matrix[test_indexes][:, selected_indexes]
            test_predictions = model.predict(test_features)
            test_probabilities = model.predict_proba(test_features)
            subject_results.append(
                {
                    "subject": str(subject),
                    "train_metrics": classification_metrics(
                        dataset.labels[train_indexes], train_predictions, labels=labels
                    ),
                    "metrics": classification_metrics(
                        dataset.labels[test_indexes], test_predictions, labels=labels
                    ),
                    "confidence": probability_metrics(
                        dataset.labels[test_indexes], test_probabilities, labels=labels
                    ),
                }
            )

        feature_results[feature_set_name] = {
            "feature_names": list(feature_names),
            "num_features": len(feature_names),
            "subjects": subject_results,
            "aggregate": _aggregate_subject_metrics(
                subject_results, bootstrap_samples=bootstrap_samples, random_seed=random_seed
            ),
        }

    result = {
        "audit_name": "leave-one-subject-out-robustness",
        "model_name": model_name,
        "data_path": str(Path(data_path)),
        "sample_rate": sample_rate,
        "num_samples": dataset.num_samples,
        "num_subjects": dataset.num_subjects,
        "split_strategy": "Leave-One-Subject-Out",
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "feature_sets": feature_results,
        "evaluation_seconds": round(time.perf_counter() - started, 4),
    }
    output_dir = Path(output_dir)
    save_json(result, output_dir / "loso_robustness.json")
    return result


def run_loso_selective_audit(
    data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "shrinkage-lda",
    sample_rate: float = 256.0,
    min_coverage: float = 0.3,
    inner_folds: int = 3,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Audit training-only confidence thresholds and held-out abstention behavior."""

    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in the interval (0, 1]")
    started = time.perf_counter()
    dataset = load_npz(data_path)
    feature_matrix = extract_features(dataset.signals, sample_rate=sample_rate)
    labels = sorted(np.unique(dataset.labels).astype(int).tolist())
    subjects = sorted(np.unique(dataset.subjects), key=str)
    subject_results: list[dict[str, Any]] = []

    for subject_index, subject in enumerate(subjects):
        test_mask = dataset.subjects == subject
        test_indexes = np.flatnonzero(test_mask)
        train_indexes = np.flatnonzero(~test_mask)
        model = build_baseline(model_name, random_seed=random_seed + subject_index)
        model.fit(feature_matrix[train_indexes], dataset.labels[train_indexes])
        test_probabilities = _align_probability_columns(
            model.predict_proba(feature_matrix[test_indexes]), model.classes_, labels
        )
        test_predictions = np.asarray(
            [labels[index] for index in np.argmax(test_probabilities, axis=1)]
        )

        oof_labels, oof_probabilities = _cross_fitted_training_probabilities(
            feature_matrix,
            dataset.labels,
            dataset.subjects,
            train_indexes,
            model_name,
            labels,
            random_seed + subject_index,
            inner_folds,
        )
        threshold = fit_confidence_threshold(
            oof_labels,
            oof_probabilities,
            labels=labels,
            min_coverage=min_coverage,
        )
        subject_results.append(
            {
                "subject": str(subject),
                "train_subjects": sorted(
                    {str(value) for value in dataset.subjects[train_indexes]}
                ),
                "threshold": threshold,
                "threshold_selection": {
                    "method": (
                        "maximum OOF class-balanced selective F1 at predefined minimum coverage"
                    ),
                    "min_coverage": min_coverage,
                    "oof_metrics": selective_metrics(
                        oof_labels,
                        oof_probabilities,
                        labels=labels,
                        threshold=threshold,
                    ),
                },
                "metrics": classification_metrics(
                    dataset.labels[test_indexes], test_predictions, labels=labels
                ),
                "confidence": probability_metrics(
                    dataset.labels[test_indexes], test_probabilities, labels=labels
                ),
                "selective": selective_metrics(
                    dataset.labels[test_indexes],
                    test_probabilities,
                    labels=labels,
                    threshold=threshold,
                ),
            }
        )

    result = {
        "audit_name": "leave-one-subject-out-selective-audit",
        "model_name": model_name,
        "data_path": str(Path(data_path)),
        "sample_rate": sample_rate,
        "num_samples": dataset.num_samples,
        "num_subjects": dataset.num_subjects,
        "split_strategy": "Leave-One-Subject-Out",
        "threshold_selection": {
            "method": "maximum OOF class-balanced selective F1 at predefined minimum coverage",
            "minimum_coverage": min_coverage,
            "inner_folds": inner_folds,
            "candidate_thresholds": [
                round(float(value), 2) for value in np.linspace(0.5, 0.95, 19)
            ],
            "training_only": True,
        },
        "random_seed": random_seed,
        "subjects": subject_results,
        "aggregate": _aggregate_selective_metrics(
            subject_results,
            labels=labels,
            bootstrap_samples=2000,
            random_seed=random_seed,
        ),
        "evaluation_seconds": round(time.perf_counter() - started, 4),
    }
    output_dir = Path(output_dir)
    save_json(result, output_dir / "selective_audit.json")
    return result


def _cross_fitted_training_probabilities(
    feature_matrix: np.ndarray,
    labels_array: np.ndarray,
    subjects: np.ndarray,
    indexes: np.ndarray,
    model_name: str,
    labels: list[int],
    random_seed: int,
    num_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OOF probabilities using only the training subjects of one LOSO split."""

    local_features = feature_matrix[indexes]
    local_labels = labels_array[indexes]
    local_subjects = subjects[indexes]
    subject_count = len(np.unique(local_subjects))
    splitter = StratifiedGroupKFold(
        n_splits=min(num_folds, subject_count),
        shuffle=True,
        random_state=random_seed,
    )
    probabilities: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []
    for inner_fold, (inner_train, inner_validation) in enumerate(
        splitter.split(local_features, local_labels, groups=local_subjects),
        start=1,
    ):
        model = build_baseline(model_name, random_seed=random_seed + inner_fold)
        model.fit(local_features[inner_train], local_labels[inner_train])
        probabilities.append(
            _align_probability_columns(
                model.predict_proba(local_features[inner_validation]), model.classes_, labels
            )
        )
        true_labels.append(local_labels[inner_validation])
    return np.concatenate(true_labels), np.vstack(probabilities)


def _align_probability_columns(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    labels: list[int],
) -> np.ndarray:
    """Align model probabilities to the complete dataset label order."""

    aligned = np.zeros((len(probabilities), len(labels)), dtype=np.float64)
    label_indexes = {int(label): index for index, label in enumerate(labels)}
    for source_index, label in enumerate(model_classes):
        aligned[:, label_indexes[int(label)]] = probabilities[:, source_index]
    return aligned


def _aggregate_selective_metrics(
    subject_results: list[dict[str, Any]],
    labels: list[int],
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    """Aggregate held-out selective metrics over participants."""

    scalar_metrics = (
        "threshold",
        "coverage",
        "rejection_rate",
        "selective_accuracy",
        "selective_macro_f1",
    )
    aggregate: dict[str, Any] = {}
    for metric in scalar_metrics:
        values = np.asarray([result["selective"][metric] for result in subject_results])
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "bootstrap_95ci": list(_bootstrap_mean_ci(values, bootstrap_samples, random_seed)),
        }

    classification_metrics_to_aggregate = ("accuracy", "balanced_accuracy", "macro_f1")
    aggregate["classification"] = {
        metric: {
            "mean": float(np.mean([result["metrics"][metric] for result in subject_results])),
            "std": float(np.std([result["metrics"][metric] for result in subject_results])),
        }
        for metric in classification_metrics_to_aggregate
    }
    per_class_metrics = ("mean_confidence", "coverage", "correct_coverage", "accepted_precision")
    aggregate["per_class"] = {
        str(label): {
            metric: {
                "mean": float(
                    np.mean(
                        [
                            result["selective"]["per_class"][str(label)][metric]
                            for result in subject_results
                        ]
                    )
                ),
                "std": float(
                    np.std(
                        [
                            result["selective"]["per_class"][str(label)][metric]
                            for result in subject_results
                        ]
                    )
                ),
            }
            for metric in per_class_metrics
        }
        for label in labels
    }
    oof_thresholds = np.asarray(
        [result["threshold_selection"]["oof_metrics"]["threshold"] for result in subject_results]
    )
    oof_coverages = np.asarray(
        [result["threshold_selection"]["oof_metrics"]["coverage"] for result in subject_results]
    )
    aggregate["oof_threshold"] = {
        "mean": float(oof_thresholds.mean()),
        "std": float(oof_thresholds.std()),
    }
    aggregate["oof_coverage"] = {
        "mean": float(oof_coverages.mean()),
        "std": float(oof_coverages.std()),
    }
    return aggregate


def _aggregate_subject_metrics(
    subject_results: list[dict[str, Any]],
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    scalar_metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    aggregate: dict[str, Any] = {}
    for metric in scalar_metrics:
        values = np.asarray([result["metrics"][metric] for result in subject_results])
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "bootstrap_95ci": list(_bootstrap_mean_ci(values, bootstrap_samples, random_seed)),
        }
    confidence_metrics = (
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "mean_confidence",
    )
    aggregate["confidence"] = {
        metric: {
            "mean": float(np.mean([result["confidence"][metric] for result in subject_results])),
            "std": float(np.std([result["confidence"][metric] for result in subject_results])),
        }
        for metric in confidence_metrics
    }
    labels = subject_results[0]["metrics"]["labels"]
    aggregate["per_class"] = {
        str(label): {
            metric: {
                "mean": float(
                    np.mean(
                        [
                            result["metrics"]["per_class"][str(label)][metric]
                            for result in subject_results
                        ]
                    )
                ),
                "std": float(
                    np.std(
                        [
                            result["metrics"]["per_class"][str(label)][metric]
                            for result in subject_results
                        ]
                    )
                ),
                "bootstrap_95ci": list(
                    _bootstrap_mean_ci(
                        np.asarray(
                            [
                                result["metrics"]["per_class"][str(label)][metric]
                                for result in subject_results
                            ]
                        ),
                        bootstrap_samples,
                        random_seed,
                    )
                ),
            }
            for metric in ("precision", "recall", "f1-score")
        }
        for label in labels
    }
    pooled_confusion = np.sum(
        [result["metrics"]["confusion_matrix"] for result in subject_results], axis=0
    ).astype(int)
    row_totals = pooled_confusion.sum(axis=1, keepdims=True)
    aggregate["pooled_confusion_matrix"] = pooled_confusion.tolist()
    aggregate["row_normalized_confusion_matrix"] = np.divide(
        pooled_confusion,
        row_totals,
        out=np.zeros_like(pooled_confusion, dtype=np.float64),
        where=row_totals != 0,
    ).tolist()
    aggregate["mean_train_macro_f1"] = float(
        np.mean([result["train_metrics"]["macro_f1"] for result in subject_results])
    )
    return aggregate


def _bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    indexes = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    means = values[indexes].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)
