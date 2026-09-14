"""Subject-domain robustness and feature ablation experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_npz
from .evaluation import classification_metrics, probability_metrics, save_json
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
