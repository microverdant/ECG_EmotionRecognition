"""Dataset contracts and subject-level splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ECGDataset:
    """Prepared fixed-length ECG windows and their metadata."""

    signals: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray

    def validate(self, num_classes: int | None = None) -> None:
        """Validate shapes, finite values, and label ranges."""

        signals = np.asarray(self.signals)
        labels = np.asarray(self.labels)
        subjects = np.asarray(self.subjects)

        if signals.ndim != 2:
            raise ValueError("signals must have shape (n_samples, signal_length)")
        if labels.ndim != 1 or subjects.ndim != 1:
            raise ValueError("labels and subjects must be one-dimensional arrays")
        if not (len(signals) == len(labels) == len(subjects)):
            raise ValueError("signals, labels, and subjects must have the same number of rows")
        if len(signals) == 0:
            raise ValueError("the dataset must contain at least one window")
        if not np.isfinite(signals).all():
            raise ValueError("signals must contain only finite values")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be integer encoded")
        if num_classes is not None and ((labels < 0).any() or (labels >= num_classes).any()):
            raise ValueError("labels contain values outside the configured class range")
        if np.any(np.asarray([str(subject) for subject in subjects]) == ""):
            raise ValueError("subjects must not contain empty identifiers")

    @property
    def num_samples(self) -> int:
        return int(self.signals.shape[0])

    @property
    def signal_length(self) -> int:
        return int(self.signals.shape[1])

    @property
    def num_subjects(self) -> int:
        return int(np.unique(self.subjects).size)


def load_npz(path: str | Path, num_classes: int | None = None) -> ECGDataset:
    """Load a prepared dataset archive following the project data contract."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {"signals", "labels", "subjects"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
        dataset = ECGDataset(
            signals=np.asarray(archive["signals"], dtype=np.float32),
            labels=np.asarray(archive["labels"], dtype=np.int64),
            subjects=np.asarray(archive["subjects"]),
        )

    dataset.validate(num_classes=num_classes)
    return dataset


def save_npz(path: str | Path, dataset: ECGDataset) -> None:
    """Save a validated dataset archive."""

    dataset.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signals=np.asarray(dataset.signals, dtype=np.float32),
        labels=np.asarray(dataset.labels, dtype=np.int64),
        subjects=np.asarray(dataset.subjects),
    )


def split_by_subject(
    dataset: ECGDataset,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create deterministic train/validation/test indexes without subject leakage."""

    dataset.validate()
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1")

    unique_subjects = np.unique(dataset.subjects)
    if len(unique_subjects) < 3:
        raise ValueError("at least three subjects are required for a train/validation/test split")

    rng = np.random.default_rng(random_seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_size)))
    n_validation = max(1, int(round(len(shuffled) * validation_size)))
    if n_test + n_validation >= len(shuffled):
        n_validation = 1
        n_test = 1
    if n_test + n_validation >= len(shuffled):
        raise ValueError("split ratios leave no subject for training")

    test_subjects = set(shuffled[:n_test].tolist())
    validation_subjects = set(shuffled[n_test : n_test + n_validation].tolist())

    subject_values = np.asarray(dataset.subjects)
    test_mask = np.isin(subject_values, list(test_subjects))
    validation_mask = np.isin(subject_values, list(validation_subjects))
    train_mask = ~(test_mask | validation_mask)

    result = {
        "train": np.flatnonzero(train_mask),
        "validation": np.flatnonzero(validation_mask),
        "test": np.flatnonzero(test_mask),
    }
    if any(len(indexes) == 0 for indexes in result.values()):
        raise ValueError("one split is empty; provide more subjects or adjust split ratios")
    return result


