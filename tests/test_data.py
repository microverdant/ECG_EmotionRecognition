import numpy as np
import pytest

from ecg_emotion.data import ECGDataset, split_by_subject


def make_dataset() -> ECGDataset:
    signals = np.ones((12, 32), dtype=np.float32)
    labels = np.arange(12, dtype=np.int64) % 4
    subjects = np.repeat(["S01", "S02", "S03", "S04"], 3)
    return ECGDataset(signals, labels, subjects)


def test_subject_split_has_no_overlap() -> None:
    dataset = make_dataset()
    splits = split_by_subject(dataset, test_size=0.25, validation_size=0.25, random_seed=7)
    subject_sets = {
        name: set(dataset.subjects[indexes].tolist()) for name, indexes in splits.items()
    }
    assert subject_sets["train"].isdisjoint(subject_sets["validation"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["validation"].isdisjoint(subject_sets["test"])


def test_dataset_rejects_mismatched_rows() -> None:
    dataset = ECGDataset(np.ones((2, 4)), np.zeros(1, dtype=int), np.array(["S01", "S02"]))
    with pytest.raises(ValueError, match="same number of rows"):
        dataset.validate()

