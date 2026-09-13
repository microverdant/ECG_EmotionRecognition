from pathlib import Path

import numpy as np

from ecg_emotion.synthetic import generate_demo_dataset
from ecg_emotion.training import train_feature_baseline
from ecg_emotion.wesad import load_wesad_subject


def test_feature_baseline_saves_reproducible_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo.npz"
    output_dir = tmp_path / "artifacts"
    generate_demo_dataset(
        dataset_path,
        sample_rate=64,
        seconds=4,
        num_subjects=6,
        windows_per_subject=8,
        random_seed=3,
    )
    result = train_feature_baseline(
        dataset_path,
        output_dir,
        sample_rate=64,
        random_seed=3,
    )
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert result["split_subjects"]["train"]
    assert 0.0 <= result["metrics"]["test"]["macro_f1"] <= 1.0


def test_wesad_adapter_maps_supported_labels(tmp_path: Path) -> None:
    import pickle

    sample_rate = 100
    signal = np.sin(2 * np.pi * 1.2 * np.arange(sample_rate * 6) / sample_rate)
    labels = np.full(len(signal), 1, dtype=int)
    payload = {"signal": {"chest": {"ECG": signal}}, "label": labels}
    subject_path = tmp_path / "S99.pkl"
    with subject_path.open("wb") as file:
        pickle.dump(payload, file)

    dataset = load_wesad_subject(
        subject_path,
        sample_rate=sample_rate,
        window_seconds=2,
        stride_seconds=2,
        line_frequency=None,
    )
    assert dataset.labels.tolist() == [0, 0, 0]
    assert dataset.subjects.tolist() == ["S99", "S99", "S99"]
