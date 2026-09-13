from pathlib import Path

from ecg_emotion.synthetic import generate_demo_dataset
from ecg_emotion.training import train_feature_baseline


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

