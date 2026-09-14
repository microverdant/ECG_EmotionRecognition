from pathlib import Path

from ecg_emotion.robustness import run_loso_audit
from ecg_emotion.synthetic import generate_demo_dataset


def test_loso_audit_reports_ablation_and_bootstrap_ci(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo.npz"
    output_dir = tmp_path / "robustness"
    generate_demo_dataset(
        dataset_path,
        sample_rate=64,
        seconds=4,
        num_subjects=6,
        windows_per_subject=4,
        random_seed=6,
    )
    result = run_loso_audit(
        dataset_path,
        output_dir,
        sample_rate=64,
        bootstrap_samples=50,
        random_seed=6,
    )
    assert result["split_strategy"] == "Leave-One-Subject-Out"
    assert set(result["feature_sets"]) == {
        "all_features",
        "rate_hrv",
        "morphology_spectral",
        "without_amplitude",
    }
    ci = result["feature_sets"]["all_features"]["aggregate"]["macro_f1"]["bootstrap_95ci"]
    assert len(ci) == 2
    assert ci[0] <= ci[1]
    aggregate = result["feature_sets"]["all_features"]["aggregate"]
    assert set(aggregate["per_class"]) == {"0", "1", "2", "3"}
    assert len(aggregate["pooled_confusion_matrix"]) == 4
    assert len(aggregate["row_normalized_confusion_matrix"]) == 4
    assert (output_dir / "loso_robustness.json").exists()
