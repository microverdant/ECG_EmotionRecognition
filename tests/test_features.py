import numpy as np

from ecg_emotion.features import FEATURE_NAMES, extract_features


def test_feature_extraction_returns_finite_matrix() -> None:
    sample_rate = 100
    time = np.arange(sample_rate * 10) / sample_rate
    signals = np.vstack(
        [np.sin(2 * np.pi * 1.2 * time), np.sin(2 * np.pi * 0.8 * time)]
    ).astype(np.float32)
    features = extract_features(signals, sample_rate=sample_rate)
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()

