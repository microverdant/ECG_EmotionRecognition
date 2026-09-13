import numpy as np

from ecg_emotion.features import FEATURE_NAMES, detect_r_peaks, extract_features


def test_feature_extraction_returns_finite_matrix() -> None:
    sample_rate = 100
    time = np.arange(sample_rate * 10) / sample_rate
    signals = np.vstack(
        [np.sin(2 * np.pi * 1.2 * time), np.sin(2 * np.pi * 0.8 * time)]
    ).astype(np.float32)
    features = extract_features(signals, sample_rate=sample_rate)
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_r_peak_detector_finds_repeated_impulses() -> None:
    sample_rate = 100
    signal = np.zeros(sample_rate * 10, dtype=np.float32)
    signal[np.arange(sample_rate, len(signal), sample_rate)] = 2.0
    kernel = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)
    signal = np.convolve(signal, kernel, mode="same")
    peaks = detect_r_peaks(signal, sample_rate)
    assert 7 <= len(peaks) <= 10
