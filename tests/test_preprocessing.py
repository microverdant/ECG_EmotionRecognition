import numpy as np
import pytest

from ecg_emotion.preprocessing import (
    clean_ecg,
    make_windows,
    normalize_windows,
    notch_filter,
    zscore,
)


def test_make_windows_filters_transition_windows() -> None:
    signal = np.arange(20, dtype=np.float32)
    labels = np.array([0] * 10 + [1] * 10)
    windows, window_labels, subjects = make_windows(
        signal,
        window_size=10,
        stride=5,
        subject_id="S01",
        labels=labels,
        min_label_purity=0.8,
    )
    assert windows.shape == (2, 10)
    assert window_labels.tolist() == [0, 1]
    assert subjects.tolist() == ["S01", "S01"]


def test_normalize_windows_handles_constant_signal() -> None:
    normalized = normalize_windows(np.ones((2, 8), dtype=np.float32))
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized, 0.0)


def test_zscore_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        zscore(np.array([]))


def test_notch_filter_preserves_signal_shape_and_finite_values() -> None:
    sample_rate = 256
    time = np.arange(sample_rate * 4) / sample_rate
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.2 * np.sin(2 * np.pi * 50 * time)
    filtered = notch_filter(signal, sample_rate=sample_rate, line_frequency=50.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_clean_ecg_does_not_normalize_recording_by_default() -> None:
    sample_rate = 100
    time = np.arange(sample_rate * 4) / sample_rate
    signal = 5.0 + np.sin(2 * np.pi * 1.2 * time)
    cleaned = clean_ecg(signal, sample_rate=sample_rate, line_frequency=None)
    assert cleaned.shape == signal.shape
    assert not np.isclose(float(cleaned.mean()), 0.0, atol=1e-3)
