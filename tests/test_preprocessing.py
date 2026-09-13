import numpy as np
import pytest

from ecg_emotion.preprocessing import make_windows, normalize_windows, zscore


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

