"""Signal cleaning and leakage-safe windowing helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    low_cut: float = 0.5,
    high_cut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to a one-dimensional ECG."""

    signal = _as_signal(signal)
    if not 0 < low_cut < high_cut < sample_rate / 2:
        raise ValueError("cutoff frequencies must satisfy 0 < low < high < Nyquist")
    sos = butter(order, [low_cut, high_cut], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def notch_filter(
    signal: np.ndarray,
    sample_rate: float,
    line_frequency: float = 50.0,
) -> np.ndarray:
    """Remove a power-line frequency using a zero-phase notch filter."""

    signal = _as_signal(signal)
    if not 0 < line_frequency < sample_rate / 2:
        raise ValueError("line_frequency must be below the Nyquist frequency")
    b, a = iirnotch(line_frequency, Q=30.0, fs=sample_rate)
    return filtfilt(b, a, signal).astype(np.float32)


def clean_ecg(
    signal: np.ndarray,
    sample_rate: float,
    line_frequency: float | None = 50.0,
) -> np.ndarray:
    """Apply the default ECG cleaning chain."""

    cleaned = bandpass_filter(signal, sample_rate)
    if line_frequency is not None:
        cleaned = notch_filter(cleaned, sample_rate, line_frequency)
    return zscore(cleaned)


def zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a signal without introducing NaNs for constant input."""

    signal = _as_signal(signal)
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return ((signal - mean) / max(std, eps)).astype(np.float32)


def normalize_windows(signals: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply per-window z-score normalization to a batch of ECG windows."""

    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError("signals must have shape (n_samples, signal_length)")
    means = signals.mean(axis=1, keepdims=True)
    stds = signals.std(axis=1, keepdims=True)
    return ((signals - means) / np.maximum(stds, eps)).astype(np.float32)


def make_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    subject_id: str | int,
    labels: Iterable[int] | None = None,
    min_label_purity: float = 0.8,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Create fixed-size windows and optionally assign majority labels.

    Labels are assigned by majority vote. Windows crossing a label transition are discarded when
    the majority label does not meet ``min_label_purity``.
    """

    signal = _as_signal(signal)
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if not 0 < min_label_purity <= 1:
        raise ValueError("min_label_purity must be in the interval (0, 1]")

    label_array = None if labels is None else np.asarray(list(labels))
    if label_array is not None and len(label_array) != len(signal):
        raise ValueError("labels must have the same length as signal")

    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    subjects: list[str | int] = []
    for start in range(0, len(signal) - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[start:end])
        subjects.append(subject_id)
        if label_array is not None:
            values, counts = np.unique(label_array[start:end], return_counts=True)
            majority_index = int(np.argmax(counts))
            purity = float(counts[majority_index] / window_size)
            if purity < min_label_purity:
                windows.pop()
                subjects.pop()
                continue
            window_labels.append(int(values[majority_index]))

    if not windows:
        empty = np.empty((0, window_size), dtype=np.float32)
        empty_subjects = np.empty((0,), dtype=str)
        empty_labels = None if label_array is None else np.empty((0,), dtype=np.int64)
        return empty, empty_labels, empty_subjects

    return (
        np.asarray(windows, dtype=np.float32),
        None if label_array is None else np.asarray(window_labels, dtype=np.int64),
        np.asarray(subjects),
    )


def _as_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")
    return signal
