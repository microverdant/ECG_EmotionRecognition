"""Fast, interpretable features for ECG baseline models."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, periodogram

FEATURE_NAMES = (
    "mean",
    "std",
    "rms",
    "range",
    "energy",
    "peak_count",
    "peak_rate",
    "rr_mean",
    "rr_std",
    "rr_rmssd",
    "low_band_power",
    "mid_band_power",
    "high_band_power",
)


def extract_features(signals: np.ndarray, sample_rate: float = 256.0) -> np.ndarray:
    """Extract a compact feature matrix from fixed-length ECG windows."""

    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2:
        raise ValueError("signals must have shape (n_samples, signal_length)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    matrix = np.zeros((len(signals), len(FEATURE_NAMES)), dtype=np.float32)
    for row, signal in enumerate(signals):
        matrix[row] = _extract_single(signal, sample_rate)
    return matrix


def _extract_single(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(np.std(signal))
    peaks, _ = find_peaks(
        signal,
        distance=max(1, int(sample_rate * 0.25)),
        prominence=max(std * 0.3, 1e-6),
    )
    rr = np.diff(peaks) / sample_rate
    freqs, power = periodogram(signal, fs=sample_rate)

    def band_power(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(trapezoid(power[mask], freqs[mask])) if mask.any() else 0.0

    values = np.array(
        [
            np.mean(signal),
            std,
            np.sqrt(np.mean(signal**2)),
            np.ptp(signal),
            np.mean(signal**2),
            len(peaks),
            len(peaks) / (len(signal) / sample_rate),
            np.mean(rr) if len(rr) else 0.0,
            np.std(rr) if len(rr) else 0.0,
            np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0.0,
            band_power(0.04, 0.15),
            band_power(0.15, 0.4),
            band_power(0.4, 1.0),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
