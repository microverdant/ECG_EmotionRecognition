"""Fast, interpretable ECG and short-window HRV features."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

FEATURE_NAMES = (
    "signal_std",
    "signal_rms",
    "signal_range",
    "signal_line_length",
    "signal_skewness",
    "signal_kurtosis",
    "r_peak_count",
    "heart_rate_mean",
    "heart_rate_std",
    "rr_mean",
    "rr_median",
    "rr_std",
    "rr_iqr",
    "rr_rmssd",
    "rr_pnn50",
    "r_amplitude_mean",
    "r_amplitude_std",
    "qrs_band_power",
    "high_band_power",
    "spectral_entropy",
    "hrv_lf_power",
    "hrv_hf_power",
    "hrv_lf_hf_ratio",
)


def extract_features(signals: np.ndarray, sample_rate: float = 256.0) -> np.ndarray:
    """Extract a compact domain-informed feature matrix from ECG windows."""

    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2:
        raise ValueError("signals must have shape (n_samples, signal_length)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    matrix = np.zeros((len(signals), len(FEATURE_NAMES)), dtype=np.float32)
    for row, signal in enumerate(signals):
        matrix[row] = _extract_single(signal, sample_rate)
    return matrix


def detect_r_peaks(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """Detect R-peak locations using an energy envelope and local refinement."""

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(signal) < max(16, int(sample_rate)):
        return np.empty(0, dtype=np.int64)

    high_cut = min(20.0, sample_rate / 2 - 1.0)
    if high_cut <= 5.0:
        return np.empty(0, dtype=np.int64)
    sos = butter(2, [5.0, high_cut], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, signal)
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = derivative**2
    integration_width = max(1, int(round(0.12 * sample_rate)))
    envelope = np.convolve(
        energy,
        np.ones(integration_width, dtype=np.float64) / integration_width,
        mode="same",
    )
    threshold = float(np.median(envelope) + 0.5 * np.std(envelope))
    candidates, _ = find_peaks(
        envelope,
        distance=max(1, int(round(0.3 * sample_rate))),
        height=threshold,
    )

    search_radius = max(1, int(round(0.12 * sample_rate)))
    refined: list[int] = []
    for candidate in candidates:
        start = max(0, candidate - search_radius)
        end = min(len(filtered), candidate + search_radius + 1)
        refined.append(start + int(np.argmax(np.abs(filtered[start:end]))))
    return np.asarray(sorted(set(refined)), dtype=np.int64)


def _extract_single(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    centered = signal - np.median(signal)
    peaks = detect_r_peaks(centered, sample_rate)
    rr = np.diff(peaks) / sample_rate
    heart_rate = 60.0 / rr if len(rr) else np.empty(0)

    freqs, power = periodogram(centered, fs=sample_rate)
    total_power = float(np.sum(power))
    normalized_power = power / total_power if total_power > 0 else np.zeros_like(power)
    positive_power = normalized_power[normalized_power > 0]
    spectral_entropy = (
        float(-np.sum(positive_power * np.log(positive_power))) if len(positive_power) else 0.0
    )

    def signal_band_power(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(trapezoid(power[mask], freqs[mask])) if np.count_nonzero(mask) > 1 else 0.0

    lf_power, hf_power = _hrv_frequency_features(peaks, rr, sample_rate)
    amplitudes = centered[peaks] if len(peaks) else np.empty(0)
    values = np.array(
        [
            np.std(centered),
            np.sqrt(np.mean(centered**2)),
            np.ptp(centered),
            np.mean(np.abs(np.diff(centered))) if len(centered) > 1 else 0.0,
            skew(centered, bias=False),
            kurtosis(centered, bias=False),
            len(peaks),
            np.mean(heart_rate) if len(heart_rate) else 0.0,
            np.std(heart_rate) if len(heart_rate) else 0.0,
            np.mean(rr) if len(rr) else 0.0,
            np.median(rr) if len(rr) else 0.0,
            np.std(rr) if len(rr) else 0.0,
            np.subtract(*np.percentile(rr, [75, 25])) if len(rr) else 0.0,
            np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0.0,
            np.mean(np.abs(np.diff(rr)) > 0.05) if len(rr) > 1 else 0.0,
            np.mean(amplitudes) if len(amplitudes) else 0.0,
            np.std(amplitudes) if len(amplitudes) else 0.0,
            signal_band_power(5.0, 15.0),
            signal_band_power(15.0, 40.0),
            spectral_entropy,
            lf_power,
            hf_power,
            lf_power / max(hf_power, 1e-8),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _hrv_frequency_features(
    peaks: np.ndarray,
    rr_intervals: np.ndarray,
    sample_rate: float,
) -> tuple[float, float]:
    if len(rr_intervals) < 6:
        return 0.0, 0.0

    beat_times = peaks[1:] / sample_rate
    interpolation_rate = 4.0
    interpolation_times = np.arange(beat_times[0], beat_times[-1], 1.0 / interpolation_rate)
    if len(interpolation_times) < 8:
        return 0.0, 0.0
    tachogram = np.interp(interpolation_times, beat_times, rr_intervals)
    tachogram -= np.mean(tachogram)
    frequencies, power = welch(
        tachogram,
        fs=interpolation_rate,
        nperseg=min(256, len(tachogram)),
    )

    def band_power(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)
        if np.count_nonzero(mask) <= 1:
            return 0.0
        return float(trapezoid(power[mask], frequencies[mask]))

    return band_power(0.04, 0.15), band_power(0.15, 0.4)

