"""Synthetic ECG-like data for tests and local smoke runs.

This data is deliberately not a physiological dataset. It exists only to verify that the complete
pipeline can run without distributing participant data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import ECGDataset, save_npz


def generate_demo_dataset(
    output_path: str | Path,
    sample_rate: int = 128,
    seconds: int = 10,
    num_subjects: int = 6,
    windows_per_subject: int = 12,
    random_seed: int = 42,
) -> ECGDataset:
    """Generate a small balanced dataset with subject-specific variation."""

    rng = np.random.default_rng(random_seed)
    length = sample_rate * seconds
    time = np.arange(length, dtype=np.float64) / sample_rate
    class_rates = np.array([1.0, 1.25, 0.8, 1.45])
    signals: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []

    for subject_index in range(num_subjects):
        subject_scale = 0.9 + 0.04 * subject_index
        subject_offset = rng.normal(0, 0.03)
        for window_index in range(windows_per_subject):
            label = window_index % len(class_rates)
            frequency = class_rates[label] + rng.normal(0, 0.015)
            phase = rng.uniform(0, 2 * np.pi)
            pulse_train = np.zeros_like(time)
            period = 1.0 / frequency
            pulse_times = np.arange(-period, time[-1] + period, period)
            for pulse_time in pulse_times:
                pulse_train += np.exp(-0.5 * ((time - pulse_time) / 0.035) ** 2)
            signal = (
                subject_scale * pulse_train
                + 0.08 * np.sin(2 * np.pi * (0.15 + label * 0.04) * time + phase)
                + subject_offset
                + rng.normal(0, 0.04, size=length)
            )
            signals.append(signal.astype(np.float32))
            labels.append(label)
            subjects.append(f"S{subject_index + 1:02d}")

    dataset = ECGDataset(
        signals=np.asarray(signals, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects),
    )
    save_npz(output_path, dataset)
    return dataset

