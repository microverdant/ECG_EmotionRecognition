"""Standard WESAD pickle adapter for the project's prepared-data contract."""

from __future__ import annotations

import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .data import ECGDataset, save_npz
from .preprocessing import clean_ecg, make_windows

WESAD_LABEL_MAPS = {
    "core": {
        1: 0,  # baseline
        2: 1,  # stress
        3: 2,  # amusement
    },
    "extended": {
        1: 0,  # baseline
        2: 1,  # stress
        3: 2,  # amusement
        4: 3,  # meditation
    },
}


def load_wesad_subject(
    path: str | Path,
    sample_rate: float = 700.0,
    window_seconds: float = 10.0,
    stride_seconds: float = 5.0,
    min_label_purity: float = 0.8,
    line_frequency: float | None = 50.0,
    target_sample_rate: float | None = None,
    label_set: str = "core",
) -> ECGDataset:
    """Load one standard WESAD subject pickle and return prepared ECG windows."""

    path = Path(path)
    with path.open("rb") as file:
        payload = pickle.load(file, encoding="latin1")

    try:
        signal = np.asarray(payload["signal"]["chest"]["ECG"], dtype=np.float64).reshape(-1)
        labels = np.asarray(payload["label"], dtype=np.int64).reshape(-1)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Expected WESAD payload keys: signal -> chest -> ECG and label"
        ) from error

    if len(signal) != len(labels):
        raise ValueError("WESAD ECG and label arrays must have the same length")

    cleaned = clean_ecg(signal, sample_rate=sample_rate, line_frequency=line_frequency)
    effective_sample_rate = sample_rate
    if target_sample_rate is not None and target_sample_rate != sample_rate:
        cleaned, labels = resample_signal_and_labels(
            cleaned,
            labels,
            source_sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
        )
        effective_sample_rate = target_sample_rate
    windows, raw_labels, subjects = make_windows(
        cleaned,
        window_size=int(round(window_seconds * effective_sample_rate)),
        stride=int(round(stride_seconds * effective_sample_rate)),
        subject_id=path.stem,
        labels=labels,
        min_label_purity=min_label_purity,
    )
    if raw_labels is None:
        raise RuntimeError("WESAD preparation unexpectedly produced unlabeled windows")

    label_map = _get_label_map(label_set)
    keep = np.isin(raw_labels, list(label_map))
    mapped_labels = np.asarray(
        [label_map[int(label)] for label in raw_labels[keep]],
        dtype=np.int64,
    )
    return ECGDataset(windows[keep], mapped_labels, subjects[keep])


def prepare_wesad(
    input_root: str | Path,
    output_path: str | Path,
    sample_rate: float = 700.0,
    window_seconds: float = 10.0,
    stride_seconds: float = 5.0,
    min_label_purity: float = 0.8,
    line_frequency: float | None = 50.0,
    excluded_subjects: set[str] | None = None,
    target_sample_rate: float | None = None,
    label_set: str = "core",
) -> ECGDataset:
    """Prepare all available WESAD subject pickle files into one compressed archive."""

    input_root = Path(input_root)
    subject_files = sorted(input_root.rglob("S*.pkl"))
    if not subject_files:
        raise FileNotFoundError(f"No WESAD subject pickle files found under {input_root}")

    excluded_subjects = excluded_subjects or set()
    datasets: list[ECGDataset] = []
    for subject_file in subject_files:
        if subject_file.stem in excluded_subjects:
            continue
        datasets.append(
            load_wesad_subject(
                subject_file,
                sample_rate=sample_rate,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
                min_label_purity=min_label_purity,
                line_frequency=line_frequency,
                target_sample_rate=target_sample_rate,
                label_set=label_set,
            )
        )

    if not datasets:
        raise ValueError("All discovered subjects were excluded")
    combined = ECGDataset(
        signals=np.concatenate([dataset.signals for dataset in datasets]),
        labels=np.concatenate([dataset.labels for dataset in datasets]),
        subjects=np.concatenate([dataset.subjects for dataset in datasets]),
    )
    save_npz(output_path, combined)
    return combined


def resample_signal_and_labels(
    signal: np.ndarray,
    labels: np.ndarray,
    source_sample_rate: float,
    target_sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ECG with anti-aliasing and align labels by nearest source sample."""

    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    ratio = Fraction(target_sample_rate / source_sample_rate).limit_denominator(1000)
    resampled_signal = resample_poly(signal, ratio.numerator, ratio.denominator).astype(np.float32)
    source_positions = np.arange(len(resampled_signal)) * source_sample_rate / target_sample_rate
    label_indexes = np.clip(np.rint(source_positions).astype(int), 0, len(labels) - 1)
    return resampled_signal, labels[label_indexes]


def _get_label_map(label_set: str) -> dict[int, int]:
    normalized = label_set.strip().lower()
    if normalized not in WESAD_LABEL_MAPS:
        raise ValueError(f"Unsupported WESAD label set: {label_set}")
    return WESAD_LABEL_MAPS[normalized]
