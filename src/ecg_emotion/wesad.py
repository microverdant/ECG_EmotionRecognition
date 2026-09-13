"""Standard WESAD pickle adapter for the project's prepared-data contract."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .data import ECGDataset, save_npz
from .preprocessing import clean_ecg, make_windows

WESAD_LABEL_MAP = {
    1: 0,  # baseline
    2: 1,  # stress
    3: 2,  # amusement
    4: 3,  # meditation
}


def load_wesad_subject(
    path: str | Path,
    sample_rate: float = 700.0,
    window_seconds: float = 10.0,
    stride_seconds: float = 5.0,
    min_label_purity: float = 0.8,
    line_frequency: float | None = 50.0,
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
    windows, raw_labels, subjects = make_windows(
        cleaned,
        window_size=int(round(window_seconds * sample_rate)),
        stride=int(round(stride_seconds * sample_rate)),
        subject_id=path.stem,
        labels=labels,
        min_label_purity=min_label_purity,
    )
    if raw_labels is None:
        raise RuntimeError("WESAD preparation unexpectedly produced unlabeled windows")

    keep = np.isin(raw_labels, list(WESAD_LABEL_MAP))
    mapped_labels = np.asarray(
        [WESAD_LABEL_MAP[int(label)] for label in raw_labels[keep]],
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
