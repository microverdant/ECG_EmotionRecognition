"""Small inference helpers used by the local showcase demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import extract_features


def load_baseline(path: str | Path) -> dict:
    """Load a serialized feature-baseline bundle."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return joblib.load(path)


def predict_baseline(bundle: dict, signal: np.ndarray) -> dict:
    """Return a label and probabilities for one ECG signal window."""

    signal = np.asarray(signal, dtype=np.float32).reshape(1, -1)
    features = extract_features(signal, sample_rate=float(bundle["sample_rate"]))
    model = bundle["model"]
    probabilities = model.predict_proba(features)[0]
    class_ids = model.classes_.astype(int).tolist()
    best_index = int(np.argmax(probabilities))
    return {
        "label_id": class_ids[best_index],
        "label": str(class_ids[best_index]),
        "probabilities": {
            str(class_id): float(probability)
            for class_id, probability in zip(class_ids, probabilities, strict=True)
        },
    }

