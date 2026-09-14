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


def predict_baseline(
    bundle: dict,
    signal: np.ndarray,
    confidence_threshold: float | None = None,
) -> dict:
    """Return a label, probabilities, and a confidence-aware decision state."""

    signal = np.asarray(signal, dtype=np.float32).reshape(1, -1)
    features = extract_features(signal, sample_rate=float(bundle["sample_rate"]))
    model = bundle["model"]
    probabilities = model.predict_proba(features)[0]
    class_ids = model.classes_.astype(int).tolist()
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    threshold = float(bundle.get("confidence_threshold", 0.8))
    if confidence_threshold is not None:
        threshold = float(confidence_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("confidence_threshold must be in the interval (0, 1]")
    return {
        "label_id": class_ids[best_index],
        "label": str(class_ids[best_index]),
        "confidence": confidence,
        "confidence_threshold": threshold,
        "accepted": confidence >= threshold,
        "probabilities": {
            str(class_id): float(probability)
            for class_id, probability in zip(class_ids, probabilities, strict=True)
        },
    }
