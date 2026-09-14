import numpy as np

from ecg_emotion.features import FEATURE_NAMES
from ecg_emotion.inference import predict_baseline
from ecg_emotion.models import build_baseline


def test_baseline_inference_exposes_confidence_state() -> None:
    rng = np.random.default_rng(5)
    model = build_baseline("shrinkage-lda", random_seed=5)
    model.fit(rng.normal(size=(30, len(FEATURE_NAMES))), np.repeat([0, 1, 2], 10))
    result = predict_baseline(
        {
            "model": model,
            "sample_rate": 64,
            "confidence_threshold": 0.8,
        },
        rng.normal(size=256),
    )
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["accepted"], bool)
    assert np.isclose(sum(result["probabilities"].values()), 1.0)
