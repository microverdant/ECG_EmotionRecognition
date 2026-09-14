import numpy as np

from ecg_emotion.evaluation import (
    apply_temperature,
    fit_temperature,
    probability_metrics,
)


def test_probability_metrics_and_temperature_scaling_are_finite() -> None:
    labels = np.array([0, 1, 2, 1])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.4, 0.5, 0.1],
        ]
    )
    metrics = probability_metrics(labels, probabilities)
    temperature = fit_temperature(labels, probabilities)
    calibrated = apply_temperature(probabilities, temperature)

    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
    assert temperature > 0.0
    assert np.isfinite(calibrated).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)
