import numpy as np

from ecg_emotion.models import build_baseline


def test_shrinkage_lda_factory_produces_multiclass_predictions() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(30, 6))
    labels = np.repeat([0, 1, 2], 10)
    model = build_baseline("shrinkage-lda", random_seed=4)
    model.fit(features, labels)
    assert model.predict(features).shape == labels.shape
    assert np.isfinite(model.predict_proba(features)).all()


def test_balanced_shrinkage_lda_factory_produces_multiclass_predictions() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(30, 6))
    labels = np.repeat([0, 1, 2], 10)
    model = build_baseline("balanced-shrinkage-lda", random_seed=9)
    model.fit(features, labels)
    assert model.predict(features).shape == labels.shape
    assert np.allclose(model.named_steps["classifier"].priors, [1 / 3] * 3)
