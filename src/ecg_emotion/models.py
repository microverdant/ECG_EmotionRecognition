"""Model factories for CPU-friendly baseline experiments."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


class BalancedPriorLDA(LinearDiscriminantAnalysis):
    """LDA variant that assigns equal class priors at fit time."""

    def fit(self, features, labels):
        classes = np.unique(labels)
        self.priors = np.full(len(classes), 1.0 / len(classes))
        return super().fit(features, labels)


def build_logistic_regression(random_seed: int = 42) -> Pipeline:
    """Build an interpretable and fast multiclass baseline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def build_shrinkage_lda(random_seed: int = 42) -> Pipeline:
    """Build a low-variance discriminant model for small subject counts.

    Automatic covariance shrinkage stabilizes the class covariance estimate
    when the number of training participants is small relative to the feature
    dimension.  The seed is accepted for a consistent model-factory API.
    """

    del random_seed
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )


def build_balanced_shrinkage_lda(random_seed: int = 42) -> Pipeline:
    """Build a shrinkage LDA with equal class priors for balanced recall."""

    del random_seed
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                BalancedPriorLDA(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )


def build_random_forest(random_seed: int = 42) -> RandomForestClassifier:
    """Build a compact nonlinear baseline for tabular ECG features."""

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )


def build_rbf_svm(random_seed: int = 42) -> Pipeline:
    """Build a robust nonlinear baseline for compact ECG feature vectors."""

    return Pipeline(
        steps=[
            ("scaler", RobustScaler()),
            (
                "classifier",
                SVC(
                    C=3.0,
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    cache_size=1024,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def build_baseline(name: str, random_seed: int = 42):
    """Return a supported baseline model by name."""

    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"logistic", "logistic-regression", "lr"}:
        return build_logistic_regression(random_seed)
    if normalized in {"lda", "shrinkage-lda", "discriminant-analysis"}:
        return build_shrinkage_lda(random_seed)
    if normalized in {"balanced-lda", "balanced-shrinkage-lda", "class-balanced-lda"}:
        return build_balanced_shrinkage_lda(random_seed)
    if normalized in {"random-forest", "randomforest", "rf"}:
        return build_random_forest(random_seed)
    if normalized in {"svm", "rbf-svm", "svc"}:
        return build_rbf_svm(random_seed)
    raise ValueError(f"Unsupported baseline model: {name}")
