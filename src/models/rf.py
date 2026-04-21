"""Regularized random-forest builders for AWRY."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from models.common import ConstantProbabilityModel


def build_rf() -> Pipeline:
    """Build the capped random-forest pipeline."""
    # n=406 with ~7% positive class means the forest needs a hard cap
    # on depth and leaf size or it will memorize the recession months.
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=10,
                    min_samples_split=20,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_rf(X, y) -> Pipeline:
    """Fit the capped random-forest pipeline."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return ConstantProbabilityModel(float(y.mean()))
    model = build_rf()
    try:
        model.fit(X, y)
        return model
    except PermissionError:
        # Some Windows sandbox environments block joblib worker creation.
        # Retry single-threaded so evaluation remains reproducible instead of failing.
        single = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=10,
                        min_samples_split=20,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        single.fit(X, y)
        return single
