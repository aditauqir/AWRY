"""Shim: the training entry point lives in `awry_pipeline` (avoids clashes with a PyPI `pipeline` package)."""

from awry_pipeline import (  # noqa: F401
    AwryPipeline,
    HorizonModels,
    fit_awry_pipeline,
    predict_history,
    train_horizon,
)

__all__ = [
    "AwryPipeline",
    "HorizonModels",
    "fit_awry_pipeline",
    "predict_history",
    "train_horizon",
]
