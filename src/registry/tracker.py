"""MLflow logging helpers."""

from __future__ import annotations

from typing import Any

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore


def log_metrics(metrics: dict[str, float], run_name: str | None = None) -> None:
    if mlflow is None:
        return
    if run_name:
        mlflow.set_tag("mlflow.runName", run_name)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def log_params(params: dict[str, Any]) -> None:
    if mlflow is None:
        return
    mlflow.log_params(params)
