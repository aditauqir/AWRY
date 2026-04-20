"""Model-driver helpers for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _latest_feature_row(df: pd.DataFrame) -> pd.Series:
    row = df.iloc[-1]
    if isinstance(row, pd.DataFrame):
        return row.iloc[-1]
    return row


def _extract_rf_importances(pipe) -> pd.Series | None:
    rf_pipe = getattr(pipe.now, "rf", None)
    if rf_pipe is None:
        return None
    model = getattr(rf_pipe, "named_steps", {}).get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    return pd.Series(model.feature_importances_, index=pipe.x_cols).sort_values(ascending=False)


def compute_driver_items(pipe, top_n: int = 6) -> list[tuple[str, float, str, str]]:
    """Return top feature drivers using RF importances as the fallback explainer."""
    importances = _extract_rf_importances(pipe)
    if importances is None or importances.empty:
        return []

    last = _latest_feature_row(pipe.df[pipe.x_cols])
    scored = (importances * last.abs().reindex(importances.index).fillna(0.0)).sort_values(ascending=False).head(top_n)
    if scored.empty:
        return []

    max_score = max(float(scored.max()), 1e-9)
    items: list[tuple[str, float, str, str]] = []
    for col, score in scored.items():
        value = float(last.get(col, 0.0))
        signal = "recession" if value >= 0 else "expansion"
        items.append((col, float(score) / max_score, f"{value:.3f}", signal))
    return items
