"""Model-driver helpers for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _latest_feature_row(df: pd.DataFrame) -> pd.Series:
    row = df.iloc[-1]
    if isinstance(row, pd.DataFrame):
        return row.iloc[-1]
    return row


def _rf_feature_names(pipe, rf_pipe, expected_len: int) -> list[str] | None:
    x_cols = list(getattr(pipe, "x_cols", []))
    imputer = getattr(rf_pipe, "named_steps", {}).get("impute")
    if imputer is not None and hasattr(imputer, "get_feature_names_out"):
        try:
            names = list(imputer.get_feature_names_out(x_cols))
            if len(names) == expected_len:
                return names
        except Exception:
            pass

    feature_names = getattr(rf_pipe, "feature_names_in_", None)
    if feature_names is not None and len(feature_names) == expected_len:
        return list(feature_names)

    if len(x_cols) == expected_len:
        return x_cols
    return None


def _extract_rf_importances(pipe) -> pd.Series | None:
    rf_pipe = getattr(pipe.now, "rf", None)
    if rf_pipe is None:
        return None
    model = getattr(rf_pipe, "named_steps", {}).get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    feature_names = _rf_feature_names(pipe, rf_pipe, len(model.feature_importances_))
    if feature_names is None:
        return None
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


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
