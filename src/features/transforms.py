"""Log-differencing and stationarity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_diff(s: pd.Series) -> pd.Series:
    """g_t = ln(X_t) - ln(X_{t-1})."""
    x = s.astype(float).clip(lower=1e-12)
    return np.log(x).diff()


def apply_log_diff_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = log_diff(out[c])
    return out
