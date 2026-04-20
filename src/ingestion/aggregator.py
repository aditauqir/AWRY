"""Daily series → monthly: NASDAQCOM log returns (mean), VIX level (mean)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_to_monthly_nasdaqcom_log_returns(daily_levels: pd.Series) -> pd.Series:
    """Daily equity index levels → monthly average of daily log returns."""
    s = daily_levels.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    logp = np.log(s.astype(float))
    daily_ret = logp.diff()
    monthly = daily_ret.resample("ME").mean()
    monthly.name = "NASDAQCOM_mret"
    return monthly


def daily_to_monthly_vix_mean(daily_vix: pd.Series) -> pd.Series:
    """VIX daily levels → monthly average level."""
    s = daily_vix.dropna().sort_index().astype(float)
    monthly = s.resample("ME").mean()
    monthly.name = "VIXCLS_mavg"
    return monthly


def align_monthly_to_end_index(monthly: pd.Series) -> pd.Series:
    """Normalize to month-end timestamps for merging with other ME series."""
    out = monthly.copy()
    out.index = out.index + pd.offsets.MonthEnd(0)
    return out.sort_index()
