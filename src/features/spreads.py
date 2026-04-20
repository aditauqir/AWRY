"""Derived spreads — T10Y3M and BAA10Y are typically pulled directly from FRED."""

from __future__ import annotations

import pandas as pd


def ensure_spread_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through; series T10Y3M and BAA10Y are FRED levels/spreads."""
    return df
