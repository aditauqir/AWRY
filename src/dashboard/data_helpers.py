"""Cached raw monthly panel for display (levels, not transformed features)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from features.dataset_builder import build_raw_monthly_panel
from features.equity_config import DEFAULT_EQUITY_SERIES
from ingestion.fred_client import FredClient


@st.cache_data(ttl=3600, show_spinner=False)
def raw_monthly_panel(equity_series: str = DEFAULT_EQUITY_SERIES) -> pd.DataFrame:
    return build_raw_monthly_panel(FredClient(), cached=True, equity_series=equity_series)
