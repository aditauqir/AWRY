"""Cached raw monthly panel for display (levels, not transformed features)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from config import MODELS_DIR
from features.dataset_builder import build_raw_monthly_panel
from features.equity_config import DEFAULT_EQUITY_SERIES
from ingestion.fred_client import FredClient


@st.cache_data(ttl=3600, show_spinner=False)
def raw_monthly_panel(equity_series: str = DEFAULT_EQUITY_SERIES) -> pd.DataFrame:
    return build_raw_monthly_panel(FredClient(), cached=True, equity_series=equity_series)


@st.cache_data(ttl=3600, show_spinner=False)
def load_json_artifact(name: str) -> dict:
    path = MODELS_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_ablation_summary() -> pd.DataFrame:
    path = MODELS_DIR / "ablation_summary.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)
