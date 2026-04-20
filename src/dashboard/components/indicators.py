"""Feature table placeholder (SHAP-style contribution TBD)."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def show_indicator_table(df: pd.DataFrame, cols: list[str]) -> None:
    sub = df[cols].tail(1).T
    sub.columns = ["value"]
    st.dataframe(sub, use_container_width=True)
