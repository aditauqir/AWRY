"""Ablation comparison view for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def scenario_panel(summary: pd.DataFrame) -> None:
    """Render baseline vs stress-feature ablation metrics side by side."""
    if summary.empty:
        st.info("Ablation results are not available yet. Run the evaluation suite to populate them.")
        return

    st.markdown("### Baseline vs stress features")
    show = summary.copy()
    rename = {
        "ablation": "Set",
        "feature_set": "Features",
        "auroc": "AUROC",
        "brier": "Brier",
        "f1": "F1",
        "lead_2001": "Lead 2001",
        "lead_2008": "Lead 2008",
        "lead_2020": "Lead 2020",
    }
    show = show.rename(columns=rename)
    st.dataframe(show, use_container_width=True)
