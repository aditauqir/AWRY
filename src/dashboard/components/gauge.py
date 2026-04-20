"""Semicircular AWRY risk gauge (Plotly)."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.styles.theme import COLORS


def awry_gauge_figure(p: float) -> go.Figure:
    p = max(0.0, min(1.0, float(p)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "AWRY score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["yellow"]},
                "steps": [
                    {"range": [0, 30], "color": COLORS["green"]},
                    {"range": [30, 60], "color": COLORS["yellow"]},
                    {"range": [60, 100], "color": COLORS["red"]},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.8,
                    "value": p * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
    )
    return fig
