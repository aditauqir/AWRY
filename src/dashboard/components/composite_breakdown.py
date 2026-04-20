"""Stacked bar: contributions α·P_now and (1−α)·P_3m to P_AWRY."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.styles.theme import COLORS


def composite_breakdown_figure(
    p_now: float,
    p_3m: float,
    alpha: float,
    subtitle: str = "",
) -> go.Figure:
    """Horizontal stacked bar; total width equals P_AWRY."""
    a = float(alpha)
    p_now = max(0.0, min(1.0, float(p_now)))
    p_3m = max(0.0, min(1.0, float(p_3m)))
    c_now = a * p_now
    c_3m = (1.0 - a) * p_3m
    p_awry = c_now + c_3m

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=f"w·P_now ({a:.0%} on nowcast)",
            y=["P_AWRY"],
            x=[c_now],
            orientation="h",
            marker_color=COLORS["bar_green"],
            hovertemplate="w·P_now = %{x:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name=f"(1−w)·P_3m ({(1.0 - a):.0%} on 3m)",
            y=["P_AWRY"],
            x=[c_3m],
            orientation="h",
            base=[c_now],
            marker_color=COLORS["bar_orange"],
            hovertemplate="(1−w)·P_3m = %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        height=220,
        margin=dict(l=16, r=16, t=56 if subtitle else 40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"], "size": 12},
        xaxis={
            "title": {"text": ""},
            "range": [0, 1],
            "tickformat": ".0%",
            "gridcolor": "rgba(148,163,184,0.2)",
            "zeroline": False,
        },
        yaxis={"showticklabels": False},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title={
            "text": (
                f"w·P_now + (1−w)·P_3m = {p_awry:.1%}"
                + (f"<br><sub style='color:#94a3b8'>{subtitle}</sub>" if subtitle else "")
            ),
            "font": {"size": 14, "color": COLORS["text"]},
        },
        annotations=[
            dict(
                x=p_awry,
                y="P_AWRY",
                text=f" {p_awry:.1%}",
                showarrow=False,
                xanchor="left",
                font=dict(color=COLORS["text"], size=13),
            )
        ],
    )
    return fig
