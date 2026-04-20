"""Horizontal bar panel for latest macro levels (Plotly)."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.styles.theme import COLORS


def indicator_bars_figure(items: list[tuple[str, float, str, str]]) -> go.Figure:
    """
    items: (label, value_norm_0_1, display_str, signal)
    signal: 'recession' | 'expansion' | 'neutral' — drives bar color and text label.
    """
    labels = [x[0] for x in items]

    vals = []
    colors = []
    text_lines = []
    for _, v, disp, sig in items:
        vals.append(min(1.0, max(0.0, abs(float(v)))))
        if sig == "recession":
            colors.append(COLORS["bar_red"])
            text_lines.append(f"{disp}  ·  Recession signal")
        elif sig == "expansion":
            colors.append(COLORS["bar_green"])
            text_lines.append(f"{disp}  ·  Expansion signal")
        else:
            colors.append("#64748b")
            text_lines.append(f"{disp}  ·  Neutral")

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=text_lines,
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=120, r=100, t=48, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], size=12),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", range=[0, 1.15], showticklabels=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        title=dict(
            text="What's driving the signal?",
            font=dict(size=16, color=COLORS["text"]),
            x=0,
            xanchor="left",
        ),
    )
    return fig
