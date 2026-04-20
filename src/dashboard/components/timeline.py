"""Historical P_AWRY with NBER shading."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.styles.theme import COLORS


def probability_timeline(
    dates,
    p_series,
    recession: list | None = None,
    mode: str = "composite",
    *,
    show_signal_threshold: bool = False,
    signal_threshold: float = 0.30,
) -> go.Figure:
    fig = go.Figure()
    if recession is not None:
        in_rec = [bool(x) for x in recession]
        # shade recession months
        i = 0
        while i < len(dates):
            if in_rec[i]:
                j = i
                while j < len(dates) and in_rec[j]:
                    j += 1
                fig.add_vrect(
                    x0=dates[i],
                    x1=dates[j - 1],
                    fillcolor="rgba(239,68,68,0.18)",
                    line_width=0,
                )
                i = j
            else:
                i += 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=p_series,
            mode="lines",
            name=mode,
            line=dict(color=COLORS["accent"], width=2.5),
        )
    )
    if show_signal_threshold:
        fig.add_hline(
            y=signal_threshold,
            line_dash="dash",
            line_color="rgba(248, 250, 252, 0.55)",
            line_width=1.5,
            annotation_text=f"Signal threshold ({signal_threshold:.0%})",
            annotation_position="right",
            annotation_font_color=COLORS["text_muted"],
            annotation_font_size=11,
        )
    fig.update_layout(
        height=440,
        title=dict(text="Historical AWRY probability", font=dict(size=15), x=0),
        xaxis_title="Date",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        yaxis_tickformat=".0%",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS.get("card", "#111827"),
        font={"color": COLORS["text"]},
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50, l=50, r=30, b=50),
    )
    return fig
