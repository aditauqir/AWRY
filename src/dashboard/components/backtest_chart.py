"""Scenario window: AWRY vs Sahm-style gap vs yield-curve stress."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.styles.theme import COLORS


# COMMENT: OOF probabilities are more conservative than the old in-sample dashboard.
# Use a dedicated early-warning signal line for scenario lead charts instead of the
# stricter classification cutoff.
AWRY_BACKTEST_SIGNAL_THRESHOLD = 0.10

SCENARIOS = {
    "2008 GFC": ("2006-01-01", "2010-06-30", "2007-12-01"),
    "2020 COVID": ("2018-06-01", "2021-12-31", "2020-02-01"),
    "2001 Dot-com": ("1999-01-01", "2003-06-30", "2001-03-01"),
}


def scenario_comparison_figure(
    hist: pd.DataFrame,
    raw: pd.DataFrame,
    scenario: str,
) -> go.Figure:
    start, end, r0 = SCENARIOS[scenario]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    r0 = pd.Timestamp(r0)

    h = hist.loc[(hist.index >= start) & (hist.index <= end)].copy()
    if h.empty:
        return go.Figure()

    u = raw["UNRATE"].reindex(h.index).interpolate()
    t10 = raw["T10Y3M"].reindex(h.index).interpolate()
    sg = u.rolling(3, min_periods=1).mean() - u.rolling(12, min_periods=1).min()
    sg_norm = (sg / 1.5).clip(0, 1)
    yc = (-t10).clip(-2, 2).add(2).div(4).clip(0, 1)

    months_from_r0 = (h.index - r0).days / 30.44

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=months_from_r0,
            y=h["P_AWRY"].values * 100,
            name="AWRY",
            line=dict(color="#38bdf8", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months_from_r0,
            y=sg_norm.values * 100,
            name="Sahm gap (scaled)",
            line=dict(color="#fbbf24", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months_from_r0,
            y=yc.values * 100,
            name="Yield stress (−T10Y3M scaled)",
            line=dict(color="#a78bfa", width=2, dash="dot"),
        )
    )
    fig.add_vline(x=0, line_dash="solid", line_color="rgba(248,250,252,0.4)", annotation_text="R0")
    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        xaxis_title="Months vs approximate recession start (R0)",
        yaxis_title="Score (0–100)",
        yaxis_range=[0, 100],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50, l=50, r=30, b=50),
    )
    return fig


def _months_before_r0(ts: pd.Timestamp, r0: pd.Timestamp) -> int:
    return int(round((r0 - ts).days / 30.44))


def lead_months_awry(
    hist: pd.DataFrame,
    scenario: str,
    threshold: float = AWRY_BACKTEST_SIGNAL_THRESHOLD,
) -> int | None:
    """Months before R0 when AWRY first crosses the chosen decision threshold."""
    start, _, r0 = SCENARIOS[scenario]
    r0 = pd.Timestamp(r0)
    start = pd.Timestamp(start)
    h = hist.loc[(hist.index >= start) & (hist.index < r0)]
    if h.empty:
        return None
    cross = h[h["P_AWRY"] >= float(threshold)]
    if cross.empty:
        return None
    return _months_before_r0(cross.index[0], r0)


def lead_months_sahm(raw: pd.DataFrame, scenario: str) -> int | None:
    """First month Sahm gap ≥ 0.5 pp before R0."""
    start, _end, r0 = SCENARIOS[scenario]
    r0 = pd.Timestamp(r0)
    start = pd.Timestamp(start)
    w = raw.loc[(raw.index >= start) & (raw.index < r0)].copy()
    if w.empty or "UNRATE" not in w.columns:
        return None
    u = w["UNRATE"].astype(float)
    gap = u.rolling(3, min_periods=3).mean() - u.rolling(12, min_periods=12).min()
    cross = gap[gap >= 0.5].dropna()
    if cross.empty:
        return None
    return _months_before_r0(cross.index[0], r0)


def lead_months_yield(raw: pd.DataFrame, scenario: str) -> int | None:
    """First month T10Y3M < 0 (inverted) before R0."""
    start, _end, r0 = SCENARIOS[scenario]
    r0 = pd.Timestamp(r0)
    start = pd.Timestamp(start)
    w = raw.loc[(raw.index >= start) & (raw.index < r0)]
    if w.empty or "T10Y3M" not in w.columns:
        return None
    cross = w[w["T10Y3M"].astype(float) < 0.0]
    if cross.empty:
        return None
    return _months_before_r0(cross.index[0], r0)
