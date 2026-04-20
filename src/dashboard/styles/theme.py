"""Plotly / Streamlit theme — AWRY color zones."""

ZONES = {
    "green": (0.0, 0.30),
    "yellow": (0.30, 0.60),
    "red": (0.60, 1.0),
}

COLORS = {
    "green": "#22c55e",
    "yellow": "#eab308",
    "red": "#ef4444",
    "bg": "#0f172a",
    "bg_app": "#0a0a0a",
    "card": "#111827",
    "card_border": "#1f2937",
    "text": "#f8fafc",
    "text_muted": "#94a3b8",
    "accent": "#38bdf8",
    "bar_green": "#22c55e",
    "bar_orange": "#f97316",
    "bar_red": "#ef4444",
    "accent": "#38bdf8",
}

DASHBOARD_CSS = """
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #0f172a 100%);
    }
    [data-testid="stHeader"] { background-color: transparent; }
    [data-testid="stToolbar"] { background-color: transparent; }
    /* Default metric cards (diagnostics, etc.) */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.75rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricDelta"] { color: #94a3b8 !important; }
    /* KPI row — custom cards */
    .awry-kpi-card {
        border-radius: 14px;
        padding: 1.1rem 1.25rem 1.25rem 1.25rem;
        border: 1px solid;
        min-height: 7.5rem;
    }
    .awry-kpi-card.kpi-neutral {
        background: linear-gradient(160deg, #1e293b 0%, #111827 55%);
        border-color: #334155;
    }
    .awry-kpi-card.kpi-blue {
        background: linear-gradient(160deg, rgba(56, 189, 248, 0.14) 0%, #111827 50%);
        border-color: rgba(56, 189, 248, 0.45);
    }
    .awry-kpi-card.kpi-amber {
        background: linear-gradient(160deg, rgba(245, 158, 11, 0.14) 0%, #111827 50%);
        border-color: rgba(245, 158, 11, 0.45);
    }
    .awry-kpi-card .kpi-label {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        letter-spacing: 0.01em;
    }
    .awry-kpi-card .kpi-value {
        color: #f8fafc !important;
        font-size: 2.15rem !important;
        font-weight: 800 !important;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .awry-kpi-card .kpi-delta {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }
    .awry-kpi-card .kpi-interpret {
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.65rem;
        padding-top: 0.55rem;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
    }
    .awry-kpi-card .kpi-hint {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 0.5rem;
        line-height: 1.35;
    }
    /* Backtest lead — large prominence */
    .backtest-signal-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.85rem;
        margin-top: 0.75rem;
        justify-content: stretch;
    }
    .backtest-signal-card {
        flex: 1 1 28%;
        min-width: 140px;
        text-align: center;
        padding: 1.15rem 0.85rem 1.25rem 0.85rem;
        border-radius: 16px;
        border: 1px solid #334155;
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    .backtest-signal-card .bss-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
    .backtest-signal-card .bss-value {
        color: #f8fafc;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .backtest-signal-card .bss-sub {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.35rem;
    }
    /* Headings */
    h1 { color: #f8fafc !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h2, h3 { color: #e2e8f0 !important; }
    /* Radio / pills */
    .stRadio > div { gap: 0.5rem; flex-wrap: wrap; }
    /* Info boxes */
    div[data-testid="stAlert"] { border-radius: 10px; }
</style>
"""
