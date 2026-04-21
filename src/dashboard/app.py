"""Streamlit entry: AWRY dashboard (dark theme, multi-panel layout)."""

from __future__ import annotations

import sys
import json
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.backtest_chart import (
    SCENARIOS,
    lead_months_awry,
    lead_months_sahm,
    lead_months_yield,
    scenario_comparison_figure,
)
from dashboard.components.backtest_view import scenario_panel
from dashboard.components.diagnostics_panel import compute_binary_metrics, roc_figure
from dashboard.components.gauge import awry_gauge_figure
from dashboard.components.indicators import compute_driver_items
from dashboard.components.indicator_panel import indicator_bars_figure
from dashboard.components.model_explainer import render_model_explainer
from dashboard.components.timeline import probability_timeline
from dashboard.data_helpers import load_ablation_summary, raw_monthly_panel
from dashboard.export_latex import build_awry_latex_export
from dashboard.export_summary import build_awry_markdown_export
from dashboard.styles.theme import DASHBOARD_CSS
from features.dataset_builder import FEATURE_COLUMNS
from features.equity_config import DEFAULT_EQUITY_SERIES
from awry_pipeline import fit_awry_pipeline, predict_history

st.set_page_config(page_title="AWRY", layout="wide", initial_sidebar_state="expanded")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
FALLBACK_FITTED_THRESHOLD = 0.2325
FALLBACK_FITTED_ALPHA = 1.0


def _load_numeric_json_value(path: Path, keys: tuple[str, ...], fallback: float, label: str) -> float:
    if not path.exists():
        print(f"[dashboard] Warning: {path} missing; using fallback {label}={fallback}.")
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[dashboard] Warning: could not read {path} ({exc}); using fallback {label}={fallback}.")
        return fallback

    for key in keys:
        value = payload.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                print(f"[dashboard] Warning: {path}:{key} is not numeric; trying the next key.")

    print(f"[dashboard] Warning: {path} has no usable {label} key; using fallback {label}={fallback}.")
    return fallback


def load_fitted_threshold() -> float:
    """Load the model operating threshold used for fitted classification labels."""
    return _load_numeric_json_value(
        MODELS_DIR / "thresholds.json",
        ("f1_optimal", "f1_max", "tau", "operating_threshold", "threshold", "optimal_threshold", "tau_f1"),
        FALLBACK_FITTED_THRESHOLD,
        "threshold",
    )


def load_fitted_alpha() -> float:
    """Load the fitted composite alpha used for model-call blend labels."""
    keys = ("alpha", "fitted_alpha", "alpha_tuned", "best_alpha")
    candidate_paths = (
        MODELS_DIR / "alpha_tuned.json",
        MODELS_DIR / "walk_forward_summary_baseline.json",
        MODELS_DIR / "composite_baseline_metrics.json",
    )
    for path in candidate_paths:
        if path.exists():
            return _load_numeric_json_value(path, keys, FALLBACK_FITTED_ALPHA, "alpha")

    print(
        "[dashboard] Warning: no fitted alpha artifact found; "
        f"using fallback alpha={FALLBACK_FITTED_ALPHA}."
    )
    return FALLBACK_FITTED_ALPHA


def _month_end_label(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%B %Y")


def _pred_label(p: float, threshold: float = 0.5) -> str:
    return "recession" if p >= threshold else "expansion"


def _actual_label(y: float) -> str:
    return "recession" if int(round(float(y))) == 1 else "expansion"


def _match_ok(p: float, y: float, threshold: float = 0.5) -> bool:
    pred = 1 if p >= threshold else 0
    act = int(round(float(y)))
    return pred == act


def _match_pill(label: str, ok: bool | None) -> str:
    """HTML span: green Yes, red No, slate — when no outcome."""
    if ok is None:
        bg, fg, txt = "#475569", "#f8fafc", "—"
    elif ok:
        bg, fg, txt = "#22c55e", "#ffffff", "Yes"
    else:
        bg, fg, txt = "#ef4444", "#ffffff", "No"
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};padding:0.4rem 1.1rem;'
        f'border-radius:999px;font-weight:600;font-size:0.9rem;">{label}: {txt}</span>'
    )


def _one_row(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    r = df.loc[ts]
    if isinstance(r, pd.DataFrame):
        return r.iloc[0]
    return r


def _scalar_from_loc(df: pd.DataFrame, ts: pd.Timestamp, col: str) -> float | None:
    if col not in df.columns:
        return None
    try:
        v = df.loc[ts, col]
    except KeyError:
        return None
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    if pd.isna(v):
        return None
    return float(v)


def _forecast_month_end(ts: pd.Timestamp, months_ahead: int = 3) -> pd.Timestamp:
    """Month-end of `months_ahead` calendar months after `ts` (as-of month → +3 = forecast horizon month)."""
    t = pd.Timestamp(ts).normalize()
    first = t.replace(day=1)
    return (first + pd.DateOffset(months=months_ahead) + pd.offsets.MonthEnd(0)).normalize()


def _forecast_outlook_label(as_of_ts: pd.Timestamp) -> str:
    """e.g. 'June 2026 outlook' for March 2026 as-of."""
    return f"{_month_end_label(_forecast_month_end(as_of_ts, 3))} outlook"


def _realized_usrec_3m_ahead(pipe, ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    v = _scalar_from_loc(pipe.df, ts, "target_h3")
    if v is not None:
        return v
    t_end = _forecast_month_end(ts, 3)
    u = _scalar_from_loc(pipe.df, t_end, "USREC")
    if u is not None:
        return u
    idx = pipe.df.index
    pos = idx.get_indexer([t_end], method="nearest")[0]
    if pos < 0 or pos >= len(idx):
        return float("nan")
    t_near = idx[pos]
    if t_near.to_period("M") != t_end.to_period("M"):
        return float("nan")
    u2 = _scalar_from_loc(pipe.df, t_near, "USREC")
    return float(u2) if u2 is not None else float("nan")


def _norm_bar(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def _composite_risk_interpret(p_awry: float) -> str:
    """Plain-English label for headline composite (display only)."""
    if p_awry < 0.30:
        return "Low recession risk"
    if p_awry < 0.60:
        return "Elevated recession risk"
    return "High recession risk"


def _build_indicator_items(raw: pd.DataFrame) -> list[tuple[str, float, str, str]]:
    """(label, norm 0–1, display, signal) — signal is recession|expansion|neutral for UI coloring."""
    last = raw.iloc[-1]
    prev = raw.iloc[-2]
    items: list[tuple[str, float, str, str]] = []

    t10 = float(last["T10Y3M"])
    if t10 < 0:
        sig10 = "recession"
    elif t10 < 0.5:
        sig10 = "neutral"
    else:
        sig10 = "expansion"
    items.append(("T10Y3M spread", _norm_bar(t10, -2.0, 4.0), f"{t10:.2f}%", sig10))

    un = float(last["UNRATE"])
    items.append(
        ("Unemployment", _norm_bar(un, 3.0, 11.0), f"{un:.1f}%", "recession" if un >= 6.0 else "expansion")
    )

    d_pay = float(last["PAYEMS"] - prev["PAYEMS"])
    items.append(
        ("Nonfarm payrolls Δ", _norm_bar(d_pay, -400, 400), f"{d_pay:+.0f}K", "recession" if d_pay < 0 else "expansion")
    )

    vix = float(last["VIXCLS"])
    items.append(("VIX", _norm_bar(vix, 10.0, 45.0), f"{vix:.1f}", "recession" if vix >= 25 else "expansion"))

    baa = float(last["BAA10Y"])
    items.append(("BAA–10Y spread", _norm_bar(baa, 0.0, 6.0), f"{baa:.2f}%", "recession" if baa > 1.5 else "expansion"))

    hst = float(last["HOUST"])
    items.append(
        ("Housing starts", _norm_bar(hst, 400, 2000), f"{hst/1000:.2f}M", "recession" if hst < 900 else "expansion")
    )

    return items


@st.cache_resource
def _pipeline_and_history(_cache_bust: int = 6):
    # Keep the call signature compatible with older cached Streamlit imports.
    # Baseline is the strongest current feature set under the data coverage
    # available in this repo, so the dashboard uses it by default.
    pipe = fit_awry_pipeline(cached=True, equity_series=DEFAULT_EQUITY_SERIES, feature_set="baseline")
    hist = predict_history(pipe)
    return pipe, hist


def main() -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    with st.sidebar:
        if st.button("Refresh FRED data"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    pipe, hist = _pipeline_and_history()
    raw = raw_monthly_panel()
    ablation_summary = load_ablation_summary()
    scenario_hist = hist.copy()

    last = hist.iloc[-1]
    prev_row = hist.iloc[-2] if len(hist) > 1 else last
    last_ts = hist.index[-1]
    p_now = float(last["P_now"])
    p_3m = float(last["P_3m"])
    p_awry = float(last["P_AWRY"])
    nber = int(last["USREC"])
    fc_ts = _forecast_month_end(last_ts, 3)

    delta_pp = (p_awry - float(prev_row["P_AWRY"])) * 100.0
    diag_metrics = compute_binary_metrics(hist["USREC"].values, hist["P_AWRY"].values)
    threshold_payload = getattr(pipe, "thresholds", {}) or {}
    default_threshold = float(threshold_payload.get("threshold", 0.5))
    scenario_threshold = default_threshold

    # Header
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(
            '<p style="font-size:2.25rem;font-weight:800;margin:0;letter-spacing:-0.03em;">AWRY</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#94a3b8;font-size:1.05rem;margin:0.15rem 0 0 0;">Are We in a Recession Yet?</p>',
            unsafe_allow_html=True,
        )
        st.info("Displayed probabilities are out-of-sample from walk-forward CV with a 3-month purge gap.")
    with h2:
        st.caption(
            f"Latest model row: **t** = {_month_end_label(last_ts)} · "
            f"forecast horizon **t+3** = {_month_end_label(fc_ts)}"
        )
        badge = "Recession" if nber == 1 else "Expansion"
        badge_color = "#ef4444" if nber == 1 else "#22c55e"
        st.markdown(
            f'<span style="background:{badge_color};color:white;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.85rem;font-weight:600;">{badge}</span>',
            unsafe_allow_html=True,
        )

    # KPI row (custom cards — neutral / blue / amber)
    risk_line = _composite_risk_interpret(p_awry)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
<div class="awry-kpi-card kpi-neutral">
  <div class="kpi-label">AWRY composite</div>
  <div class="kpi-value">{p_awry:.1%}</div>
  <div class="kpi-delta">{delta_pp:+.1f} pp vs last month</div>
  <div class="kpi-interpret">{risk_line}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
<div class="awry-kpi-card kpi-blue">
  <div class="kpi-label">● Nowcast P(R<sub>t</sub>) · {_month_end_label(last_ts)}</div>
  <div class="kpi-value">{p_now:.1%}</div>
  <div class="kpi-hint" title="P(recession in this month), using the latest complete monthly row — not today's calendar date.">Monthly nowcast for the as-of month.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
<div class="awry-kpi-card kpi-amber">
  <div class="kpi-label">→ 3-month forecast P(R<sub>t+3</sub> | I<sub>t</sub>) · {_month_end_label(fc_ts)}</div>
  <div class="kpi-value">{p_3m:.1%}</div>
  <div class="kpi-hint" title="P(recession in the month three calendar months after t), given information through the as-of month.">Outlook for {_forecast_outlook_label(last_ts)}.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("Why isn’t the latest calendar month always shown?"):
        st.markdown(
            f"Rows need a realized **target_h3** (NBER **three months after** month *t*), so the newest model row is "
            f"usually **about three months behind** the last month in the raw FRED panel. "
            f"**Nowcast** = probability of recession in **{_month_end_label(last_ts)}**. "
            f"**Forecast** = probability of recession in **{_month_end_label(fc_ts)}** given data through **{_month_end_label(last_ts)}**. "
            f"If numbers look stale, use **Refresh FRED data** in the sidebar. "
            f"Local time: **{datetime.now().strftime('%Y-%m-%d %H:%M')}**."
        )

    # Gauge + indicators
    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(awry_gauge_figure(p_awry), use_container_width=True)
    with g2:
        # Prefer model-derived drivers. Fall back to the heuristic macro bars
        # if the explainer artifact is unavailable or the model lacks importances.
        driver_items = compute_driver_items(pipe) or _build_indicator_items(raw)
        st.plotly_chart(indicator_bars_figure(driver_items), use_container_width=True)

    # Historical chart
    st.caption("Historical AWRY probability (OOF walk-forward)")
    mode = st.radio(
        "Series",
        ["Composite", "Nowcast", "3-month forecast"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Composite":
        y = hist["P_AWRY"].values
    elif mode == "Nowcast":
        y = hist["P_now"].values
    else:
        y = hist["P_3m"].values

    st.plotly_chart(
        probability_timeline(
            hist.index,
            y,
            recession=hist["USREC"].values,
            mode=mode.lower(),
            show_signal_threshold=(mode == "Composite"),
            signal_threshold=default_threshold,
        ),
        use_container_width=True,
    )

    render_model_explainer(pipe, raw, default_threshold)

    # Bottom: backtest + diagnostics
    b1, b2 = st.columns([1, 1])

    with b1:
        st.markdown("### Backtest scenario")
        scenario = st.radio(
            "Scenario",
            list(SCENARIOS.keys()),
            horizontal=True,
        )
        st.caption(
            "Months relative to approximate recession start (R0). "
            f"Scenario overlays use the walk-forward AWRY series at the fitted {scenario_threshold:.0%} "
            "operating threshold."
        )
        st.plotly_chart(
            scenario_comparison_figure(scenario_hist, raw, scenario, threshold=scenario_threshold),
            use_container_width=True,
        )
        la = lead_months_awry(scenario_hist, scenario, threshold=scenario_threshold)
        ls = lead_months_sahm(raw, scenario)
        ly = lead_months_yield(raw, scenario)
        la_s = f"−{la}M" if la is not None else "—"
        ls_s = f"−{ls}M" if ls is not None else "—"
        ly_s = f"−{ly}M" if ly is not None else "—"
        st.markdown(
            f"""
<div class="backtest-signal-grid">
  <div class="backtest-signal-card">
    <div class="bss-label">AWRY signal (>= {scenario_threshold:.0%})</div>
    <div class="bss-value">{la_s}</div>
    <div class="bss-sub">months before R0</div>
  </div>
  <div class="backtest-signal-card">
    <div class="bss-label">Sahm gap (≥0.5pp)</div>
    <div class="bss-value">{ls_s}</div>
    <div class="bss-sub">months before R0</div>
  </div>
  <div class="backtest-signal-card">
    <div class="bss-label">Yield curve (&lt;0)</div>
    <div class="bss-value">{ly_s}</div>
    <div class="bss-sub">months before R0</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        scenario_panel(ablation_summary)

    with b2:
        st.markdown("### Model diagnostics (OOS)")
        d1, d2, d3 = st.columns(3)
        d1.metric("AUROC", f"{diag_metrics['auroc']:.3f}" if not np.isnan(diag_metrics["auroc"]) else "—")
        d2.metric("Brier score", f"{diag_metrics['brier']:.3f}")
        d3.metric("F1 (0.5)", f"{diag_metrics['f1']:.3f}")
        t0 = hist.index[0]
        t1 = hist.index[-1]
        n_m = len(hist)
        st.caption(
            "OOS window: "
            f"**{n_m}** months from {_month_end_label(t0)} to {_month_end_label(t1)}. "
            "In-sample reference metrics are saved in `artifacts/models/in_sample_metrics.json`."
        )
        st.plotly_chart(roc_figure(hist["USREC"].values, hist["P_AWRY"].values), use_container_width=True)

    with st.expander("Raw indicators (latest month)"):
        raw_df = pipe.df[FEATURE_COLUMNS].tail(3)
        st.dataframe(raw_df, use_container_width=True)

    st.divider()
    st.subheader("Historical test case (walk-forward OOF)")
    st.caption(
        "The composite P_AWRY uses the fitted α from OOF Brier optimization (not user-configurable). "
        "The sensitivity threshold slider lets you explore how classification changes at different operating points; "
        "it does not change the model."
    )
    st.caption(
        "Pick a past month. **P_now** is compared with NBER that month; "
        "**P_3m** is compared with NBER three months later."
    )
    artifact_threshold = load_fitted_threshold()
    artifact_alpha = load_fitted_alpha()
    pipe_threshold = None
    try:
        pipe_threshold = float((getattr(pipe, "thresholds", {}) or {}).get("threshold"))
    except (TypeError, ValueError):
        pipe_threshold = None
    fitted_threshold = pipe_threshold if pipe_threshold is not None and np.isfinite(pipe_threshold) else artifact_threshold
    if pipe_threshold is not None and np.isfinite(pipe_threshold) and abs(pipe_threshold - artifact_threshold) > 1e-9:
        print(
            "[dashboard] Warning: artifact threshold differs from the running pipeline; "
            f"using pipeline threshold={pipe_threshold} instead of artifact threshold={artifact_threshold}."
        )
    fitted_alpha = float(getattr(pipe, "alpha", artifact_alpha))
    if abs(fitted_alpha - artifact_alpha) > 1e-9:
        print(
            "[dashboard] Warning: artifact alpha differs from the running pipeline; "
            f"using pipeline alpha={fitted_alpha} instead of artifact alpha={artifact_alpha}."
        )
    threshold = st.slider(
        "Sensitivity threshold",
        0.1,
        0.9,
        float(fitted_threshold),
        0.0025,
        format="%.4f",
        help="Exploration only. The fitted composite label below always uses the fitted threshold.",
    )
    slider_threshold = threshold

    idx_map = {pd.Timestamp(t).strftime("%Y-%m"): pd.Timestamp(t) for t in hist.index}
    month_keys = list(idx_map.keys())
    default_2020 = next((k for k in month_keys if k.startswith("2020-")), month_keys[0])
    pick = st.selectbox("Scenario month (month-end row)", month_keys, index=month_keys.index(default_2020))
    ts = idx_map[pick]
    row = _one_row(hist, ts)
    p_now_s = float(row["P_now"])
    p_3m_s = float(row["P_3m"])
    act_now = (
        float(row["target_h0"])
        if "target_h0" in row.index and pd.notna(row.get("target_h0"))
        else float(row["USREC"])
    )
    act_3m = _realized_usrec_3m_ahead(pipe, ts)
    fc_ts_s = _forecast_month_end(ts, 3)

    # COMMENT: The fitted alpha is a research finding from OOF Brier
    # optimization, not a user preference, so the composite call is fixed here.
    p_composite = fitted_alpha * p_now_s + (1.0 - fitted_alpha) * p_3m_s
    ok_composite_fitted = _match_ok(p_composite, act_now, fitted_threshold)
    # COMMENT: The threshold slider is still useful because it represents a
    # precision/recall operating-point tradeoff for component diagnostics.
    ok_now_slider = _match_ok(p_now_s, act_now, slider_threshold)
    ok_3m_slider = _match_ok(p_3m_s, act_3m, slider_threshold) if pd.notna(act_3m) else None

    st.markdown(
        f"**Fitted composite (α = {fitted_alpha:.2f}): "
        f"P_AWRY = {p_composite:.1%} → {_pred_label(p_composite, fitted_threshold)} "
        f"at fitted threshold ({fitted_threshold:.1%}).**"
    )
    st.caption(
        f"Component labels at sensitivity threshold ({slider_threshold:.2%}): "
        f"nowcast -> {_pred_label(p_now_s, slider_threshold)}; "
        f"3-month forecast -> {_pred_label(p_3m_s, slider_threshold)}."
    )
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown(f"**As-of:** {_month_end_label(ts)}")
        st.markdown(
            f"- **Nowcast:** P = {p_now_s:.1%} → **{_pred_label(p_now_s, threshold)}**\n"
            f"- **Actual NBER ({_month_end_label(ts)}):** {_actual_label(act_now)}"
        )
    with c_b:
        st.markdown(f"**3-month forecast outlook:** {_forecast_outlook_label(ts)} (recession month **{_month_end_label(fc_ts_s)}**)")
        if pd.isna(act_3m):
            st.warning(
                "No realized **3‑month outcome** in the panel for this as-of month. "
                "Try an earlier year or clear Streamlit cache (C)."
            )
        else:
            st.markdown(
                f"- **Forecast ({_forecast_outlook_label(ts)}):** P = {p_3m_s:.1%} → **{_pred_label(p_3m_s, threshold)}** for month **{_month_end_label(fc_ts_s)}**\n"
                f"- **Actual NBER ({_month_end_label(fc_ts_s)}):** {_actual_label(act_3m)}"
            )

    pill_composite = _match_pill("Fitted composite match", ok_composite_fitted)
    slider_now = _match_pill("Slider nowcast match", ok_now_slider)
    slider_3m = _match_pill("Slider 3-month match", ok_3m_slider)
    st.markdown(
        f'<div style="text-align:center;margin:1.25rem 0 0.75rem 0;display:flex;justify-content:center;'
        f"gap:0.65rem;flex-wrap:wrap;\">{pill_composite}{slider_now}{slider_3m}</div>",
        unsafe_allow_html=True,
    )

    _gen_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _export_kw = dict(
        generated_at=_gen_at,
        pipe=pipe,
        hist=hist,
        raw_tail=pipe.df[FEATURE_COLUMNS].tail(12),
        latest_ts=last_ts,
        fc_ts_latest=fc_ts,
        test_ts=ts,
        test_threshold=float(threshold),
        diagnostics=diag_metrics,
        forecast_month_end_fn=_forecast_month_end,
        forecast_outlook_label_fn=_forecast_outlook_label,
        month_label_fn=_month_end_label,
        realized_usrec_3m_fn=_realized_usrec_3m_ahead,
    )
    md_export = build_awry_markdown_export(**_export_kw)
    tex_export = build_awry_latex_export(**_export_kw)
    _ts_fn = datetime.now().strftime("%Y%m%d_%H%M")
    _dl_l, _dl_c, _dl_r = st.columns([1, 2, 1])
    with _dl_c:
        st.download_button(
            label="Download summary (.md)",
            data=md_export.encode("utf-8"),
            file_name=f"awry_summary_{_ts_fn}.md",
            mime="text/markdown",
            key="awry_md_export",
            use_container_width=True,
            help="Full history as tables, diagnostics, raw-indicator tail, and current test-case settings.",
        )
        st.download_button(
            label="Download LaTeX (.tex)",
            data=tex_export.encode("utf-8"),
            file_name=f"awry_export_{_ts_fn}.tex",
            mime="text/plain",
            key="awry_tex_export",
            use_container_width=True,
            help="IEEEtran conference-style .tex. Needs IEEEtran.cls (TeX Live / MiKTeX). Build: pdflatex awry_export_….tex",
        )


if __name__ == "__main__":
    main()
