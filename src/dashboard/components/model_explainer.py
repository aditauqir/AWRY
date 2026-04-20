"""Interactive dashboard views that explain the AWRY math and data flow."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles.theme import COLORS


def _friendly_model_name(name: str) -> str:
    names = {
        "logit": "Regularized logistic regression",
        "rf": "Random forest",
        "xgb": "XGBoost",
    }
    return names.get(name, name)


def _base_model_names(horizon_model) -> list[str]:
    names = getattr(horizon_model, "base_names", None)
    if names:
        return list(names)
    ensemble = getattr(horizon_model, "ensemble", None)
    base_models = getattr(ensemble, "base_models", None)
    if isinstance(base_models, dict) and base_models:
        return list(base_models.keys())
    fallback: list[str] = []
    for name in ("logit", "rf", "xgb"):
        if getattr(horizon_model, name, None) is not None:
            fallback.append(name)
    return fallback


def _meta_coefficients(horizon_model) -> dict[str, float]:
    payload = getattr(horizon_model, "meta_coefficients", None)
    if isinstance(payload, dict) and payload:
        return {str(k): float(v) for k, v in payload.items()}

    ensemble = getattr(horizon_model, "ensemble", None)
    meta = getattr(ensemble, "meta", None)
    if meta is None:
        return {}

    coeffs: dict[str, float] = {}
    intercept = getattr(meta, "intercept_", None)
    coef = getattr(meta, "coef_", None)
    if intercept is not None:
        try:
            coeffs["intercept"] = float(intercept[0])
        except Exception:
            pass
    if coef is not None:
        try:
            row = coef[0]
            for idx, name in enumerate(_base_model_names(horizon_model)):
                if idx < len(row):
                    coeffs[name] = float(row[idx])
        except Exception:
            return coeffs
    return coeffs


def _source_groups(raw: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "Activity and labor": ["PAYEMS", "INDPRO", "W875RX1", "RRSFS", "UNRATE", "ICSA", "HOUST", "PERMIT"],
        "Rates and credit": ["T10Y3M", "FEDFUNDS", "BAA10Y", "CSUSHPINSA", "BAMLH0A0HYM2", "TEDRATE", "NFCI", "CFNAI"],
        "Market and uncertainty": ["NASDAQCOM", "VIXCLS", "UMCSENT", "DCOILWTICO", "USEPUNEWSINDXM"],
        "Local spreadsheet signal": ["NEWS_SENTIMENT_XLSX"],
    }


def _feature_family_counts(x_cols: list[str]) -> dict[str, int]:
    lag = [c for c in x_cols if c.endswith(("_lag1", "_lag2"))]
    trend = [c for c in x_cols if c.endswith("_trend12") or c.endswith("_ma3")]
    change = [c for c in x_cols if c.endswith("_chg12")]
    structure = [c for c in x_cols if c in {"SAHM_GAP", "T10Y3M_INVERSION_DURATION"} or c.endswith("_available")]
    used = set(lag + trend + change + structure)
    base = [c for c in x_cols if c not in used]
    return {
        "Base monthly inputs": len(base),
        "Lag features": len(lag),
        "Trend features": len(trend),
        "12-month change features": len(change),
        "Structural regime features": len(structure),
    }


def _build_flow_figure(raw: pd.DataFrame, x_cols: list[str]) -> go.Figure:
    groups = _source_groups(raw)
    group_counts = {name: len([c for c in cols if c in raw.columns]) for name, cols in groups.items()}
    families = _feature_family_counts(x_cols)
    monthly_panel_count = int(sum(group_counts.values()))
    engineered_count = int(sum(families.values()))

    labels = [
        *group_counts.keys(),
        "Monthly panel",
        *families.keys(),
        "Nowcast stack",
        "3-month stack",
        "AWRY composite",
    ]
    index = {label: i for i, label in enumerate(labels)}

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []

    for name, count in group_counts.items():
        if count <= 0:
            continue
        sources.append(index[name])
        targets.append(index["Monthly panel"])
        values.append(float(count))
        link_colors.append("rgba(56,189,248,0.35)")

    for name, count in families.items():
        if count <= 0:
            continue
        sources.append(index["Monthly panel"])
        targets.append(index[name])
        values.append(float(count))
        link_colors.append("rgba(34,197,94,0.35)")

    for name, count in families.items():
        if count <= 0:
            continue
        sources.append(index[name])
        targets.append(index["Nowcast stack"])
        values.append(float(count))
        link_colors.append("rgba(249,115,22,0.30)")

        sources.append(index[name])
        targets.append(index["3-month stack"])
        values.append(float(count))
        link_colors.append("rgba(234,179,8,0.28)")

    sources.extend([index["Nowcast stack"], index["3-month stack"]])
    targets.extend([index["AWRY composite"], index["AWRY composite"]])
    values.extend([monthly_panel_count / 2.0, monthly_panel_count / 2.0])
    link_colors.extend(["rgba(239,68,68,0.35)", "rgba(148,163,184,0.35)"])

    node_colors = [
        "#1d4ed8",
        "#2563eb",
        "#0f766e",
        "#7c3aed",
        "#334155",
        "#16a34a",
        "#22c55e",
        "#65a30d",
        "#ca8a04",
        "#f97316",
        "#eab308",
        "#ef4444",
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="rgba(255,255,255,0.08)", width=1),
                label=labels,
                color=node_colors[: len(labels)],
            ),
            link=dict(source=sources, target=targets, value=values, color=link_colors),
        )
    )
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
    )
    return fig


def _derived_feature_rows(raw: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    preferred = [
        "PAYEMS",
        "UNRATE",
        "T10Y3M",
        "BAA10Y",
        "HOUST",
        "USEPUNEWSINDXM",
        "NEWS_SENTIMENT_XLSX",
    ]
    rows: list[dict[str, str]] = []
    for base in preferred:
        if base not in raw.columns:
            continue
        derived = [c for c in x_cols if c == base or c.startswith(f"{base}_")]
        if not derived:
            continue
        rows.append(
            {
                "Raw input": base,
                "Into feature space as": ", ".join(derived[:6]) + (" ..." if len(derived) > 6 else ""),
                "Used by": "Nowcast and 3-month stacks",
            }
        )
    return pd.DataFrame(rows)


def _render_horizon_block(title: str, horizon_model, threshold: float | None = None) -> None:
    model_names = _base_model_names(horizon_model)
    model_list = ", ".join(_friendly_model_name(name) for name in model_names) if model_names else "Unavailable"
    st.markdown(f"**{title}**")
    st.markdown(f"Base models: {model_list}")

    coef_payload = _meta_coefficients(horizon_model)
    coef_rows = [
        {
            "Term": "Intercept" if name == "intercept" else _friendly_model_name(name),
            "Coefficient": float(value),
        }
        for name, value in coef_payload.items()
    ]
    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    metrics = horizon_model.metrics
    metric_cols = st.columns(3)
    metric_cols[0].metric("AUROC", f"{float(metrics.get('auroc', float('nan'))):.3f}")
    metric_cols[1].metric("Brier", f"{float(metrics.get('brier', float('nan'))):.3f}")
    metric_cols[2].metric("F1", f"{float(metrics.get('f1', float('nan'))):.3f}")

    if threshold is not None:
        st.caption(f"Dashboard decision threshold: {threshold:.3f}")


def _render_formula_view(pipe, threshold: float) -> None:
    now_terms = _base_model_names(pipe.now)
    fc_terms = _base_model_names(pipe.forecast3)
    now_coef = _meta_coefficients(pipe.now)
    fc_coef = _meta_coefficients(pipe.forecast3)

    st.markdown("#### Core equations")
    st.latex(r"X_t = g(\text{raw monthly data up to month } t)")
    st.latex(r"g(\cdot) = \{\text{levels},\ \log\Delta,\ \text{trend12},\ \Delta_{12},\ \text{lag1},\ \text{lag2}\}")

    now_expr = " + ".join(
        f"{float(now_coef.get(name, 0.0)):.2f}\\,\\mathrm{{logit}}(p_{{{name}}})"
        for name in now_terms
    )
    fc_expr = " + ".join(
        f"{float(fc_coef.get(name, 0.0)):.2f}\\,\\mathrm{{logit}}(p_{{{name}}})"
        for name in fc_terms
    )
    now_intercept = float(now_coef.get("intercept", 0.0))
    fc_intercept = float(fc_coef.get("intercept", 0.0))

    st.latex(
        rf"z_{{\mathrm{{now}}}} = {now_intercept:.2f}" + (rf" + {now_expr}" if now_expr else "")
    )
    st.latex(r"P_{\mathrm{now}} = \sigma(z_{\mathrm{now}})")
    st.latex(
        rf"z_{{\mathrm{{3m}}}} = {fc_intercept:.2f}" + (rf" + {fc_expr}" if fc_expr else "")
    )
    st.latex(r"P_{\mathrm{3m}} = \sigma(z_{\mathrm{3m}})")
    st.latex(rf"P_{{\mathrm{{AWRY}}}} = {pipe.alpha:.2f}\,P_{{\mathrm{{now}}}} + {1.0 - pipe.alpha:.2f}\,P_{{\mathrm{{3m}}}}")
    st.latex(rf"\hat y = \mathbf{{1}}[P_{{\mathrm{{AWRY}}}} \ge {threshold:.3f}]")

    st.markdown("#### Plain-English readout")
    st.markdown(
        """
The dashboard first turns monthly macro, rate, market, and news inputs into engineered features.
Those features feed three base classifiers per horizon: logistic regression, random forest, and XGBoost.
Their probabilities are stacked by a logistic meta-learner, producing a nowcast probability and a 3-month-ahead probability.
The headline AWRY score is then a weighted blend of those two horizon probabilities.
"""
    )


def render_model_explainer(pipe, raw: pd.DataFrame, threshold: float) -> None:
    """Render an interactive menu that explains the model stack, math, and data flow."""
    st.subheader("How The Model Works")
    view = st.radio(
        "Model explainer menu",
        ["Model stack", "Main math", "Data flow"],
        horizontal=True,
        key="awry_model_explainer_menu",
    )

    if view == "Model stack":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw monthly inputs", f"{len([c for c in raw.columns if c != 'USREC'])}")
        c2.metric("Engineered features", f"{len(pipe.x_cols)}")
        c3.metric("Blend alpha", f"{float(pipe.alpha):.2f}")
        c4.metric("Decision threshold", f"{float(threshold):.3f}")

        h1, h2 = st.columns(2)
        with h1:
            _render_horizon_block("Nowcast model", pipe.now, threshold)
        with h2:
            _render_horizon_block("3-month forecast model", pipe.forecast3, threshold)

    elif view == "Main math":
        _render_formula_view(pipe, threshold)

    else:
        st.caption("The flow below shows where the raw data goes before it becomes the final AWRY probability.")
        st.plotly_chart(_build_flow_figure(raw, pipe.x_cols), use_container_width=True)

        family_df = pd.DataFrame(
            [{"Feature family": name, "Count": count} for name, count in _feature_family_counts(pipe.x_cols).items()]
        )
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Feature families used by the model**")
            st.dataframe(family_df, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Examples of raw data entering feature space**")
            st.dataframe(_derived_feature_rows(raw, pipe.x_cols), use_container_width=True, hide_index=True)
