"""LaTeX document export — IEEEtran conference layout (compile with: pdflatex awry_export.tex)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from dashboard.export_summary import (
    EQUITY_DISPLAY_NAME,
    EQUITY_FRED_SERIES_ID,
    _fmt_pct,
    collect_awry_export_payload,
)


def _format_generated_author_line(generated_at: str) -> str:
    """Human-readable line for \\IEEEauthorblockA."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(generated_at.strip(), fmt)
            return dt.strftime("%B %d, %Y at %H:%M:%S")
        except ValueError:
            continue
    return generated_at


def _tex_cell(s: str) -> str:
    """Escape for LaTeX table cells (text mode)."""
    t = str(s)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("$", r"\$"),
    ):
        t = t.replace(a, b)
    return t


def _snap_field_display(field: str) -> str:
    """Map stored field label to LaTeX for KPI table."""
    m = {
        "As-of month (t)": r"As-of month ($t$)",
        "Forecast month (t+3)": r"Forecast month ($t+3$)",
        "P_AWRY": r"$P_{\mathrm{AWRY}}$",
        "P_now": r"$P_{\mathrm{now}}$",
        "P_3m": r"$P_{\mathrm{3m}}$",
        "α·P_now": r"$\alpha \cdot P_{\mathrm{now}}$",
        "(1−α)·P_3m": r"$(1-\alpha) \cdot P_{\mathrm{3m}}$",
        "NBER USREC (same month)": r"NBER USREC (same month)",
    }
    return m.get(field, _tex_cell(field))


def _kpi_table_tex(snap: pd.DataFrame) -> str:
    """IEEE-style table with vertical rules (matches sample)."""
    rows = []
    for _, row in snap.iterrows():
        f = str(row["field"])
        v = str(row["value"])
        rows.append(f"{_snap_field_display(f)} & {_tex_cell(v)} \\\\")
    body = "\n".join(rows)
    return (
        r"\begin{table}[htbp]"
        "\n"
        r"\caption{Latest Model Row (Dashboard KPIs)}"
        "\n"
        r"\begin{center}"
        "\n"
        r"\begin{tabular}{|l|r|}"
        "\n"
        r"\hline"
        "\n"
        r"\textbf{Field} & \textbf{Value} \\"
        "\n"
        r"\hline"
        "\n"
        f"{body}\n"
        r"\hline"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\label{tab:kpi}"
        "\n"
        r"\end{center}"
        "\n"
        r"\end{table}"
        "\n"
    )


def _df_to_latex(
    df: pd.DataFrame,
    *,
    longtable: bool = False,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
) -> str:
    """DataFrame fragment for inclusion in the document."""
    n = len(df.columns)
    if column_format is None:
        col_fmt = "|" + "|".join(["c"] * n) + "|" if n else "|c|"
    else:
        col_fmt = column_format
    kw: dict[str, Any] = {
        "index": False,
        "escape": True,
        "na_rep": "---",
        "column_format": col_fmt,
    }
    if longtable:
        kw["longtable"] = True
    if caption is not None:
        kw["caption"] = caption
    if label is not None:
        kw["label"] = label
    return df.to_latex(**kw)


def build_awry_latex_export(
    *,
    generated_at: str,
    pipe: Any,
    hist: pd.DataFrame,
    raw_tail: pd.DataFrame,
    latest_ts: pd.Timestamp,
    fc_ts_latest: pd.Timestamp,
    test_ts: pd.Timestamp,
    test_threshold: float,
    diagnostics: dict[str, float],
    forecast_month_end_fn: Any,
    forecast_outlook_label_fn: Any,
    month_label_fn: Any,
    realized_usrec_3m_fn: Any,
) -> str:
    """IEEEtran conference-style .tex; requires ``IEEEtran.cls`` (e.g. TeX Live / MiKTeX)."""
    p = collect_awry_export_payload(
        generated_at=generated_at,
        pipe=pipe,
        hist=hist,
        raw_tail=raw_tail,
        latest_ts=latest_ts,
        fc_ts_latest=fc_ts_latest,
        test_ts=test_ts,
        test_threshold=test_threshold,
        diagnostics=diagnostics,
        forecast_month_end_fn=forecast_month_end_fn,
        forecast_outlook_label_fn=forecast_outlook_label_fn,
        month_label_fn=month_label_fn,
        realized_usrec_3m_fn=realized_usrec_3m_fn,
    )
    alpha = p["alpha"]
    as_of_plain = _tex_cell(str(p["snap"].iloc[0]["value"]))
    author_when = _format_generated_author_line(generated_at)
    eq_name = _tex_cell(EQUITY_DISPLAY_NAME)

    kpi_block = _kpi_table_tex(p["snap"])

    hist_tex = _df_to_latex(
        p["hist"],
        longtable=True,
        caption="Historical probabilities (full in-sample series).",
        label="tab:hist",
    )
    diag_tex = _df_to_latex(p["diag"], column_format="|l|r|")
    raw_tex = _df_to_latex(
        p["raw_fmt"],
        longtable=True,
        caption="Raw indicator levels (recent rows).",
        label="tab:raw",
    )
    test_tex = _df_to_latex(p["test_tbl"], column_format="|l|r|r|")

    parts: list[str] = [
        r"\documentclass[conference,onecolumn]{IEEEtran}",
        r"\IEEEoverridecommandlockouts",
        r"\usepackage{cite}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{algorithmic}",
        r"\usepackage{graphicx}",
        r"\usepackage{textcomp}",
        r"\usepackage{xcolor}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\begin{document}",
        r"",
        r"\title{AWRY Model Analysis: Recession Probability Forecasting and Equity Benchmarking}",
        r"",
        r"\author{\IEEEauthorblockN{AWRY Tooling System}\IEEEauthorblockA{\textit{Automated Export Summary} \\",
        rf"Generated: {author_when}",
        r"}",
        r"}",
        r"",
        r"\maketitle",
        r"",
        r"\begin{abstract}",
        rf"This report provides a snapshot of the AWRY model's performance and forecasts as of {as_of_plain}. ",
        r"We detail the composite probability scoring methods, horizon ensemble configurations, ",
        r"and historical probability tables for recession forecasting.",
        r"\end{abstract}",
        r"",
        r"\begin{IEEEkeywords}",
        r"AWRY, Nowcast, Ensemble Learning, Recession Probability, NASDAQ Composite",
        r"\end{IEEEkeywords}",
        r"",
        r"\section{Introduction}",
        rf"The AWRY model utilizes a composite scoring mechanism to summarize recession risk. ",
        rf"The primary equity benchmark is the {eq_name} (FRED \texttt{{{EQUITY_FRED_SERIES_ID}}}); ",
        r"the internal feature column is \texttt{NASDAQCOM}.",
        r"",
        r"\section{Methodology}",
        r"",
        r"\subsection{Composite Scoring}",
        r"The primary metric, $P_{\mathrm{AWRY}}$, is a weighted linear combination of the nowcast and the 3-month forecast:",
        r"\begin{equation}",
        r"P_{\mathrm{AWRY}} = \alpha P_{\mathrm{now}} + (1-\alpha) P_{\mathrm{3m}}",
        r"\label{eq:composite}",
        r"\end{equation}",
        r"where $\alpha = " + f"{alpha:.6g}" + r"$ in the fitted pipeline (same as \texttt{composite\_score} in code).",
        r"",
        r"\subsection{Horizon Ensembles}",
        r"The model stacks logistic regression (Logit) and random forest (RF) within each horizon:",
        r"\begin{itemize}",
        rf"\item \textbf{{Nowcast:}} Logit {p['w_now_logit']:.4f}, RF {p['w_now_rf']:.4f}.",
        rf"\item \textbf{{3-Month Forecast:}} Logit {p['w_fc_logit']:.4f}, RF {p['w_fc_rf']:.4f}.",
        r"\end{itemize}",
        r"",
        r"\section{Current Model Indicators}",
        rf"Table~\ref{{tab:kpi}} summarizes KPIs for the latest reporting period.",
        r"",
        kpi_block,
        r"\section{Historical Probability Analysis}",
        r"Table~\ref{tab:hist} lists the full in-sample history used in the dashboard timeline.",
        hist_tex,
        r"\section{Model Diagnostics}",
        r"In-sample metrics for the composite score versus NBER are summarized in Table~\ref{tab:diag}.",
        r"\begin{table}[htbp]",
        r"\caption{Diagnostics (composite vs.\ NBER)}",
        r"\begin{center}",
        diag_tex,
        r"\label{tab:diag}",
        r"\end{center}",
        r"\end{table}",
        r"\section{Raw Indicator Levels}",
        r"Recent raw feature levels appear in Table~\ref{tab:raw}.",
        raw_tex,
        r"\section{Historical Test Case}",
        rf"Scenario month \textbf{{{_tex_cell(p['month_label_test'])}}}, decision threshold ${p['test_threshold']:.2f}$, ",
        rf"outlook \textit{{{_tex_cell(p['forecast_outlook'])}}}.",
        r"\subsection{Probabilities vs.\ NBER}",
        r"\begin{table}[htbp]",
        r"\caption{Test-case probabilities}",
        r"\begin{center}",
        test_tex,
        r"\label{tab:test}",
        r"\end{center}",
        r"\end{table}",
        r"\subsection{Composite Decomposition (selected month)}",
        rf"\begin{{equation}}",
        rf"P_{{\mathrm{{AWRY}}}} = {alpha:.6f} \times {p['p_now_t']:.6f} + {1.0 - alpha:.6f} \times {p['p_3m_t']:.6f} = {p['p_awry_t']:.6f}",
        rf"\label{{eq:decomp}}",
        rf"\end{{equation}}",
        r"\begin{itemize}",
        rf"\item $\alpha \cdot P_{{\mathrm{{now}}}}$ = {_fmt_pct(alpha * p['p_now_t'])}",
        rf"\item $(1-\alpha) \cdot P_{{\mathrm{{3m}}}}$ = {_fmt_pct((1.0 - alpha) * p['p_3m_t'])}",
        r"\end{itemize}",
        r"\end{document}",
        "",
    ]
    return "\n".join(parts)
