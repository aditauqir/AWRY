"""In-sample AUROC / Brier / F1 and ROC curve from history table."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from dashboard.styles.theme import COLORS


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_hat = (y_score >= threshold).astype(int)
    out: dict[str, float] = {
        "brier": float(brier_score_loss(y_true, y_score)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
    return out


def roc_figure(y_true: np.ndarray, y_score: np.ndarray, name: str = "Ensemble") -> go.Figure:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="rgba(148,163,184,0.5)", dash="dash"))
    )
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=name,
                line=dict(color="#38bdf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.15)",
            )
        )
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], size=11),
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,0.15)"),
        legend=dict(orientation="h", y=1.1),
        title=dict(text="ROC (composite, in-sample)", font=dict(size=13)),
    )
    return fig
