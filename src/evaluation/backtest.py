"""Rolling-window backtest (no shuffle; time-ordered splits)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_ordered_split(
    df: pd.DataFrame,
    train_end_idx: int,
    val_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = rows [:train_end_idx], validation = next val_months rows."""
    train = df.iloc[:train_end_idx]
    val = df.iloc[train_end_idx : train_end_idx + val_months]
    return train, val


def expanding_window_indices(
    n_rows: int,
    min_train: int = 120,
    val_months: int = 12,
    step: int = 12,
) -> list[tuple[int, int]]:
    """
    Yields (train_end_idx, val_end_idx) for rolling evaluation.
    train_end_idx is exclusive end of training slice.
    """
    out: list[tuple[int, int]] = []
    t = min_train
    while t + val_months <= n_rows:
        out.append((t, t + val_months))
        t += step
    return out
