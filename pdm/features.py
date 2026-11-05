"""Labels and rolling-window features.

All rolling features are trailing (causal): the value at cycle t uses only
cycles t-w+1 .. t of the same engine. At the start of a trajectory, where
fewer than w cycles exist, the window is truncated to the cycles available
(so every row gets features and no information crosses engine boundaries).

Per sensor and window the features are:
  mean  - arithmetic mean
  std   - population standard deviation (ddof=0; 0 for a 1-point window)
  slope - least-squares slope against cycle number, per cycle (0 for a 1-point window)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

FEATURE_KINDS = ("mean", "std", "slope")


def add_failure_label(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Add `label` = 1 if the engine fails within `horizon` cycles (RUL <= horizon), else 0."""
    out = df.copy()
    out["label"] = (out["rul"] <= horizon).astype(np.int64)
    return out


def informative_sensors(train: pd.DataFrame, sensors: Sequence[str], tol: float = 1e-8) -> list[str]:
    """Sensors whose training-set standard deviation exceeds `tol` (constant sensors carry no signal)."""
    return [s for s in sensors if train[s].std(ddof=0) > tol]


def feature_names(sensors: Sequence[str]) -> list[str]:
    return [f"{s}_{k}" for s in sensors for k in FEATURE_KINDS]


def _window_stats(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trailing-window mean, std (ddof=0) and slope for each column of `values` (n x m).

    Computed exactly per window (two-pass, centred), not with running sums,
    so results don't depend on the magnitude of the raw values.
    """
    n, m = values.shape
    padded = np.vstack([np.full((window - 1, m), np.nan), values])
    win = sliding_window_view(padded, window, axis=0)  # (n, m, window), oldest first
    valid = ~np.isnan(win)
    count = valid.sum(axis=-1)  # >= 1 because the current value is always present

    y = np.where(valid, win, 0.0)
    mean = y.sum(axis=-1) / count
    yc = np.where(valid, win - mean[..., None], 0.0)
    std = np.sqrt((yc**2).sum(axis=-1) / count)

    # Positions 0..window-1 within the window; consecutive rows are consecutive cycles.
    x = np.broadcast_to(np.arange(window, dtype=np.float64), win.shape)
    xmean = np.where(valid, x, 0.0).sum(axis=-1) / count
    xc = np.where(valid, x - xmean[..., None], 0.0)
    sxx = (xc**2).sum(axis=-1)
    sxy = (xc * yc).sum(axis=-1)
    slope = np.divide(sxy, sxx, out=np.zeros_like(sxy), where=sxx > 0)
    return mean, std, slope


def rolling_features(df: pd.DataFrame, sensors: Sequence[str], window: int) -> pd.DataFrame:
    """Trailing rolling mean/std/slope per engine for each sensor.

    `df` must contain `unit` and `cycle` with contiguous cycles per unit
    (guaranteed for C-MAPSS by the Stage 1 tests). Returns a DataFrame with
    the same index as `df` and columns `feature_names(sensors)`.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    sensors = list(sensors)
    out = np.empty((len(df), 3 * len(sensors)))
    order = np.argsort(df["unit"].to_numpy(), kind="stable")
    sorted_df = df.iloc[order]
    if not (sorted_df.groupby("unit")["cycle"].diff().dropna() == 1).all():
        raise ValueError("cycles must be sorted and contiguous within each unit")
    vals = sorted_df[sensors].to_numpy(dtype=np.float64)
    units = sorted_df["unit"].to_numpy()
    bounds = np.flatnonzero(np.diff(units)) + 1
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(units)]):
        mean, std, slope = _window_stats(vals[start:stop], window)
        block = np.stack([mean, std, slope], axis=-1).reshape(stop - start, -1)  # sensor-major
        out[order[start:stop]] = block
    return pd.DataFrame(out, index=df.index, columns=feature_names(sensors))
