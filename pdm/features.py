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


def corrected_rolling_features(df: pd.DataFrame, sensors: Sequence[str], window: int, drift) -> pd.DataFrame:
    """Stage 4: rolling mean/std/slope recomputed from drift-corrected streams.

    `drift` is the pdm.drift.DriftResult for `df` (same index; `df` sorted by
    unit and cycle). At cycle t, let F(t) be the set of sensors flagged as
    drifting. For a sensor in F(t), its features at t are computed over the
    window of corrected values x_j(s) - d_j(s), s = t-w+1..t, where d(s) is
    the drift estimate for the set F(t) from the residual at cycle s (so the
    whole window is corrected consistently and the flag onset does not create
    an artificial step). Sensors not flagged at t keep their raw features.
    Uses only cycles <= t.
    """
    from pdm.drift import _check_sorted, _unit_blocks  # local import: drift is optional for baseline use

    sensors = list(sensors)
    dcols = list(drift.flags.columns)
    if missing := set(sensors) - set(dcols):
        raise ValueError(f"no drift output for sensors {sorted(missing)}")
    if not drift.flags.index.equals(df.index):
        raise ValueError("drift result is not aligned with df")
    _check_sorted(df)

    raw = rolling_features(df, sensors, window)
    out = raw.to_numpy().copy()
    col_of = {s: 3 * i for i, s in enumerate(sensors)}
    flags = drift.flags.to_numpy()
    values = df[dcols].to_numpy(dtype=np.float64)
    for start, stop in _unit_blocks(df["unit"].to_numpy()):
        fl = flags[start:stop]
        if not fl.any():
            continue
        patterns, inverse = np.unique(fl, axis=0, return_inverse=True)
        engine_rows = np.arange(start, stop)
        for p_idx, pattern in enumerate(patterns):
            F = np.flatnonzero(pattern)
            if F.size == 0:
                continue
            at = np.flatnonzero(inverse.ravel() == p_idx)  # cycles of this engine whose flag set is F
            corrected = values[start:stop][:, F] - drift.estimate_with(F, engine_rows)
            mean, std, slope = _window_stats(corrected, window)
            for k, j in enumerate(F):
                c = col_of.get(dcols[j])
                if c is None:
                    continue
                out[start + at, c] = mean[at, k]
                out[start + at, c + 1] = std[at, k]
                out[start + at, c + 2] = slope[at, k]
    return pd.DataFrame(out, index=df.index, columns=raw.columns)


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
