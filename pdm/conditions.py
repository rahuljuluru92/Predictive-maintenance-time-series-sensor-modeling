"""Operating conditions and per-condition sensor normalization (Stage 6).

FD002 and FD004 switch between six operating conditions (altitude, Mach,
throttle) from one cycle to the next, which moves most sensors far more than
degradation does. Each cycle is assigned to its condition by rounding the
three operational settings (the six condition points are well separated; the
spread within each is < 0.01). Each sensor is then z-scored with the mean and
std of that condition in the training data. Sensors that are constant within
a condition get 0 for that condition. For single-condition subsets (FD001,
FD003) this is a per-sensor affine transform, which leaves Random Forest
splits and the noise-scaled drift detector unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pdm.data import SETTING_COLS, SUBSETS, load_subset

ROUNDING = {"setting_1": 0, "setting_2": 2, "setting_3": 0}


def condition_ids(df: pd.DataFrame) -> pd.Series:
    """Operating condition per row as a string key like '42|0.84|100'."""
    r = df[SETTING_COLS].round(ROUNDING) + 0.0  # + 0.0 turns -0.0 into 0.0
    keys = r["setting_1"].map("{:g}".format) + "|" + r["setting_2"].map("{:g}".format) + "|" + r["setting_3"].map("{:g}".format)
    return keys.rename("condition")


@dataclass
class ConditionNormalizer:
    tol: float = 1e-8
    stats_: dict = field(default_factory=dict, init=False, repr=False)  # condition -> (mean, std) arrays
    sensors_: list[str] = field(default_factory=list, init=False)

    def fit(self, train: pd.DataFrame, sensors: Sequence[str]) -> ConditionNormalizer:
        self.sensors_ = list(sensors)
        cond = condition_ids(train)
        for key, g in train[self.sensors_].groupby(cond):
            self.stats_[key] = (g.mean().to_numpy(), g.std(ddof=0).to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cond = condition_ids(df)
        unseen = set(cond.unique()) - set(self.stats_)
        if unseen:
            raise ValueError(f"operating conditions not seen in training: {sorted(unseen)}")
        x = df[self.sensors_].to_numpy(dtype=np.float64)
        z = np.zeros_like(x)
        for key, (mean, std) in self.stats_.items():
            m = (cond == key).to_numpy()
            if not m.any():
                continue
            ok = std > self.tol
            z[np.ix_(m, ok)] = (x[np.ix_(m, ok)] - mean[ok]) / std[ok]
        out = df.copy()
        out[self.sensors_] = z
        return out


def load_pooled(subsets: Sequence[str] = SUBSETS) -> dict[str, pd.DataFrame]:
    """Train and test data of several subsets in one frame each.

    Adds `subset` and keeps the original engine number as `engine`; `unit`
    becomes a globally unique id (1000 * subset number + engine, e.g. FD003
    engine 7 -> 3007) so engines stay sorted and distinct.
    """
    out = {"train": [], "test": []}
    for s in subsets:
        d = load_subset(s)
        for split, df in d.items():
            df = df.assign(subset=s, engine=df["unit"])
            df["unit"] = int(s[-1]) * 1000 + df["engine"]
            out[split].append(df)
    return {k: pd.concat(v, ignore_index=True).sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True) for k, v in out.items()}
