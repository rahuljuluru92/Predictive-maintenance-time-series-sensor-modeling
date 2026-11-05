"""The two pipelines compared in this project, sharing everything but drift correction.

baseline : rolling features of the raw sensor streams -> Random Forest
corrected: Isolation Forest drift detector (fitted on training engines) ->
           rolling features of the drift-corrected streams -> Random Forest
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pdm.drift import DriftDetector, DriftResult
from pdm.features import corrected_rolling_features, rolling_features
from pdm.model import WINDOW, make_classifier

PIPELINES = ("baseline", "corrected")


@dataclass
class FittedPipeline:
    kind: str
    sensors: list[str]
    classifier: RandomForestClassifier
    detector: DriftDetector | None = None
    window: int = WINDOW

    def features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, DriftResult | None]:
        if self.kind == "baseline":
            return rolling_features(df, self.sensors, self.window), None
        drift = self.detector.transform(df)
        return corrected_rolling_features(df, self.sensors, self.window, drift), drift

    def predict_proba(self, df: pd.DataFrame) -> tuple[np.ndarray, DriftResult | None]:
        X, drift = self.features(df)
        return self.classifier.predict_proba(X)[:, 1], drift


def fit_pipeline(kind: str, train: pd.DataFrame, sensors: Sequence[str], window: int = WINDOW) -> FittedPipeline:
    """Fit a pipeline on labelled training data (`label` column, sorted by unit, cycle)."""
    if kind not in PIPELINES:
        raise ValueError(f"unknown pipeline {kind!r}")
    sensors = list(sensors)
    detector = DriftDetector().fit(train, sensors) if kind == "corrected" else None
    fitted = FittedPipeline(kind, sensors, make_classifier(), detector, window)
    X, _ = fitted.features(train)
    fitted.classifier.fit(X, train["label"].to_numpy())
    return fitted


def flag_summary(df: pd.DataFrame, drift: DriftResult) -> dict:
    """How much of `df` the drift detector touched."""
    per_engine = drift.flags.groupby(df["unit"]).any()
    return {
        "engines_with_any_flag": int(per_engine.any(axis=1).sum()),
        "engines": int(df["unit"].nunique()),
        "rows_with_any_flag": int(drift.flags.any(axis=1).sum()),
        "rows": len(df),
        "flagged_engines_by_sensor": {k: int(v) for k, v in per_engine.sum().items() if v > 0},
    }
