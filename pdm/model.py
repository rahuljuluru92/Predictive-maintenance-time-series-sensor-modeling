"""Classifier configuration and evaluation helpers shared by all pipelines.

Hyperparameters are fixed up front (not tuned on any test data) and are
identical for the baseline and drift-corrected models, so the only
difference between the two is the input features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pdm.metrics import evaluate

HORIZON = 30  # cycles; label = 1 when RUL <= HORIZON
WINDOW = 10  # cycles in each trailing rolling window
THRESHOLD = 0.5  # alert when predicted probability >= THRESHOLD
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}


def make_classifier(**overrides) -> RandomForestClassifier:
    return RandomForestClassifier(**{**RF_PARAMS, **overrides})


def fit_predict(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, **overrides) -> tuple[RandomForestClassifier, np.ndarray]:
    """Fit the Random Forest and return (model, P(label=1) for X_test)."""
    clf = make_classifier(**overrides)
    clf.fit(X_train, np.asarray(y_train))
    return clf, clf.predict_proba(X_test)[:, 1]


def evaluation_report(test: pd.DataFrame, proba: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Metrics over every test cycle, plus at each engine's last observed cycle.

    `test` must have `unit`, `cycle` and `label` columns aligned with `proba`.
    """
    per_cycle = evaluate(test["label"].to_numpy(), proba, threshold)
    last = test.assign(_p=proba).sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    per_engine_last = evaluate(last["label"].to_numpy(), last["_p"].to_numpy(), threshold)
    return {"per_cycle": per_cycle, "last_cycle_per_engine": per_engine_last}


def top_importances(clf: RandomForestClassifier, columns, k: int = 10) -> list[tuple[str, float]]:
    imp = pd.Series(clf.feature_importances_, index=list(columns)).sort_values(ascending=False)
    return [(name, round(float(v), 4)) for name, v in imp.head(k).items()]
