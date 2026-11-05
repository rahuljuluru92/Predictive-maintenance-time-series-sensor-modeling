"""Evaluation metrics for binary failure-within-horizon prediction.

Definitions used throughout the project:
  recall           = TP / (TP + FN)  share of failing-soon cycles that raise an alert
  false-alert rate = FP / (FP + TN)  share of healthy cycles that raise an alert
  precision        = TP / (TP + FP)  share of alerts that are correct
  ROC-AUC          = probability a random positive scores above a random
                     negative (ties count 1/2), via the Mann-Whitney U statistic
"""

from __future__ import annotations

import numpy as np


def _as_binary(y) -> np.ndarray:
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return y.astype(np.int64)


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    t, p = _as_binary(y_true), _as_binary(y_pred)
    if t.shape != p.shape:
        raise ValueError("shape mismatch")
    return {
        "tp": int(((t == 1) & (p == 1)).sum()),
        "fp": int(((t == 0) & (p == 1)).sum()),
        "tn": int(((t == 0) & (p == 0)).sum()),
        "fn": int(((t == 1) & (p == 0)).sum()),
    }


def _ratio(num: int, den: int) -> float:
    return float(num) / den if den else float("nan")


def recall(y_true, y_pred) -> float:
    c = confusion_counts(y_true, y_pred)
    return _ratio(c["tp"], c["tp"] + c["fn"])


def false_alert_rate(y_true, y_pred) -> float:
    c = confusion_counts(y_true, y_pred)
    return _ratio(c["fp"], c["fp"] + c["tn"])


def precision(y_true, y_pred) -> float:
    c = confusion_counts(y_true, y_pred)
    return _ratio(c["tp"], c["tp"] + c["fp"])


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given the average of the ranks they span."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    t = _as_binary(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if t.shape != s.shape:
        raise ValueError("shape mismatch")
    n_pos, n_neg = int(t.sum()), int((1 - t).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _average_ranks(s)
    u = r[t == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    """All metrics at a fixed decision threshold (alert when score >= threshold) plus ROC-AUC."""
    s = np.asarray(scores, dtype=np.float64)
    pred = (s >= threshold).astype(np.int64)
    c = confusion_counts(y_true, pred)
    return {
        "threshold": threshold,
        "n": int(len(s)),
        "n_pos": c["tp"] + c["fn"],
        **c,
        "recall": _ratio(c["tp"], c["tp"] + c["fn"]),
        "false_alert_rate": _ratio(c["fp"], c["fp"] + c["tn"]),
        "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
        "roc_auc": roc_auc(y_true, s),
    }
