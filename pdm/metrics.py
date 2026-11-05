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


def relative_change(new: float, old: float) -> float:
    """(new - old) / old; NaN when old is 0."""
    return (new - old) / old if old else float("nan")


def engine_counts(units, y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Confusion counts per engine: returns (engine ids, array n_engines x 4 of tp, fp, tn, fn)."""
    u = np.asarray(units)
    t, p = _as_binary(y_true), _as_binary(y_pred)
    ids, inv = np.unique(u, return_inverse=True)
    cells = np.column_stack([(t == 1) & (p == 1), (t == 0) & (p == 1), (t == 0) & (p == 0), (t == 1) & (p == 0)])
    counts = np.zeros((len(ids), 4), dtype=np.int64)
    np.add.at(counts, inv.ravel(), cells.astype(np.int64))
    return ids, counts


def _rates(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp, fp, tn, fn = (counts[..., i].astype(np.float64) for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fn), fp / (fp + tn)


def paired_bootstrap(units, y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Engine-level paired bootstrap of (b - a) for recall and false-alert rate.

    Engines are resampled with replacement; both models are always evaluated
    on the same resampled engines. Returns point estimates and percentile
    intervals for the absolute differences and the relative changes (b - a) / a.
    """
    _, ca = engine_counts(units, y_true, pred_a)
    _, cb = engine_counts(units, y_true, pred_b)
    n = len(ca)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)  # (n_boot, n_engines)
    ra, fa = _rates(w @ ca)
    rb, fb = _rates(w @ cb)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    pa, pfa = _rates(ca.sum(axis=0))
    pb, pfb = _rates(cb.sum(axis=0))

    def interval(x):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_r, rel_f = (rb - ra) / ra, (fb - fa) / fa
    return {
        "n_boot": n_boot,
        "level": level,
        "recall_diff": float(pb - pa),
        "recall_diff_ci": interval(rb - ra),
        "recall_rel_change": float(relative_change(pb, pa)),
        "recall_rel_change_ci": interval(rel_r),
        "false_alert_rate_diff": float(pfb - pfa),
        "false_alert_rate_diff_ci": interval(fb - fa),
        "false_alert_rate_rel_change": float(relative_change(pfb, pfa)),
        "false_alert_rate_rel_change_ci": interval(rel_f),
    }


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
