"""Metric code checked against hand-computed values (independent of any model)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pdm import metrics
from pdm.model import evaluation_report

# y_true = 1 1 0 0 0 1
# y_pred = 1 0 1 0 0 1   -> TP=2 (idx 0,5)  FN=1 (idx 1)  FP=1 (idx 2)  TN=2 (idx 3,4)
Y_TRUE = [1, 1, 0, 0, 0, 1]
Y_PRED = [1, 0, 1, 0, 0, 1]


def test_confusion_counts():
    assert metrics.confusion_counts(Y_TRUE, Y_PRED) == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}


def test_recall_false_alert_rate_precision():
    assert metrics.recall(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert metrics.false_alert_rate(Y_TRUE, Y_PRED) == pytest.approx(1 / 3)
    assert metrics.precision(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)


def test_rates_undefined_without_denominator():
    assert np.isnan(metrics.recall([0, 0], [1, 0]))  # no positives
    assert np.isnan(metrics.false_alert_rate([1, 1], [1, 0]))  # no negatives
    assert np.isnan(metrics.precision([1, 0], [0, 0]))  # no alerts


def test_roc_auc_hand_computed():
    # Positives 0.35 and 0.8 vs negatives 0.1 and 0.4:
    # 0.35>0.1 yes, 0.35>0.4 no, 0.8>0.1 yes, 0.8>0.4 yes -> 3/4
    assert metrics.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_roc_auc_ties_count_half():
    # Positives 0.5, 0.9 vs negatives 0.5, 0.2:
    # 0.5 vs 0.5 tie (1/2), 0.5>0.2 (1), 0.9 beats both (2) -> 3.5/4
    assert metrics.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9]) == pytest.approx(0.875)


def test_roc_auc_perfect_inverted_and_degenerate():
    assert metrics.roc_auc([0, 0, 1], [0.1, 0.2, 0.3]) == 1.0
    assert metrics.roc_auc([0, 0, 1], [0.3, 0.2, 0.1]) == 0.0
    assert metrics.roc_auc([0, 0, 0], [0.3, 0.2, 0.1]) != metrics.roc_auc([0, 0, 0], [0.3, 0.2, 0.1])  # NaN


def test_roc_auc_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)  # rounding creates many ties
    assert metrics.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        metrics.recall([0, 2], [0, 1])


def test_evaluate_threshold_is_inclusive():
    r = metrics.evaluate([1, 0, 1, 0], [0.5, 0.49, 0.9, 0.7], threshold=0.5)
    # alerts at 0.5, 0.9, 0.7 -> TP=2 FP=1 TN=1 FN=0
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 1, 0)
    assert r["recall"] == 1.0 and r["false_alert_rate"] == 0.5 and r["precision"] == pytest.approx(2 / 3)
    assert r["n"] == 4 and r["n_pos"] == 2


def test_evaluation_report_last_cycle_per_engine():
    test = pd.DataFrame(
        {"unit": [2, 1, 1, 2, 1], "cycle": [1, 1, 2, 2, 3], "label": [0, 0, 0, 1, 1]}
    )
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    r = evaluation_report(test, proba, threshold=0.5)
    assert r["per_cycle"]["n"] == 5
    # Last cycles: unit 1 cycle 3 (label 1, p 0.9 -> TP); unit 2 cycle 2 (label 1, p 0.4 -> FN)
    last = r["last_cycle_per_engine"]
    assert (last["n"], last["tp"], last["fn"], last["fp"], last["tn"]) == (2, 1, 1, 0, 0)


# ------------------------------------------------ Stage 5: comparison helpers


def test_relative_change():
    assert metrics.relative_change(0.5, 0.4) == pytest.approx(0.25)
    assert metrics.relative_change(0.3, 0.4) == pytest.approx(-0.25)
    assert np.isnan(metrics.relative_change(0.3, 0.0))


def test_engine_counts_hand_computed():
    # engine 7: (t,p) = (1,1) (0,1) (0,0)  -> tp 1, fp 1, tn 1, fn 0
    # engine 3: (t,p) = (1,0) (1,1)        -> tp 1, fp 0, tn 0, fn 1
    units = [7, 3, 7, 3, 7]
    y = [1, 1, 0, 1, 0]
    p = [1, 0, 1, 1, 0]
    ids, counts = metrics.engine_counts(units, y, p)
    assert ids.tolist() == [3, 7]
    assert counts.tolist() == [[1, 0, 0, 1], [1, 1, 1, 0]]
    assert counts.sum(axis=0).tolist() == list(metrics.confusion_counts(y, p).values())


def test_paired_bootstrap_identical_models_have_zero_difference():
    rng = np.random.default_rng(1)
    units = np.repeat(np.arange(20), 10)
    y = rng.integers(0, 2, 200)
    p = rng.integers(0, 2, 200)
    r = metrics.paired_bootstrap(units, y, p, p, n_boot=200, seed=0)
    assert r["recall_diff"] == 0 and r["recall_diff_ci"] == [0.0, 0.0]
    assert r["false_alert_rate_diff"] == 0 and r["false_alert_rate_diff_ci"] == [0.0, 0.0]


def test_paired_bootstrap_point_estimates_and_interval():
    # Model b catches every positive that a misses and adds no false alerts.
    units = np.repeat(np.arange(10), 4)
    y = np.tile([1, 1, 0, 0], 10)
    a = np.tile([1, 0, 1, 0], 10)  # recall 0.5, false-alert rate 0.5
    b = np.tile([1, 1, 1, 0], 10)  # recall 1.0, false-alert rate 0.5
    r = metrics.paired_bootstrap(units, y, a, b, n_boot=500, seed=3)
    assert r["recall_diff"] == pytest.approx(0.5)
    assert r["recall_rel_change"] == pytest.approx(1.0)
    assert r["false_alert_rate_diff"] == 0.0
    # every engine is identical, so every resample gives the same answer
    assert r["recall_diff_ci"] == pytest.approx([0.5, 0.5])
    assert r["recall_rel_change_ci"] == pytest.approx([1.0, 1.0])


def test_paired_bootstrap_is_seeded():
    rng = np.random.default_rng(2)
    units = np.repeat(np.arange(30), 5)
    y = rng.integers(0, 2, 150)
    a, b = rng.integers(0, 2, 150), rng.integers(0, 2, 150)
    r1 = metrics.paired_bootstrap(units, y, a, b, n_boot=300, seed=5)
    r2 = metrics.paired_bootstrap(units, y, a, b, n_boot=300, seed=5)
    assert r1 == r2
    lo, hi = r1["recall_diff_ci"]
    assert lo <= r1["recall_diff"] <= hi
