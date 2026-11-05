"""Stage 6: per-subset generalization across FD001-FD004 for both pipelines.

1. Stratified group 5-fold CV over all pooled training engines (grouped by
   engine, stratified by subset); per-subset metrics on out-of-fold predictions.
2. Pooled model -> each subset's official test set.
3. Leave-one-subset-out: train on three subsets -> official test set of the fourth.
Design fixed in DECISIONS.md before running. Writes results/cross_validation.json.

Usage: python scripts/run_cross_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdm.conditions import ConditionNormalizer, load_pooled  # noqa: E402
from pdm.data import SENSOR_COLS, SUBSETS  # noqa: E402
from pdm.features import add_failure_label, informative_sensors  # noqa: E402
from pdm.metrics import evaluate  # noqa: E402
from pdm.model import HORIZON, THRESHOLD  # noqa: E402
from pdm.pipeline import PIPELINES, fit_pipeline, flag_summary  # noqa: E402

N_SPLITS = 5
KEYS = ("roc_auc", "recall", "false_alert_rate", "precision")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_and_predict(train: pd.DataFrame, evals: dict[str, pd.DataFrame]) -> tuple[dict, dict, list[str]]:
    """Normalize per condition (train stats), fit both pipelines, predict each eval frame."""
    norm = ConditionNormalizer().fit(train, SENSOR_COLS)
    trn = norm.transform(train)
    sensors = informative_sensors(trn, SENSOR_COLS)
    evals_n = {k: norm.transform(v) for k, v in evals.items()}
    probs, flags = {}, {}
    for kind in PIPELINES:
        fitted = fit_pipeline(kind, trn, sensors)
        for name, df in evals_n.items():
            p, drift = fitted.predict_proba(df)
            probs[(kind, name)] = p
            if drift is not None:
                flags[name] = {s: flag_summary(df[df["subset"] == s], drift.flags[df["subset"] == s]) for s in df["subset"].unique()}
    return probs, flags, sensors


def per_subset(df: pd.DataFrame, proba: np.ndarray) -> dict:
    out = {}
    for s in SUBSETS:
        m = (df["subset"] == s).to_numpy()
        if m.any():
            out[s] = evaluate(df.loc[m, "label"].to_numpy(), proba[m], THRESHOLD)
    return out


def main() -> None:
    pooled = load_pooled()
    train = add_failure_label(pooled["train"], HORIZON)
    test = add_failure_label(pooled["test"], HORIZON)
    result = {"subsets": list(SUBSETS), "horizon": HORIZON, "threshold": THRESHOLD}

    # ------------------------------------------------------------ 1. CV
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    oof = {kind: np.full(len(train), np.nan) for kind in PIPELINES}
    folds, fold_flags = [], []
    for k, (tr_idx, va_idx) in enumerate(cv.split(train, y=train["subset"], groups=train["unit"])):
        tr = train.iloc[np.sort(tr_idx)].reset_index(drop=True)
        va = train.iloc[np.sort(va_idx)].reset_index(drop=True)
        log(f"fold {k}: {tr['unit'].nunique()} train engines, {va['unit'].nunique()} validation engines")
        probs, flags, sensors = fit_and_predict(tr, {"val": va})
        fold = {"fold": k, "n_sensors": len(sensors),
                "validation_engines_by_subset": va.groupby("subset")["unit"].nunique().to_dict(),
                "validation_positive_share_by_subset": va.groupby("subset")["label"].mean().round(4).to_dict()}
        for kind in PIPELINES:
            p = probs[(kind, "val")]
            oof[kind][np.sort(va_idx)] = p
            fold[kind] = per_subset(va, p)
        folds.append(fold)
        fold_flags.append(flags["val"])
        log("  " + "  ".join(f"{kind} " + " ".join(f"{s}:{fold[kind][s]['roc_auc']:.4f}" for s in SUBSETS) for kind in PIPELINES))

    cv_summary = {}
    for kind in PIPELINES:
        cv_summary[kind] = {}
        for s in SUBSETS:
            vals = {key: [f[kind][s][key] for f in folds] for key in KEYS}
            m = (train["subset"] == s).to_numpy()
            cv_summary[kind][s] = {
                "fold_mean_std": {key: [float(np.mean(v)), float(np.std(v))] for key, v in vals.items()},
                "out_of_fold": evaluate(train.loc[m, "label"].to_numpy(), oof[kind][m], THRESHOLD),
            }
    flag_totals = {}
    for s in SUBSETS:
        eng = sum(f[s]["engines_with_any_flag"] for f in fold_flags)
        tot = sum(f[s]["engines"] for f in fold_flags)
        flag_totals[s] = {"validation_engines_with_any_flag": eng, "engines": tot}
    result["cross_validation"] = {"protocol": "StratifiedGroupKFold(5, shuffle, seed 0): groups=engine, stratified by subset",
                                  "folds": folds, "summary": cv_summary, "drift_flags_out_of_fold": flag_totals}

    # -------------------------------------------------- 2. official test sets
    log("pooled model -> official test sets")
    probs, flags, sensors = fit_and_predict(train, {"test": test})
    result["official_test_sets"] = {kind: per_subset(test, probs[(kind, "test")]) for kind in PIPELINES}
    result["official_test_sets"]["drift_flags"] = flags["test"]
    result["official_test_sets"]["n_sensors"] = len(sensors)

    # --------------------------------------------- 3. leave one subset out
    loso = {}
    for held in SUBSETS:
        log(f"leave-one-subset-out: holding out {held}")
        tr = train[train["subset"] != held].reset_index(drop=True)
        te = test[test["subset"] == held].reset_index(drop=True)
        probs, flags, _ = fit_and_predict(tr, {"test": te})
        loso[held] = {kind: evaluate(te["label"].to_numpy(), probs[(kind, "test")], THRESHOLD) for kind in PIPELINES}
        loso[held]["drift_flags"] = flags["test"][held]
    result["leave_one_subset_out"] = loso

    # ------------------------------------------------ generalization check
    check = {}
    for kind in PIPELINES:
        aucs = {s: cv_summary[kind][s]["fold_mean_std"]["roc_auc"][0] for s in SUBSETS}
        drops = {s: result["official_test_sets"][kind][s]["roc_auc"] - loso[s][kind]["roc_auc"] for s in SUBSETS}
        check[kind] = {
            "cv_mean_auc_by_subset": aucs,
            "cv_auc_spread_max_minus_min": max(aucs.values()) - min(aucs.values()),
            "loso_auc_drop_vs_pooled_test": drops,
            "criterion_cv_spread_le_0.02": max(aucs.values()) - min(aucs.values()) <= 0.02,
            "criterion_loso_drop_le_0.02": all(v <= 0.02 for v in drops.values()),
        }
    result["generalization_check"] = check

    path = ROOT / "results" / "cross_validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    log(f"wrote {path}")

    for kind in PIPELINES:
        print(f"\n== {kind} ==")
        print("subset  CV AUC (mean±std)   CV recall        CV false-alert      test AUC  LOSO AUC  test recall  LOSO recall")
        for s in SUBSETS:
            f = cv_summary[kind][s]["fold_mean_std"]
            t, lo = result["official_test_sets"][kind][s], loso[s][kind]
            print(f"{s}   {f['roc_auc'][0]:.4f}±{f['roc_auc'][1]:.4f}   {f['recall'][0]:.4f}±{f['recall'][1]:.4f}   "
                  f"{f['false_alert_rate'][0]:.5f}±{f['false_alert_rate'][1]:.5f}   {t['roc_auc']:.4f}    {lo['roc_auc']:.4f}    "
                  f"{t['recall']:.4f}       {lo['recall']:.4f}")
        print({k: v for k, v in check[kind].items() if k.startswith("criterion") or k == "cv_auc_spread_max_minus_min"})
    print("\nout-of-fold drift flags:", flag_totals)


if __name__ == "__main__":
    main()
