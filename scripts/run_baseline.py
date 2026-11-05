"""Stage 2: baseline Random Forest on raw (not drift-corrected) rolling features.

Trains on all training engines of a subset and evaluates on that subset's
official held-out test set (test_FDxxx.txt labelled with RUL_FDxxx.txt).
Writes results/baseline_<subset>.json.

Usage: python scripts/run_baseline.py [--subset FD001]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdm.data import SENSOR_COLS, load_subset  # noqa: E402
from pdm.features import add_failure_label, informative_sensors, rolling_features  # noqa: E402
from pdm.model import HORIZON, RF_PARAMS, THRESHOLD, WINDOW, evaluation_report, fit_predict, top_importances  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    args = ap.parse_args()

    d = load_subset(args.subset)
    train = add_failure_label(d["train"], HORIZON)
    test = add_failure_label(d["test"], HORIZON)
    sensors = informative_sensors(train, SENSOR_COLS)

    X_train = rolling_features(train, sensors, WINDOW)
    X_test = rolling_features(test, sensors, WINDOW)
    clf, proba = fit_predict(X_train, train["label"], X_test)
    report = evaluation_report(test, proba)

    result = {
        "pipeline": "baseline (raw rolling features, no drift correction)",
        "subset": args.subset,
        "train_rows": len(train),
        "train_units": int(train["unit"].nunique()),
        "test_rows": len(test),
        "test_units": int(test["unit"].nunique()),
        "horizon": HORIZON,
        "window": WINDOW,
        "threshold": THRESHOLD,
        "rf_params": RF_PARAMS,
        "sensors": sensors,
        "n_features": X_train.shape[1],
        **report,
        "top_importances": top_importances(clf, X_train.columns),
    }
    out = ROOT / "results" / f"baseline_{args.subset}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("per_cycle", "last_cycle_per_engine")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
