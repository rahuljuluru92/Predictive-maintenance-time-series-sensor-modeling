"""Stage 6 follow-up: why does leaving FD004 out of training hurt FD004 so much?

Hypothesis: FD004 is the only subset combining the fan fault mode with the
five non-sea-level operating conditions. Without FD004, the training fleet
has fan degradation only at sea level (FD003), so the model and the
per-condition normalization have never seen fan degradation in the other
conditions. If so, the leave-one-out loss should be concentrated in cycles
flown at non-sea-level conditions.

Baseline pipeline only (the drift stage is not needed for this question).
Compares the pooled model (all four subsets) with the leave-FD004-out model
on the FD004 official test set, split by the current cycle's operating
condition. Writes results/investigate_fd004.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdm.conditions import ConditionNormalizer, condition_ids, load_pooled  # noqa: E402
from pdm.data import SENSOR_COLS  # noqa: E402
from pdm.features import add_failure_label, informative_sensors  # noqa: E402
from pdm.metrics import evaluate  # noqa: E402
from pdm.model import HORIZON, THRESHOLD  # noqa: E402
from pdm.pipeline import fit_pipeline  # noqa: E402

SEA_LEVEL = "0|0|100"


def run(train, test):
    norm = ConditionNormalizer().fit(train, SENSOR_COLS)
    trn, ten = norm.transform(train), norm.transform(test)
    sensors = informative_sensors(trn, SENSOR_COLS)
    p, _ = fit_pipeline("baseline", trn, sensors).predict_proba(ten)
    return p


def split_metrics(test, p):
    cond = condition_ids(test).to_numpy()
    y = test["label"].to_numpy()
    out = {"all": evaluate(y, p, THRESHOLD)}
    for name, m in (("sea_level", cond == SEA_LEVEL), ("other_conditions", cond != SEA_LEVEL)):
        out[name] = evaluate(y[m], p[m], THRESHOLD)
    # engines with at least one false alert
    fa = (p >= THRESHOLD) & (y == 0)
    out["engines_with_a_false_alert"] = int(test.assign(fa=fa).groupby("unit")["fa"].any().sum())
    out["engines"] = int(test["unit"].nunique())
    return out


def main() -> None:
    pooled = load_pooled()
    train = add_failure_label(pooled["train"], HORIZON)
    test = add_failure_label(pooled["test"], HORIZON)
    fd004 = test[test["subset"] == "FD004"].reset_index(drop=True)

    p_pooled = run(train, fd004)
    p_loso = run(train[train["subset"] != "FD004"].reset_index(drop=True), fd004)
    res = {
        "hypothesis": "leave-FD004-out loss concentrated in non-sea-level cycles (fan fault never seen there in training)",
        "share_of_fd004_test_cycles_at_sea_level": float((condition_ids(fd004) == SEA_LEVEL).mean()),
        "pooled_model": split_metrics(fd004, p_pooled),
        "leave_fd004_out_model": split_metrics(fd004, p_loso),
    }
    path = ROOT / "results" / "investigate_fd004.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    for model in ("pooled_model", "leave_fd004_out_model"):
        r = res[model]
        print(model, "| engines with a false alert:", r["engines_with_a_false_alert"], "/", r["engines"])
        for part in ("all", "sea_level", "other_conditions"):
            m = r[part]
            print(f"  {part:16s} n {m['n']:6d} pos {m['n_pos']:4d}  AUC {m['roc_auc']:.4f}  recall {m['recall']:.4f}  false-alert {m['false_alert_rate']:.5f}  precision {m['precision']:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
