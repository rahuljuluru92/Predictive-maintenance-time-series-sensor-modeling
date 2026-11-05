"""Stage 5: drift-corrected classifier vs. the Stage 2 baseline.

1. Clean comparison on the official held-out test set (primary result).
2. Injected-drift stress test on the same test set (secondary, synthetic drift).
Design fixed in DECISIONS.md before running. Writes results/comparison_<subset>.json.

Usage: python scripts/run_comparison.py [--subset FD001] [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdm.data import SENSOR_COLS, load_subset  # noqa: E402
from pdm.drift import drift_injection_rates, inject_drift, random_drift_plan  # noqa: E402
from pdm.features import add_failure_label, informative_sensors  # noqa: E402
from pdm.metrics import paired_bootstrap, relative_change  # noqa: E402
from pdm.model import HORIZON, THRESHOLD, evaluation_report  # noqa: E402
from pdm.pipeline import fit_pipeline, flag_summary  # noqa: E402

KEYS = ("recall", "false_alert_rate", "precision", "roc_auc")


def mean_std(values: list[float]) -> list[float]:
    return [float(np.mean(values)), float(np.std(values))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    d = load_subset(args.subset)
    train = add_failure_label(d["train"], HORIZON)
    test = add_failure_label(d["test"], HORIZON)
    sensors = informative_sensors(train, SENSOR_COLS)

    base = fit_pipeline("baseline", train, sensors)
    corr = fit_pipeline("corrected", train, sensors)
    p_base, _ = base.predict_proba(test)
    p_corr, drift_test = corr.predict_proba(test)
    _, drift_train = corr.features(train)

    rep_base = evaluation_report(test, p_base)
    rep_corr = evaluation_report(test, p_corr)

    # The baseline must reproduce Stage 2 exactly.
    stage2 = json.loads((ROOT / "results" / f"baseline_{args.subset}.json").read_text())
    assert rep_base["per_cycle"] == stage2["per_cycle"], "baseline does not reproduce Stage 2"

    y = test["label"].to_numpy()
    pred_b = (p_base >= THRESHOLD).astype(int)
    pred_c = (p_corr >= THRESHOLD).astype(int)
    clean = {
        "baseline": rep_base,
        "corrected": rep_corr,
        "relative_change_per_cycle": {
            k: relative_change(rep_corr["per_cycle"][k], rep_base["per_cycle"][k]) for k in KEYS
        },
        "cycles_where_alert_decision_differs": int((pred_b != pred_c).sum()),
        "max_abs_probability_difference": float(np.abs(p_base - p_corr).max()),
        "bootstrap_corrected_minus_baseline": paired_bootstrap(test["unit"], y, pred_b, pred_c, n_boot=2000, seed=0),
        "detector_flags_train_in_sample": flag_summary(train, drift_train),
        "detector_flags_test": flag_summary(test, drift_test),
    }

    # ---------------------------------------------- injected-drift stress test
    rates, direction = drift_injection_rates(train, sensors, corr.detector.noise_sigma_)
    stress = {"design": {
        "rates_raw_per_cycle": {s: round(r, 5) for s, r in rates.items()},
        "degradation_direction": direction,
        "onset": f"uniform over cycles {corr.detector.baseline + 1} .. last-1 of each test engine",
        "seeds": list(range(args.seeds)),
        "note": "models trained on clean training data; drift injected into the test set only",
    }}
    for m_idx, mode in enumerate(("mimic", "mask")):
        runs = {"baseline": {k: [] for k in KEYS}, "corrected": {k: [] for k in KEYS}, "caught": [], "fp": {"baseline": [], "corrected": []}}
        for seed in range(args.seeds):
            rng = np.random.default_rng([m_idx, seed])
            plan = random_drift_plan(test, rates, rng, min_onset=corr.detector.baseline + 1, sign=mode, reference_sign=direction)
            drifted = inject_drift(test, plan)
            pb, _ = base.predict_proba(drifted)
            pc, dr = corr.predict_proba(drifted)
            for name, p in (("baseline", pb), ("corrected", pc)):
                r = evaluation_report(drifted, p)["per_cycle"]
                for k in KEYS:
                    runs[name][k].append(r[k])
                runs["fp"][name].append(r["fp"])
            caught = [bool(dr.flags.loc[drifted["unit"] == u, s].any()) for u, s in zip(plan["unit"], plan["sensor"])]
            runs["caught"].append(float(np.mean(caught)))
        summary = {
            name: {k: mean_std(v) for k, v in runs[name].items()} for name in ("baseline", "corrected")
        }
        summary["mean_false_alert_count"] = {n: float(np.mean(v)) for n, v in runs["fp"].items()}
        summary["relative_change_of_means"] = {
            k: relative_change(summary["corrected"][k][0], summary["baseline"][k][0]) for k in KEYS
        }
        summary["share_of_engines_where_injected_sensor_was_flagged"] = mean_std(runs["caught"])
        summary["per_seed"] = {
            name: {k: [round(v, 6) for v in runs[name][k]] for k in KEYS} for name in ("baseline", "corrected")
        }
        summary["per_seed"]["false_alert_count"] = runs["fp"]
        summary["seeds_where_corrected_has_fewer_false_alerts"] = int(
            sum(c < b for b, c in zip(runs["fp"]["baseline"], runs["fp"]["corrected"]))
        )
        summary["seeds_where_corrected_has_higher_recall"] = int(
            sum(c > b for b, c in zip(runs["baseline"]["recall"], runs["corrected"]["recall"]))
        )
        stress[mode] = summary

    out = {
        "subset": args.subset,
        "test_rows": len(test),
        "test_positive_rows": int(y.sum()),
        "clean_test_set": clean,
        "injected_drift_stress_test": stress,
    }
    path = ROOT / "results" / f"comparison_{args.subset}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    def row(name, r):
        return f"{name:10s} recall {r['recall']:.4f}  false-alert {r['false_alert_rate']:.5f}  precision {r['precision']:.4f}  AUC {r['roc_auc']:.4f}  (TP {r['tp']} FP {r['fp']} FN {r['fn']})"

    print("CLEAN TEST SET (per cycle)")
    print(row("baseline", rep_base["per_cycle"]))
    print(row("corrected", rep_corr["per_cycle"]))
    print("relative change:", {k: round(v, 4) for k, v in clean["relative_change_per_cycle"].items()})
    print("cycles with a different alert decision:", clean["cycles_where_alert_decision_differs"])
    print("bootstrap:", json.dumps(clean["bootstrap_corrected_minus_baseline"]))
    print("flags test:", clean["detector_flags_test"], "| train:", clean["detector_flags_train_in_sample"])
    for mode in ("mimic", "mask"):
        s = stress[mode]
        print(f"\nINJECTED DRIFT ({mode}), mean ± std over {args.seeds} seeds")
        for name in ("baseline", "corrected"):
            print(f"{name:10s} " + "  ".join(f"{k} {s[name][k][0]:.4f}±{s[name][k][1]:.4f}" for k in KEYS))
        print("relative change of means:", {k: round(v, 4) for k, v in s["relative_change_of_means"].items()})
        print("mean false-alert count:", s["mean_false_alert_count"], "| injected sensor flagged:", s["share_of_engines_where_injected_sensor_was_flagged"])
        print("seeds with fewer false alerts (corrected):", s["seeds_where_corrected_has_fewer_false_alerts"], "| seeds with higher recall (corrected):", s["seeds_where_corrected_has_higher_recall"], f"of {args.seeds}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
