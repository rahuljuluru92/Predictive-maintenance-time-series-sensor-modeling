"""Stage 3 check of the drift detector on real C-MAPSS engines.

Uses training engines only (the official test set is reserved for Stage 5):
5-fold split over engines; the detector is fitted on 4/5 of the engines and
evaluated on the held-out 1/5, both as recorded (clean) and with one injected
drifting sensor per engine (design in DECISIONS.md). Writes
results/drift_detection_<subset>.json.

Usage: python scripts/run_drift_detection.py [--subset FD001]
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
from pdm.drift import DriftDetector, drift_injection_rates, inject_drift, random_drift_plan  # noqa: E402
from pdm.features import informative_sensors  # noqa: E402

N_FOLDS = 5
SEED = 7


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    det = df[df["detected"]]
    bins = pd.cut(df["final_drift_noise"], [0, 2, 4, 8, np.inf], right=False)
    return {
        "engines": len(df),
        "detected_on_injected_sensor": float(df["detected"].mean()),
        "first_flag_is_injected_sensor": float(df["first_correct"].mean()),
        "any_other_sensor_flagged": float(df["other_flagged"].mean()),
        "detection_rate_by_final_drift_noise_units": {
            str(k): [float(v), int(n)] for (k, v), n in zip(df.groupby(bins, observed=False)["detected"].mean().items(),
                                                          df.groupby(bins, observed=False).size())
        },
        "median_drift_at_detection_noise_units": float(det["drift_at_detection_noise"].median()),
        "median_detection_delay_cycles": float(det["delay"].median()),
        "median_correction_mae_noise_units": float(det["mae_noise"].median()),
        "median_share_of_final_drift_removed": float(det["share_removed"].median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    args = ap.parse_args()

    train = load_subset(args.subset)["train"]
    sensors = informative_sensors(train, SENSOR_COLS)
    units = train["unit"].unique()
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(units), N_FOLDS)

    clean_flagged, results = [], {"mimic": [], "mask": []}
    info = {}
    for fold, held in enumerate(folds):
        fit_df = train[~train["unit"].isin(held)]
        held_df = train[train["unit"].isin(held)]
        det = DriftDetector().fit(fit_df, sensors)
        rates, direction = drift_injection_rates(fit_df, sensors, det.noise_sigma_)
        sigma = pd.Series(det.noise_sigma_, index=sensors)
        if fold == 0:
            info = {
                "k": det.k_,
                "eigenvalue_ratio_to_median": np.round(det.eigenvalues_ / np.median(det.eigenvalues_), 2).tolist(),
                "detectable_share_1_minus_Pjj": dict(zip(sensors, np.round(1 - (det.components_**2).sum(1), 3).tolist())),
                "injection_rates_raw_per_cycle": {s: round(r, 5) for s, r in rates.items()},
                "injection_rates_noise_units_per_cycle": {s: round(r / sigma[s], 4) for s, r in rates.items()},
            }
        clean = det.transform(held_df)
        clean_flagged += clean.flags.groupby(held_df["unit"]).any().any(axis=1).tolist()

        for mode in ("mimic", "mask"):
            plan = random_drift_plan(held_df, rates, rng, min_onset=det.baseline + 1, sign=mode, reference_sign=direction)
            drifted = inject_drift(held_df, plan)
            r = det.transform(drifted)
            corrected = r.corrected_streams(drifted)
            for p in plan.itertuples(index=False):
                m = (drifted["unit"] == p.unit).to_numpy()
                f = r.flags[m]
                cyc = drifted.loc[m, "cycle"].to_numpy()
                final = abs(p.rate) * (cyc[-1] - p.onset) / sigma[p.sensor]
                row = {
                    "unit": p.unit, "sensor": p.sensor, "final_drift_noise": final,
                    "detected": bool(f[p.sensor].any()),
                    "other_flagged": bool(f.drop(columns=p.sensor).to_numpy().any()),
                    "first_correct": False,
                }
                if f.to_numpy().any():
                    first_idx = int(np.argmax(f.to_numpy().any(axis=1)))
                    row["first_correct"] = bool(f[p.sensor].to_numpy()[first_idx])
                if row["detected"]:
                    k = int(np.argmax(f[p.sensor].to_numpy()))
                    row["delay"] = int(cyc[k] - p.onset)
                    row["drift_at_detection_noise"] = abs(p.rate) * (cyc[k] - p.onset) / sigma[p.sensor]
                    err = (corrected.loc[m, p.sensor].to_numpy() - held_df.loc[m, p.sensor].to_numpy())[k:]
                    row["mae_noise"] = float(np.abs(err).mean() / sigma[p.sensor])
                    injected_end = abs(p.rate) * (cyc[-1] - p.onset)
                    row["share_removed"] = float(1 - abs(err[-1]) / injected_end)
                results[mode].append(row)

    out = {
        "subset": args.subset,
        "protocol": f"{N_FOLDS}-fold split over training engines; detector fitted on the other folds",
        "detector": {k: getattr(DriftDetector(), k) for k in ("window", "baseline", "eig_ratio", "alpha", "z_thresh", "persistence")},
        "fold0_model": info,
        "clean_engines_with_any_flag": [int(np.sum(clean_flagged)), len(clean_flagged)],
        "injected": {mode: summarize(rows) for mode, rows in results.items()},
    }
    path = ROOT / "results" / f"drift_detection_{args.subset}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
