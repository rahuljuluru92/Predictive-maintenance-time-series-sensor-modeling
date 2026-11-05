"""Stage 3: the drift detector must tell injected sensor drift apart from
injected degradation on synthetic data before it is trusted on C-MAPSS.

Synthetic fleet (pdm.synthetic): 8 noisy sensors (noise std 1), two fault
modes whose degradation moves several sensors together and accelerates
toward failure. Drift = a linear calibration ramp on one sensor only.
"""

import numpy as np
import pandas as pd
import pytest

from pdm.drift import DriftDetector, inject_drift, random_drift_plan
from pdm.synthetic import SENSORS, SIGNATURES, make_fleet

RATE = 0.05  # drift, noise units per cycle


@pytest.fixture(scope="module")
def detector():
    train = make_fleet(60, np.random.default_rng(0))  # drift-free training fleet
    return DriftDetector().fit(train, SENSORS)


def _plan(unit, sensor, onset, rate):
    return pd.DataFrame([(unit, sensor, onset, rate)], columns=["unit", "sensor", "onset", "rate"])


@pytest.fixture(scope="module")
def drift_grid(detector):
    """One engine per (fault mode, drifting sensor, drift direction): 32 cases."""
    cases = []
    seed = 100
    for mode in (0, 1):
        for sensor in SENSORS:
            for sign in (1.0, -1.0):
                seed += 1
                clean = make_fleet(1, np.random.default_rng(seed), length_range=(200, 260), modes=(mode,))
                onset = int(0.4 * len(clean))
                drifted = inject_drift(clean, _plan(1, sensor, onset, sign * RATE))
                cases.append((mode, sensor, onset, clean, drifted, detector.transform(drifted)))
    return cases


# ------------------------------------------------------------ learned model


def test_noise_scale_estimated(detector):
    assert detector.noise_sigma_ == pytest.approx(np.ones(len(SENSORS)), rel=0.05)


def test_degradation_subspace_contains_both_fault_signatures(detector):
    assert detector.k_ == 2
    V = detector.components_
    for sig in SIGNATURES:
        s = sig / np.linalg.norm(sig)
        assert np.linalg.norm(V.T @ s) > 0.99


# ---------------------------------------------- degradation is not drift


def test_true_degradation_is_not_flagged(detector):
    held_out = make_fleet(40, np.random.default_rng(1), first_unit=1000)
    flags = detector.transform(held_out).flags
    flagged_engines = flags.groupby(held_out["unit"]).any().any(axis=1).sum()
    assert flagged_engines <= 1  # of 40


def test_faster_degradation_is_still_not_flagged(detector):
    # Twice the training severity: larger deltas, but still in the degradation subspace.
    steep = make_fleet(20, np.random.default_rng(2), severity=16.0, first_unit=2000)
    flags = detector.transform(steep).flags
    assert flags.groupby(steep["unit"]).any().any(axis=1).sum() <= 1  # of 20


def test_clean_streams_left_unchanged(detector):
    held_out = make_fleet(10, np.random.default_rng(3))
    r = detector.transform(held_out)
    corrected = r.corrected_streams(held_out)
    unflagged = ~r.flags
    for s in SENSORS:
        assert (corrected.loc[unflagged[s], s] == held_out.loc[unflagged[s], s]).all()


# ---------------------------------------------------- drift is detected


def test_drift_detected_on_the_right_sensor_only(drift_grid):
    for mode, sensor, onset, _clean, drifted, r in drift_grid:
        flagged = list(r.flags.columns[r.flags.any()])
        assert flagged == [sensor], (mode, sensor, flagged)
        first = drifted.loc[r.flags[sensor], "cycle"].min()
        assert first > onset, (mode, sensor)


def test_drift_detected_while_still_small(drift_grid):
    # Degradation moves sensors by up to ~8 noise units at failure; drift should
    # be caught well before it reaches that size.
    sizes = []
    for _mode, sensor, onset, _clean, drifted, r in drift_grid:
        first = drifted.loc[r.flags[sensor], "cycle"].min()
        sizes.append(RATE * (first - onset))
    assert max(sizes) < 8.0
    assert np.median(sizes) < 4.0


def test_correction_recovers_the_clean_stream(drift_grid):
    # Expected estimate noise: the drift estimate inherits the noise of the
    # 10-cycle trailing mean and of the engine's fixed 10-cycle baseline, each
    # variance 0.1, scaled by 1/(1-P_jj) (~1.4 here): std ~0.5 noise units, of
    # which ~0.37 is a per-engine constant (baseline). So a single engine's mean
    # error can sit near 1 noise unit by chance; bounds are set at ~4 std of the
    # per-engine constant (1.5) and, across engines, a median of 0.75.
    maes = []
    for mode, sensor, onset, clean, drifted, r in drift_grid:
        corrected = r.corrected_streams(drifted)
        after = r.flags[sensor]
        err = (corrected.loc[after, sensor] - clean.loc[after, sensor]).abs()
        injected_end = RATE * (drifted["cycle"].iloc[-1] - onset)
        maes.append(err.mean())
        assert err.mean() < 1.5, (mode, sensor)
        assert err.iloc[-1] < 0.25 * injected_end, (mode, sensor)  # >= 75% of the drift removed
        others = [s for s in SENSORS if s != sensor]
        assert (corrected[others] == drifted[others]).all().all()  # nothing else touched
    assert np.median(maes) < 0.75


def test_drift_on_a_healthy_engine(detector):
    clean = make_fleet(1, np.random.default_rng(4), length_range=(250, 250), degrade=False)
    drifted = inject_drift(clean, _plan(1, "x3", 60, RATE))
    r = detector.transform(drifted)
    assert list(r.flags.columns[r.flags.any()]) == ["x3"]
    assert detector.transform(clean).flags.values.sum() == 0


# ------------------------------------------------------------- mechanics


def test_outputs_are_causal(detector):
    df = inject_drift(make_fleet(2, np.random.default_rng(5), length_range=(220, 220)), _plan(1, "x5", 80, -RATE))
    full = detector.transform(df)
    cut = 150
    part_df = df[df["cycle"] <= cut]
    part = detector.transform(part_df)
    for name in ("delta", "residual", "flags", "estimate"):
        a = getattr(full, name).loc[part_df.index].to_numpy()
        b = getattr(part, name).to_numpy()
        assert np.array_equal(a, b), name
    assert np.array_equal(full.score.loc[part_df.index].to_numpy(), part.score.to_numpy())


def test_short_engine_gets_no_flags(detector):
    short = make_fleet(1, np.random.default_rng(6), length_range=(8, 8))
    r = detector.transform(short)
    assert r.flags.values.sum() == 0 and (r.delta.to_numpy() == 0).all()


def test_rejects_unsorted_input(detector):
    df = make_fleet(1, np.random.default_rng(7), length_range=(30, 30))
    with pytest.raises(ValueError, match="sorted"):
        detector.transform(df.iloc[::-1])


def test_inject_drift_is_exact():
    df = make_fleet(2, np.random.default_rng(8), length_range=(20, 20))
    out = inject_drift(df, _plan(2, "x1", 5, 0.5))
    added = out["x1"] - df["x1"]
    expected = np.where((df["unit"] == 2) & (df["cycle"] > 5), 0.5 * (df["cycle"] - 5), 0.0)
    assert np.allclose(added, expected, atol=1e-12)
    assert out.drop(columns="x1").equals(df.drop(columns="x1"))


def test_random_drift_plan_modes():
    df = make_fleet(30, np.random.default_rng(9), length_range=(40, 60))
    rates = {"x0": 0.1, "x2": 0.3}
    ref = {"x0": -2.0, "x2": 5.0}
    rng = np.random.default_rng(10)
    mimic = random_drift_plan(df, rates, rng, min_onset=10, sign="mimic", reference_sign=ref)
    mask = random_drift_plan(df, rates, rng, min_onset=10, sign="mask", reference_sign=ref)
    assert len(mimic) == 30 and set(mimic["sensor"]) <= set(rates)
    last = df.groupby("unit")["cycle"].max()
    assert (mimic["onset"] >= 10).all() and (mimic["onset"] <= mimic["unit"].map(last) - 1).all()
    for plan, factor in ((mimic, 1.0), (mask, -1.0)):
        expect = plan["sensor"].map(lambda s: factor * np.sign(ref[s]) * rates[s])
        assert np.allclose(plan["rate"], expect)
