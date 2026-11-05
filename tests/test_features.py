"""Rolling-window features and labels, checked against hand-calculated values."""

import math

import numpy as np
import pandas as pd
import pytest

from pdm.features import add_failure_label, feature_names, informative_sensors, rolling_features


def _frame(units_values: dict[int, list[float]], interleave: bool = False) -> pd.DataFrame:
    rows = [(u, c + 1, v) for u, vals in units_values.items() for c, v in enumerate(vals)]
    if interleave:  # mix units together; cycles stay in order within each unit
        rows.sort(key=lambda r: (r[1], r[0]))
    return pd.DataFrame(rows, columns=["unit", "cycle", "s"])


# Hand calculations, window = 3, population std (ddof=0), slope per cycle.
# Unit 1: y = 1, 2, 4, 8, 16
#   t1 [1]         mean 1     std 0          slope 0
#   t2 [1,2]       mean 1.5   std 0.5        slope 1
#   t3 [1,2,4]     mean 7/3   std sqrt(14)/3  slope (-1*1 + 1*4)/2   = 1.5
#   t4 [2,4,8]     mean 14/3  std sqrt(56)/3  slope (-1*2 + 1*8)/2   = 3
#   t5 [4,8,16]    mean 28/3  std sqrt(224)/3 slope (-1*4 + 1*16)/2  = 6
# Unit 2: y = 10, 10, 7
#   t1 [10]        mean 10    std 0          slope 0
#   t2 [10,10]     mean 10    std 0          slope 0
#   t3 [10,10,7]   mean 9     std sqrt(2)    slope (-1*10 + 1*7)/2   = -1.5
EXPECTED = {
    1: {
        "mean": [1, 1.5, 7 / 3, 14 / 3, 28 / 3],
        "std": [0, 0.5, math.sqrt(14) / 3, math.sqrt(56) / 3, math.sqrt(224) / 3],
        "slope": [0, 1, 1.5, 3, 6],
    },
    2: {"mean": [10, 10, 9], "std": [0, 0, math.sqrt(2)], "slope": [0, 0, -1.5]},
}


@pytest.mark.parametrize("interleave", [False, True])
def test_rolling_features_match_hand_calculation(interleave):
    df = _frame({1: [1, 2, 4, 8, 16], 2: [10, 10, 7]}, interleave=interleave)
    f = rolling_features(df, ["s"], window=3)
    assert list(f.columns) == ["s_mean", "s_std", "s_slope"]
    assert f.index.equals(df.index)
    for unit, exp in EXPECTED.items():
        rows = f[df["unit"] == unit]
        for kind in ("mean", "std", "slope"):
            assert rows[f"s_{kind}"].to_numpy() == pytest.approx(exp[kind], abs=1e-12), (unit, kind)


def test_no_leakage_across_units():
    # Unit 2's first window must not see unit 1's values, whatever their size.
    a = rolling_features(_frame({1: [1e6, -1e6, 5e5], 2: [3, 4]}), ["s"], window=5)
    b = rolling_features(_frame({2: [3, 4]}), ["s"], window=5)
    assert a.iloc[3:].to_numpy() == pytest.approx(b.to_numpy(), abs=1e-12)


def test_window_one_is_identity():
    df = _frame({1: [3.0, -2.0, 7.5]})
    f = rolling_features(df, ["s"], window=1)
    assert f["s_mean"].tolist() == [3.0, -2.0, 7.5]
    assert f["s_std"].tolist() == [0.0, 0.0, 0.0]
    assert f["s_slope"].tolist() == [0.0, 0.0, 0.0]


def test_large_offset_does_not_lose_precision():
    # Same shape as unit 1 but shifted by 1e6: mean shifts, std/slope identical.
    base = rolling_features(_frame({1: [1, 2, 4, 8, 16]}), ["s"], window=3)
    shifted = rolling_features(_frame({1: [1e6 + v for v in [1, 2, 4, 8, 16]]}), ["s"], window=3)
    assert (shifted["s_mean"] - 1e6).to_numpy() == pytest.approx(base["s_mean"].to_numpy(), abs=1e-9)
    assert shifted["s_std"].to_numpy() == pytest.approx(base["s_std"].to_numpy(), abs=1e-9)
    assert shifted["s_slope"].to_numpy() == pytest.approx(base["s_slope"].to_numpy(), abs=1e-9)


def test_multiple_sensors_column_order():
    df = _frame({1: [1, 2, 4]}).assign(t=[5.0, 5.0, 5.0])
    f = rolling_features(df, ["s", "t"], window=2)
    assert list(f.columns) == feature_names(["s", "t"]) == ["s_mean", "s_std", "s_slope", "t_mean", "t_std", "t_slope"]
    assert f["t_mean"].tolist() == [5.0, 5.0, 5.0] and f["s_slope"].tolist() == [0.0, 1.0, 2.0]


def test_non_contiguous_cycles_rejected():
    df = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 2, 4], "s": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="contiguous"):
        rolling_features(df, ["s"], window=2)


def test_failure_label_horizon_inclusive():
    df = pd.DataFrame({"rul": [100, 31, 30, 29, 1, 0]})
    assert add_failure_label(df, 30)["label"].tolist() == [0, 0, 1, 1, 1, 1]


def test_informative_sensors_drops_constants():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0], "c": [np.pi] * 3})
    assert informative_sensors(df, ["a", "b", "c"]) == ["b"]


# ------------------------------------------------- Stage 4: corrected features

from pdm.drift import DriftDetector, DriftResult, inject_drift  # noqa: E402
from pdm.features import corrected_rolling_features  # noqa: E402
from pdm.synthetic import SENSORS, make_fleet  # noqa: E402


def _drift_result(df, flags, resid, Q=None, sigma=None):
    """Hand-built DriftResult: flags/residuals per sensor, drift estimate = Q_FF^-1 r_F * sigma."""
    cols = list(flags)
    m = len(cols)
    zeros = pd.DataFrame(0.0, index=df.index, columns=cols)
    return DriftResult(
        delta=zeros,
        residual=pd.DataFrame({c: np.asarray(resid[c], dtype=float) for c in cols}, index=df.index),
        score=pd.Series(0.0, index=df.index),
        anomalous=pd.Series(False, index=df.index),
        flags=pd.DataFrame({c: np.asarray(flags[c], dtype=bool) for c in cols}, index=df.index),
        estimate=zeros,
        complement=np.eye(m) if Q is None else np.asarray(Q, dtype=float),
        noise_sigma=np.ones(m) if sigma is None else np.asarray(sigma, dtype=float),
    )


F_, T_ = False, True

# One engine, window 3, x = 1 2 4 8 16, drift estimate d = 0 0 1 2 3, flagged from cycle 4.
#   cycles 1-3 not flagged -> raw features (see EXPECTED[1] above)
#   cycle 4: window cycles 2..4 corrected: [2-0, 4-1, 8-2] = [2, 3, 6]
#            mean 11/3; deviations -5/3, -2/3, 7/3 -> var (25+4+49)/27 = 26/9 -> std sqrt(26)/3
#            slope (-1*2 + 1*6)/2 = 2
#   cycle 5: window cycles 3..5 corrected: [4-1, 8-2, 16-3] = [3, 6, 13]
#            mean 22/3; deviations -13/3, -4/3, 17/3 -> var (169+16+289)/27 = 158/9 -> std sqrt(158)/3
#            slope (-1*3 + 1*13)/2 = 5
# Note cycle 3's value is corrected inside the cycle-4 window even though cycle 3
# itself was not flagged: the whole window uses the flag set known at cycle 4.
CORRECTED_EXPECTED = {
    "mean": [1, 1.5, 7 / 3, 11 / 3, 22 / 3],
    "std": [0, 0.5, math.sqrt(14) / 3, math.sqrt(26) / 3, math.sqrt(158) / 3],
    "slope": [0, 1, 1.5, 2, 5],
}


def test_corrected_features_match_hand_calculation():
    df = _frame({1: [1, 2, 4, 8, 16]})
    drift = _drift_result(df, {"s": [F_, F_, F_, T_, T_]}, {"s": [0, 0, 1, 2, 3]})
    f = corrected_rolling_features(df, ["s"], 3, drift)
    for kind, exp in CORRECTED_EXPECTED.items():
        assert f[f"s_{kind}"].to_numpy() == pytest.approx(exp, abs=1e-12), kind


def test_corrected_features_convert_noise_units_to_raw_units():
    # sigma = 2 and residual halved gives the same raw-unit estimate as above.
    df = _frame({1: [1, 2, 4, 8, 16]})
    drift = _drift_result(df, {"s": [F_, F_, F_, T_, T_]}, {"s": [0, 0, 0.5, 1, 1.5]}, sigma=[2.0])
    f = corrected_rolling_features(df, ["s"], 3, drift)
    assert f["s_mean"].to_numpy() == pytest.approx(CORRECTED_EXPECTED["mean"], abs=1e-12)
    assert f["s_slope"].to_numpy() == pytest.approx(CORRECTED_EXPECTED["slope"], abs=1e-12)


def test_unflagged_sensor_keeps_raw_features():
    df = _frame({1: [1, 2, 4, 8, 16]}).assign(t=[5.0, 3.0, 9.0, 1.0, 2.0])
    drift = _drift_result(df, {"s": [F_, F_, F_, T_, T_], "t": [F_] * 5}, {"s": [0, 0, 1, 2, 3], "t": [7, 7, 7, 7, 7]})
    f = corrected_rolling_features(df, ["s", "t"], 3, drift)
    raw = rolling_features(df, ["s", "t"], 3)
    assert f[["t_mean", "t_std", "t_slope"]].equals(raw[["t_mean", "t_std", "t_slope"]])


def test_joint_estimate_changes_when_a_second_sensor_is_flagged():
    # Constant streams a = b = 10, residuals r_a = 3, r_b = 0, Q_FF = [[1, .5], [.5, 1]].
    # Cycles 2-3: only a flagged -> d_a = r_a / Q_aa = 3            -> a = 7
    # Cycles 4-5: a and b flagged -> d = Q_FF^-1 r = (4/3*3, -2/3*3) = (4, -2) -> a = 6, b = 12
    df = pd.DataFrame({"unit": 1, "cycle": [1, 2, 3, 4, 5], "a": 10.0, "b": 10.0})
    drift = _drift_result(
        df,
        {"a": [F_, T_, T_, T_, T_], "b": [F_, F_, F_, T_, T_]},
        {"a": [3.0] * 5, "b": [0.0] * 5},
        Q=[[1.0, 0.5], [0.5, 1.0]],
    )
    f = corrected_rolling_features(df, ["a", "b"], 2, drift)
    assert f["a_mean"].to_numpy() == pytest.approx([10, 7, 7, 6, 6], abs=1e-12)
    assert f["b_mean"].to_numpy() == pytest.approx([10, 10, 10, 12, 12], abs=1e-12)
    assert f["a_std"].abs().max() < 1e-12 and f["a_slope"].abs().max() < 1e-12


def test_flags_in_one_engine_do_not_affect_another():
    df = _frame({1: [1, 2, 4, 8, 16], 2: [1, 2, 4, 8, 16]})
    flags = [F_] * 5 + [F_, F_, F_, T_, T_]
    drift = _drift_result(df, {"s": flags}, {"s": [9, 9, 9, 9, 9, 0, 0, 1, 2, 3]})
    f = corrected_rolling_features(df, ["s"], 3, drift)
    raw = rolling_features(df, ["s"], 3)
    assert f[df["unit"] == 1].equals(raw[df["unit"] == 1])
    assert f.loc[df["unit"] == 2, "s_mean"].to_numpy() == pytest.approx(CORRECTED_EXPECTED["mean"], abs=1e-12)


def test_no_flags_means_identical_to_raw_features():
    df = _frame({1: [1, 2, 4, 8, 16], 2: [10, 10, 7]})
    drift = _drift_result(df, {"s": [F_] * 8}, {"s": np.arange(8.0)})
    assert corrected_rolling_features(df, ["s"], 3, drift).equals(rolling_features(df, ["s"], 3))


def test_window_one_matches_online_corrected_stream():
    # Integration with the real detector: with a 1-cycle window the corrected
    # mean is exactly the online corrected stream x(t) - flag(t) * estimate(t).
    det = DriftDetector().fit(make_fleet(30, np.random.default_rng(0)), SENSORS)
    df = make_fleet(3, np.random.default_rng(11), length_range=(220, 220))
    plan = pd.DataFrame([(2, "x4", 60, 0.08)], columns=["unit", "sensor", "onset", "rate"])
    df = inject_drift(df, plan)
    drift = det.transform(df)
    assert drift.flags["x4"].any()  # the injected drift is detected
    f = corrected_rolling_features(df, SENSORS, 1, drift)
    online = drift.corrected_streams(df)
    for s in SENSORS:
        assert f[f"{s}_mean"].to_numpy() == pytest.approx(online[s].to_numpy(), abs=1e-9)


def test_corrected_features_reject_misaligned_drift_result():
    df = _frame({1: [1, 2, 4]})
    drift = _drift_result(df, {"s": [F_, F_, T_]}, {"s": [0, 0, 1]})
    with pytest.raises(ValueError, match="aligned"):
        corrected_rolling_features(df.set_index(df.index + 100), ["s"], 2, drift)
    with pytest.raises(ValueError, match="no drift output"):
        corrected_rolling_features(df.assign(t=1.0), ["s", "t"], 2, drift)
