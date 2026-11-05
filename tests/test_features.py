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
