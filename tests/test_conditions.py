"""Operating-condition identification and per-condition normalization (Stage 6)."""

import numpy as np
import pandas as pd
import pytest

from pdm.conditions import ConditionNormalizer, condition_ids, load_pooled
from pdm.data import DEFAULT_DATA_DIR


def _settings(rows):
    return pd.DataFrame(rows, columns=["setting_1", "setting_2", "setting_3"])


def test_condition_ids_round_noisy_settings():
    df = _settings([(-0.0004, 0.0003, 100.0), (41.9985, 0.8406, 100.0), (25.002, 0.6201, 60.0), (0.0087, -0.0006, 100.0)])
    assert condition_ids(df).tolist() == ["0|0|100", "42|0.84|100", "25|0.62|60", "0|0|100"]


def test_normalizer_hand_computed():
    # Condition A (0|0|100): s = 1, 3 -> mean 2, population std 1 -> z = -1, +1
    # Condition B (42|0.84|100): s = 10, 10, 10 -> std 0 -> z = 0
    train = _settings([(0, 0, 100), (0, 0, 100), (42, 0.84, 100), (42, 0.84, 100), (42, 0.84, 100)]).assign(s=[1.0, 3.0, 10.0, 10.0, 10.0])
    norm = ConditionNormalizer().fit(train, ["s"])
    assert norm.transform(train)["s"].tolist() == [-1.0, 1.0, 0.0, 0.0, 0.0]
    # New rows use the training statistics: A: (5 - 2) / 1 = 3; B: constant -> 0
    new = _settings([(0.001, 0.0, 100), (42.002, 0.8401, 100)]).assign(s=[5.0, 99.0])
    assert norm.transform(new)["s"].tolist() == [3.0, 0.0]


def test_normalizer_leaves_other_columns_alone():
    train = _settings([(0, 0, 100), (0, 0, 100)]).assign(s=[1.0, 3.0], unit=[1, 1], cycle=[1, 2])
    out = ConditionNormalizer().fit(train, ["s"]).transform(train)
    assert out[["unit", "cycle", "setting_1"]].equals(train[["unit", "cycle", "setting_1"]])


def test_normalizer_rejects_unseen_condition():
    norm = ConditionNormalizer().fit(_settings([(0, 0, 100), (0, 0, 100)]).assign(s=[1.0, 2.0]), ["s"])
    with pytest.raises(ValueError, match="not seen"):
        norm.transform(_settings([(20, 0.7, 100)]).assign(s=[1.0]))


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / "train_FD001.txt").exists(), reason="C-MAPSS data not downloaded")
def test_load_pooled_real_data():
    d = load_pooled()
    train, test = d["train"], d["test"]
    assert train.groupby("subset")["engine"].nunique().to_dict() == {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}
    assert test.groupby("subset")["engine"].nunique().to_dict() == {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
    for df in (train, test):
        assert (df["unit"] == df["subset"].str[-1].astype(int) * 1000 + df["engine"]).all()
        assert df["unit"].is_monotonic_increasing
        assert (df.groupby("unit")["cycle"].diff().dropna() == 1).all()
    # Six operating conditions in total, shared by all subsets; FD001/FD003 use one.
    conds = condition_ids(train)
    assert conds.nunique() == 6
    assert set(conds[train["subset"].isin(["FD001", "FD003"])]) == {"0|0|100"}
    assert set(condition_ids(test)) <= set(conds)
    # After normalization each sensor that varies within conditions has mean 0, std 1 per condition.
    norm = ConditionNormalizer().fit(train, ["s2", "s11"])
    z = norm.transform(train)
    stats = z.groupby(conds)[["s2", "s11"]].agg(["mean", "std"])
    assert np.allclose(stats.xs("mean", axis=1, level=1), 0, atol=1e-9)
    assert np.allclose(stats.xs("std", axis=1, level=1), 1, atol=1e-3)
