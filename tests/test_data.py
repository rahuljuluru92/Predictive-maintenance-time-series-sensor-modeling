"""Tests for pdm.data.

The synthetic-file tests always run. The real-data tests check every subset
against the dataset's documented format and skip if the data hasn't been
downloaded (run scripts/download_data.py).
"""

import numpy as np
import pandas as pd
import pytest

from pdm import data
from pdm.data import COLUMNS, DEFAULT_DATA_DIR, SENSOR_COLS, SETTING_COLS, SUBSETS


def _write(path, rows):
    # Mimic the raw files: space separated with trailing spaces.
    path.write_text("".join(" ".join(str(v) for v in r) + "  \n" for r in rows))


def _row(unit, cycle, base=0.0):
    return [unit, cycle, 0.1, 0.2, 100.0] + [base + i for i in range(1, 22)]


# ---------------------------------------------------------------- synthetic


def test_read_file_columns_dtypes_and_sorting(tmp_path):
    p = tmp_path / "train_X.txt"
    _write(p, [_row(2, 1), _row(1, 2), _row(1, 1)])
    df = data.read_cmapss_file(p)
    assert list(df.columns) == COLUMNS
    assert df.shape == (3, 26)
    assert df["unit"].dtype == np.int64 and df["cycle"].dtype == np.int64
    assert all(df[c].dtype == np.float64 for c in SETTING_COLS + SENSOR_COLS)
    assert df[["unit", "cycle"]].values.tolist() == [[1, 1], [1, 2], [2, 1]]
    assert df.loc[0, "s1"] == 1.0 and df.loc[0, "s21"] == 21.0


def test_read_file_rejects_wrong_column_count(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("1 1 0 0 100 1 2 3\n")
    with pytest.raises(ValueError, match="expected 26 columns"):
        data.read_cmapss_file(p)


def test_train_rul_counts_down_to_zero(tmp_path):
    p = tmp_path / "train.txt"
    _write(p, [_row(1, c) for c in range(1, 5)] + [_row(2, c) for c in range(1, 3)])
    df = data.add_train_rul(data.read_cmapss_file(p))
    assert df["rul"].tolist() == [3, 2, 1, 0, 1, 0]


def test_test_rul_offsets_from_final_rul(tmp_path):
    p = tmp_path / "test.txt"
    r = tmp_path / "rul.txt"
    _write(p, [_row(1, c) for c in range(1, 4)] + [_row(2, c) for c in range(1, 3)])
    r.write_text("10\n5\n")
    rul = data.read_rul_file(r)
    assert rul.index.tolist() == [1, 2] and rul.tolist() == [10, 5]
    df = data.add_test_rul(data.read_cmapss_file(p), rul)
    assert df["rul"].tolist() == [12, 11, 10, 6, 5]


def test_test_rul_unit_mismatch_raises(tmp_path):
    p = tmp_path / "test.txt"
    r = tmp_path / "rul.txt"
    _write(p, [_row(1, 1), _row(2, 1)])
    r.write_text("10\n")
    with pytest.raises(ValueError):
        data.add_test_rul(data.read_cmapss_file(p), data.read_rul_file(r))


def test_unknown_subset_raises():
    with pytest.raises(ValueError):
        data.load_subset("FD005")


def test_sensor_info_covers_all_sensors():
    assert list(data.SENSOR_INFO) == SENSOR_COLS


# ---------------------------------------------------------------- real data

needs_data = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "train_FD001.txt").exists(),
    reason="C-MAPSS data not downloaded; run scripts/download_data.py",
)

# Engine counts as found in the files. The dataset readme agrees for
# FD001-FD003 but lists FD004 as 248 train / 249 test; the files contain
# 249 train / 248 test (and RUL_FD004 has 248 entries, matching the test file).
EXPECTED = {
    #         train units, test units, train rows, test rows, operating conditions
    "FD001": (100, 100, 20631, 13096, 1),
    "FD002": (260, 259, 53759, 33991, 6),
    "FD003": (100, 100, 24720, 16596, 1),
    "FD004": (249, 248, 61249, 41214, 6),
}


@pytest.fixture(scope="module", params=SUBSETS)
def loaded(request):
    return request.param, data.load_subset(request.param)


@needs_data
def test_shapes_and_units(loaded):
    subset, d = loaded
    n_train, n_test, r_train, r_test, _ = EXPECTED[subset]
    for split, n_units, n_rows in (("train", n_train, r_train), ("test", n_test, r_test)):
        df = d[split]
        assert list(df.columns) == COLUMNS + ["rul"]
        assert df.shape == (n_rows, 27)
        assert df["unit"].nunique() == n_units
        assert sorted(df["unit"].unique()) == list(range(1, n_units + 1))
        assert not df.isna().any().any()


@needs_data
def test_cycles_contiguous_from_one(loaded):
    _, d = loaded
    for df in d.values():
        g = df.groupby("unit")["cycle"]
        assert (g.min() == 1).all()
        assert (g.max() == g.count()).all()  # no gaps or duplicates


@needs_data
def test_rul_properties(loaded):
    _, d = loaded
    train, test = d["train"], d["test"]
    assert (train.groupby("unit")["rul"].min() == 0).all()
    assert (train["rul"] >= 0).all()
    assert (test["rul"] > 0).all()  # test trajectories end before failure
    # RUL decreases by exactly 1 per cycle within each unit.
    for df in (train, test):
        assert (df.groupby("unit")["rul"].diff().dropna() == -1).all()


@needs_data
def test_operating_condition_count(loaded):
    subset, d = loaded
    n_cond = EXPECTED[subset][4]
    # Settings are noisy around discrete condition points; rounding recovers them.
    conds = d["train"][SETTING_COLS].round({"setting_1": 0, "setting_2": 2, "setting_3": 0}).drop_duplicates()
    assert len(conds) == n_cond


@needs_data
def test_sensor_values_physically_plausible():
    # FD001 runs at a single (sea-level) condition: fan-inlet temperature T2
    # should be the ISA sea-level value 518.67 degR and fan-inlet pressure P2
    # about 14.62 psia, per Table 2 units.
    train = data.load_subset("FD001")["train"]
    assert np.allclose(train["s1"], 518.67)
    assert np.allclose(train["s5"], 14.62)
    # Temperatures increase through the engine: T24 < T30 and T24 < T50.
    assert (train["s2"] < train["s3"]).all() and (train["s2"] < train["s4"]).all()
    # Physical fan speed Nf is ~2388 rpm at this condition.
    assert train["s8"].between(2380, 2400).all()
