"""Loading utilities for the NASA C-MAPSS turbofan degradation dataset.

Raw file format (per the dataset readme): whitespace-separated text, one row
per engine per operational cycle, 26 numeric columns:
unit number, cycle, 3 operational settings, 21 sensor measurements.
Files end each line with trailing spaces, which whitespace splitting absorbs.

Sensor symbols and units follow Table 2 of Saxena et al. (PHM08),
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation",
which ships with the dataset. The readme itself labels columns only as
"sensor measurement 1..21"; the s1..s21 -> symbol mapping assumes the
columns are in Table 2 order (see ASSUMPTIONS.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

# (symbol, description, units) in Table 2 order, keyed by column name.
SENSOR_INFO = {
    "s1": ("T2", "Total temperature at fan inlet", "degR"),
    "s2": ("T24", "Total temperature at LPC outlet", "degR"),
    "s3": ("T30", "Total temperature at HPC outlet", "degR"),
    "s4": ("T50", "Total temperature at LPT outlet", "degR"),
    "s5": ("P2", "Pressure at fan inlet", "psia"),
    "s6": ("P15", "Total pressure in bypass-duct", "psia"),
    "s7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "s11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "-"),
    "s16": ("farB", "Burner fuel-air ratio", "-"),
    "s17": ("htBleed", "Bleed enthalpy", "-"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "CMAPSSData"


def _check_subset(subset: str) -> None:
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}")


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Parse one train_/test_ file into a DataFrame with named columns.

    `unit` and `cycle` are int64; all other columns are float64. Rows are
    sorted by (unit, cycle). Raises ValueError if the column count is wrong.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, found {raw.shape[1]}")
    raw.columns = COLUMNS
    df = raw.astype({c: np.float64 for c in COLUMNS[2:]})
    df["unit"] = raw["unit"].astype(np.int64)
    df["cycle"] = raw["cycle"].astype(np.int64)
    return df.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)


def read_rul_file(path: str | Path) -> pd.Series:
    """Parse a RUL_ file: one integer per line, the true RUL of test unit i (1-based)."""
    values = [int(line.split()[0]) for line in Path(path).read_text().splitlines() if line.strip()]
    return pd.Series(values, index=pd.RangeIndex(1, len(values) + 1, name="unit"), name="rul", dtype=np.int64)


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add an `rul` column to run-to-failure training data.

    Each training trajectory ends at failure, so RUL at a cycle is
    (last cycle of that unit) - (current cycle); the final row has RUL 0.
    """
    out = df.copy()
    out["rul"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    return out


def add_test_rul(df: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    """Add an `rul` column to truncated test data using the provided final-cycle RULs.

    RUL at a cycle = true RUL at the unit's last observed cycle
    + (last observed cycle - current cycle).
    """
    units = np.sort(df["unit"].unique())
    if len(units) != len(rul) or not np.array_equal(units, rul.index.to_numpy()):
        raise ValueError(f"Test units {len(units)} do not match RUL entries {len(rul)}")
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = out["unit"].map(rul).to_numpy() + (last - out["cycle"])
    return out


def load_subset(subset: str, data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load train and test data for one subset, both with an `rul` column.

    Returns {"train": DataFrame, "test": DataFrame}.
    """
    _check_subset(subset)
    d = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    train = add_train_rul(read_cmapss_file(d / f"train_{subset}.txt"))
    test = add_test_rul(read_cmapss_file(d / f"test_{subset}.txt"), read_rul_file(d / f"RUL_{subset}.txt"))
    return {"train": train, "test": test}
