"""Small synthetic turbofan-like fleet for controlled drift-vs-degradation tests.

Each engine has `n_sensors` noisy sensors (noise std 1) with an engine-specific
offset (initial wear / manufacturing variation). Degradation follows one of
two fault modes; each mode moves several sensors together in fixed
proportions (its signature), growing slowly and then faster toward failure:
    signal(t) = severity * (exp(4 t / L) - 1) / (exp(4) - 1) * signature
Sensor drift is added separately with pdm.drift.inject_drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNATURES = np.array(
    [
        [1.0, 0.8, -0.6, 0.9, 0.5, -0.7, 0.0, 0.0],  # mode A
        [0.0, 0.3, 0.9, -0.2, 0.0, 0.6, -0.8, 0.7],  # mode B
    ]
)
N_SENSORS = SIGNATURES.shape[1]
SENSORS = [f"x{i}" for i in range(N_SENSORS)]


def health(length: int, severity: float) -> np.ndarray:
    t = np.arange(1, length + 1)
    return severity * (np.exp(4.0 * t / length) - 1.0) / (np.exp(4.0) - 1.0)


def make_fleet(
    n_units: int,
    rng: np.random.Generator,
    length_range: tuple[int, int] = (150, 300),
    severity: float = 8.0,
    modes: tuple[int, ...] = (0, 1),
    degrade: bool = True,
    first_unit: int = 1,
) -> pd.DataFrame:
    """Return a long frame with columns unit, cycle, mode, x0..x7 (sorted by unit, cycle)."""
    frames = []
    for i in range(n_units):
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        mode = int(modes[i % len(modes)])
        offset = rng.normal(0.0, 2.0, N_SENSORS)
        noise = rng.normal(0.0, 1.0, (length, N_SENSORS))
        signal = np.outer(health(length, severity), SIGNATURES[mode]) if degrade else 0.0
        x = 100.0 + offset + signal + noise
        f = pd.DataFrame(x, columns=SENSORS)
        f.insert(0, "mode", mode)
        f.insert(0, "cycle", np.arange(1, length + 1))
        f.insert(0, "unit", first_unit + i)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
