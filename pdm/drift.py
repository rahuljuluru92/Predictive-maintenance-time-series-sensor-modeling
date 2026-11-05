"""Stage 3: unsupervised sensor-drift detection and correction.

C-MAPSS has no drift labels, so "drift" is operationalized as follows
(full rationale in the README / DECISIONS.md):

* Temporal sensor deltas. Each sensor is scaled by its measurement-noise
  level (std of cycle-to-cycle differences / sqrt 2), and for every engine and
  cycle t we take delta(t) = trailing `window`-cycle mean - the engine's own
  mean over its first `baseline` cycles. This is how far each sensor has moved
  since the engine was new.
* True degradation is a physical process seen by many sensors at once, in
  fixed proportions, so on drift-free training engines the delta vectors lie
  close to a low-dimensional "degradation subspace". It is estimated with an
  uncentred PCA of the training deltas (a delta of 0 means "no change").
* Sensor drift (gradual calibration loss) moves one sensor on its own and so
  pushes the delta vector out of that subspace. The residual
  r(t) = delta(t) - projection onto the subspace removes the degradation
  signal and keeps what degradation cannot explain.
* An Isolation Forest is fitted on two summaries of the training residuals,
  computed per cycle from the standardized residuals z_j = r_j / std_train(r_j):
  max_j |z_j| (the largest single-sensor deviation, which a one-sensor drift
  drives up) and sum_j z_j^2 (total unexplained change). A cycle is anomalous
  when its score exceeds the (1 - alpha) quantile of training scores. (An
  Isolation Forest on the raw 15-dimensional residual was tried first; it
  isolates a deviation confined to one coordinate poorly, so drift was caught
  only once it was about twice as large. See DECISIONS.md.)
* Attribution: a drift in one sensor j leaves the residual (I - P) e_j * size,
  which spills into other sensors' residuals too. The sensor that best
  explains the residual on its own is the one with the largest standardized
  residual |r_j| / std_train(r_j) (reconstruction-based contribution; under
  isotropic noise the true sensor always maximizes it). One drifting sensor
  per engine at a time is assumed.
* A sensor is confirmed as drifting once, for `persistence` consecutive
  cycles, the cycle was anomalous, that sensor was the best explanation, and
  its standardized residual exceeded `z_thresh`. Drift is gradual and
  persistent while noise spikes are not; calibration loss does not heal
  itself, so a confirmed sensor stays flagged (latched) for the rest of that
  engine's trajectory. The correction itself is re-estimated every cycle, so
  a false confirmation costs little: its estimate stays near zero.
* Drift estimate: with Q = I - P (P = projector onto the subspace) and F the
  set of sensors flagged at cycle t, the drift sizes are the least-squares
  fit of the residual by single-sensor shifts on F:
  drift_F(t) = Q[F, F]^-1 r_F(t) noise units, converted back to raw units.
  For one sensor this is r_j / (1 - P_jj): 1 - P_jj is the share of a pure
  single-sensor shift that lands in the residual. Estimating flagged sensors
  jointly means a sensor wrongly confirmed early (picking up another sensor's
  spill-over) has its estimate pulled back toward zero once the true sensor is
  also confirmed. Sensors that are not flagged get no correction.

Everything is causal: outputs at cycle t use only that engine's cycles 1..t.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _unit_blocks(units: np.ndarray):
    bounds = np.flatnonzero(np.diff(units)) + 1
    return zip(np.r_[0, bounds], np.r_[bounds, len(units)])


def _check_sorted(df: pd.DataFrame) -> None:
    u = df["unit"].to_numpy()
    c = df["cycle"].to_numpy()
    same = u[1:] == u[:-1]
    if (u[1:] < u[:-1]).any() or not (c[1:][same] == c[:-1][same] + 1).all():
        raise ValueError("rows must be sorted by unit then cycle, with contiguous cycles")


@dataclass
class DriftResult:
    """Per-row outputs of DriftDetector.transform (rows aligned with the input frame)."""

    delta: pd.DataFrame  # temporal deltas, noise units
    residual: pd.DataFrame  # deltas minus degradation-subspace projection, noise units
    score: pd.Series  # Isolation Forest anomaly score (higher = more anomalous)
    anomalous: pd.Series  # score above the training threshold
    flags: pd.DataFrame  # bool per sensor: drifting at this cycle (persistent + attributed)
    estimate: pd.DataFrame  # causal drift estimate per sensor, raw sensor units

    def corrected_streams(self, df: pd.DataFrame) -> pd.DataFrame:
        """Online corrected stream: x_j(t) - flag_j(t) * estimate_j(t), raw units.

        Sensors that are not flagged at cycle t keep their raw value.
        """
        cols = list(self.flags.columns)
        out = df.copy()
        out[cols] = df[cols].to_numpy() - self.flags.to_numpy() * self.estimate.to_numpy()
        return out


@dataclass
class DriftDetector:
    window: int = 10  # trailing smoothing window for deltas (cycles)
    baseline: int = 10  # first cycles of each engine define its reference level
    eig_ratio: float = 5.0  # keep components with eigenvalue > eig_ratio * median eigenvalue
    alpha: float = 0.01  # anomaly threshold = (1 - alpha) quantile of training scores
    z_thresh: float = 4.0  # standardized residual needed to attribute drift to a sensor
    persistence: int = 10  # consecutive cycles required before a sensor is confirmed (= window)
    n_estimators: int = 200
    random_state: int = 0
    # Fitted attributes
    sensors_: list[str] = field(default_factory=list, init=False)
    noise_sigma_: np.ndarray = field(default=None, init=False, repr=False)
    eigenvalues_: np.ndarray = field(default=None, init=False, repr=False)
    components_: np.ndarray = field(default=None, init=False, repr=False)  # (m, k)
    resid_std_: np.ndarray = field(default=None, init=False, repr=False)
    threshold_: float = field(default=None, init=False)
    forest_: IsolationForest = field(default=None, init=False, repr=False)

    @property
    def k_(self) -> int:
        return self.components_.shape[1]

    # ----------------------------------------------------------------- core

    def _deltas(self, df: pd.DataFrame) -> np.ndarray:
        """Temporal deltas (n x m) in noise units; 0 before cycle `baseline`."""
        z = df[self.sensors_].to_numpy(dtype=np.float64) / self.noise_sigma_
        cycles = df["cycle"].to_numpy()
        out = np.zeros_like(z)
        for start, stop in _unit_blocks(df["unit"].to_numpy()):
            zu = z[start:stop]
            if len(zu) < self.baseline:
                continue
            y = zu - zu[: self.baseline].mean(axis=0)  # centre on the engine's own baseline
            cs = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)])
            idx = np.arange(1, len(y) + 1)
            lo = np.maximum(idx - self.window, 0)
            trailing = (cs[idx] - cs[lo]) / (idx - lo)[:, None]
            ready = cycles[start:stop] >= self.baseline
            out[start:stop][ready] = trailing[ready]
        return out

    def _residuals(self, delta: np.ndarray) -> np.ndarray:
        V = self.components_
        return delta - (delta @ V) @ V.T

    def _joint_estimate(self, resid: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """Drift size per flagged sensor (noise units): Q[F,F]^-1 r_F for each row's flag set F."""
        V = self.components_
        Q = np.eye(V.shape[0]) - V @ V.T
        est = np.zeros_like(resid)
        patterns, inverse = np.unique(flags, axis=0, return_inverse=True)
        for p_idx, pattern in enumerate(patterns):
            F = np.flatnonzero(pattern)
            if F.size == 0:
                continue
            rows = np.flatnonzero(inverse.ravel() == p_idx)
            # pinv guards against nearly collinear sensors (poorly isolable pairs)
            est[np.ix_(rows, F)] = resid[np.ix_(rows, F)] @ np.linalg.pinv(Q[np.ix_(F, F)]).T
        return est

    def fit(self, df: pd.DataFrame, sensors: Sequence[str]) -> DriftDetector:
        """Fit on drift-free engines (training data). `df` sorted by unit, cycle."""
        _check_sorted(df)
        self.sensors_ = list(sensors)
        diffs = df.groupby("unit")[self.sensors_].diff().to_numpy()
        sigma = np.nanstd(diffs, axis=0) / np.sqrt(2.0)
        if (sigma <= 0).any():
            raise ValueError("a sensor has no cycle-to-cycle variation; drop constant sensors first")
        self.noise_sigma_ = sigma

        ready = df["cycle"].to_numpy() >= self.baseline
        delta = self._deltas(df)[ready]
        _, s, vt = np.linalg.svd(delta, full_matrices=False)  # uncentred: origin = no change
        self.eigenvalues_ = s**2 / len(delta)
        k = max(1, int((self.eigenvalues_ > self.eig_ratio * np.median(self.eigenvalues_)).sum()))
        self.components_ = vt[:k].T

        resid = self._residuals(delta)
        self.resid_std_ = resid.std(axis=0)
        X = self._forest_features(resid)
        self.forest_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state)
        self.forest_.fit(X)
        train_scores = -self.forest_.score_samples(X)
        self.threshold_ = float(np.quantile(train_scores, 1.0 - self.alpha))
        return self

    def _forest_features(self, resid: np.ndarray) -> np.ndarray:
        """Isolation Forest inputs: [max_j |z_j|, sum_j z_j^2] of the standardized residuals."""
        z = resid / self.resid_std_
        return np.column_stack([np.abs(z).max(axis=1), (z**2).sum(axis=1)])

    def transform(self, df: pd.DataFrame) -> DriftResult:
        _check_sorted(df)
        delta = self._deltas(df)
        resid = self._residuals(delta)
        ready = df["cycle"].to_numpy() >= self.baseline
        score = np.where(ready, -self.forest_.score_samples(self._forest_features(resid)), -np.inf)
        anomalous = score > self.threshold_

        zres = np.abs(resid) / self.resid_std_
        best = zres.argmax(axis=1)
        n, m = zres.shape
        candidate = np.zeros((n, m), dtype=bool)
        hit = anomalous & (zres[np.arange(n), best] > self.z_thresh)
        candidate[np.arange(n)[hit], best[hit]] = True

        flags = np.zeros_like(candidate)
        p = self.persistence
        for start, stop in _unit_blocks(df["unit"].to_numpy()):
            c = candidate[start:stop].astype(np.int64)
            if len(c) < p:
                continue
            run = np.cumsum(np.vstack([np.zeros((1, m), dtype=np.int64), c]), axis=0)
            confirmed = np.zeros_like(c, dtype=bool)
            confirmed[p - 1 :] = (run[p:] - run[:-p]) == p
            flags[start:stop] = np.maximum.accumulate(confirmed, axis=0)  # latch

        estimate = self._joint_estimate(resid, flags) * self.noise_sigma_

        idx, cols = df.index, self.sensors_
        return DriftResult(
            delta=pd.DataFrame(delta, index=idx, columns=cols),
            residual=pd.DataFrame(resid, index=idx, columns=cols),
            score=pd.Series(score, index=idx, name="score"),
            anomalous=pd.Series(anomalous, index=idx, name="anomalous"),
            flags=pd.DataFrame(flags, index=idx, columns=cols),
            estimate=pd.DataFrame(estimate, index=idx, columns=cols),
        )


# --------------------------------------------------------------- injection


def typical_degradation_shift(train: pd.DataFrame, sensors: Sequence[str], n: int = 10) -> pd.DataFrame:
    """Per sensor, how far it moves over an engine's life (raw units):
    for each training engine, mean of its last n cycles - mean of its first n;
    returns columns `mean` (signed average over engines) and `mean_abs`."""
    g = train.sort_values(["unit", "cycle"]).groupby("unit")[list(sensors)]
    shift = g.tail(n).groupby(train["unit"]).mean() - g.head(n).groupby(train["unit"]).mean()
    return pd.DataFrame({"mean": shift.mean(), "mean_abs": shift.abs().mean()})


def drift_injection_rates(
    train: pd.DataFrame, sensors: Sequence[str], noise_sigma: Sequence[float], ramp_cycles: int = 50, min_shift: float = 2.0
) -> tuple[dict[str, float], dict[str, float]]:
    """Rates for the drift-injection experiments (fixed design, see DECISIONS.md).

    Eligible sensors move by at least `min_shift` noise units over a typical
    life (a clear degradation trend). Each drifts at mean_abs_shift /
    `ramp_cycles` raw units per cycle, i.e. a full degradation-sized shift
    after `ramp_cycles` cycles. Returns (rates, degradation direction sign).
    """
    shift = typical_degradation_shift(train, sensors)
    sigma = pd.Series(np.asarray(noise_sigma, dtype=float), index=list(sensors))
    eligible = shift.index[shift["mean_abs"] >= min_shift * sigma]
    rates = {s: float(shift.loc[s, "mean_abs"] / ramp_cycles) for s in eligible}
    direction = {s: float(np.sign(shift.loc[s, "mean"]) or 1.0) for s in eligible}
    return rates, direction


def inject_drift(df: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Add linear calibration drift: for each plan row (unit, sensor, onset, rate),
    sensor += rate * (cycle - onset) for cycles > onset of that unit. Raw units."""
    out = df.copy()
    for row in plan.itertuples(index=False):
        m = (out["unit"] == row.unit) & (out["cycle"] > row.onset)
        out.loc[m, row.sensor] = out.loc[m, row.sensor] + row.rate * (out.loc[m, "cycle"] - row.onset)
    return out


def random_drift_plan(
    df: pd.DataFrame,
    rates: dict[str, float],
    rng: np.random.Generator,
    min_onset: int,
    sign: str = "random",
    reference_sign: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One drifting sensor per engine, chosen uniformly from `rates`' keys, with
    onset uniform over the engine's cycles [min_onset, last cycle - 1] and
    |rate| = rates[sensor] raw units per cycle.

    sign: "random" (+/- with equal probability), "mimic" (same sign as
    reference_sign[sensor], i.e. the direction the sensor moves under real
    degradation) or "mask" (opposite sign).
    """
    sensors = list(rates)
    rows = []
    for unit, last in df.groupby("unit")["cycle"].max().items():
        sensor = sensors[rng.integers(len(sensors))]
        onset = int(rng.integers(min_onset, max(min_onset, last - 1) + 1))
        if sign == "random":
            sgn = rng.choice([-1.0, 1.0])
        elif sign in ("mimic", "mask"):
            sgn = np.sign(reference_sign[sensor]) * (1.0 if sign == "mimic" else -1.0)
        else:
            raise ValueError(f"unknown sign mode {sign!r}")
        rows.append((unit, sensor, onset, sgn * rates[sensor]))
    return pd.DataFrame(rows, columns=["unit", "sensor", "onset", "rate"])
