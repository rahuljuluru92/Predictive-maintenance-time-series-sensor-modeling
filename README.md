# Predictive maintenance on NASA C-MAPSS: separating sensor drift from engine degradation

A two-stage, classical machine-learning pipeline on the NASA C-MAPSS turbofan
degradation benchmark:

1. **Drift detection (Isolation Forest).** Flags gradual calibration drift in
   individual sensors, i.e. change that the engine's coordinated degradation
   pattern cannot explain, and estimates and removes it, using only past data.
2. **Failure prediction (Random Forest).** Rolling-window features from the
   (drift-corrected) sensor streams predict whether an engine will fail within
   the next 30 cycles.

The drift-corrected pipeline is compared with a baseline that skips stage 1,
and both are cross-validated across the four C-MAPSS subsets (FD001–FD004).
Every number below comes from the scripts in this repository, run on the real
data; nothing is estimated or rounded up.

## Results at a glance

| question | answer (measured) |
|---|---|
| How good is the baseline? (FD001 official test set, every cycle) | recall **0.6295**, false-alert rate **0.345 %**, precision 0.826, ROC-AUC **0.9918** |
| Does drift correction help on the real data? | **No measurable difference.** Recall 0.6325 vs 0.6295 and false-alert rate 0.353 % vs 0.345 %; 95 % bootstrap intervals for both differences include zero. The detector flagged none of the 100 test engines; the dataset documents no calibration drift. |
| Does it help when drift is present? (synthetic drift injected into the FD001 test set) | **Yes, with trade-offs.** Drift that looks like wear: **21 % fewer false alerts** (recall −4.4 %). Drift that hides wear: **recall +5.0 %** (false alerts +14 %, back toward the clean level). |
| Does it generalize across FD001–FD004 when all subsets are in training? | **Yes.** 5-fold CV ROC-AUC 0.989–0.994 on every subset, spread ≤ 0.0052; the minority subsets FD001/FD003 are not worse. |
| Does it generalize to a subset never seen in training? | **Not for FD004.** Leave-one-subset-out ROC-AUC drops by ≤ 0.004 for FD001–FD003 but by **0.028–0.029 for FD004**, and the drift detector then flags 79 of 248 FD004 engines. |

## Contents

- [Dataset](#dataset)
- [Setup](#setup)
- [How drift is operationalized](#how-drift-is-operationalized)
- [Evaluation design](#evaluation-design)
- [Results](#results)
- [Running the tests](#running-the-tests)
- [Reproducing the numbers](#reproducing-the-numbers)
- [Limitations](#limitations)
- [Project layout](#project-layout)

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set**, NASA Ames
Prognostics Center of Excellence. Downloaded from NASA's Prognostics Data
Repository mirror:
`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`

- **License / attribution:** publicly released by NASA (U.S. Government work).
  The bundled readme asks users to cite:
  A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage Propagation Modeling for
  Aircraft Engine Run-to-Failure Simulation"*, Proceedings of the 1st
  International Conference on Prognostics and Health Management (PHM08),
  Denver, CO, October 2008.
- The data files are **not** committed to this repository. `scripts/download_data.py`
  downloads the archive, checks its SHA-256, and extracts it to `data/raw/CMAPSSData/`.

| subset | operating conditions | fault modes | train engines | test engines |
|---|---|---|---|---|
| FD001 | 1 (sea level) | HPC degradation | 100 | 100 |
| FD002 | 6 | HPC degradation | 260 | 259 |
| FD003 | 1 (sea level) | HPC + fan degradation | 100 | 100 |
| FD004 | 6 | HPC + fan degradation | 249 | 248 |

Each row is one operating cycle of one engine: engine id, cycle, 3 operational
settings, and 21 sensor readings (temperatures, pressures, speeds, flows; names
and units from Table 2 of the paper are in `pdm/data.py`). Training engines
run until failure; test engines stop before failure, and their true remaining
useful life (RUL) at the last cycle is given separately.
(The bundled readme lists FD004 as 248 train / 249 test engines; the files
actually contain 249 / 248. The tests check the real counts.)

## Setup

Tested with Python 3.13 on Linux. No GPU, no API keys, no services.

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_data.py
```

## How drift is operationalized

C-MAPSS has **no drift labels**, so "sensor drift" has to be defined. The
definition used here (`pdm/drift.py`) rests on one physical observation:
real degradation (HPC or fan efficiency and flow loss) moves *many sensors
together in fixed proportions*, while a calibration problem moves *one sensor
on its own*.

1. **Temporal sensor deltas.** Each sensor is scaled by its measurement-noise
   level (std of cycle-to-cycle differences / √2). For every engine and cycle,
   delta = trailing 10-cycle mean − that engine's own mean over its first 10
   cycles: how far each sensor has moved since the engine was new. Using each
   engine's own start removes engine-to-engine offsets (initial wear,
   manufacturing variation), which the dataset describes as normal.
2. **Degradation subspace.** An uncentred PCA of the training engines' deltas
   keeps the components whose eigenvalue exceeds 5× the median eigenvalue
   (the median estimates the noise floor). On FD001 that gives 2 components,
   and the part of the delta *outside* them stays flat from a new engine to
   failure. Real degradation is fully absorbed, so it is not mistaken for drift.
3. **Isolation Forest.** It is fitted on the training engines' residuals
   (delta minus its projection onto the degradation subspace), summarized per
   cycle as the largest standardized single-sensor residual and the total
   residual energy. A cycle is anomalous above the 99th percentile of training
   scores.
4. **Attribution and confirmation.** The sensor that best explains the residual
   on its own (largest standardized residual) is the suspect. It is confirmed
   once, for 10 consecutive cycles, the cycle is anomalous, the same sensor is
   the suspect, and its standardized residual exceeds 4. Confirmed sensors stay
   flagged for the rest of the engine's life (calibration loss does not repair
   itself).
5. **Correction.** The drift of every flagged sensor is re-estimated each cycle
   by least squares, Q[F,F]⁻¹ r_F (Q projects onto the residual space, F is
   the flagged set), and subtracted. Unflagged sensors are never touched.
   Everything is causal: what is used at cycle t comes only from cycles ≤ t of
   that engine.

The detector was first validated on a **synthetic fleet** (8 sensors, two
fault modes, controlled injected drift; `pdm/synthetic.py`, `tests/test_drift.py`):

- Degradation-only engines are not flagged: 0 of 40, and 0 of 20 at twice the
  training severity.
- Every one of 32 injected drifts (2 fault modes × 8 sensors × 2 directions)
  is flagged on exactly the right sensor, and only after onset, at a median
  drift size of 2.9 noise units.
- The median remaining error after correction is 4.9% of the injected drift.

On **real FD001 engines** (5-fold split of the training engines; drift
injected into held-out engines; `scripts/run_drift_detection.py`):

- 4 of 100 clean engines get a false flag.
- Injected drift is caught on the right sensor in 74–79% of engines, and in
  100% of the engines whose drift grew to 8 or more noise units before the
  end of their record.
- Detection happens at a median drift of 3.8–3.9 noise units, about 35 cycles
  after onset.
- About 95–96% of the drift is removed.

The design went through two documented revisions on synthetic and training
data before any test-set evaluation: confirmation persistence 5 → 10 cycles,
and Isolation Forest inputs changed from the raw residual vector to the two
residual summaries, with the attribution threshold raised from 3 to 4.

## Evaluation design

All settings were fixed before the models were evaluated, and both pipelines
share them, so the drift-correction stage is the only difference between them.

- **Label:** 1 if the engine fails within the next 30 cycles (RUL ≤ 30), else 0;
  every cycle is a prediction.
- **Features:** trailing 10-cycle rolling mean, standard deviation and slope of
  each informative sensor (FD001: 15 sensors; 6 constant ones dropped). The
  operational settings and the cycle counter are not used.
- **Classifier:** `RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
  max_features="sqrt", random_state=42)`. An alert is raised when P(failure
  within 30 cycles) ≥ 0.5. Nothing is tuned.
- **Metrics:**
  - recall = TP/(TP+FN)
  - **false-alert rate = FP/(FP+TN)**, the share of healthy cycles that raise
    an alert
  - precision
  - ROC-AUC

  All are implemented in `pdm/metrics.py` and unit-tested against
  hand-computed values.
- **Held-out data for Stages 2 and 5:** the official FD001 test set
  (13,096 cycles, 100 engines, 332 positive cycles), used once.

## Results

### Baseline (no drift correction), FD001 official test set

| per cycle | recall | false-alert rate | precision | ROC-AUC |
|---|---|---|---|---|
| baseline | 0.6295 (209/332) | 0.00345 (44/12,764) | 0.8261 | 0.9918 |

At each engine's last observed cycle (100 decisions): recall 0.72 (18/25),
false-alert rate 0.0133 (1/75), ROC-AUC 0.9808.

### Drift-corrected model vs. baseline, FD001 official test set

| per cycle | recall | false-alert rate | precision | ROC-AUC | TP / FP / FN |
|---|---|---|---|---|---|
| baseline | 0.6295 | 0.00345 | 0.8261 | 0.9918 | 209 / 44 / 123 |
| drift-corrected | 0.6325 | 0.00353 | 0.8235 | 0.9918 | 210 / 45 / 122 |
| relative change | +0.48% | +2.27% | −0.31% | −0.00% | |

**On the real data, drift correction makes no measurable difference.** A
paired bootstrap over test engines (2,000 resamples) gives 95% intervals of
−0.015 to +0.029 for the recall difference and −0.00016 to +0.00037 for the
false-alert-rate difference; both include zero. The detector flagged none of
the 100 test engines, so the test features were identical for both models.
The small differences (the alert decision differs on 12 of 13,096 cycles)
come from 2 of 100 training engines it corrected (sensors s6 and s8), which
slightly changed what the Random Forest was trained on. This fits how the data
was made: the C-MAPSS documentation describes sensor noise and engine-to-engine
variation but no calibration drift, so on this data there is little or nothing
for the drift stage to remove.

### Stress test: synthetic drift injected into the FD001 test set

To see what the drift stage does when drift *is* present, one sensor per test
engine was given a linear drift:

- **Rate:** the sensor's typical lifetime degradation shift reached after
  50 cycles.
- **Onset:** a random cycle after the engine's 10-cycle reference period.
- **Direction:** either the direction degradation moves that sensor ("mimic",
  which looks like wear) or the opposite ("mask", which hides wear).

Both models are the ones trained on clean data. Values are mean ± std over
10 random drift assignments. **These numbers depend on this synthetic drift
model.**

| scenario | model | recall | false-alert rate | mean false alerts | ROC-AUC |
|---|---|---|---|---|---|
| mimic | baseline | 0.6750 ± 0.0127 | 0.0049 ± 0.0008 | 62.2 | 0.9879 |
| mimic | drift-corrected | 0.6455 ± 0.0088 | 0.0038 ± 0.0003 | 49.0 | 0.9905 |
| mask | baseline | 0.5765 ± 0.0148 | 0.0027 ± 0.0003 | 35.1 | 0.9916 |
| mask | drift-corrected | 0.6054 ± 0.0105 | 0.0031 ± 0.0001 | 40.0 | 0.9915 |

- **Mimic drift.** False alerts fall by **21.2%** relative to the drifted
  baseline, with fewer false alerts in 10 of 10 runs. That removes 72.5% of
  the false-alert increase the drift caused. Recall falls by 4.4%, because
  part of the drifted baseline's recall came from drift that looked like wear.
- **Mask drift.** Recall rises by **5.0%** (higher in 8 of 10 runs),
  recovering 54.5% of the recall the drift took away. False alerts rise by
  14%, back toward the clean level, which the drift had pushed down.
- The detector caught the injected sensor in about 60% of test engines. Most
  of the rest got drift that started too late in their (truncated) record to
  become detectable.

### Cross-validation across FD001–FD004

**Operating conditions.** FD002 and FD004 switch between six operating
conditions (altitude, Mach, throttle) from one cycle to the next, and that
moves the sensors far more than wear does. Each cycle's condition is
identified by rounding its three settings. The six condition points are
shared by all subsets and well separated; FD001 and FD003 run only at sea
level. Each sensor is then z-scored per condition using training statistics
(`pdm/conditions.py`). Both pipelines get this step. For the single-condition
subsets it is a per-sensor rescaling, which changes neither the Random Forest
nor the noise-scaled drift detector.

**Setup.** One fleet-wide model per pipeline, trained on the pooled training
engines of all four subsets (709 engines; FD002 + FD004 provide ~72 % of the
cycles). All settings are frozen from the earlier stages; nothing was retuned.
Three evaluations:

1. **Stratified group 5-fold CV** (`StratifiedGroupKFold`):
   - All cycles of an engine stay in the same fold. Row-level K-fold would put
     neighbouring cycles of one engine on both sides and inflate the scores.
   - Folds are stratified by subset, so every fold holds 20 FD001, 52 FD002,
     20 FD003 and 49–50 FD004 engines.
   - The normalizer, drift detector and classifier are fitted inside each
     training fold only.
2. **Official test sets:** the pooled model is scored on each subset's own
   test set.
3. **Leave-one-subset-out:** train on three subsets, then test on the official
   test set of the fourth.

The criterion was written down before the run: generalization "holds" if every
subset's CV ROC-AUC is within 0.02 of the best subset, *and* leaving a subset
out of training costs it at most 0.02 ROC-AUC.

**5-fold CV, per subset** (mean ± std over folds; validation engines are full
run-to-failure trajectories):

| subset | pipeline | ROC-AUC | recall | false-alert rate |
|---|---|---|---|---|
| FD001 | baseline | 0.9903 ± 0.0024 | 0.851 ± 0.028 | 0.0193 ± 0.0047 |
| FD001 | drift-corrected | 0.9901 ± 0.0026 | 0.854 ± 0.025 | 0.0189 ± 0.0048 |
| FD002 | baseline | 0.9888 ± 0.0013 | 0.835 ± 0.016 | 0.0182 ± 0.0041 |
| FD002 | drift-corrected | 0.9887 ± 0.0011 | 0.835 ± 0.017 | 0.0179 ± 0.0041 |
| FD003 | baseline | 0.9937 ± 0.0010 | 0.856 ± 0.026 | 0.0123 ± 0.0048 |
| FD003 | drift-corrected | 0.9939 ± 0.0009 | 0.857 ± 0.024 | 0.0123 ± 0.0042 |
| FD004 | baseline | 0.9904 ± 0.0018 | 0.858 ± 0.014 | 0.0180 ± 0.0030 |
| FD004 | drift-corrected | 0.9904 ± 0.0018 | 0.857 ± 0.012 | 0.0183 ± 0.0029 |

ROC-AUC spread across subsets: 0.0049 (baseline) and 0.0052 (drift-corrected).
**First criterion met.** The subsets that are a minority in training, FD001
and FD003, do as well as the dominant FD002 and FD004; FD003 scores highest.

**Official test sets: pooled model vs. leave-one-subset-out** (ROC-AUC):

| subset | baseline, pooled | baseline, subset left out | drop | drift-corrected, pooled | drift-corrected, subset left out | drop |
|---|---|---|---|---|---|---|
| FD001 | 0.9914 | 0.9909 | 0.0005 | 0.9912 | 0.9909 | 0.0003 |
| FD002 | 0.9855 | 0.9839 | 0.0016 | 0.9866 | 0.9837 | 0.0029 |
| FD003 | 0.9959 | 0.9926 | 0.0033 | 0.9961 | 0.9922 | 0.0039 |
| FD004 | 0.9912 | **0.9633** | **0.0279** | 0.9913 | **0.9624** | **0.0289** |

Recall and false-alert rate of the pooled model on the official test sets
(baseline / drift-corrected):

- FD001: 0.639 / 0.651 and 0.35 % / 0.35 %
- FD002: 0.732 / 0.731 and 0.84 % / 0.80 %
- FD003: 0.643 / 0.667 and 0.17 % / 0.15 %
- FD004: 0.589 / 0.589 and 0.45 % / 0.47 %

**Second criterion not met, for FD004 (both pipelines).** Trained without
FD004, the model loses 0.028–0.029 ROC-AUC on FD004. Its false-alert rate is
2.2–2.7× higher (1.0–1.2 %), and precision falls from 0.73–0.74 to 0.53–0.56.

**Why FD004?** FD004 is the only subset that combines the fan fault mode with
the five non-sea-level conditions. Without it, training sees fan degradation
only at sea level (FD003). The follow-up (`scripts/investigate_fd004.py`)
splits FD004 test cycles by the condition they were flown at:

| FD004 test cycles | ROC-AUC change | false-alert rate change |
|---|---|---|
| non-sea-level (85 % of cycles) | 0.9913 → 0.9624 | ×3.0 |
| sea level | 0.9902 → 0.9690 | ×1.5 |

The loss is larger where that combination was never seen. Sea-level cycles
also suffer, partly because every 10-cycle window mixes conditions. So this
supports the explanation but does not prove it. The remedy would be training
data that covers that fault-mode/condition combination; no modelling change
was made.

**Drift correction across subsets.** It changes ROC-AUC by at most 0.0011 in
any evaluation, and recall and false-alert differences are small and mixed in
sign, consistent with FD001. The detector itself:

- Pooled model: it flags few engines, with 0 out-of-fold flags on FD001 and
  FD002, 14/100 on FD003 and 7/249 on FD004.
- Trained without FD004: it flags **79 of 248 FD004 test engines** (sensors
  s10, s6, s12). Degradation behaviour the training fleet never showed looks
  like drift to a detector that defines drift as "change outside the learned
  degradation pattern". This is a real limitation of the approach.

**Why the CV and test-set rates differ.** CV recall (~0.85) is higher than on
the official test sets (0.59–0.73), and CV false-alert rates (1.2–1.9 %) are
higher too (test: 0.15–0.84 %). CV validates on complete trajectories. The
official test trajectories stop before failure (FD001: 7 to 145 cycles
before), so their positive cycles sit mostly 10–30 cycles before failure,
which is harder, and their negatives are mostly early-life, which is easier.
The label mix differs; this is not over-fitting. ROC-AUC is comparable between
the two.

## Running the tests

```bash
python -m pytest
```

75 tests in `tests/`:

| file | what it checks |
|---|---|
| `test_data.py` | parsing, shapes, columns, engine and cycle structure, RUL labels, operating conditions and sensor plausibility for all four subsets (real-data tests skip if the data is not downloaded) |
| `test_features.py` | rolling mean, std and slope, and drift-corrected features, against hand-calculated values; no leakage across engines; 30-cycle label boundary |
| `test_drift.py` | synthetic drift vs. degradation: subspace recovery, no flags on degradation (including steeper degradation), every injected drift caught on the right sensor, correction accuracy, causality, injection helpers |
| `test_metrics.py` | confusion counts, recall, false-alert rate, precision, ROC-AUC (with ties, checked against scikit-learn), engine-level bootstrap, against hand-computed values |
| `test_conditions.py` | operating-condition identification and per-condition normalization |

## Reproducing the numbers

All scripts are deterministic (fixed seeds) and write JSON to `results/`.

| script | stage | output | runtime (4 CPU cores) |
|---|---|---|---|
| `scripts/run_baseline.py` | baseline, FD001 test set | `results/baseline_FD001.json` | ~10 s |
| `scripts/run_drift_detection.py` | detector check on held-out FD001 training engines | `results/drift_detection_FD001.json` | ~10 s |
| `scripts/run_comparison.py` | drift-corrected vs. baseline + stress test | `results/comparison_FD001.json` | ~40 s |
| `scripts/run_cross_validation.py` | cross-subset CV, official test sets, leave-one-subset-out | `results/cross_validation.json` | ~33 min |
| `scripts/investigate_fd004.py` | follow-up on the FD004 leave-one-out drop | `results/investigate_fd004.json` | ~3 min |

## Limitations

- **No known drift in the data.** C-MAPSS is documented as simulated with
  sensor noise and engine-to-engine variation, with no calibration drift and
  no drift labels. The benefit of drift correction can therefore only be shown
  with injected drift, and it depends on the drift model used (linear ramp, one
  sensor at a time, sized to each sensor's degradation range). Step changes,
  noise-level changes, or several sensors drifting together were not tested.
  The training engines are assumed drift-free; any drift in them would be
  learned as normal.
- **Some drift cannot be told apart from degradation.** The part of a drift
  that happens to lie along the degradation pattern cannot be separated from
  real wear. That share is largest for s14 and s9 (Nc, NRc) in FD001.
- **Fixed alert threshold.** All recall and false-alert numbers use the
  untuned threshold of 0.5. ROC-AUC shows the ranking quality independently
  of the threshold.
- **Unit of evaluation.** Metrics count cycles, and cycles of the same engine
  are correlated. The bootstrap intervals therefore resample whole engines.
- **Generalization to unseen subsets is not guaranteed.** Leaving FD004 out of
  training costs it 0.028–0.029 ROC-AUC (see above). A fleet-wide model needs
  training data that covers each fault-mode and operating-condition
  combination.
- **The drift detector depends on the training fleet.** It treats any
  sensor-specific change outside the degradation pattern learned from
  training engines as drift. Unfamiliar but genuine degradation is therefore
  "corrected" too: 79 of 248 FD004 engines when FD004 was left out of
  training, and 14 of 100 FD003 engines out-of-fold in the pooled CV.

## Project layout

```
pdm/
  data.py         loading and RUL labels
  features.py     30-cycle labels, rolling features (raw and drift-corrected)
  drift.py        drift detector, drift injection
  synthetic.py    synthetic fleet for controlled tests
  conditions.py   operating conditions, per-condition normalization, pooled loading
  model.py        fixed classifier settings, evaluation report
  metrics.py      metrics and engine-level bootstrap
  pipeline.py     baseline and drift-corrected pipelines
scripts/          download + one script per stage (see above)
tests/            pytest suite
results/          JSON outputs of the scripts (committed)
```
