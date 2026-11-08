# Retrofit AI: physics-informed hybrid fault detection for pumps

**Since AI Hackathon 2026 · Business Turku / Elomatic challenge**

An operator assistant that watches pumps with whatever sensors they already have, raises **early** alerts,
says **what** is wrong and whether it is the **machine or a lying sensor**, shows its **evidence and confidence**,
and recommends an action for a human to approve.

It combines two parts:

* **Physics rules** (pump affinity laws, pump curve, NPSH cavitation margin, bearing-frequency analysis, sensor
  plausibility checks) that decide *when* to alert and give a first diagnosis with a written explanation.
* **A machine-learning diagnostician** (LightGBM, trained on 450 simulated pumps) that confirms or corrects the
  diagnosis with a **calibrated** confidence, only when it is reliably better than the rules.

### Results (official scorer, `src/retrofit_sim/scoring.py`)

| Composite score (0–100) | Train PUMP-01..07 | Validation PUMP-08/09 | **90 held-out pumps** |
|---|---|---|---|
| B0 SCADA alarms (organisers) | 27.6 | 23.1 | – |
| B1 mode-aware z (organisers, best baseline) | 58.2 | 78.0 | 60.0 |
| B2 CUSUM (organisers) | 53.3 | 50.9 | – |
| B3 Isolation Forest (organisers) | 39.4 | 69.2 | – |
| Ours: physics rules | 86.7 | 95.8 | 70.4 |
| **Ours: physics rules + hybrid ML** | 85.9 | **96.1** | **70.8** |

The **held-out** column is the least biased number: 90 pumps from 15 unseen simulator seeds (random ratings, sites,
instrumentation levels and fault schedules, 239 faults) that were never used for development.
The validation set is tiny (2 pumps, 9 faults) and the physics rules were developed while looking at PUMP-01..09,
so the validation number is optimistic; see [Evaluation](#7-evaluation).

---

## 1. Repository layout

```
├── src/ data/ configs/ baselines/ submission_template/ make_dataset.py ...   organisers' package (unchanged)
├── joice_physics_rules/diag2.py      physics rules: detector + rule-based diagnosis (final version)
├── generate_training_data.py         step 1  simulate extra labelled pumps (parallel, resumable)
├── physics_pipeline.py               step 2  normalisation + physics + detector on every pump; scores vs B1
├── build_training_table.py           step 3  causal feature snapshots + true labels -> training table
├── train_diagnostician.py            step 4  LightGBM, grouped cross-validation, held-out test
├── integrate_hybrid.py               step 5  calibration + hybrid decision, scored with the official scorer
├── make_submission.py                step 6  alerts for the hidden test pumps PUMP-10/11/12
├── compare_baselines.py              step 7  comparison with B0-B3 under the official protocol
├── build/
│   ├── generated/    simulated data (85 seeds)            ├── pipeline/   per-minute features + alerts per seed
│   ├── training/     train_table.parquet                  ├── models/     diagnostician.joblib, cv_predictions.csv
│   ├── integration/  calibrator.joblib, scores.csv        └── comparison/ comparison.md / .png / .csv
└── submission/predictions_test.csv   <- the file handed in
```

## 2. Setup

Requirements: Python 3.11+ and (for cloning) [Git](https://git-scm.com/downloads).

### Get the code

**Option A: clone with Git**

```bash
git clone https://github.com/KalebYohannese/RETROFIT_AI_PARTICIPANTS_2026.git
cd RETROFIT_AI_PARTICIPANTS_2026
```

**Option B: download a ZIP**

On the repository page, click **Code → Download ZIP**, unzip it, and open a terminal in the unzipped folder.

**What the repository contains.** The scripts, the physics rules (`joice_physics_rules/diag2.py`), the organisers'
package (`src/`, `data/full/`, `make_dataset.py`, `configs/`, `baselines/`) and the trained model
(`build/models/diagnostician.joblib`, `build/integration/calibrator.joblib`). The generated training data
(`build/generated/`, ~2.6 GB) is **not** included; it is only needed to retrain and can be regenerated exactly
with step 1 of the [full replication](#4-full-replication-from-scratch).

### Install

Open a terminal in the repository folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

On Windows use `python` instead of `python3` if `python3` is not found.

## 3. Quick start: reproduce the submission with the trained model

With `build/models/` and `build/integration/` present:

```bash
python3 make_submission.py
```

Expected: `format check test: OK`, **33 alerts** (PUMP-10: 6, PUMP-11: 6, PUMP-12: 21), 1 ML override, and the
validation sanity check `rules 95.8 hybrid 96.1`. Output: `submission/predictions_test.csv` (template format),
plus `predictions_test_rules_only.csv` and `predictions_validation.csv` for reference.

## 4. Full replication from scratch

All steps are deterministic (fixed seeds). Times are for a normal laptop/PC.

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Generate data | `python3 generate_training_data.py` | ~40 min (11 workers) | `build/generated/` (85 seeds, 2.6 GB) |
| 2. Physics pipeline | `python3 physics_pipeline.py` | ~20–30 min | `build/pipeline/` |
| 3. Training table | `python3 build_training_table.py` | ~5 min | `build/training/train_table.parquet` |
| 4. Train model | `python3 train_diagnostician.py` | ~5 min | `build/models/` |
| 5. Hybrid + calibration | `python3 integrate_hybrid.py` | ~5 min | `build/integration/` |
| 6. Submission | `python3 make_submission.py` | ~2 min | `submission/` |
| 7. Baseline comparison | `python3 compare_baselines.py` | ~10 min | `build/comparison/` |

Tips: run `python3 generate_training_data.py --quick` first (4 seeds) to test the chain end to end.
Step 1 can be stopped and restarted; finished seeds are skipped. Keep the computer awake during step 1.

---

## 5. How it works

```
 telemetry (1 Hz or 1-min) + pump datasheet (+ operating context, vibration snapshots if available)
        │
 [1] Normalise ─ each pump against ITS OWN first 24 h of steady running, per 5 % speed band (robust z-scores)
        │
 [2] Physics ── head & power corrected for speed (affinity laws), head factor vs pump curve, NPSH margin,
        │       bearing envelope at the bearing defect frequency, 1×/2× vibration ratios, current-vs-power line,
        │       bearing-vs-fluid temperature, in-minute ranges, warm-up after starts
        │
 [3] Detect ─── WHEN: sustained deviation (|z| > 6 for 15 min) or a physics alarm
        │       (NPSH margin < 1.15, bearing index, sensor plausibility checks); one alert per fault per 12 h
        │
 [4] Diagnose ─ WHAT: rule verdict + ML model (276 causal features) -> hybrid decision, calibrated confidence
        │
 [5] Output ─── organisers' template: asset, time, MACHINE/SENSOR, fault type, confidence, evidence, action
```

### 5.1 Physics rules (`joice_physics_rules/diag2.py`)

| Indicator | Physics | Detects |
|---|---|---|
| Head and power vs reference, corrected with affinity laws (H ∝ n², P ∝ n³) | removes the effect of speed changes | restriction, impeller wear, air |
| Head factor = measured head / pump-curve head (parabolic curve from datasheet) | healthy ≈ 1.0 | impeller wear, air entrainment |
| NPSH margin = NPSHa / NPSHr (vapour pressure from Antoine equation) | cavitation below ~1.15 | cavitation, strainer clogging |
| Bearing index = envelope amplitude at BPFO / high-frequency RMS | separates bearing damage from cavitation noise | bearing wear |
| 1× and 2× running-speed vibration ratios | imbalance at 1×, misalignment at 1× + 2× | imbalance, misalignment |
| Motor current vs power line (fixed reference from the first hours) | an honest current sensor follows power | current-sensor drift / shift |
| Bearing temperature vs fluid temperature | fixed physical reference | temperature-sensor drift, stuck wireless sensor |
| In-minute range (max − min) vs the pump's normal range | spikes hide inside 1-minute averages | noise spikes |
| One channel moves, physically linked channels do not | machine faults move several channels together | sensor vs machine |

Every alert carries this evidence as text, e.g. *"NPSH margin 0.9, cavitates below 1.15"*.

### 5.2 Training data (`generate_training_data.py`)

The official data has only 36 labelled faults. The organisers' simulator (included in the package) generates
more, fully labelled:

| Seeds | Mode | Role | Content |
|---|---|---|---|
| 100–159 | stress | train | 360 new random pumps, 10 common fault types |
| 160–174 | stress | **held-out test** | 90 pumps, never used for training or tuning |
| 300–309 | full | extra | re-simulations of the public pumps; the only source of cavitation, impeller wear and air entrainment |

Total: 540 pumps, 1,529 observable faults. Labels come from the simulator's answer key (`private/`), which is
used **only as labels and for checking, never as model input**.

### 5.3 Training table (`build_training_table.py`)

One row = one moment on one pump, features computed **only from data up to that moment**:

* **EVENT** rows: 0.5, 1, 3, 6 and 12 h after each fault becomes detectable (learn faults early and developed)
* **ALERT** rows: every detector alert; alerts outside any fault are labelled `NO_FAULT` (learn false alarms)
* **NORMAL / BENIGN** rows: healthy moments, start-ups, load changes (`NO_FAULT`)

276 features: current z-scores and physics indicators, trends over 30 min / 3 h / 12 h, step-vs-ramp and
"moves alone" hints, instrumentation tier and available sensors, and the rule verdict. **No raw sensor values,
no pump names**, so the model cannot memorise individual pumps.

### 5.4 Model (`train_diagnostician.py`)

LightGBM multiclass (13 fault types + `NO_FAULT`), class-balanced. Evaluated with **grouped cross-validation by
pump** (a pump is never in both training and test; re-simulated copies of the same public pump form one group)
and on the held-out seeds.

| Fault-type accuracy (per moment) | Rules | Model | Hybrid |
|---|---|---|---|
| Cross-validation, 450 pumps | 0.415 | 0.466 | 0.473 |
| Held-out test, 90 pumps | 0.403 | 0.516 | 0.503 |

### 5.5 Hybrid decision (`integrate_hybrid.py`)

At every detector alert (alert times are never changed):

1. **Rule-priority types** keep the rule's answer: cavitation, air entrainment, impeller wear, communication loss,
   calibration shift, noise spikes. These are the types where the rule was **equal or better than the model in
   cross-validation** (chosen before looking at test results).
2. Otherwise the ML answer is used if its **calibrated** confidence ≥ 0.5.
3. Otherwise the rule's answer stands.
4. The model never deletes an alert; an override that would clash with a neighbouring alert is undone.

---

## 6. Design principles

### Generalisability: works on pumps it has never seen

* **Each pump is compared with itself.** "Normal" is learned per pump from its own first 24 h of running and per
  speed band, so size, rating and duty do not matter.
* **Physics removes operating-point effects.** Affinity laws and the pump curve turn raw pressures and powers into
  quantities that mean the same thing on any pump.
* **The model sees no pump identity**: only normalised and physical features, no raw values, no names.
* **Trained on variety**: 450 random pumps across ferries, food, chemical and paper plants, three instrumentation
  levels and four control types (fixed speed, speed drive on pressure or temperature, flow valve).
* **Measured on unseen pumps**: grouped cross-validation and 90 held-out pumps (70.8 vs best baseline 60.0).
* **Transfer to other rotating machines**: normalisation, sensor-fault checks, bearing analysis, the hybrid logic
  and the operator workflow carry over unchanged. Only the physics module (pump curve → fan curve or compressor map)
  and the fault list change.

### Uncertainty: says how sure it is, and says "unclear" when it is

* **Calibrated confidence.** The raw model is overconfident (says 0.93, right 74 % of the time). An isotonic
  calibration fitted on cross-validation predictions maps it to how often it is actually right, so 0.7 means ~70 %.
* **`UNKNOWN` is a valid answer**: "something is wrong, cause unclear", instead of a confident guess.
* **Two opinions per alert**: the output records the rule verdict, the ML verdict and who decided
  (`RULE`, `ML`, `RULE+ML agree`). Agreement raises confidence; disagreement is visible to the operator.
* **Evidence with every alert** (channels, deviations, physical quantities) so a human can check the reasoning.
* **Honest limits**: alerts start only after ~24 h of baseline; without vibration data bearing faults are reported
  with lower confidence or as `UNKNOWN`.

### Retrofitability: works with the sensors a pump already has

| Level | Sensors | What it can do |
|---|---|---|
| T0 legacy | motor current, discharge pressure, temperatures (+ speed/power with a drive) | hydraulic faults, sensor faults, broad diagnosis |
| T1 retrofit | + suction pressure, wireless vibration and bearing temperature every 5 min | + cavitation margin, bearing, imbalance, misalignment |
| T2 full | + 1 s vibration, flow, winding temperature | most precise diagnosis |

* **Reads which sensors exist** (`channels_available`) and runs only the checks those sensors support; it degrades
  gracefully instead of failing.
* **No OEM data or machine changes needed**: only the nameplate/datasheet values and the existing signals.
* **Advice only**: it recommends inspections; a human approves; it never acts on the machine.
* **Clear upgrade path**: a cheap wireless vibration sensor (T0 → T1) is what enables early bearing, imbalance and
  misalignment diagnosis, which makes the business case for retrofit sensors concrete.

### Robustness: hard to fool, safe to run

* **Causal**: every alert uses only data up to its timestamp, so it can be re-run on truncated data with the same
  result.
* **Sensor vs machine**: physically linked channels and fixed physical references separate a failing pump from a
  failing instrument (drift, stuck, calibration shift, noise, communication loss).
* **Few false alarms**: warm-up suppression after starts, 15-minute persistence, one alert per fault per 12 h,
  `NO_FAULT` training examples; about 0.1 false alerts per pump per day on held-out pumps.
* **Missing data**: gaps are detected as communication loss; 5-minute wireless data is handled, not treated as zero.
* **The ML cannot make detection worse**: alert times come from the physics detector; the model only names the fault,
  and only when calibrated-confident and allowed for that fault type.
* **Reproducible**: fixed seeds, deterministic simulator, format check against the organisers' rules before writing
  the submission.

---

## 7. Evaluation

* **Protocol.** For the comparison, a separate model is trained **without any data of PUMP-08/09** (their
  re-simulated copies are removed; 652 rows). Baselines B0–B3 are the organisers' own results, tuned on training
  pumps; the full comparison with our models is in the
  [Results table](#results-official-scorer-srcretrofit_simscoringpy) at the top. Test pumps PUMP-10..12 were never used for training, tuning or decisions.
* **Caveat.** The physics thresholds are physically motivated but were developed while looking at the faults of
  PUMP-01..09; validation has only 9 faults (one fault ≈ 11 points of fault-type recall). The **held-out** results
  are therefore the main evidence.
* **Physics checked against the simulator's hidden truth** (540 pumps): head error ~1 % on T1/T2 and ~3.8 % on T0;
  NPSH margin ~4 % error with rank correlation 0.95; cavitation flagged in 145/145 true cavitation minutes with no
  false minutes; 2× vibration ratio vs true misalignment 0.85, 1× ratio vs true imbalance 0.84.

| Held-out (90 pumps, 239 faults) | Rules | Hybrid |
|---|---|---|
| Composite | 70.4 | 70.8 |
| Event F1 | 0.772 | 0.772 |
| Timeliness | 0.634 | 0.634 |
| Fault-type recall | 0.586 | 0.598 |
| Machine/sensor class recall | 0.658 | 0.662 |
| False alerts per pump-day | 0.108 | 0.108 |
| Faults detected | 170 / 239 | 170 / 239 |

Generate the full comparison (tables, chart, caveats) with `python3 compare_baselines.py`.

## 8. Limitations and next steps

* **Detection is the main gap**: 69 of 239 held-out faults are never detected. Weakest: sensor drift (5/22 correct),
  imbalance (8/21), misalignment (5/14). Next: drift detection against model-predicted linked channels, 1×/2×
  vibration alarms.
* **Faults present from the first minute** (e.g. impeller wear) become part of the learned "normal"; only datasheet
  comparisons can catch them.
* **T0 legacy pumps** rely on an assumed suction pressure; a learned "virtual suction sensor" trained on
  instrumented pumps would improve them.
* **Simulated data**: trained and tested on the organisers' simulator. Real ships and plants need a pilot to confirm
  thresholds and calibration.
* **Rare types** (air entrainment, impeller wear, cavitation) come from re-simulations of a few public pumps;
  for those the rules have priority.

## 9. Compliance and provenance

* Fit on training pumps, select on validation, no tuning on test pumps; test predictions produced once.
* Extra data generated with the organisers' simulator (`make_dataset.py`), as allowed by the package licence;
  simulator ground truth used only as labels.
* Organisers' code (`src/`, baselines, scorer) is MIT licensed and used unchanged.

## 10. Credits

Business Turku / Elomatic challenge package, Since AI Hackathon 2026.
