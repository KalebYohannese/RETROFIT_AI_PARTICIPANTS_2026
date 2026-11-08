# Metrics comparison: our system vs the organisers' baselines

All numbers from the official scorer (`src/retrofit_sim/scoring.py`). Baselines B0-B3 are the organisers' own results (`baselines/results/`), tuned on the training pumps.

## Validation pumps PUMP-08/09 (official protocol)

| Method | Composite | Event F1 | Timeliness | Fault-type recall | Class recall | False alerts / pump-day |
|---|---|---|---|---|---|---|
| B0 SCADA alarms | 23.1 | 0.20 | 0.11 | 0.11 | 0.11 | 0.000 |
| B1 mode-aware z | 78.0 | 0.89 | 0.86 | 0.44 | 0.89 | 0.071 |
| B2 CUSUM | 50.9 | 0.45 | 0.77 | 0.33 | 0.67 | 1.071 |
| B3 Isolation Forest | 69.2 | 0.74 | 0.78 | 0.44 | 0.78 | 0.214 |
| Ours: Joice v4 physics rules | 95.8 | 1.00 | 0.93 | 0.89 | 1.00 | 0.000 |
| Ours: v4 rules + hybrid ML | 96.1 | 1.00 | 0.95 | 0.89 | 1.00 | 0.000 |

## Training pumps PUMP-01..07

| Method | Composite | Event F1 | Timeliness | Fault-type recall | Class recall | False alerts / pump-day |
|---|---|---|---|---|---|---|
| B0 SCADA alarms | 27.6 | 0.31 | 0.21 | 0.04 | 0.23 | 0.143 |
| B1 mode-aware z | 58.2 | 0.72 | 0.58 | 0.27 | 0.54 | 0.122 |
| B2 CUSUM | 53.3 | 0.48 | 0.81 | 0.27 | 0.81 | 0.939 |
| B3 Isolation Forest | 39.4 | 0.47 | 0.41 | 0.08 | 0.38 | 0.204 |
| Ours: Joice v4 physics rules | 86.7 | 0.92 | 0.78 | 0.81 | 0.88 | 0.041 |
| Ours: v4 rules + hybrid ML | 85.9 | 0.88 | 0.78 | 0.85 | 0.85 | 0.041 |

## 90 held-out generated pumps (15 unseen simulator seeds, 239 faults)

| Method | Composite | Event F1 | Timeliness | Fault-type recall | Class recall | False alerts / pump-day |
|---|---|---|---|---|---|---|
| B1 mode-aware z | 60.0 | 0.71 | 0.57 | 0.37 | 0.52 | 0.130 |
| Ours: Joice v4 physics rules | 70.4 | 0.77 | 0.63 | 0.59 | 0.66 | 0.108 |
| Ours: v4 rules + hybrid ML | 70.8 | 0.77 | 0.63 | 0.60 | 0.66 | 0.108 |

## How to read this

* **Protocol.** The ML diagnostician used here was trained without any data from PUMP-08/09 (re-simulated copies removed); it uses generated pumps (new random pumps from the provided simulator) and re-simulations of PUMP-01..07. Test pumps PUMP-10..12 were never used.
* **Caveat on the physics rules.** Their thresholds come from pump physics and datasheets, but they were developed while looking at the events of PUMP-01..09, so the official validation score is optimistic for the rules. The validation set is also tiny (2 pumps, 9 faults: one fault = ~11 points of fault-type recall).
* **The least biased evidence** is the held-out generated pumps: 90 unseen pumps with random ratings, instrumentation levels and fault schedules, never used for development.
