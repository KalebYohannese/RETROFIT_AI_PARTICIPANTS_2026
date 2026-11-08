"""Metrics comparison deliverable: our system vs the organisers' baselines B0-B3, following their protocol
(fit on PUMP-01..07, evaluate on PUMP-08/09), scored with the official scorer.

Run from the RETROFIT_AI_PARTICIPANTS_2026 folder :
    python3 compare_baselines.py

Protocol for the ML part: a separate "protocol model" is trained WITHOUT any data of the validation pumps
(the re-simulated copies of PUMP-08/09 in the extra seeds are removed); generated stress pumps are new random pumps.
Outputs in build/comparison/: comparison.csv, comparison.md (table + caveats), comparison.png (if matplotlib is installed)
"""
import argparse, importlib.util, subprocess, sys, warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent


def load_local(name):
    spec = importlib.util.spec_from_file_location(f"_local_{name}", ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


ms = load_local("make_submission")           # run_split(): Joice rules + hybrid on an official split
ih = ms.ih
from retrofit_sim.scoring import score       # noqa: E402

TEMPLATE = ms.TEMPLATE
METRICS = ["composite_score", "event_f1", "timeliness", "fault_type_recall", "class_recall", "false_alerts_per_asset_day"]
VALIDATION_PUMPS = {"PUMP-08", "PUMP-09"}
NAMES = {"B0_scada_alarms": "B0 SCADA alarms", "B1_mode_aware_z": "B1 mode-aware z", "B2_cusum": "B2 CUSUM",
         "B3_isolation_forest": "B3 Isolation Forest"}


def train_protocol_model(table: Path, out: Path):
    T = pd.read_parquet(table)
    leak = T["seed_folder"].str.startswith("full") & T["asset_id"].isin(VALIDATION_PUMPS)
    T = T[~leak]
    out.mkdir(parents=True, exist_ok=True)
    T.to_parquet(out / "train_table_protocol.parquet", index=False)
    print(f"protocol training table: removed {int(leak.sum())} rows from re-simulated PUMP-08/09, "
          f"{int(T.role.isin(['train', 'extra']).sum())} training rows left")
    log = out / "train_protocol.log"
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, str(ROOT / "train_diagnostician.py"), "--table", str(out / "train_table_protocol.parquet"),
                            "--out", str(out / "model_protocol")], stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"protocol model training failed, see {log}")
    import joblib
    bundle = joblib.load(out / "model_protocol" / "diagnostician.joblib")
    calib = ih.fit_calibrator(out / "model_protocol" / "cv_predictions.csv")
    joblib.dump(calib, out / "model_protocol" / "calibrator.joblib")
    return bundle, calib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/full")
    p.add_argument("--table", default="build/training/train_table.parquet")
    p.add_argument("--pipeline", default="build/pipeline", help="for the held-out generated pumps (B1 / rules)")
    p.add_argument("--integration", default="build/integration", help="for the held-out generated pumps (hybrid)")
    p.add_argument("--rule-priority", nargs="*", default=None)
    p.add_argument("--out", default="build/comparison")
    a = p.parse_args()
    data, out = ROOT / a.data, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.rule_priority is not None:
        ih.RULE_PRIORITY = set(a.rule_priority)

    bundle, calib = train_protocol_model(ROOT / a.table, out)

    rows = []
    # ---- organisers' baselines (their own measured results, tuned on train)
    bm = pd.read_csv(ROOT / "baselines" / "results" / "baseline_metrics.csv")
    for _, r in bm.iterrows():
        rows.append(dict(method=NAMES.get(r.baseline, r.baseline), split=r.split, **{m: r[m] for m in METRICS}))

    # ---- ours, on the official train and validation pumps
    bn = pd.read_csv(data / "benign_windows_train_validation.csv")
    for split, evf in (("train", "events_train.csv"), ("validation", "events_validation.csv")):
        assets, rules, hyb = ms.run_split(data, split, bundle, calib, 0.5)
        ev = pd.read_csv(data / evf)
        for name, pred in (("Ours: Joice v4 physics rules", rules), ("Ours: v4 rules + hybrid ML", hyb)):
            r = score(pred[TEMPLATE], ev, assets, bn, split)
            rows.append(dict(method=name, split=split, **{m: r[m] for m in METRICS}))
            if split == "validation":
                pred[TEMPLATE].to_csv(out / f"predictions_{'hybrid' if 'hybrid' in name else 'rules'}_validation.csv", index=False)
        print(f"scored {split}")

    # ---- held-out generated pumps (90 unseen pumps, 239 faults): the less biased check
    held = {}
    try:
        s = pd.read_csv(ROOT / a.pipeline / "scores.csv"); s = s[s.seed.isin([f"stress_{i}" for i in range(160, 175)])]
        g = s.groupby("system")[METRICS].mean()
        held["B1 mode-aware z"] = g.loc["B1"]; held["Ours: Joice v4 physics rules"] = g.loc["JOICE"]
        s2 = pd.read_csv(ROOT / a.integration / "scores.csv")
        held["Ours: v4 rules + hybrid ML"] = s2[s2.version == "HYBRID"][METRICS].mean()
        for k, v in held.items():
            rows.append(dict(method=k, split="held-out generated (90 pumps)", **v.to_dict()))
    except Exception as e:
        print("held-out generated results not found, skipped:", e)

    R = pd.DataFrame(rows)
    R.to_csv(out / "comparison.csv", index=False)
    order = ["B0 SCADA alarms", "B1 mode-aware z", "B2 CUSUM", "B3 Isolation Forest",
             "Ours: Joice v4 physics rules", "Ours: v4 rules + hybrid ML"]
    piv = R.pivot_table(index="method", columns="split", values="composite_score").reindex(order)
    print("\nComposite score (official scorer):"); print(piv.round(1).to_string())

    # ---- markdown report
    def table(split):
        t = R[R.split == split].set_index("method").reindex([o for o in order if o in set(R[R.split == split].method)])
        lines = ["| Method | Composite | Event F1 | Timeliness | Fault-type recall | Class recall | False alerts / pump-day |",
                 "|---|---|---|---|---|---|---|"]
        for m, r in t.iterrows():
            lines.append(f"| {m} | {r.composite_score:.1f} | {r.event_f1:.2f} | {r.timeliness:.2f} | {r.fault_type_recall:.2f} "
                         f"| {r.class_recall:.2f} | {r.false_alerts_per_asset_day:.3f} |")
        return "\n".join(lines)

    md = ["# Metrics comparison: our system vs the organisers' baselines", "",
          "All numbers from the official scorer (`src/retrofit_sim/scoring.py`). Baselines B0-B3 are the organisers' own "
          "results (`baselines/results/`), tuned on the training pumps.", "",
          "## Validation pumps PUMP-08/09 (official protocol)", "", table("validation"), "",
          "## Training pumps PUMP-01..07", "", table("train"), ""]
    if held:
        md += ["## 90 held-out generated pumps (15 unseen simulator seeds, 239 faults)", "", table("held-out generated (90 pumps)"), ""]
    md += ["## How to read this", "",
           "* **Protocol.** The ML diagnostician used here was trained without any data from PUMP-08/09 "
           "(re-simulated copies removed); it uses generated pumps (new random pumps from the provided simulator) and "
           "re-simulations of PUMP-01..07. Test pumps PUMP-10..12 were never used.",
           "* **Caveat on the physics rules.** Their thresholds come from pump physics and datasheets, but they were developed "
           "while looking at the events of PUMP-01..09, so the official validation score is optimistic for the rules. "
           "The validation set is also tiny (2 pumps, 9 faults: one fault = ~11 points of fault-type recall).",
           "* **The least biased evidence** is the held-out generated pumps: 90 unseen pumps with random ratings, "
           "instrumentation levels and fault schedules, never used for development.", ""]
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    # ---- chart
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        splits = [s for s in ("train", "validation", "held-out generated (90 pumps)") if s in piv.columns]
        ax = piv[splits].plot(kind="barh", figsize=(9, 5), width=0.8)
        ax.set_xlabel("Composite score (official scorer, 0-100)"); ax.set_ylabel(""); ax.invert_yaxis()
        ax.legend(title="Evaluated on", loc="lower right"); ax.set_xlim(0, 100)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f", fontsize=8, padding=2)
        plt.tight_layout(); plt.savefig(out / "comparison.png", dpi=150)
        print(f"chart: {out / 'comparison.png'}")
    except Exception as e:
        print("chart skipped (pip install matplotlib):", e)
    print(f"report: {out / 'comparison.md'}")


if __name__ == "__main__":
    main()