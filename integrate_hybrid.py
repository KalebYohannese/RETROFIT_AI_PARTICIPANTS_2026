"""Plug the hybrid diagnostician into the alert pipeline and score it with the official scorer.

Run from the RETROFIT_AI_PARTICIPANTS_2026 folder, after physics_pipeline.py and train_diagnostician.py:
    python3 integrate_hybrid.py                       # all test-role seeds (held out)
    python3 integrate_hybrid.py --seeds stress_160 stress_161

For every alert of Joice's detector (same alert times), the type and class are decided by:
  1. RULE-PRIORITY types (cavitation, air entrainment, impeller wear, comms loss, calibration shift,
     noise spikes): keep Joice's answer
  2. otherwise the ML model, if its CALIBRATED confidence >= --min-conf
  3. otherwise Joice's answer
Three versions are scored on the same seeds with the official scorer (src/retrofit_sim/scoring.py):
  JOICE            Joice's detector + rules (as in physics_pipeline.py)
  HYBRID           alert times unchanged, hybrid decides type/class
  HYBRID_SUPPRESS  same, but an alert is dropped when the model says NO_FAULT with calibrated confidence >= --suppress-conf

Alerts are never dropped (except HYBRID_SUPPRESS); an override that would clash with a neighbouring
alert for the scorer is undone instead.
Outputs in build/integration/: alerts/<seed>_<version>.csv, scores.csv, per_event.csv, calibrator.joblib
"""
import argparse, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "joice_physics_rules"))
from retrofit_sim import baselines as B                       # noqa: E402
from retrofit_sim.scoring import score                        # noqa: E402
import importlib.util                                          # noqa: E402


def _load_local(name):
    """load OUR script by file path (Joice's folder has a build_training_table.py with the same name)"""
    spec = importlib.util.spec_from_file_location(f"_local_{name}", ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


_btt, _td = _load_local("build_training_table"), _load_local("train_diagnostician")
features_at, ts = _btt.features_at, _btt.ts                    # same features as training
make_X, CLASS_OF = _td.make_X, _td.CLASS_OF

# Final choice (from the cross-validation table of the final model: types where the rule >= the model)
RULE_PRIORITY = {"CAVITATION", "AIR_ENTRAINMENT", "IMPELLER_WEAR", "COMMUNICATION_LOSS",
                 "SENSOR_CALIBRATION_SHIFT", "SENSOR_NOISE_SPIKES"}
ACTIONS = {
    "BEARING_WEAR": "Plan bearing replacement; check lubrication.",
    "SUCTION_STRAINER_CLOGGING": "Inspect / clean suction strainer.",
    "CAVITATION": "Raise suction level / clean strainer / reduce flow.",
    "DISCHARGE_RESTRICTION": "Check discharge valves and heat-exchanger fouling.",
    "MISALIGNMENT": "Check coupling alignment (likely after maintenance).",
    "IMBALANCE": "Inspect impeller for fouling; balance.",
    "IMPELLER_WEAR": "Performance test; inspect impeller / wear rings.",
    "AIR_ENTRAINMENT": "Check sea-chest venting / basin level; air in suction.",
    "SENSOR_DRIFT": "Verify the instrument against a reference gauge; recalibrate.",
    "SENSOR_STUCK": "Check transmitter / wireless sensor.",
    "SENSOR_CALIBRATION_SHIFT": "Verify the instrument against a reference gauge; recalibrate.",
    "SENSOR_NOISE_SPIKES": "Check cabling / grounding of the instrument.",
    "COMMUNICATION_LOSS": "Check gateway / network; data gap.",
    "UNKNOWN": "Inspect pump; cause unclear.",
}
COLS = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"]


def fit_calibrator(cv_file):
    """Isotonic map: model's raw top-class probability -> probability the answer is actually right
    (fitted on the cross-validation predictions, i.e. on pumps the fold models had not seen)."""
    cv = pd.read_csv(cv_file)
    cv = cv[cv["MODEL"].astype(str) != ""]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(cv["MODEL_conf"].values, (cv["MODEL"] == cv["label_type"]).astype(float).values)
    return iso


def load_seed(d: Path):
    pub, priv = d / "public", d / "private"
    assets = pd.read_csv(pub / "assets.csv")
    if (priv / "events_stress_all.csv").exists():
        ev, bnf = pd.read_csv(priv / "events_stress_all.csv"), pub / "benign_windows_stress.csv"
    else:
        ev = pd.concat([pd.read_csv(pub / f) for f in ("events_train.csv", "events_validation.csv") if (pub / f).exists()])
        bnf = pub / "benign_windows_train_validation.csv"
    return assets, ev, (pd.read_csv(bnf) if bnf.exists() else None)


def resolve_conflicts(rows):
    """Never drop an alert Joice kept. If an ML override would create a conflict for the scorer
    (same fault on the same pump within 12 h, or a different guess within 5 min of the previous alert),
    undo that override and keep Joice's original answer instead. rows: list of (time, original, new)."""
    out, last_any, last_type = [], {}, {}
    for t, orig, new in sorted(rows, key=lambda x: (x[1]["asset_id"], x[0])):
        a = orig["asset_id"]
        if new["fault_type"] != orig["fault_type"]:
            ft = new["fault_type"]
            repeat = ft != "CAVITATION" and (a, ft) in last_type and t - last_type[(a, ft)] < B.REFRACTORY
            burst = a in last_any and t - last_any[a][0] <= pd.Timedelta(minutes=5) and last_any[a][1] != ft
            if repeat or burst:
                new = orig
        out.append(new)
        last_any[a] = (t, new["fault_type"]); last_type[(a, new["fault_type"])] = t
    return out


def hybrid_alerts(zp, assets, joice, bundle, calib, min_conf, suppress_conf):
    """returns (hybrid, hybrid_suppress) alert tables built on Joice's alert times"""
    model, cols = bundle["model"], bundle["columns"]
    A = assets.set_index("asset_id")
    out_h, out_s = [], []
    for aid, g in zp.groupby("asset_id", sort=False):
        al = joice[joice.asset_id == aid]
        if al.empty:
            continue
        g = g.sort_values("window_start_utc").reset_index(drop=True)
        tend = g["window_start_utc"] + pd.Timedelta(minutes=1)
        feats, idx = [], []
        for i, r in al.iterrows():
            j = int(tend.searchsorted(ts(pd.Series([r.alert_time_utc]))[0], side="right") - 1)
            if j < 0:
                continue
            feats.append(features_at(g, j, A.loc[aid], rule_override=(r.fault_type, r.event_class, float(r.confidence))))
            idx.append(i)
        if not feats:
            continue
        F = pd.DataFrame(feats, index=idx)
        F["rule_type"] = F["rule_type"].fillna("UNKNOWN")
        X = make_X(F, cols)
        P = pd.DataFrame(model.predict_proba(X), index=idx, columns=model.classes_)
        for i in idx:
            r = al.loc[i]
            m_type = P.loc[i].idxmax(); raw = float(P.loc[i].max())
            cal = float(calib.predict([raw])[0])
            base = r.to_dict(); t_al = ts(pd.Series([r.alert_time_utc]))[0]
            rule_type = r.fault_type
            if rule_type in RULE_PRIORITY or m_type == "NO_FAULT" or cal < min_conf or m_type == rule_type:
                new = dict(base)                          # keep Joice's answer
                if m_type == rule_type:
                    new["confidence"] = max(float(r.confidence), cal)
                    new["evidence"] = f"{r.evidence} | ML agrees ({cal:.2f})"
            else:
                new = dict(base, fault_type=m_type, event_class=CLASS_OF(m_type), confidence=round(cal, 3),
                           recommended_action=ACTIONS.get(m_type, base["recommended_action"]),
                           evidence=f"ML {m_type} ({cal:.2f}) overrides rule {rule_type}; rule evidence: {r.evidence}")
            out_h.append((t_al, base, new))
            if not (m_type == "NO_FAULT" and cal >= suppress_conf and rule_type not in RULE_PRIORITY):
                out_s.append((t_al, base, new))
    to_df = lambda rows: pd.DataFrame(resolve_conflicts(rows), columns=COLS) if rows else pd.DataFrame(columns=COLS)
    return to_df(out_h), to_df(out_s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="build/generated")
    p.add_argument("--pipeline", default="build/pipeline")
    p.add_argument("--models", default="build/models")
    p.add_argument("--out", default="build/integration")
    p.add_argument("--seeds", nargs="*", help="default: all seeds with role 'test'")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--suppress-conf", type=float, default=0.8)
    p.add_argument("--rule-priority", nargs="*", default=None,
                   help="fault types where the rule always wins (default: built-in list). Choose from CROSS-VALIDATION results, not test")
    a = p.parse_args()
    data, pipe, mdir, out = Path(a.data), Path(a.pipeline), Path(a.models), Path(a.out)
    (out / "alerts").mkdir(parents=True, exist_ok=True)

    global RULE_PRIORITY
    if a.rule_priority is not None:
        RULE_PRIORITY = set(a.rule_priority)
    print("Rule-priority types:", sorted(RULE_PRIORITY))
    bundle = joblib.load(mdir / "diagnostician.joblib")
    calib = fit_calibrator(mdir / "cv_predictions.csv"); joblib.dump(calib, out / "calibrator.joblib")
    grid = np.array([0.3, 0.5, 0.7, 0.9, 0.97])
    print("Calibration (raw model confidence -> calibrated):",
          ", ".join(f"{r:.2f}->{c:.2f}" for r, c in zip(grid, calib.predict(grid))))

    if a.seeds:
        seeds = a.seeds
    else:
        aa = pd.read_csv(data / "assets_all.csv")
        seeds = sorted({Path(d).name for d, r in zip(aa.data_dir, aa.role) if r == "test"})
    if not seeds:
        sys.exit("No seeds to evaluate.")

    scores, per = [], []
    for k, name in enumerate(seeds, 1):
        zp = pd.read_parquet(pipe / "features" / f"{name}.parquet"); zp["window_start_utc"] = ts(zp["window_start_utc"])
        joice = pd.read_csv(pipe / "alerts" / f"{name}_JOICE.csv")
        assets, ev, bn = load_seed(data / name)
        hyb, sup = hybrid_alerts(zp, assets, joice, bundle, calib, a.min_conf, a.suppress_conf)
        line = []
        for ver, al in (("JOICE", joice), ("HYBRID", hyb), ("HYBRID_SUPPRESS", sup)):
            al.to_csv(out / "alerts" / f"{name}_{ver}.csv", index=False)
            r = score(al, ev, assets, bn, None)
            n_over = int(al["evidence"].astype(str).str.startswith("ML ").sum()) if len(al) else 0
            scores.append(dict(seed=name, version=ver, alerts=len(al), ml_overrides=n_over, **{m: round(float(r[m]), 3) for m in (
                "composite_score", "event_f1", "timeliness", "fault_type_recall", "class_recall", "false_alerts_per_asset_day")}))
            pe = pd.DataFrame(r["per_event"])
            if len(pe):
                pe.insert(0, "version", ver); pe.insert(0, "seed", name); per.append(pe)
            line.append(f"{ver} {r['composite_score']:.1f}")
        print(f"[{k}/{len(seeds)}] {name}: " + "  ".join(line), flush=True)

    S = pd.DataFrame(scores); S.to_csv(out / "scores.csv", index=False)
    PE = pd.concat(per, ignore_index=True); PE.to_csv(out / "per_event.csv", index=False)
    print("\n=== Official scorer, average over seeds ===")
    print(S.drop(columns="seed").groupby("version").mean().round(3).to_string())
    PE["type_correct"] = PE["type_correct"].astype(bool)
    t = PE.pivot_table(index="fault_type", columns="version", values="type_correct", aggfunc="sum", fill_value=0)
    t["events"] = PE[PE.version == "JOICE"].groupby("fault_type").size()
    print("\n=== Events with the correct fault type (first alert decides) ===")
    print(t.astype(int).to_string())
    det = PE.pivot_table(index="version", values="detected", aggfunc="sum")
    print("\nEvents detected per version:", det["detected"].astype(int).to_dict())


if __name__ == "__main__":
    main()