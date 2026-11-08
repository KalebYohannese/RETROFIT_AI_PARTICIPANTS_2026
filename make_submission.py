"""Make the submission file for the hidden test pumps (PUMP-10, 11, 12) with Joice's rules + the hybrid.

Run from the RETROFIT_AI_PARTICIPANTS_2026 folder (with the Joice version the model was trained on):
    python3 make_submission.py

Writes submission/predictions_test.csv            <- the file to hand in (template format)
       submission/predictions_validation.csv      <- same pipeline on PUMP-08/09, scored locally as a sanity check
       submission/predictions_test_rules_only.csv <- Joice's rules alone, for reference
Nothing here looks at test labels (there are none), and nothing is tuned on the test pumps.
"""
import argparse, importlib.util, sys, warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent


def load_local(name):
    """load OUR script by file path (Joice's folder contains scripts with the same names)"""
    spec = importlib.util.spec_from_file_location(f"_local_{name}", ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


pp = load_local("physics_pipeline")          # loading, vibration features, Joice detector
ih = load_local("integrate_hybrid")          # hybrid decision (same code that was scored on the test seeds)
from retrofit_sim.scoring import FAULT_TYPES, score   # noqa: E402

TEMPLATE = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"]


def run_split(data: Path, split: str, bundle, calib, min_conf):
    assets = pd.read_csv(data / "assets.csv")
    assets = assets[assets["split"] == split].reset_index(drop=True)
    feat = pd.read_parquet(data / "features_1min.parquet")
    feat = feat[feat["asset_id"].isin(assets["asset_id"])]
    ctx = pd.read_csv(data / "operating_context.csv")
    ctx = ctx[ctx["asset_id"].isin(assets["asset_id"])]
    snaps = pd.read_parquet(data / "vibration_snapshots.parquet")
    snaps = snaps[snaps["asset_id"].isin(assets["asset_id"])]
    zp, _, rules = pp.run_systems(feat, assets, ctx, snaps)
    hyb, _ = ih.hybrid_alerts(zp, assets, rules, bundle, calib, min_conf, 2.0)   # 2.0 = never suppress
    return assets, rules, hyb


def check_format(df: pd.DataFrame, assets: pd.DataFrame, label: str):
    """the checks the organisers' loader applies, plus a few sanity checks"""
    problems = []
    if list(df.columns) != TEMPLATE:
        problems.append(f"columns are {list(df.columns)}")
    if len(df):
        if not set(df["asset_id"]) <= set(assets["asset_id"]):
            problems.append(f"unknown pumps {sorted(set(df['asset_id']) - set(assets['asset_id']))}")
        bad_t = ~df["fault_type"].isin(FAULT_TYPES | {"UNKNOWN"})
        if bad_t.any():
            problems.append(f"invalid fault types {sorted(df.loc[bad_t, 'fault_type'].unique())}")
        if not df["event_class"].isin(["MACHINE", "SENSOR"]).all():
            problems.append("event_class must be MACHINE or SENSOR")
        c = pd.to_numeric(df["confidence"], errors="coerce")
        if c.isna().any() or (c < 0).any() or (c > 1).any():
            problems.append("confidence must be a number between 0 and 1")
        t = pd.to_datetime(df["alert_time_utc"], utc=True, errors="coerce")
        if t.isna().any():
            problems.append("unparseable alert_time_utc")
        win = assets.set_index("asset_id")
        lo = pd.to_datetime(df["asset_id"].map(win["window_start_utc"]), utc=True)
        hi = pd.to_datetime(df["asset_id"].map(win["window_end_utc"]), utc=True)
        if ((t < lo) | (t > hi)).any():
            problems.append("alerts outside the pump's data window")
    print(f"format check {label}: " + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)))
    return not problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/full")
    p.add_argument("--models", default="build/models")
    p.add_argument("--calibrator", default="build/integration/calibrator.joblib")
    p.add_argument("--rule-priority", nargs="*", default=None)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--out", default="submission")
    a = p.parse_args()
    data, out = ROOT / a.data, ROOT / a.out
    out.mkdir(exist_ok=True)
    if a.rule_priority is not None:
        ih.RULE_PRIORITY = set(a.rule_priority)
    print("Rule-priority types:", sorted(ih.RULE_PRIORITY))
    bundle = joblib.load(ROOT / a.models / "diagnostician.joblib")
    calib = joblib.load(ROOT / a.calibrator)

    # ---- sanity check on the validation pumps (labels available)
    va_assets, va_rules, va_hyb = run_split(data, "validation", bundle, calib, a.min_conf)
    va_hyb[TEMPLATE].to_csv(out / "predictions_validation.csv", index=False)
    ev = pd.read_csv(data / "events_validation.csv"); bn = pd.read_csv(data / "benign_windows_train_validation.csv")
    s_r = score(va_rules[TEMPLATE], ev, va_assets, bn, "validation")
    s_h = score(va_hyb[TEMPLATE], ev, va_assets, bn, "validation")
    print(f"validation PUMP-08/09: rules {s_r['composite_score']:.1f}   hybrid {s_h['composite_score']:.1f}"
          "   (optimistic: the extra training seeds re-simulate these pumps)")

    # ---- the deliverable: test pumps
    te_assets, te_rules, te_hyb = run_split(data, "test", bundle, calib, a.min_conf)
    sub = te_hyb[TEMPLATE].sort_values(["asset_id", "alert_time_utc"]).reset_index(drop=True)
    ok = check_format(sub, te_assets, "test")
    sub.to_csv(out / "predictions_test.csv", index=False)
    te_rules[TEMPLATE].to_csv(out / "predictions_test_rules_only.csv", index=False)
    print(f"\nwrote {out / 'predictions_test.csv'}: {len(sub)} alerts" + ("" if ok else "  <-- fix the problems above"))
    print("\nAlerts per test pump:")
    for aid, g in sub.groupby("asset_id"):
        tier = te_assets.set_index("asset_id").loc[aid, "instrumentation_tier"]
        print(f"  {aid} ({tier}): {len(g)} alerts  " + ", ".join(f"{k} x{v}" for k, v in g.fault_type.value_counts().items()))
    n_over = int(te_hyb["evidence"].astype(str).str.startswith("ML ").sum())
    print(f"\nML overrides on the test pumps: {n_over} of {len(sub)} alerts")


if __name__ == "__main__":
    main()