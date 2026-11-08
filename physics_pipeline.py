"""Run B1 and Joice's physics system on generated simulator data, score both, and
check Joice's physics numbers against the simulator's hidden truth.

Run from the RETROFIT_AI_PARTICIPANTS_2026 folder (needs joice_physics_rules/ next to src/):
    python3 physics_pipeline.py                         # all finished seeds in build/generated
    python3 physics_pipeline.py --seeds stress_100 full_300

Outputs (in build/pipeline/):
  features/<seed>.parquet   per-minute z-scores + Joice physics features (input for the ML step)
  alerts/<seed>_B1.csv, alerts/<seed>_JOICE.csv
  scores.csv                B1 vs Joice per seed (composite, F1, fault-type recall, ...)
  per_event.csv             every event: detected? predicted type? for both systems
  physics_check.csv         Joice's computed physics vs simulator truth, per pump

The simulator truth (private/truth_1min.parquet) is used ONLY for checking here, never as a model input.
"""
import argparse, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "joice_physics_rules"))
from retrofit_sim import baselines as B          # noqa: E402
from retrofit_sim.scoring import score           # noqa: E402
import diag2                                      # noqa: E402

UTC_NS = "datetime64[ns, UTC]"


def ts(s):
    """parse to UTC timestamps with one fixed resolution (avoids pandas merge errors)"""
    return pd.to_datetime(s, utc=True).astype(UTC_NS)


# ---------------------------------------------------------------- loading
def load_seed(d: Path):
    pub, priv = d / "public", d / "private"
    feat = pd.read_parquet(pub / "features_1min.parquet")
    assets = pd.read_csv(pub / "assets.csv")
    ctx = pd.read_csv(pub / "operating_context.csv")
    snaps = pd.read_parquet(pub / "vibration_snapshots.parquet") if (pub / "vibration_snapshots.parquet").exists() else None
    if (priv / "events_stress_all.csv").exists():
        ev = pd.read_csv(priv / "events_stress_all.csv")
        bn_file = pub / "benign_windows_stress.csv"
    else:
        ev = pd.concat([pd.read_csv(pub / f) for f in ("events_train.csv", "events_validation.csv") if (pub / f).exists()],
                       ignore_index=True)
        bn_file = pub / "benign_windows_train_validation.csv"
    bn = pd.read_csv(bn_file) if bn_file.exists() else None
    truth = pd.read_parquet(priv / "truth_1min.parquet") if (priv / "truth_1min.parquet").exists() else None
    return feat, assets, ctx, snaps, ev, bn, truth


# ---------------------------------------------------------------- vibration features (Joice's code as a function)
def vibration_features(snaps, assets):
    cols = ["asset_id", "time", "rpm", "a_1x", "a_2x", "a_vane", "env_bpfo", "hf_rms", "crest"]
    if snaps is None or len(snaps) == 0:
        return pd.DataFrame(columns=cols)
    A = assets.set_index("asset_id")

    def amp_at(freqs, spec, f, tol=0.03):
        m = (freqs > f * (1 - tol)) & (freqs < f * (1 + tol))
        return float(spec[m].max()) if m.any() else np.nan

    rows = []
    for _, s in snaps.iterrows():
        a = A.loc[s.asset_id]
        x = np.asarray(s.samples, dtype=float); x -= x.mean()
        fs = s.sample_rate_hz; f1 = s.rpm_at_capture / 60.0
        win = np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x * win)) * 2 / win.sum()
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        bp = butter(4, [2000, 3600], btype="band", fs=fs, output="sos")
        env = np.abs(hilbert(sosfiltfilt(bp, x))); env -= env.mean()
        espec = np.abs(np.fft.rfft(env * win)) * 2 / win.sum()
        hp = butter(4, 4000, btype="high", fs=fs, output="sos")
        rows.append(dict(asset_id=s.asset_id, time=s.timestamp_utc, rpm=s.rpm_at_capture,
                         a_1x=amp_at(freqs, spec, f1), a_2x=amp_at(freqs, spec, 2 * f1),
                         a_vane=amp_at(freqs, spec, a.impeller_vanes * f1),
                         env_bpfo=amp_at(freqs, espec, a.bearing_bpfo_order * f1),
                         hf_rms=float(np.sqrt(np.mean(sosfiltfilt(hp, x) ** 2))),
                         crest=float(np.abs(x).max() / np.sqrt(np.mean(x ** 2)))))
    return pd.DataFrame(rows, columns=cols).sort_values(["asset_id", "time"])


def empty_snap_ratios():
    return pd.DataFrame({"asset_id": pd.Series(dtype=str), "time": pd.Series(dtype=UTC_NS),
                         **{k: pd.Series(dtype=float) for k in ("r_bpfo", "r_1x", "r_2x", "r_hf", "b_idx", "r_bidx")}})


# ---------------------------------------------------------------- B1 + Joice on one seed
def run_systems(feat, assets, ctx, snaps):
    zf = B.zscores(B.regime_frame(feat, assets))
    zf["window_start_utc"] = ts(zf["window_start_utc"])
    ctx = ctx.copy(); ctx["t"] = ts(ctx["timestamp_utc"])
    vib = vibration_features(snaps, assets)
    if len(vib):
        snapf = diag2.snap_ratios(vib); snapf["time"] = ts(snapf["time"])
    else:
        snapf = empty_snap_ratios()
    zp = diag2.prep(zf, ctx, snapf, assets)
    b1 = B.b1_threshold(zf, 6.0, 15)          # same settings the organisers chose for B1
    jo = diag2.detect(zp, assets)
    return zp, b1, jo


def score_both(name, b1, jo, ev, assets, bn):
    rows, per = [], []
    for sysname, pred in (("B1", b1), ("JOICE", jo)):
        r = score(pred, ev, assets, bn, None)
        rows.append(dict(seed=name, system=sysname, **{k: round(float(r[k]), 3) for k in (
            "composite_score", "event_f1", "timeliness", "fault_type_recall", "class_recall",
            "false_alerts_per_asset_day")}))
        pe = pd.DataFrame(r["per_event"])
        if len(pe):
            pe.insert(0, "system", sysname); pe.insert(0, "seed", name); per.append(pe)
    return rows, per


# ---------------------------------------------------------------- physics vs truth
def physics_check(name, zp, truth, assets):
    """Compare Joice's computed physics with the simulator's hidden truth, per pump (steady running minutes)."""
    if truth is None:
        return []
    t = truth.copy(); t["window_start_utc"] = ts(t["timestamp_utc"])
    m = zp.merge(t, on=["asset_id", "window_start_utc"], how="inner", suffixes=("", "_true"))
    m = m[m["steady"].astype(bool) & (m["since_start"] >= 120)]
    A = assets.set_index("asset_id")
    out = []
    for a, g in m.groupby("asset_id"):
        s = A.loc[a]
        r = (g["motor_speed_rpm__mean"] / s.motor_rated_rpm).clip(lower=0.3) if "motor_speed_rpm__mean" in g and g["motor_speed_rpm__mean"].notna().any() else 1.0
        head_est = g["ph_Ht"] * s.pump_rated_head_m * r ** 2
        row = dict(seed=name, asset_id=a, tier=s.instrumentation_tier, control=s.control, minutes=len(g))

        def pct_err(est, true):
            ok = est.notna() & true.notna() & (true.abs() > 1e-6)
            return float(((est[ok] - true[ok]).abs() / true[ok].abs()).median() * 100) if ok.sum() > 30 else np.nan

        def rho(x, y):
            ok = x.notna() & y.notna()
            if ok.sum() < 30 or x[ok].nunique() < 3 or y[ok].nunique() < 3:
                return np.nan
            return float(spearmanr(x[ok], y[ok]).correlation)

        # 1) direct physical quantities
        row["head_median_err_pct"] = pct_err(head_est, g["head_m"])
        row["npsh_margin_median_err_pct"] = pct_err(g["ph_sigma"], g["npsh_margin"])
        row["npsh_margin_rank_corr"] = rho(g["ph_sigma"], g["npsh_margin"])
        # cavitation call: Joice says margin < 1.15  vs  simulator says bubbles present
        if g["ph_sigma"].notna().any():
            pred, true = g["ph_sigma"] < 1.15, g["cavitation_intensity"] > 0
            row["cav_minutes_true"] = int(true.sum()); row["cav_minutes_flagged"] = int(pred.sum())
            row["cav_minutes_both"] = int((pred & true).sum())
        # 2) fault indicators vs hidden degradation level (rank correlation; NaN if that fault never moved)
        for feat, latent in (("b_idx", "latent_bearing_wear"), ("r_2x", "latent_misalignment"),
                             ("r_1x", "latent_unbalance"), ("ph_dH", "latent_impeller_wear"),
                             ("ph_dfH", "latent_impeller_wear"), ("ph_sigma", "latent_clog")):
            if feat in g and latent in g:
                sign = -1 if feat in ("ph_dH", "ph_dfH", "ph_sigma") else 1   # these fall as the fault grows
                row[f"corr_{feat}__{latent}"] = rho(sign * g[feat], g[latent])
        out.append(row)
    return out


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="build/generated")
    p.add_argument("--out", default="build/pipeline")
    p.add_argument("--seeds", nargs="*", help="folder names like stress_100 full_300 (default: all finished)")
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    (out / "features").mkdir(parents=True, exist_ok=True); (out / "alerts").mkdir(exist_ok=True)
    seeds = a.seeds or sorted(d.name for d in data.iterdir() if (d / "DONE").exists())
    if not seeds:
        sys.exit(f"No finished seeds in {data}")

    scores, per_event, checks = [], [], []
    for i, name in enumerate(seeds, 1):
        t0 = time.time()
        feat, assets, ctx, snaps, ev, bn, truth = load_seed(data / name)
        zp, b1, jo = run_systems(feat, assets, ctx, snaps)
        zp.to_parquet(out / "features" / f"{name}.parquet")
        b1.to_csv(out / "alerts" / f"{name}_B1.csv", index=False)
        jo.to_csv(out / "alerts" / f"{name}_JOICE.csv", index=False)
        s, pe = score_both(name, b1, jo, ev, assets, bn); scores += s; per_event += pe
        checks += physics_check(name, zp, truth, assets)
        b, j = s[0], s[1]
        print(f"[{i}/{len(seeds)}] {name}: B1 {b['composite_score']:.1f} (type {b['fault_type_recall']:.2f})  "
              f"JOICE {j['composite_score']:.1f} (type {j['fault_type_recall']:.2f})  {time.time() - t0:.0f}s", flush=True)

    S = pd.DataFrame(scores); S.to_csv(out / "scores.csv", index=False)
    PE = pd.concat(per_event, ignore_index=True) if per_event else pd.DataFrame()
    PE.to_csv(out / "per_event.csv", index=False)
    C = pd.DataFrame(checks); C.to_csv(out / "physics_check.csv", index=False)

    print("\n=== Average over seeds ===")
    print(S.drop(columns="seed").groupby("system").mean().round(3).to_string())
    if len(PE):
        print("\n=== Correct fault type per true type (events across all seeds) ===")
        PE["type_correct"] = PE["type_correct"].astype(bool)
        t = PE.pivot_table(index="fault_type", columns="system", values="type_correct", aggfunc="sum", fill_value=0)
        t["events"] = PE[PE.system == "B1"].groupby("fault_type").size()
        print(t.astype(int).to_string())
    if len(C):
        print("\n=== Joice's physics vs simulator truth (median over pumps) ===")
        num = C.select_dtypes("number")
        print(num.median().round(3).to_string())
        print("\nhead error % by tier:", C.groupby("tier")["head_median_err_pct"].median().round(2).to_dict())


if __name__ == "__main__":
    main()