"""Build the training table for the ML diagnostician.

Run from the RETROFIT_AI_PARTICIPANTS_2026 folder, after generate_training_data.py and physics_pipeline.py:
    python3 build_training_table.py

One row = one moment on one pump:
  * EVENT rows    : several moments inside each observable fault event (0.5, 1, 3, 6, 12 h after it
                    became detectable), so the model learns each fault both early and well developed
  * ALERT rows    : every moment Joice's detector actually fired (what the model sees in practice);
                    alerts outside any event are labelled NO_FAULT (false alarms to learn to suppress)
  * NORMAL rows   : random healthy moments and benign moments (start-ups, load changes) -> NO_FAULT

Features use ONLY data up to that moment (causal). Labels come from the simulator answer key.
Output: build/training/train_table.parquet (+ a short summary printed)
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "joice_physics_rules"))
from retrofit_sim import baselines as B   # noqa: E402
import diag2                               # noqa: E402

UTC_NS = "datetime64[ns, UTC]"
ts = lambda s: pd.to_datetime(s, utc=True).astype(UTC_NS)

CH = list(B.CH)                                                        # channel short names: pd, ps, q, i, ...
LAST_COLS = ([f"z_{c}" for c in CH] + [f"zs_{c}" for c in CH] +
             ["since_start", "prev_stop_min", "speed_ratio", "sea_state_hs_m", "suction_static_head_m",
              "ph_Ht", "ph_q", "ph_fH", "ph_dH", "ph_dP", "ph_dfH", "ph_dfH6h", "ph_sigma", "ph_iP_res",
              "r_bpfo", "r_1x", "r_2x", "r_hf", "b_idx", "r_bidx",
              # Joice v4 additions (NaN when running with v3)
              "ph_bt_dev", "ph_bt_same_h"] +
             [f"ph_rr_{c}" for c in ("flow_m3_h", "discharge_pressure_kpa", "suction_pressure_kpa",
                                     "motor_current_a", "motor_power_kw", "motor_speed_rpm")])
RR_COLS = [c for c in LAST_COLS if c.startswith("ph_rr_")]
TREND_COLS = [f"z_{c}" for c in CH] + ["ph_dH", "ph_dP", "ph_dfH", "ph_sigma", "b_idx"]
WINDOWS = (30, 180, 720)                                               # minutes
EVENT_OFFSETS_H = (0.5, 1, 3, 6, 12)
TIERS = ["T0_LEGACY", "T1_RETROFIT", "T2_FULL"]
CONTROLS = ["FIXED_SPEED", "VFD_PRESSURE", "VFD_TEMPERATURE", "FLOW_VALVE"]
N_NORMAL, N_BENIGN = 8, 4                                              # per pump


def features_at(g, j, asset, k=6.0, rule_override=None):
    """All features for pump frame g (sorted, 0..n-1) at row j, using rows <= j only.
    rule_override = (fault_type, event_class, confidence) of a detector alert at this moment: the rule verdict
    is then the alert itself (v4 puts its new sensor checks in the detector, not in diagnose2)."""
    f = {}
    row = g.iloc[j]
    for c in LAST_COLS:
        f[c] = float(row[c]) if c in g and pd.notna(row[c]) else np.nan
    for c in CH:
        f[f"flat_{c}"] = float(bool(row.get(f"flat_{c}", False)) if pd.notna(row.get(f"flat_{c}", np.nan)) else 0.0)
    for w in WINDOWS:
        lo = max(0, j - w)
        win = g.iloc[lo:j + 1]
        f[f"gap_frac_{w}"] = float(win["gap"].astype(float).mean())
        f[f"steady_frac_{w}"] = float(win["steady"].astype(float).mean())
        for c in TREND_COLS:
            if c not in g:
                continue
            x = win[c].astype(float)
            if x.notna().sum() < 3:
                f[f"{c}_d{w}"] = f[f"{c}_std{w}"] = f[f"{c}_absmax{w}"] = np.nan
                continue
            head, tail = x.iloc[:5].median(), x.iloc[-5:].median()
            f[f"{c}_d{w}"] = tail - head                      # change over the window (trend)
            f[f"{c}_std{w}"] = x.std()                        # roughness / noise
            f[f"{c}_absmax{w}"] = x.abs().max()
    # in-minute range (v4): biggest range ratio now and its peak over 30 min / 3 h (intermittent noise)
    rr = [c for c in RR_COLS if c in g]
    if rr:
        R = g[rr].iloc[max(0, j - 180):j + 1].astype(float).max(axis=1)
        f["ph_rr_max"] = float(R.iloc[-1]) if pd.notna(R.iloc[-1]) else np.nan
        f["ph_rr_max_peak30"] = float(R.iloc[-31:].max()) if R.notna().any() else np.nan
        f["ph_rr_max_peak180"] = float(R.max()) if R.notna().any() else np.nan
    else:
        f["ph_rr_max"] = f["ph_rr_max_peak30"] = f["ph_rr_max_peak180"] = np.nan
    # step vs ramp hint for the most deviating channel: biggest 10-min jump / total change in last 6 h
    zc = {c: abs(f.get(f"z_{c}", np.nan)) for c in CH if pd.notna(f.get(f"z_{c}", np.nan))}
    if zc:
        top = max(zc, key=zc.get)
        x = g[f"z_{top}"].iloc[max(0, j - 360):j + 1].astype(float).ffill().fillna(0).values
        total = abs(x[-1] - np.median(x[:20])) if len(x) >= 30 else np.nan
        jump = np.nanmax(np.abs(x[10:] - x[:-10])) if len(x) > 10 else np.nan
        f["top_jump_ratio"] = float(jump / total) if total and total > 1e-6 else np.nan
        # does the top channel move alone? (sensor-fault hint) = max |z| of its physically coupled channels / its |z|
        cpl = [abs(f.get(f"z_{x}", np.nan)) for x in B.COUPLED.get(top, []) if pd.notna(f.get(f"z_{x}", np.nan))]
        f["top_coupled_ratio"] = float(max(cpl) / zc[top]) if cpl and zc[top] > 1e-6 else np.nan
    # pump description (no pump identity, no profile name, so nothing to memorise)
    for t in TIERS:
        f[f"tier_{t}"] = float(asset.instrumentation_tier == t)
    for c in CONTROLS:
        f[f"ctl_{c}"] = float(asset.control == c)
    chans = str(asset.channels_available)
    for c, col in B.CH.items():
        f[f"has_{c}"] = float(col in chans)
    # Joice's rule verdict as an input
    if rule_override is not None:
        ft, cls, conf = rule_override
    else:
        try:
            cls, ft, conf, _, _ = diag2.diagnose2(g, j, k, asset.control)
        except Exception:
            cls, ft, conf = "MACHINE", "UNKNOWN", 0.0
    f["rule_type"], f["rule_class"], f["rule_conf"] = ft, cls, float(conf)
    return f


def label_at(t, ev):
    """True label at time t: the active observable event with the latest onset, else NO_FAULT."""
    act = ev[(ev.onset_utc <= t) & (t <= ev.end_utc)]
    if len(act) == 0:
        return "NO_FAULT", "NONE", None
    e = act.sort_values("onset_utc").iloc[-1]
    return e.fault_type, e.event_class, e.event_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="build/generated")
    p.add_argument("--pipeline", default="build/pipeline")
    p.add_argument("--out", default="build/training")
    p.add_argument("--seed", type=int, default=0, help="random seed for picking normal moments")
    a = p.parse_args()
    data, pipe, out = Path(a.data), Path(a.pipeline), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    roles = {}
    if (data / "assets_all.csv").exists():
        aa = pd.read_csv(data / "assets_all.csv")
        roles = {Path(d).name: r for d, r in zip(aa.data_dir, aa.role)}

    rows = []
    for fp in sorted((pipe / "features").glob("*.parquet")):
        name = fp.stem
        d = data / name
        if not d.exists():
            print(f"skip {name}: no data folder"); continue
        role = roles.get(name, "unknown")
        prefix = f"{name.split('_')[0][0].upper()}{name.split('_')[1]}"
        zp = pd.read_parquet(fp); zp["window_start_utc"] = ts(zp["window_start_utc"])
        assets = pd.read_csv(d / "public" / "assets.csv").set_index("asset_id")
        if (d / "private" / "events_stress_all.csv").exists():
            ev = pd.read_csv(d / "private" / "events_stress_all.csv")
            bnf = d / "public" / "benign_windows_stress.csv"
        else:
            ev = pd.concat([pd.read_csv(d / "public" / f) for f in ("events_train.csv", "events_validation.csv")
                            if (d / "public" / f).exists()], ignore_index=True)
            bnf = d / "public" / "benign_windows_train_validation.csv"
        for c in ("onset_utc", "detectable_utc", "end_utc"):
            ev[c] = ts(ev[c])
        ev["end_utc"] = ev["end_utc"].fillna(pd.Timestamp("2100-01-01", tz="UTC"))   # empty end = still ongoing
        ev["observable"] = ev["observable"].astype(str).str.lower().isin(["true", "1"])
        bn = pd.read_csv(bnf) if bnf.exists() else pd.DataFrame(columns=["asset_id", "start_utc", "end_utc"])
        alerts = pd.read_csv(pipe / "alerts" / f"{name}_JOICE.csv")

        for aid, g in zp.groupby("asset_id", sort=False):
            g = g.sort_values("window_start_utc").reset_index(drop=True)
            asset = assets.loc[aid]
            tend = g["window_start_utc"] + pd.Timedelta(minutes=1)     # a row is fully known at its window end
            t_first, t_last = tend.iloc[0], tend.iloc[-1]
            e_all = ev[ev.asset_id == aid]
            e_obs = e_all[e_all.observable]
            samples = []                                                # (time, kind)
            for _, e in e_obs.iterrows():
                start = e.detectable_utc if pd.notna(e.detectable_utc) else e.onset_utc
                base_end = t_first + pd.Timedelta(hours=24)
                # pre-existing faults start inside the 24 h baseline: also sample after the baseline
                starts = [start] if start >= base_end else [start, base_end]
                ev_times = sorted({s + pd.Timedelta(hours=h) for s in starts for h in EVENT_OFFSETS_H})
                for t in ev_times:
                    if t <= e.end_utc and base_end <= t <= t_last:
                        samples.append((t, "EVENT", None))
            aa = alerts[alerts.asset_id == aid]
            for t, ft_, cl_, cf_ in zip(ts(aa["alert_time_utc"]), aa["fault_type"], aa["event_class"], aa["confidence"]):
                samples.append((t, "ALERT", (ft_, cl_, float(cf_))))
            # healthy moments: after the 24 h baseline, outside every event (observable or not)
            cand = tend[(tend >= t_first + pd.Timedelta(hours=24))]
            busy = np.zeros(len(cand), bool)
            for _, e in e_all.iterrows():
                busy |= ((cand >= e.onset_utc - pd.Timedelta(hours=1)) & (cand <= e.end_utc + pd.Timedelta(hours=1))).values
            free = cand[~busy]
            if len(free):
                for t in rng.choice(free.values, min(N_NORMAL, len(free)), replace=False):
                    samples.append((pd.Timestamp(t).tz_localize("UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t), "NORMAL", None))
            b = bn[bn.asset_id == aid]
            if len(b):
                mids = ts(b.start_utc) + (ts(b.end_utc) - ts(b.start_utc)) / 2
                mids = mids[(mids >= t_first + pd.Timedelta(hours=24)) & (mids <= t_last)]
                for t in rng.permutation(mids.values)[:N_BENIGN]:
                    samples.append((pd.Timestamp(t).tz_localize("UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t), "BENIGN", None))

            for t, kind, rov in samples:
                j = int(tend.searchsorted(t, side="right") - 1)      # last row fully known at t
                if j < 0:
                    continue
                ftype, fclass, eid = label_at(t, e_obs)
                if kind in ("NORMAL", "BENIGN") and ftype != "NO_FAULT":
                    continue
                f = features_at(g, j, asset, rule_override=rov)
                rows.append(dict(seed_folder=name, role=role, asset_key=f"{prefix}-{aid}", asset_id=aid,
                                 tier=asset.instrumentation_tier, t=t, sample_kind=kind, event_id=eid,
                                 label_type=ftype, label_class=fclass, **f))
        print(f"{name}: {sum(r['seed_folder'] == name for r in rows)} rows", flush=True)

    T = pd.DataFrame(rows)
    T.to_parquet(out / "train_table.parquet", index=False)
    print(f"\nSaved {out / 'train_table.parquet'}: {len(T)} rows, "
          f"{T.shape[1] - 11} features, {T.asset_key.nunique()} pumps")
    print("\nRows per label and kind:")
    print(T.pivot_table(index="label_type", columns="sample_kind", values="t", aggfunc="count", fill_value=0).to_string())
    print("\nRows per role:", T.role.value_counts().to_dict())
    ok = T[T.label_type != "NO_FAULT"]
    print(f"Joice's rule already correct on fault rows: {(ok.rule_type == ok.label_type).mean():.2f}")


if __name__ == "__main__":
    main()
