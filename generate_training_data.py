"""Generate simulated training data for the RETROFIT AI diagnostician.

Run from the RETROFIT_AI_PARTICIPANTS_2026 folder:
    python3 generate_training_data.py                  # full plan (~85 seeds)
    python3 generate_training_data.py --quick          # 3 stress + 1 full seed, to test the pipeline

What it makes (defaults):
  stress seeds 100-159 -> role "train"   (new random pumps, 10 common fault types)
  stress seeds 160-174 -> role "test"    (held out: never train on these)
  full   seeds 300-309 -> role "extra"   (the 9 public pumps re-simulated: adds CAVITATION,
                                          IMPELLER_WEAR, AIR_ENTRAINMENT, which stress mode never makes.
                                          Near-copies of PUMP-01..09, so don't evaluate on 01..09
                                          with a model trained on these.)

Output: build/generated/
  stress_<seed>/ , full_<seed>/      raw simulator output (public/ = model inputs, private/ = answer key)
  events_all.csv                     every fault event, all seeds, with seed / role / unique asset key
  assets_all.csv                     every generated pump, with seed / role / unique asset key
  generation_log.csv                 per-seed status and runtime

Already finished seeds are skipped, so you can stop and re-run safely.
Use private/ files only as labels or for checking, never as model inputs.
"""
import argparse, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def seed_range(text):
    """'100-159' or '100,105,110' -> list of ints"""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_one(job):
    tier, seed, role, out_root, days = job
    final = Path(out_root) / f"{tier}_{seed}"
    if (final / "DONE").exists():
        return dict(tier=tier, seed=seed, role=role, status="skipped", seconds=0.0)
    tmp = Path(out_root) / f".tmp_{tier}_{seed}"
    shutil.rmtree(tmp, ignore_errors=True)
    cmd = [sys.executable, "make_dataset.py", "--tier", tier, "--seed", str(seed), "--output", str(tmp)]
    if tier == "stress" and days:
        cmd += ["--days", str(days)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    sec = round(time.time() - t0, 1)
    if r.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return dict(tier=tier, seed=seed, role=role, status="FAILED", seconds=sec,
                    error=(r.stderr or r.stdout)[-500:])
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    (final / "DONE").write_text(f"{tier} seed {seed} role {role}\n")
    return dict(tier=tier, seed=seed, role=role, status="ok", seconds=sec)


def load_seed(out_root, tier, seed, role):
    """Return (events, assets) for one finished seed, with a globally unique asset key."""
    d = Path(out_root) / f"{tier}_{seed}"
    pub, priv = d / "public", d / "private"
    if tier == "stress":
        ev = pd.read_csv(priv / "events_stress_all.csv")          # ALL events (public file has only ~30 %)
    else:
        ev = pd.concat([pd.read_csv(pub / f) for f in ("events_train.csv", "events_validation.csv")
                        if (pub / f).exists()], ignore_index=True)
    assets = pd.read_csv(pub / "assets.csv")
    prefix = f"{tier[0].upper()}{seed}"                            # e.g. S100, F300
    for df in (ev, assets):
        df.insert(0, "asset_key", prefix + "-" + df["asset_id"].astype(str))
        df.insert(1, "seed", seed); df.insert(2, "tier", tier); df.insert(3, "role", role)
        df.insert(4, "data_dir", str(d))
    return ev, assets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stress-train", default="100-159")
    p.add_argument("--stress-test", default="160-174")
    p.add_argument("--full-extra", default="300-309")
    p.add_argument("--days", type=float, default=None, help="days per stress pump (simulator default 3; 7 = closer to the official data, ~2x time)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="build/generated")
    p.add_argument("--quick", action="store_true", help="tiny run to test the pipeline")
    a = p.parse_args()

    if not Path("make_dataset.py").exists():
        sys.exit("Run this from the RETROFIT_AI_PARTICIPANTS_2026 folder (make_dataset.py not found).")
    if a.quick:
        a.stress_train, a.stress_test, a.full_extra = "100-101", "160", "300"

    jobs = ([("stress", s, "train", a.out, a.days) for s in seed_range(a.stress_train)] +
            [("stress", s, "test", a.out, a.days) for s in seed_range(a.stress_test)] +
            [("full", s, "extra", a.out, None) for s in seed_range(a.full_extra)])
    Path(a.out).mkdir(parents=True, exist_ok=True)

    # rough disk check: ~25 MB per stress seed (3 days), ~110 MB per full seed with snapshots
    days_factor = (a.days or 3) / 3
    need_gb = (sum(j[0] == "stress" for j in jobs) * 25 * days_factor + sum(j[0] == "full" for j in jobs) * 110) / 1024
    free_gb = shutil.disk_usage(a.out).free / 1024 ** 3
    print(f"{len(jobs)} seeds, {a.workers} parallel workers, about {need_gb:.1f} GB needed, {free_gb:.1f} GB free")
    if free_gb < need_gb * 1.2:
        sys.exit("Not enough free disk space.")

    log, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); log.append(r)
            msg = f"[{i}/{len(jobs)}] {r['tier']:6s} seed {r['seed']:4d} ({r['role']}): {r['status']} {r['seconds']}s"
            if r["status"] == "FAILED":
                msg += "\n    " + r.get("error", "").replace("\n", "\n    ")
            print(msg, flush=True)
    pd.DataFrame(log).sort_values(["tier", "seed"]).to_csv(Path(a.out) / "generation_log.csv", index=False)

    # combined index of all finished seeds
    evs, ass = [], []
    for tier, seed, role, *_ in jobs:
        if (Path(a.out) / f"{tier}_{seed}" / "DONE").exists():
            e, s = load_seed(a.out, tier, seed, role); evs.append(e); ass.append(s)
    if not evs:
        sys.exit("No finished seeds.")
    events, assets = pd.concat(evs, ignore_index=True), pd.concat(ass, ignore_index=True)
    events.to_csv(Path(a.out) / "events_all.csv", index=False)
    assets.to_csv(Path(a.out) / "assets_all.csv", index=False)

    failed = sum(r["status"] == "FAILED" for r in log)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. {failed} failed.")
    print(f"{len(assets)} pumps, {len(events)} events ({int(events['observable'].sum())} observable)\n")
    obs = events[events["observable"].astype(bool)]
    table = obs.pivot_table(index=["event_class", "fault_type"], columns="role", values="event_id",
                            aggfunc="count", fill_value=0)
    print("Observable events per fault type and role:")
    print(table.to_string())
    print("\nPumps per instrumentation tier and role:")
    print(assets.pivot_table(index="instrumentation_tier", columns="role", values="asset_key",
                             aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()