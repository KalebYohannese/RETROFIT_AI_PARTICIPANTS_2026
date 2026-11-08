"""Train the ML diagnostician and compare it with Joice's rules.

Run from the RETROFIT_AI_PARTICIPANTS_2026 folder, after build_training_table.py:
    pip install lightgbm scikit-learn joblib
    python3 train_diagnostician.py

Data roles (set by generate_training_data.py):
  train + extra -> used for training (cross-validated by pump)
  test          -> held-out seeds, only used for the final evaluation

Reports three diagnosticians on the same rows:
  RULE   : Joice's rule verdict
  MODEL  : LightGBM
  HYBRID : MODEL when its confidence >= --min-conf (NO_FAULT only above 0.8), otherwise RULE
Outputs in build/models/: diagnostician.joblib, cv_predictions.csv, test_predictions.csv, feature_importance.csv
"""
import argparse, json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("Install LightGBM first:  pip install lightgbm")

META = ["seed_folder", "role", "asset_key", "asset_id", "tier", "t", "sample_kind", "event_id",
        "label_type", "label_class"]
SENSOR_TYPES = {"SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_CALIBRATION_SHIFT", "SENSOR_NOISE_SPIKES", "COMMUNICATION_LOSS"}
CLASS_OF = lambda ft: "NONE" if ft == "NO_FAULT" else ("SENSOR" if ft in SENSOR_TYPES else "MACHINE")


def make_X(df, columns=None):
    """numeric features + one-hot of the rule verdict; align to training columns if given"""
    X = df.drop(columns=[c for c in META if c in df] + ["rule_type", "rule_class"])
    X = pd.concat([X, pd.get_dummies(df["rule_type"], prefix="rule").astype(float)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def new_model(seed=0):
    return lgb.LGBMClassifier(
        objective="multiclass", n_estimators=400, learning_rate=0.04, num_leaves=15,
        min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_lambda=1.0, class_weight="balanced", random_state=seed, verbose=-1)


def predict(model, X, classes):
    """probabilities as a DataFrame over ALL classes (a fold model may not have seen every class)"""
    P = pd.DataFrame(0.0, index=X.index, columns=classes)
    P[list(model.classes_)] = model.predict_proba(X)
    return P


def evaluate(df, pred_col, title):
    """accuracy of fault type on fault rows, class accuracy, false-alarm handling, per type and per tier"""
    f = df[df.label_type != "NO_FAULT"]
    n = df[df.label_type == "NO_FAULT"]
    out = {"rows": len(df),
           "type_acc_on_faults": (f[pred_col] == f.label_type).mean(),
           "class_acc_on_faults": (f[pred_col].map(CLASS_OF) == f.label_class).mean(),
           "no_fault_recognised": (n[pred_col] == "NO_FAULT").mean() if len(n) else np.nan,
           "faults_called_no_fault": (f[pred_col] == "NO_FAULT").mean()}
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def compare(df, title, min_conf):
    df = df.copy()
    # the model may overrule the rule above min_conf; it may only suppress an alert (NO_FAULT) when very sure
    use_model = (df["MODEL_conf"] >= min_conf) & ((df["MODEL"] != "NO_FAULT") | (df["MODEL_conf"] >= 0.8))
    df["HYBRID"] = np.where(use_model, df["MODEL"], df["RULE"])
    print(f"\n=== {title} ({len(df)} rows, {df.asset_key.nunique()} pumps) ===")
    summ = pd.DataFrame({s: evaluate(df, s, title) for s in ("RULE", "MODEL", "HYBRID")}).T
    print(summ.to_string())
    f = df[df.label_type != "NO_FAULT"].assign(
        RULE_ok=lambda d: d.RULE == d.label_type, MODEL_ok=lambda d: d.MODEL == d.label_type,
        HYBRID_ok=lambda d: d.HYBRID == d.label_type)
    agg = dict(rows=("label_type", "size"), RULE=("RULE_ok", "mean"), MODEL=("MODEL_ok", "mean"), HYBRID=("HYBRID_ok", "mean"))
    print("\nFault-type accuracy per true type:")
    print(f.groupby("label_type").agg(**agg).round(2).to_string())
    print("\nFault-type accuracy per instrumentation tier:")
    print(f.groupby("tier").agg(**agg).round(2).to_string())
    # is the confidence honest? accuracy per confidence band
    d = df.assign(band=pd.cut(df["MODEL_conf"], [0, 0.4, 0.6, 0.8, 1.0], include_lowest=True),
                  ok=df.MODEL == df.label_type)
    cal = d.groupby("band", observed=True).agg(rows=("ok", "size"), mean_conf=("MODEL_conf", "mean"), actually_correct=("ok", "mean"))
    print("\nConfidence check (MODEL): a well-calibrated model has mean_conf close to actually_correct")
    print(cal.round(2).to_string())
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="build/training/train_table.parquet")
    p.add_argument("--out", default="build/models")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-conf", type=float, default=0.5, help="HYBRID uses the model only above this confidence")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    T = pd.read_parquet(a.table)
    T["rule_type"] = T["rule_type"].fillna("UNKNOWN")
    T["RULE"] = T["rule_type"]          # the rule always names a fault (it assumes the detector fired)
    train = T[T.role.isin(["train", "extra"])].reset_index(drop=True)
    test = T[T.role == "test"].reset_index(drop=True)
    if len(train) == 0:
        raise SystemExit("No train/extra rows in the table.")
    classes = sorted(T.label_type.unique())
    X_tr = make_X(train)
    cols = list(X_tr.columns)
    y_tr = train.label_type.values
    print(f"Training rows {len(train)} from {train.asset_key.nunique()} pumps; "
          f"test rows {len(test)} from {test.asset_key.nunique()} pumps; {len(cols)} features; {len(classes)} classes")

    # ---- grouped cross-validation on the training pumps (never test on a pump you trained on)
    # copies of the same public pump from different full-tier seeds are near-duplicates -> one group
    groups = np.where(train.seed_folder.str.startswith("full"), "FULL-" + train.asset_id, train.asset_key)
    k = min(a.folds, len(set(groups)))
    cv = train.copy(); cv["MODEL"] = ""; cv["MODEL_conf"] = 0.0
    for fold, (i_tr, i_va) in enumerate(GroupKFold(n_splits=k).split(X_tr, y_tr, groups=groups)):
        m = new_model(fold).fit(X_tr.iloc[i_tr], y_tr[i_tr])
        P = predict(m, X_tr.iloc[i_va], classes)
        cv.loc[i_va, "MODEL"] = P.idxmax(axis=1).values
        cv.loc[i_va, "MODEL_conf"] = P.max(axis=1).values
    cv.to_csv(out / "cv_predictions.csv", index=False)
    summ_cv = compare(cv, f"Cross-validation on training pumps ({k} folds, grouped by pump)", a.min_conf)

    # ---- final model on all training rows
    model = new_model(0).fit(X_tr, y_tr)
    imp = pd.DataFrame({"feature": cols, "importance": model.booster_.feature_importance("gain")})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    print("\nTop 15 features (gain):")
    print(imp.head(15).to_string(index=False))
    joblib.dump({"model": model, "columns": cols, "classes": classes, "min_conf": a.min_conf}, out / "diagnostician.joblib")

    # ---- held-out test seeds
    if len(test):
        P = predict(model, make_X(test, cols), classes)
        te = test.copy(); te["MODEL"] = P.idxmax(axis=1).values; te["MODEL_conf"] = P.max(axis=1).values
        te.to_csv(out / "test_predictions.csv", index=False)
        summ_te = compare(te, "HELD-OUT TEST SEEDS (the honest number)", a.min_conf)
        json.dump({"cv": summ_cv.to_dict(), "test": summ_te.to_dict()}, open(out / "summary.json", "w"), indent=2, default=str)
    print(f"\nSaved model and reports to {out}")


if __name__ == "__main__":
    main()