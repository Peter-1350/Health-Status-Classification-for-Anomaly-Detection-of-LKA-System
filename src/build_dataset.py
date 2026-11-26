"""Step 1-3: raw CAN CSVs -> cached frame-level feature tables + feature selection.

A "frame" is one can_XXXX.csv (~300 rows, 3 s sliding window @100 Hz).
Label (stage-1 decision, LABEL_MODE='record'): mean LKA_error over the whole
record -> threshold 0.25; every frame of that record inherits the label.

Outputs (data/processed/):
  frames_train.parquet, frames_test.parquet  one row per frame
  feature_meta.json                          selection rules, columns, stats
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CORR_MAX = 0.95
NEAR_CONST_FRAC = 0.99
MAX_CATEGORIES = 10
ARRAY_ROW_STATS = ("mean", "std", "min", "max", "first", "last")


def schema_from_reference():
    """Column groups are taken from one full TRAIN file; the Test split has 2 extra columns."""
    ref = sorted(glob.glob(os.path.join(C.RAW, "Train", "record*", "can_*.csv")))
    for path in ref:
        df = pd.read_csv(path, low_memory=False)
        if not len(df):
            continue
        df = df.drop(columns=[c for c in ("Time.1",) if c in df.columns])
        num = [c for c in df.select_dtypes("number").columns if c not in C.TIME_COLS + C.LEAK_COLS]
        boolean = list(df.select_dtypes("bool").columns)
        text = [c for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)]
        arrays = [c for c in text if str(df[c].iloc[0]).lstrip().startswith("[")]
        strings = [c for c in text if c not in arrays and c not in C.TIME_COLS]
        return num, boolean, strings, arrays
    raise RuntimeError("no non-empty reference CSV found")


def array_features(s: pd.Series) -> dict:
    """Per-row trajectory stats (mean/std/min/max/first/last), then aggregated over the window."""
    vecs = []
    for raw in s:
        try:
            v = np.fromstring(str(raw).strip().strip("[]"), sep=",")
        except ValueError:
            continue
        if v.size and np.isfinite(v).any():
            vecs.append(v)
    if len(vecs) < 2:
        vecs = vecs * 2 if vecs else [np.zeros(1), np.zeros(1)]
    M = np.vstack(vecs)
    rows = {
        "mean": M.mean(axis=1),
        "std": M.std(axis=1),
        "min": M.min(axis=1),
        "max": M.max(axis=1),
        "first": M[:, 0],
        "last": M[:, -1],
    }
    feats = {}
    for name, arr in rows.items():
        feats[f"__row{name}__mean"] = float(np.mean(arr))
        feats[f"__row{name}__std"] = float(np.std(arr))
    return feats


def frame_row(path, num, boolean, strings, arrays):
    df = pd.read_csv(path, low_memory=False)
    if not len(df):  # header-only file: record has no usable CAN data
        return None
    df = df.drop(columns=[c for c in ("Time.1",) if c in df.columns])

    row = {}
    if num:
        X = df[[c for c in num if c in df.columns]].astype("float64")
        d = X.diff()
        stats = {
            "mean": X.mean(),
            "std": X.std(ddof=0),
            "min": X.min(),
            "max": X.max(),
            "range": X.max() - X.min(),
            "last_first": X.iloc[-1] - X.iloc[0],
            "meanabsdiff": d.abs().mean(),
            "stddiff": d.std(ddof=0),
        }
        for s, ser in stats.items():
            row.update({f"{c}__{s}": v for c, v in ser.items()})
    for c in boolean:
        if c in df.columns:
            row[f"{c}__frac"] = float(df[c].mean())
    for c in strings:
        if c in df.columns:
            vc = df[c].value_counts()
            row[c] = vc.index[0] if len(vc) else "unknown"
    for c in arrays:
        if c in df.columns:
            row.update({f"{c}{k}": v for k, v in array_features(df[c]).items()})

    row["n_rows"] = int(len(df))
    row["duration"] = float(df["Time"].max() - df["Time"].min()) if "Time" in df else np.nan
    lka = float(df["LKA_error"].mean())  # this frame's own mean LKA error
    tl = df[["Time", "LKA_error"]].to_numpy(dtype="float64") if "Time" in df else np.empty((0, 2))
    return row, lka, tl


def build_split(split, schema):
    num, boolean, strings, arrays = schema
    rows, meta = [], []
    for rec_dir in sorted(glob.glob(os.path.join(C.RAW, split, "record*")),
                          key=lambda p: int(os.path.basename(p).replace("record", ""))):
        record_id = os.path.basename(rec_dir)
        frames, lka_means, times = [], [], []
        for i, path in enumerate(sorted(glob.glob(os.path.join(rec_dir, "can_*.csv")))):
            out = frame_row(path, num, boolean, strings, arrays)
            if out is None:
                continue
            row, lka, t = out
            frames.append((i, row))
            lka_means.append(lka)
            times.append(t)
        if not frames:  # record100 / record177: nothing to learn or predict from
            meta.append({"record_id": record_id, "n_frames": 0, "label": None})
            continue
        rec_mean = float(np.mean(lka_means))  # equal weight per frame, per Proposal §1
        tl = np.vstack(times) if times else np.empty((0, 2))
        _, first = np.unique(tl[:, 0], return_index=True)  # overlapping windows -> keep first sample
        pooled = float(tl[first, 1].mean()) if len(first) else float("nan")
        y = int(rec_mean >= C.LKA_THRESH)
        for i, row in frames:
            rows.append({"split": split, "record_id": record_id, "frame_idx": i,
                         "y": y, "lka_record_mean": rec_mean, "lka_record_pooled": pooled, **row})
        meta.append({"record_id": record_id, "n_frames": len(frames),
                     "label": y, "lka_mean": rec_mean, "lka_pooled": pooled})
    df = pd.DataFrame(rows)
    print(f"  {split}: {len(meta)} records, {len(df)} frames, "
          f"{int(df['y'].sum()) if len(df) else 0} anomaly frames, "
          f"{sum(1 for m in meta if m['n_frames'] == 0)} empty records")
    return df, meta


def encode_categories(train, test, strings):
    """One-hot the mode-of-window categorical columns; categories are learned on TRAIN only."""
    cats = {}
    for c in strings:
        keep = train[c].value_counts().index[:MAX_CATEGORIES].tolist() if c in train.columns else []
        cats[c] = keep
        for df in (train, test):
            key = (df[c].where(df[c].isin(keep), "other") if c in df.columns
                   else pd.Series("other", index=df.index))
            for cat in keep + ["other"]:
                df[f"{c}_{cat}"] = (key == cat).astype("int8")
            df.drop(columns=[c], inplace=True, errors="ignore")
    return cats


def select_features(train, feats):
    """Drops are fitted on TRAIN only and then applied to both splits."""
    n = len(train)
    dropped = {"all_nan": [], "constant": [], "near_constant": [], "collinear": []}

    cand = []
    for c in feats:
        col = train[c]
        if col.notna().sum() == 0:
            dropped["all_nan"].append(c)
        elif col.std(ddof=0) == 0 or col.nunique(dropna=True) <= 1:
            dropped["constant"].append(c)
        elif col.value_counts(dropna=True).iloc[0] / n > NEAR_CONST_FRAC:
            dropped["near_constant"].append(c)
        else:
            cand.append(c)

    Z = train[cand].fillna(train[cand].median())
    corr = Z.corr().abs()
    order = Z.std(ddof=0).sort_values(ascending=False).index
    keep = []
    for c in order:
        if all(corr.at[c, k] <= CORR_MAX for k in keep):
            keep.append(c)
        else:
            dropped["collinear"].append(c)
    return keep, dropped


def main(force=False):
    C.ensure_dirs()
    train_path = os.path.join(C.PROC, "frames_train.parquet")
    test_path = os.path.join(C.PROC, "frames_test.parquet")
    if os.path.exists(train_path) and os.path.exists(test_path) and not force:
        print("cache exists -> skip (use --force to rebuild)")
        return

    t0 = time.time()
    schema = schema_from_reference()
    print("schema:", {k: len(v) for k, v in zip(("num", "bool", "str", "array"), schema)})
    train, train_meta = build_split("Train", schema)
    test, test_meta = build_split("Test", schema)

    cats = encode_categories(train, test, schema[2])
    feats = [c for c in train.columns if c not in C.TF_COLS]
    keep, dropped = select_features(train, feats)

    train["split"], test["split"] = "Train", "Test"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    meta = {
        "label_mode": C.LABEL_MODE,
        "lka_thresh": C.LKA_THRESH,
        "n_splits": C.N_SPLITS,
        "raw_columns": {k: v for k, v in zip(("num", "bool", "str", "array"), schema)},
        "leak_cols_removed": C.LEAK_COLS,
        "time_cols_removed": C.TIME_COLS,
        "categorical_columns": cats,
        "n_features_built": len(feats),
        "n_features_selected": len(keep),
        "selected_features": keep,
        "dropped": {k: {"n": len(v), "cols": v} for k, v in dropped.items()},
        "train": {"records": len(train_meta),
                  "records_used": sum(1 for m in train_meta if m["n_frames"]),
                  "frames": len(train), "anomaly_frames": int(train["y"].sum()),
                  "empty_records": [m["record_id"] for m in train_meta if not m["n_frames"]]},
        "test": {"records": len(test_meta),
                 "records_used": sum(1 for m in test_meta if m["n_frames"]),
                 "frames": len(test), "anomaly_frames": int(test["y"].sum()),
                 "empty_records": [m["record_id"] for m in test_meta if not m["n_frames"]]},
        "records": {"Train": train_meta, "Test": test_meta},
        "build_seconds": round(time.time() - t0, 1),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    C.dump_json(meta, os.path.join(C.PROC, "feature_meta.json"))

    both = train[["record_id", "lka_record_mean", "lka_record_pooled"]].drop_duplicates()
    flips = int(((both["lka_record_mean"] >= C.LKA_THRESH) != (both["lka_record_pooled"] >= C.LKA_THRESH)).sum())
    print(f"\nfeatures built {len(feats)} -> selected {len(keep)}")
    for k, v in dropped.items():
        print(f"  dropped {k}: {len(v)}")
    print(f"frame label vs pooled-row label mismatches on records: {flips}")
    print(f"saved -> {C.PROC}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
