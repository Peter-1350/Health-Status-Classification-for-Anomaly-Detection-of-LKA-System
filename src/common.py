"""Shared paths, constants and metrics for the LKA stage-1 CAN-only pipeline."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "TESLA_AP3_MODEL3_v1")
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")

SEED = 42
LKA_THRESH = 0.25
LABEL_MODE = "record"  # stage-1 decision: record label, every frame inherits it
N_SPLITS = 5
RECALL_TARGET = 0.90

# LKA_error defines the label; the two terms below reconstruct it exactly -> leakage.
LEAK_COLS = ["LKA_error", "op_left_laneline", "op_right_laneline"]
# Absolute time / duplicated column: no health information, dropped from features.
TIME_COLS = ["Time", "Time.1", "unix_time", "can_mono_time", "wallTimeCentiseconds", "easternTime"]

TF_COLS = ["record_id", "split", "frame_idx", "n_rows", "y", "lka_record_mean", "lka_record_pooled"]


def ensure_dirs():
    for d in (PROC, MODELS, RESULTS, os.path.join(RESULTS, "figures")):
        os.makedirs(d, exist_ok=True)


def load_meta():
    return load_json(os.path.join(PROC, "feature_meta.json"))


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_frames(split):
    return pd.read_parquet(os.path.join(PROC, f"frames_{split}.parquet"))


def make_models(pos_weight=1.0):
    """One fixed configuration per algorithm: stage 1 does no hyperparameter search."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    imp = lambda: SimpleImputer(strategy="median")
    return {
        "logreg": Pipeline([
            ("imp", imp()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]),
        "tree": Pipeline([
            ("imp", imp()),
            ("clf", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=SEED)),
        ]),
        "rf": Pipeline([
            ("imp", imp()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        ]),
        "xgb": Pipeline([
            ("imp", imp()),
            ("clf", XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                n_jobs=-1, random_state=SEED)),
        ]),
    }


def record_prob(df, prob):
    """Record-level probability = mean of its frame probabilities."""
    return df.assign(p=prob).groupby("record_id")["p"].mean()


def record_truth(df):
    """Records with no usable CAN frame get no label (excluded from metrics)."""
    return df.groupby("record_id")["y"].first()


def metrics(y_true, p, thresh):
    """Classification metrics at a fixed operating threshold + threshold-free AUCs."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thresh).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "threshold": float(thresh),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "anomaly_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "anomaly_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "tn_fp_fn_tp": confusion_matrix(y_true, pred, labels=[0, 1]).ravel().tolist(),
    }
    single_class = y_true.min() == y_true.max()
    out["pr_auc"] = float("nan") if single_class else float(average_precision_score(y_true, p))
    out["roc_auc"] = float("nan") if single_class else float(roc_auc_score(y_true, p))
    return out


def pick_threshold(y_true, p, target=RECALL_TARGET):
    """Max macro-F1 subject to anomaly recall >= target; target=None -> plain argmax F1."""
    y_true = np.asarray(y_true).astype(int)
    grid = np.arange(0.01, 0.991, 0.01)
    cands = [(t,) + _f1_recall(y_true, p, t) for t in grid]
    if target is not None:
        cands = [c for c in cands if c[2] >= target] or cands
    return float(max(cands, key=lambda c: (c[1], c[0]))[0])


def _f1_recall(y_true, p, t):
    pred = (p >= t).astype(int)
    return (
        f1_score(y_true, pred, average="macro", zero_division=0),
        recall_score(y_true, pred, pos_label=1, zero_division=0),
    )
