"""Step 4: GroupKFold(5) by record -> OOF threshold -> refit on all train, save models."""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    C.ensure_dirs()
    meta = C.load_meta()
    feats = meta["selected_features"]
    train = C.load_frames("Train")
    X = train[feats].to_numpy(dtype="float32")
    y = train["y"].to_numpy()
    groups = train["record_id"].to_numpy()
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"train {X.shape[0]} frames x {X.shape[1]} features, "
          f"{len(np.unique(groups))} records, anomaly frames {y.sum()}")

    models = C.make_models(pos_weight)
    cv = GroupKFold(n_splits=C.N_SPLITS)
    summary, fold_rows, thresholds, f1_thresholds, oof_store = [], [], {}, {}, {}

    for name, pipe in models.items():
        oof = np.zeros(len(y))
        for k, (tr, va) in enumerate(cv.split(X, y, groups)):
            pipe.fit(X[tr], y[tr])
            oof[va] = pipe.predict_proba(X[va])[:, 1]
            p_rec = C.record_prob(train.iloc[va], oof[va])
            m = C.metrics(C.record_truth(train.iloc[va]).to_numpy(), p_rec.to_numpy(), 0.5)
            fold_rows.append({"model": name, "fold": k + 1, **{f"rec_{k2}": v for k2, v in m.items()
                                                              if k2 in ("accuracy", "macro_f1", "anomaly_recall")}})
        oof_store[name] = oof
        p_rec = C.record_prob(train, oof)
        y_rec = C.record_truth(train)
        t = C.pick_threshold(y_rec.to_numpy(), p_rec.to_numpy())
        t_f1 = C.pick_threshold(y_rec.to_numpy(), p_rec.to_numpy(), target=None)
        thresholds[name] = t
        f1_thresholds[name] = t_f1
        rec_m = C.metrics(y_rec.to_numpy(), p_rec.to_numpy(), t)
        frame_m = C.metrics(y, oof, t)
        folds = [r for r in fold_rows if r["model"] == name]
        summary.append({
            "model": name, "threshold": t,
            **{f"record_{k}": v for k, v in rec_m.items()},
            "frame_pr_auc": frame_m["pr_auc"],
            "frame_macro_f1": frame_m["macro_f1"],
            "frame_accuracy": frame_m["accuracy"],
            "cv_record_macro_f1_mean": float(np.mean([r["rec_macro_f1"] for r in folds])),
            "cv_record_macro_f1_std": float(np.std([r["rec_macro_f1"] for r in folds])),
            "cv_record_recall_mean": float(np.mean([r["rec_anomaly_recall"] for r in folds])),
        })
        print(f"  {name:7s} record macro-F1(OOF)={rec_m['macro_f1']:.3f} "
              f"recall={rec_m['anomaly_recall']:.3f} pr_auc={rec_m['pr_auc']:.3f} "
              f"t={t:.2f} frame_pr_auc={frame_m['pr_auc']:.3f}")

    df_sum = pd.DataFrame(summary).sort_values("record_pr_auc", ascending=False)
    df_sum.round(4).to_csv(os.path.join(C.RESULTS, "cv_summary.csv"), index=False)
    pd.DataFrame(fold_rows).round(4).to_csv(os.path.join(C.RESULTS, "cv_folds.csv"), index=False)
    C.dump_json({k: v.tolist() for k, v in oof_store.items()}, os.path.join(C.RESULTS, "oof_train_probs.json"))

    best = df_sum.iloc[0]["model"]
    C.dump_json({"best": best, "thresholds": thresholds, "f1_thresholds": f1_thresholds,
                 "features": feats, "n_features": len(feats), "pos_weight": pos_weight},
                os.path.join(C.MODELS, "selection.json"))

    final = C.make_models(pos_weight)
    for name, pipe in final.items():
        pipe.fit(X, y)
        joblib.dump(pipe, os.path.join(C.MODELS, f"{name}.joblib"))
    print(f"\nbest model by OOF record PR-AUC: {best}")
    print(f"models -> {C.MODELS}\ncv tables -> {C.RESULTS}")


if __name__ == "__main__":
    main()
