"""Step 5: test-set prediction at record level + metrics + figures."""

import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    C.ensure_dirs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    selection = C.load_json(os.path.join(C.MODELS, "selection.json"))
    feats = selection["features"]
    test = C.load_frames("Test")
    X = test[feats].to_numpy(dtype="float32")
    y_rec = C.record_truth(test)

    # every officially listed test record, so empty ones stay visible instead of vanishing
    all_records = [os.path.basename(p) for p in sorted(
        glob.glob(os.path.join(C.RAW, "Test", "record*")),
        key=lambda p: int(os.path.basename(p).replace("record", "")))]

    preds = pd.DataFrame(index=pd.Index(all_records, name="record_id"))
    preds["n_frames"] = test.groupby("record_id").size().reindex(all_records).fillna(0).astype(int)
    preds["y_true"] = y_rec.reindex(all_records).astype("Int64")
    metrics_out = {}

    for name in selection["thresholds"]:
        pipe = joblib.load(os.path.join(C.MODELS, f"{name}.joblib"))
        p_frame = pipe.predict_proba(X)[:, 1]
        p_rec = C.record_prob(test, p_frame).reindex(all_records)
        t = selection["thresholds"][name]
        preds[f"p_record_{name}"] = p_rec
        preds[f"y_pred_{name}"] = (p_rec >= t).astype(float).where(p_rec.notna()).astype("Int64")
        ok = preds["y_true"].notna()
        metrics_out[name] = {
            "record": C.metrics(preds.loc[ok, "y_true"].to_numpy(), p_rec[ok].to_numpy(), t),
            "record_f1opt": C.metrics(preds.loc[ok, "y_true"].to_numpy(), p_rec[ok].to_numpy(),
                                       selection["f1_thresholds"][name]),
            "frame": C.metrics(test["y"].to_numpy(), p_frame, t),
        }

    best = selection["best"]
    preds["p_record"] = preds[f"p_record_{best}"]
    preds["y_pred"] = preds[f"y_pred_{best}"]
    order = ["n_frames", "y_true", "y_pred", "p_record"] + [c for name in selection["thresholds"]
                                                             for c in (f"p_record_{name}", f"y_pred_{name}")]
    preds = preds[order]
    preds.round(4).to_csv(os.path.join(C.RESULTS, "predictions_test.csv"))

    yt = preds.loc[preds["y_true"].notna(), "y_true"].to_numpy(dtype=int)
    pp = preds.loc[preds["y_true"].notna(), "p_record"].to_numpy()
    cm = C.metrics(yt, pp, selection["thresholds"][best])["tn_fp_fn_tp"]
    C.dump_json({"best_model": best, "thresholds": selection["thresholds"],
                 "n_test_records": len(all_records),
                 "n_test_records_scored": int(preds["y_true"].notna().sum()),
                 "no_data_records": preds.index[preds["y_true"].isna()].tolist(),
                 "metrics": metrics_out}, os.path.join(C.RESULTS, "test_metrics.json"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cmx = np.array(cm).reshape(2, 2)
    ax[0].imshow(cmx, cmap="Blues")
    ax[0].set_xticks([0, 1], ["pred Normal", "pred Anomaly"])
    ax[0].set_yticks([0, 1], ["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, cmx[i, j], ha="center", va="center",
                       color="white" if cmx[i, j] > cmx.max() / 2 else "black")
    ax[0].set_title(f"{best} record-level confusion matrix")
    for name in selection["thresholds"]:
        p_rec = preds[f"p_record_{name}"].to_numpy()
        ok = ~np.isnan(p_rec)
        if len(np.unique(yt)) > 1:
            pr, rc, _ = precision_recall_curve(preds.loc[ok, "y_true"].to_numpy(dtype=int), p_rec[ok])
            ax[1].plot(rc, pr, label=f"{name} (AP={metrics_out[name]['record']['pr_auc']:.3f})")
    ax[1].set_xlabel("anomaly recall")
    ax[1].set_ylabel("anomaly precision")
    ax[1].legend()
    ax[1].set_title("record-level PR curves (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(C.RESULTS, "figures", "test_eval.png"), dpi=150)
    plt.close(fig)

    rows = []
    for name, m in metrics_out.items():
        r = m["record"]
        rows.append({"model": name, "threshold": r["threshold"], "record_accuracy": r["accuracy"],
                     "record_macro_f1": r["macro_f1"], "record_anomaly_recall": r["anomaly_recall"],
                     "record_anomaly_precision": r["anomaly_precision"],
                     "record_pr_auc": r["pr_auc"], "record_roc_auc": r["roc_auc"],
                     "frame_pr_auc": m["frame"]["pr_auc"], "frame_macro_f1": m["frame"]["macro_f1"],
                     "f1opt_threshold": m["record_f1opt"]["threshold"],
                     "f1opt_record_accuracy": m["record_f1opt"]["accuracy"],
                     "f1opt_record_macro_f1": m["record_f1opt"]["macro_f1"],
                     "f1opt_record_anomaly_recall": m["record_f1opt"]["anomaly_recall"]})
    df = pd.DataFrame(rows).sort_values("record_macro_f1", ascending=False)
    df.round(4).to_csv(os.path.join(C.RESULTS, "test_summary.csv"), index=False)

    # top features of the best box/tree model
    pipe = joblib.load(os.path.join(C.MODELS, f"{best}.joblib"))
    clf = pipe.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None and hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    if imp is not None:
        pd.DataFrame({"feature": feats, "importance": imp}) \
            .sort_values("importance", ascending=False).head(20) \
            .to_csv(os.path.join(C.RESULTS, "feature_importance.csv"), index=False)

    print(df.to_string(index=False))
    print(f"\nscored {int(preds['y_true'].notna().sum())}/{len(all_records)} test records "
          f"(no CAN data: {preds.index[preds['y_true'].isna()].tolist()})")
    print(f"predictions -> {os.path.join(C.RESULTS, 'predictions_test.csv')}")


if __name__ == "__main__":
    main()
