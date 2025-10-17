import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

# Allow running as a script without package install
try:
    from .data import load_dataset, split_xy
except Exception:
    import sys as _sys, os as _os
    _SRC_DIR = _os.path.dirname(os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_SRC_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from src.data import load_dataset, split_xy


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(
    tier1_dir: str = "outputs/1_tier1",
    out_dir: str = "outputs/4_report",
    data_path: str = "UCI_Credit_Card.csv",
):
    ensure_dir(out_dir)

    # Load OOF predictions and metrics
    with open(os.path.join(tier1_dir, "oof_metrics.json"), "r", encoding="utf-8") as f:
        m = json.load(f)
    oof = pd.read_csv(os.path.join(tier1_dir, "oof_preds.csv"))["oof"].to_numpy()

    # Load ground truth for visualization
    df = load_dataset(data_path)
    _, y_s = split_xy(df)
    y_true = y_s.to_numpy()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, oof)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OOF)")
    plt.legend(loc="lower right")
    _savefig(os.path.join(out_dir, "roc.png"))

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, oof)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (OOF)")
    plt.legend(loc="lower left")
    _savefig(os.path.join(out_dir, "pr.png"))

    # Calibration (reliability) curve
    frac_pos, mean_pred = calibration_curve(y_true, oof, n_bins=10, strategy="uniform")
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (OOF)")
    _savefig(os.path.join(out_dir, "calibration.png"))

    # Probability histogram per class
    plt.figure(figsize=(5, 4))
    plt.hist(oof[y_true == 0], bins=30, alpha=0.6, label="y=0")
    plt.hist(oof[y_true == 1], bins=30, alpha=0.6, label="y=1")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("OOF Probability Histogram")
    plt.legend()
    _savefig(os.path.join(out_dir, "prob_hist.png"))

    # Confusion matrix at best F1 threshold
    thr = float(m.get("threshold", 0.5))
    y_pred = (oof >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix @ thr={thr:.2f}")
    _savefig(os.path.join(out_dir, "confusion_matrix.png"))

    # Error CSV (sorted by absolute probability error)
    out_err = pd.DataFrame({
        "ID": df["ID"] if "ID" in df.columns else np.arange(len(df)),
        "y_true": y_true,
        "prob": oof,
        "pred": y_pred,
    })
    out_err["error"] = (out_err["y_true"] != out_err["pred"]).astype(int)
    out_err["abs_error"] = np.abs(out_err["y_true"] - out_err["prob"])
    out_err.sort_values("abs_error", ascending=False).to_csv(os.path.join(out_dir, "errors.csv"), index=False)

    # Report markdown
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Tier-1 Report\n\n")
        f.write(f"- OOF ROC-AUC: {m['roc_auc']:.4f}\n")
        f.write(f"- OOF PR-AUC: {m['pr_auc']:.4f}\n")
        f.write(f"- Best F1: {m['f1']:.4f} @ thr={thr:.2f}\n")
        f.write("\n## Figures\n")
        f.write("- roc.png\n")
        f.write("- pr.png\n")
        f.write("- calibration.png\n")
        f.write("- prob_hist.png\n")
        f.write("- confusion_matrix.png\n")
        f.write("\n## Tables\n")
        f.write("- errors.csv (sorted by abs_error desc)\n")

    print("[Report] report.md 및 시각화 생성:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1", type=str, default="outputs/1_tier1")
    parser.add_argument("--out", type=str, default="outputs/4_report")
    parser.add_argument("--data", type=str, default="UCI_Credit_Card.csv")
    args = parser.parse_args()
    main(args.tier1, args.out, args.data)

