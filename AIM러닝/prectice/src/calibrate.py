import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# 패키지/스크립트 실행 양쪽 호환을 위한 임포트 처리
try:
    from .data import load_dataset, split_xy
    from .features import add_features
    from .metrics import compute_metrics, to_dict
except Exception:
    import sys as _sys, os as _os
    _SRC_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_SRC_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from src.data import load_dataset, split_xy
    from src.features import add_features
    from src.metrics import compute_metrics, to_dict


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def main(
    data_path: str,
    out_dir_tier1: str = "outputs/1_tier1",
    out_dir: str = "outputs/3_tier3",
):
    ensure_dir(out_dir)

    df = load_dataset(data_path)
    X_df, y_s = split_xy(df)
    X_fe = add_features(X_df)
    X = X_fe.to_numpy(dtype=np.float32)
    y = y_s.to_numpy(dtype=np.int32)

    # Base estimator for calibration (refit on full data)
    base = LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_estimators=2000,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    # scikit-learn >=1.4 uses 'estimator' instead of deprecated 'base_estimator'
    try:
        calib = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    except TypeError:
        # fallback for older versions
        calib = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=5)
    calib.fit(X, y)

    prob = calib.predict_proba(X)[:, 1]
    m = compute_metrics(y, prob)

    joblib.dump(calib, os.path.join(out_dir, "calibration.pkl"))
    with open(os.path.join(out_dir, "calibration_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(to_dict(m), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"best_f1_threshold": m.threshold}, f, ensure_ascii=False, indent=2)

    print("[Tier-3] 완료:")
    print("  - calibration.pkl, thresholds.json 저장:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="UCI_Credit_Card.csv")
    parser.add_argument("--tier1", type=str, default="outputs/1_tier1")
    parser.add_argument("--out", type=str, default="outputs/3_tier3")
    args = parser.parse_args()

    main(
        data_path=args.data,
        out_dir_tier1=args.tier1,
        out_dir=args.out,
    )
