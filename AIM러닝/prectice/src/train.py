import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import lightgbm as lgb

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


def train_fold(X: np.ndarray, y: np.ndarray, tr_idx, va_idx, params: dict, stopping_rounds: int = 200):
    clf = LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        max_depth=params.get("max_depth", -1),
        min_child_samples=params.get("min_child_samples", 40),
        subsample=params.get("subsample", 0.8),
        subsample_freq=params.get("subsample_freq", 1),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 0.0),
        n_estimators=params.get("n_estimators", 5000),
        class_weight=params.get("class_weight", None),
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
    )
    va_prob = clf.predict_proba(X_va)[:, 1]
    return clf, va_prob


def main(
    data_path: str,
    out_dir: str = "outputs/1_tier1",
    n_splits: int = 9,
    seed: int = 42,
    class_weight: Optional[str] = None,
    learning_rate: Optional[float] = None,
    num_leaves: Optional[int] = None,
    max_depth: Optional[int] = None,
    min_child_samples: Optional[int] = None,
    subsample: Optional[float] = None,
    subsample_freq: Optional[int] = None,
    colsample_bytree: Optional[float] = None,
    reg_alpha: Optional[float] = None,
    reg_lambda: Optional[float] = None,
    n_estimators: Optional[int] = None,
    stopping_rounds: int = 200,
):
    ensure_dir(out_dir)
    models_dir = os.path.join(out_dir, "lgbm_models")
    ensure_dir(models_dir)

    df = load_dataset(data_path)
    X_df, y_s = split_xy(df)
    X_fe = add_features(X_df)
    X = X_fe.to_numpy(dtype=np.float32)
    y = y_s.to_numpy(dtype=np.int32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    params = {
        "learning_rate": 0.05 if learning_rate is None else learning_rate,
        "num_leaves": 63 if num_leaves is None else num_leaves,
        "max_depth": -1 if max_depth is None else max_depth,
        "min_child_samples": 40 if min_child_samples is None else min_child_samples,
        "subsample": 0.8 if subsample is None else subsample,
        "subsample_freq": 1 if subsample_freq is None else subsample_freq,
        "colsample_bytree": 0.8 if colsample_bytree is None else colsample_bytree,
        "reg_alpha": 0.0 if reg_alpha is None else reg_alpha,
        "reg_lambda": 0.0 if reg_lambda is None else reg_lambda,
        "n_estimators": 5000 if n_estimators is None else n_estimators,
        "class_weight": class_weight,
    }

    oof = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        clf, va_prob = train_fold(X, y, tr_idx, va_idx, params, stopping_rounds=stopping_rounds)
        oof[va_idx] = va_prob

        # save model
        booster = clf.booster_
        booster.save_model(os.path.join(models_dir, f"lgbm_fold{fold}.txt"))

        # fold metrics
        m = compute_metrics(y[va_idx], va_prob)
        fold_metrics.append({"fold": fold, **to_dict(m)})

    # OOF metrics
    oof_metrics = compute_metrics(y, oof)
    pd.DataFrame({"oof": oof}).to_csv(os.path.join(out_dir, "oof_preds.csv"), index=False)
    with open(os.path.join(out_dir, "fold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "oof_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(to_dict(oof_metrics), f, ensure_ascii=False, indent=2)

    print("[Tier-1] 완료:")
    print("  - 모델 저장:", models_dir)
    print("  - OOF/메트릭:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="UCI_Credit_Card.csv")
    parser.add_argument("--out", type=str, default="outputs/1_tier1")
    parser.add_argument("--splits", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-child-samples", type=int, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--subsample-freq", type=int, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--reg-alpha", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--stopping-rounds", type=int, default=200)
    args = parser.parse_args()

    main(
        data_path=args.data,
        out_dir=args.out,
        n_splits=args.splits,
        seed=args.seed,
        class_weight=args.class_weight,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        n_estimators=args.n_estimators,
        stopping_rounds=args.stopping_rounds,
    )
