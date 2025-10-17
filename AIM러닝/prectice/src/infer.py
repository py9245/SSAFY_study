import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import Booster

# 패키지/스크립트 실행 양쪽 호환을 위한 임포트 처리
try:
    from .features import add_features
except Exception:
    import sys as _sys, os as _os
    _SRC_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_SRC_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from src.features import add_features


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_latest_fold_model(models_dir: str) -> Booster:
    txt_models = sorted([p for p in Path(models_dir).glob("lgbm_fold*.txt")])
    if not txt_models:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {models_dir}")
    # pick first (or best)
    return Booster(model_file=str(txt_models[0]))


def main(data_path: str, out_path: str, tier1_dir: str, tier3_dir: str | None = None):
    ensure_dir(Path(out_path).parent)

    df = pd.read_csv(data_path)
    X_df = df.drop(columns=[c for c in ["ID", "default.payment.next.month"] if c in df.columns])
    X_fe = add_features(X_df)
    X = X_fe.to_numpy(dtype=np.float32)

    booster = load_latest_fold_model(os.path.join(tier1_dir, "lgbm_models"))
    raw_prob = booster.predict(X)

    prob = raw_prob
    thr = 0.5
    if tier3_dir and Path(os.path.join(tier3_dir, "calibration.pkl")).exists():
        calib = joblib.load(os.path.join(tier3_dir, "calibration.pkl"))
        prob = calib.predict_proba(X)[:, 1]
        thr_path = os.path.join(tier3_dir, "thresholds.json")
        if Path(thr_path).exists():
            with open(thr_path, "r", encoding="utf-8") as f:
                thr = float(json.load(f).get("best_f1_threshold", 0.5))

    pred = (prob >= thr).astype(int)

    out_df = pd.DataFrame({"prob": prob, "pred": pred})
    out_df.to_csv(out_path, index=False)
    print("[Infer] 저장:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/preds.csv")
    parser.add_argument("--tier1", type=str, default="outputs/1_tier1")
    parser.add_argument("--tier3", type=str, default=None)
    args = parser.parse_args()

    main(
        data_path=args.data,
        out_path=args.output,
        tier1_dir=args.tier1,
        tier3_dir=args.tier3,
    )
