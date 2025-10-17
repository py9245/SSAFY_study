import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 패키지/스크립트 실행 양쪽 호환을 위한 임포트 처리
try:
    from .data import load_dataset, split_xy
    from .metrics import compute_metrics, to_dict
except Exception:
    import sys as _sys, os as _os
    _SRC_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_SRC_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from src.data import load_dataset, split_xy
    from src.metrics import compute_metrics, to_dict


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def main(tier1_dir: str = "outputs/1_tier1", out_dir: str = "outputs/2_tier2", data_path: str = "UCI_Credit_Card.csv"):
    ensure_dir(out_dir)

    # Single-model case: treat Tier-1 OOF as ensemble baseline
    oof_path = os.path.join(tier1_dir, "oof_preds.csv")
    if not os.path.exists(oof_path):
        raise FileNotFoundError("Tier-1 산출물이 없습니다. 먼저 train.py를 실행하세요.")

    df = load_dataset(data_path)
    _, y_s = split_xy(df)
    y = y_s.to_numpy()

    oof = pd.read_csv(oof_path)["oof"].to_numpy()
    m = compute_metrics(y, oof)

    pd.DataFrame({"oof": oof}).to_csv(os.path.join(out_dir, "oof_preds_ensemble.csv"), index=False)
    with open(os.path.join(out_dir, "oof_metrics_ensemble.json"), "w", encoding="utf-8") as f:
        json.dump(to_dict(m), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "ensemble.json"), "w", encoding="utf-8") as f:
        json.dump({"type": "weighted_mean", "weights": {"lgbm": 1.0}}, f, ensure_ascii=False, indent=2)

    print("[Tier-2] 완료:")
    print("  - ensemble.json, OOF 저장:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1", type=str, default="outputs/1_tier1")
    parser.add_argument("--out", type=str, default="outputs/2_tier2")
    parser.add_argument("--data", type=str, default="UCI_Credit_Card.csv")
    args = parser.parse_args()
    main(args.tier1, args.out, args.data)
