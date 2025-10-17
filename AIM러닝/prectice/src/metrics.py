from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    f1: float
    threshold: float


def sweep_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr_grid=None) -> Tuple[float, float]:
    if thr_grid is None:
        thr_grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    best_f1, best_thr = -1.0, 0.5
    for t in thr_grid:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_f1, best_thr


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr_grid=None) -> Metrics:
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    best_f1, best_thr = sweep_threshold(y_true, y_prob, thr_grid=thr_grid)
    return Metrics(roc_auc=roc, pr_auc=pr, f1=best_f1, threshold=best_thr)


def to_dict(m: Metrics) -> Dict:
    return {
        "roc_auc": float(m.roc_auc),
        "pr_auc": float(m.pr_auc),
        "f1": float(m.f1),
        "threshold": float(m.threshold),
    }

