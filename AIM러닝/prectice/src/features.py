import numpy as np
import pandas as pd


def _signed_log1p(series: pd.Series) -> pd.Series:
    arr = series.to_numpy()
    return np.sign(arr) * np.log1p(np.abs(arr))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # Column groups expected in UCI Credit dataset
    bill_cols = [c for c in X.columns if c.startswith("BILL_AMT")]
    pay_cols = [c for c in X.columns if c.startswith("PAY_AMT")]
    pay_status_cols = [c for c in X.columns if c.startswith("PAY_") and c not in pay_cols]

    # Basic ratios/summaries
    if "LIMIT_BAL" in X.columns and "BILL_AMT1" in X.columns:
        X["util_recent"] = X["BILL_AMT1"] / (X["LIMIT_BAL"].replace(0, np.nan))
        X["util_recent"] = X["util_recent"].fillna(0.0)

    if bill_cols:
        X["bill_sum"] = X[bill_cols].sum(axis=1)
        X["util_mean"] = (
            X[bill_cols].mean(axis=1) / (X.get("LIMIT_BAL", pd.Series(1.0, index=X.index)).replace(0, np.nan))
        ).fillna(0.0)

    if pay_cols:
        X["pay_sum"] = X[pay_cols].sum(axis=1)
        X["pay_ratio1"] = (X.get("PAY_AMT1", pd.Series(0, index=X.index)) /
                            (X.get("BILL_AMT1", pd.Series(np.nan, index=X.index)).replace(0, np.nan))).fillna(0.0)
        X["pay_ratio_mean"] = (
            X[pay_cols].mean(axis=1) /
            (X[bill_cols].replace(0, np.nan).mean(axis=1) if bill_cols else pd.Series(np.nan, index=X.index))
        ).fillna(0.0)

    # Delay patterns
    if pay_status_cols:
        pay_status = X[pay_status_cols].copy()
        pos_mask = (pay_status > 0)
        X["delay_cnt_pos"] = pos_mask.sum(axis=1)
        X["delay_max"] = pay_status.max(axis=1)
        X["delay_sum_pos"] = pay_status.where(pay_status > 0, 0).sum(axis=1)
        # longest consecutive positive run
        X["delay_consecutive_pos_max"] = _longest_consecutive_positive(pay_status)

    # Trends and diffs
    def _trend_and_diffs(cols: list[str], prefix: str):
        if not cols:
            return
        cols_sorted = sorted(cols, key=lambda c: int(''.join([ch for ch in c if ch.isdigit()]) or 0))
        first, last = cols_sorted[0], cols_sorted[-1]
        X[f"{prefix}_trend"] = X[first] - X[last]
        for i in range(len(cols_sorted) - 1):
            c1, c2 = cols_sorted[i], cols_sorted[i + 1]
            X[f"{prefix}_diff_{i+1}"] = X[c1] - X[c2]

    _trend_and_diffs(bill_cols, "bill")
    _trend_and_diffs(pay_cols, "pay")

    # Signed log1p transforms for monetary columns
    money_cols = ["LIMIT_BAL"] + bill_cols + pay_cols
    for c in money_cols:
        if c in X.columns:
            X[f"{c}_slog1p"] = _signed_log1p(X[c])

    # Replace infs and NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def _longest_consecutive_positive(frame: pd.DataFrame) -> pd.Series:
    arr = (frame.to_numpy() > 0).astype(np.int32)
    n, m = arr.shape
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        best, cur = 0, 0
        for j in range(m):
            if arr[i, j] == 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out[i] = best
    return pd.Series(out, index=frame.index)
