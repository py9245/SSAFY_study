import os
from typing import List, Tuple

import pandas as pd


TARGET_COL = "default.payment.next.month"
ID_COL = "ID"


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV by searching common project locations.

    Priority order:
    1) The path as given
    2) Project root (parent of src) + basename(path)
    3) Project root + path (for relative paths)
    Returns the first readable CSV.
    """
    candidates: List[str] = [path]

    # Project root (parent of src directory)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(src_dir, os.pardir))

    base = os.path.basename(path)
    candidates.append(os.path.join(proj_root, base))
    candidates.append(os.path.join(proj_root, path))

    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)

    tried = "\n".join(os.path.abspath(p) for p in candidates)
    cwd = os.getcwd()
    raise FileNotFoundError(
        "데이터 파일을 찾을 수 없습니다. 아래 경로들을 확인하세요.\n"
        f"- CWD: {cwd}\n"
        f"- Tried:\n{tried}"
    )


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"타겟 컬럼({TARGET_COL})이 데이터프레임에 없습니다.")
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore")
    return X, y

