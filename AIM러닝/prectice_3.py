# -*- coding: utf-8 -*-
"""
UCI_Credit_Card: K-Fold로 BCE(log_loss) 최소 모델 찾기
- 동일한 CV로 여러 모델/그리드 공정 비교
- 평가: neg_log_loss(최대화) -> BCE = -neg_log_loss(최소화)
- 보조 지표: ROC-AUC
- 베스트 모델/파라미터/점수 표 형태로 출력, 모델 저장 (joblib)
"""

import os
import numpy as np
import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
import joblib

# -----------------------------
# 0) 데이터 로드 & 기본 설정
# -----------------------------
CSV_PATH = "UCI_Credit_Card.csv"
TARGET = "default.payment.next.month"
SEED = 42
N_SPLITS = 9

df = pd.read_csv(CSV_PATH)

# 특징/타깃 분리 (ID 제거)
X = df.drop(columns=["ID", TARGET])
y = df[TARGET].astype(int).values

# 금액 계열 컬럼(heavy-tail)
AMOUNT_COLS = (
    ["LIMIT_BAL"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

# -----------------------------
# 1) 전처리: 음수 안전 로그 (CHANGED)
#    sign(x) * log1p(|x|) -> 음수도 OK, inf/NaN 방지
# -----------------------------
def signed_log1p(X):
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))

log_amounts = ColumnTransformer(
    transformers=[
        ("slog1p", FunctionTransformer(signed_log1p, feature_names_out="one-to-one"), AMOUNT_COLS),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

linear_preprocess = Pipeline([
    ("log1p", log_amounts),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
])

tree_preprocess = Pipeline([
    ("log1p", log_amounts),
])

# -----------------------------
# 1.5) 변환 유효성 사전 체크 (CHANGED: 디버그에 도움)
# -----------------------------
X_check = log_amounts.fit_transform(X, y)
if not np.isfinite(X_check).all():
    raise ValueError("전처리 결과에 NaN/inf가 있습니다. 변환 단계를 확인하세요.")

# -----------------------------
# 2) 베이스라인
# -----------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
baseline = DummyClassifier(strategy="prior")
baseline_scores = cross_val_score(baseline, X, y, cv=cv, scoring="neg_log_loss", n_jobs=-1)
print(f"[Baseline] mean BCE (prior only): {(-baseline_scores.mean()):.6f}")

# -----------------------------
# 3) 후보 모델 & 그리드
# -----------------------------
candidates = []

# 3-1) Logistic Regression
pipe_lr = Pipeline([
    ("prep", linear_preprocess),
    ("model", LogisticRegression(solver="saga", max_iter=3000, random_state=SEED))
])
grid_lr = {
    "model__penalty": ["l2", "l1"],
    "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    "model__class_weight": [None, "balanced"],
}
candidates.append(("LogisticRegression", pipe_lr, grid_lr))

# 3-2) SGDClassifier
pipe_sgd = Pipeline([
    ("prep", linear_preprocess),
    ("model", SGDClassifier(loss="log_loss", max_iter=3000, random_state=SEED, tol=1e-3))
])
grid_sgd = [
    {"model__penalty": ["l2"], "model__alpha": [1e-4, 1e-3, 1e-2]},
    {"model__penalty": ["l1"], "model__alpha": [1e-4, 1e-3, 1e-2]},
    {"model__penalty": ["elasticnet"], "model__alpha": [1e-4, 1e-3], "model__l1_ratio": [0.15, 0.5]},
]
candidates.append(("SGDClassifier", pipe_sgd, grid_sgd))

# 3-3) Gradient Boosting
pipe_gb = Pipeline([
    ("prep", tree_preprocess),
    ("model", GradientBoostingClassifier(random_state=SEED))
])
grid_gb = {
    "model__n_estimators": [200, 400],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
    "model__subsample": [1.0, 0.8],
    "model__min_samples_leaf": [1, 20],
}
candidates.append(("GradientBoosting", pipe_gb, grid_gb))

# 3-4) Random Forest
pipe_rf = Pipeline([
    ("prep", tree_preprocess),
    ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1))
])
grid_rf = {
    "model__n_estimators": [400, 800],
    "model__max_depth": [None, 12, 20],
    "model__min_samples_leaf": [1, 5],
    "model__max_features": ["sqrt", 0.5],
    "model__class_weight": [None, "balanced"],
}
candidates.append(("RandomForest", pipe_rf, grid_rf))

# -----------------------------
# 4) GridSearchCV로 BCE 최소 모델 찾기
# -----------------------------
rows = []
best_obj = {"name": None, "bce": np.inf, "auc": None, "params": None, "estimator": None}

for name, pipe, grid in candidates:
    print(f"\n=== {name} : GridSearchCV (scoring=neg_log_loss) ===")
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        # 디버깅이 필요하면 아래 주석 해제:
        # error_score='raise',
    )
    gs.fit(X, y)

    mean_neg_ll = gs.best_score_
    std_neg_ll  = gs.cv_results_["std_test_score"][gs.best_index_]
    bce = -mean_neg_ll

    auc_scores = cross_val_score(gs.best_estimator_, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    auc_mean = auc_scores.mean()

    rows.append({
        "model": name,
        "best_params": gs.best_params_,
        "mean_BCE": bce,
        "std_BCE": std_neg_ll,
        "mean_AUC": auc_mean,
    })

    if bce < best_obj["bce"]:
        best_obj.update({
            "name": name,
            "bce": bce,
            "auc": auc_mean,
            "params": gs.best_params_,
            "estimator": gs.best_estimator_,
        })

# -----------------------------
# 5) 결과 표 + 베스트 모델 저장
# -----------------------------
result_df = pd.DataFrame(rows).sort_values("mean_BCE", ascending=True).reset_index(drop=True)
print("\n===== CV 결과 (낮은 BCE가 더 좋음) =====")
with pd.option_context("display.max_colwidth", None):
    print(result_df)

BEST_PATH = "best_credit_default_model.joblib"
joblib.dump(best_obj["estimator"], BEST_PATH)
print(f"\n[Best] {best_obj['name']}")
print(f"  BCE  : {best_obj['bce']:.6f}")
print(f"  AUC  : {best_obj['auc']:.6f}")
print(f"  PARAM: {best_obj['params']}")
print(f"→ 저장 완료: {BEST_PATH}")

# -----------------------------
# 6) 샘플 예측 확인
# -----------------------------
proba_head = best_obj["estimator"].predict_proba(X)[:5, 1]
print("\n샘플 예측 확률(상위 5개):", np.round(proba_head, 6))
