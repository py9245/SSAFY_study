import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif

# ----------------------------
# 0) 데이터 로드 & 분리
# ----------------------------
df = pd.read_csv("UCI_Credit_Card.csv")
print(df.info())
#
# TARGET = "default.payment.next.month"
# X_df = df.drop(columns=["ID", TARGET]).astype(np.float64)   # 23개 피처
# y = df[TARGET].values.astype(np.float64).reshape(-1, 1)
#
# # ----------------------------
# # 1) 피처 상위 1/2 선택 (Mutual Information)
# #    23 -> 11개로 축소
# # ----------------------------
# mi = mutual_info_classif(X_df.values, y.ravel(), discrete_features='auto', random_state=42)
# mi_series = pd.Series(mi, index=X_df.columns).sort_values(ascending=False)
#
# HALF = len(X_df.columns) // 2            # 23 -> 11
# FEATURE_POOL = mi_series.index.tolist()[:HALF]
# print("[INFO] 상위 1/2 피처(개수=%d):" % len(FEATURE_POOL), FEATURE_POOL)
#
# # ----------------------------
# # 2) 로지스틱 회귀(시그모이드) + GD
# #    - 학습: BCE, 평가: MSE
# # ----------------------------
# def sigmoid(z):
#     z = np.clip(z, -30, 30)
#     return 1.0 / (1.0 + np.exp(-z))
#
# def train_logistic_gd(Phi_tr, y_tr, epochs=1000, lr=0.05, l2=0.0):
#     m, d = Phi_tr.shape
#     w = np.zeros((d, 1))
#     b = 0.0
#     for _ in range(epochs):
#         z = Phi_tr @ w + b
#         p = sigmoid(z)
#         grad = (p - y_tr)                 # (m,1)
#         dw = (Phi_tr.T @ grad) / m + (l2/m) * w
#         db = grad.mean()
#         w -= lr * dw
#         b -= lr * db
#     return w, b
#
# def mse(y_true, y_prob):
#     return float(((y_true - y_prob)**2).mean())
#
# # ----------------------------
# # 3) 스케일링 + 상호작용항(자승 X, 편항 포함)
# #    combo_size = r이면 degree=r 로 설정 → ab, abc 등 r차 교차항까지 생성
# # ----------------------------
# def make_design(X_tr_raw, X_va_raw):
#     scaler = StandardScaler()
#     Xs_tr = scaler.fit_transform(X_tr_raw)
#     Xs_va = scaler.transform(X_va_raw)
#
#     r = X_tr_raw.shape[1]                     # 조합에 포함된 변수 개수
#     poly = PolynomialFeatures(
#         degree=r,                             # r차까지 교차항 생성 (자승 X)
#         interaction_only=True,                # 제곱/세제곱 같은 단항 거부
#         include_bias=True                     # 편항 1 포함
#     )
#     Phi_tr = poly.fit_transform(Xs_tr)
#     Phi_va = poly.transform(Xs_va)
#     return Phi_tr, Phi_va
#
# # ----------------------------
# # 4) 한 조합의 9-Fold CV MSE 계산
# # ----------------------------
# def cv_mse_for_combo(combo_cols, epochs=1000, lr=0.05, kfold=9, l2=0.0, seed=42):
#     kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
#     X_sub = X_df[list(combo_cols)].values
#
#     fold_mses = []
#     for tr_idx, va_idx in kf.split(X_sub):
#         X_tr_raw, X_va_raw = X_sub[tr_idx], X_sub[va_idx]
#         y_tr, y_va = y[tr_idx], y[va_idx]
#
#         Phi_tr, Phi_va = make_design(X_tr_raw, X_va_raw)
#         w, b = train_logistic_gd(Phi_tr, y_tr, epochs=epochs, lr=lr, l2=l2)
#         p_va = sigmoid(Phi_va @ w + b)
#         fold_mses.append(mse(y_va, p_va))
#
#     return float(np.mean(fold_mses))
#
# # ----------------------------
# # 5) 초기 후보 조합 생성 (r=1~3)
# #    r을 더 늘리면 설계행렬이 폭증 → 먼저 1~3만 권장
# # ----------------------------
# MAX_COMB_SIZE = 3
# candidates = []
# for r in range(1, MAX_COMB_SIZE + 1):
#     for combo in combinations(FEATURE_POOL, r):
#         candidates.append(combo)
# print(f"[INFO] 초기 후보 조합 수: {len(candidates)}")
#
# # ----------------------------
# # 6) 토너먼트 서치
# #    - 각 라운드: 전 후보 CV → 상위 1/2만 생존
# #    - epoch *= 1.3
# # ----------------------------
# def tournament_search(candidates, init_epochs=1000, lr=0.05, kfold=9, l2=0.0, epoch_factor=1.3):
#     rounds = 0
#     epochs = int(init_epochs)
#     cur = candidates[:]
#     history = []
#
#     while len(cur) > 1:
#         rounds += 1
#         print(f"\n[ROUND {rounds}] 후보 {len(cur)}개 | epochs={epochs}")
#
#         scores = []
#         for combo in cur:
#             m = cv_mse_for_combo(combo, epochs=epochs, lr=lr, kfold=kfold, l2=l2, seed=42)
#             scores.append((m, combo))
#
#         scores.sort(key=lambda x: x[0]) # MSE 오름차순
#         best_mse, best_combo = scores[0]
#         history.append((rounds, best_mse, best_combo))
#         print(f"  - best@round{rounds}: MSE={best_mse:.6f}, combo={best_combo}")
#
#         keep_n = max(1, len(scores) // 2) # 상위 1/2 생존
#         cur = [c for (m, c) in scores[:keep_n]]
#
#         epochs = max(1, int(epochs * epoch_factor)) # epoch 1.3배
#
#     final_combo = cur[0]
#     final_mse = cv_mse_for_combo(final_combo, epochs=epochs, lr=lr, kfold=kfold, l2=l2, seed=42)
#     return final_combo, final_mse, history
#
# # ----------------------------
# # 7) 실행
# # ----------------------------
# INIT_EPOCHS = 1000
# LR = 0.05
# KFOLD = 9
# L2 = 0.0
#
# best_combo, best_mse, search_hist = tournament_search(
#     candidates,
#     init_epochs=INIT_EPOCHS,
#     lr=LR,
#     kfold=KFOLD,
#     l2=L2,
#     epoch_factor=1.3,
# )
#
# print("\n[RESULT]")
# print("최종 베스트 조합:", best_combo)
# print("최종 CV MSE    :", best_mse)
# print("라운드 히스토리:", search_hist)
