import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# 1. 데이터 로드
# ----------------------------
df = pd.read_csv("study_data.csv", encoding="utf-8-sig")

np.random.seed(24)

x = df.iloc[:, :-1].to_numpy(copy=False)
y = df.iloc[:, -1].to_numpy(copy=False)

# ----------------------------
# 2. PolynomialFeatures 생성
# ----------------------------
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
x_poly = poly.fit_transform(x)
feature_names = poly.get_feature_names_out(df.columns[:-1])

# ----------------------------
# 3. 스케일링
# ----------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_poly)

# ----------------------------
# 4. 경사하강법 함수
# ----------------------------
def gradient_descent(X, y, lr=0.003, epochs=20000):
    X = np.c_[np.ones(X.shape[0]), X]
    beta = np.zeros(X.shape[1])
    for _ in range(epochs):
        y_pred = X @ beta
        error = y_pred - y
        grad = (1 / len(y)) * (X.T @ error)
        beta -= lr * grad
    mse = mean_squared_error(y, y_pred)
    return beta, mse, y_pred

# ----------------------------
# 5. 모든 조합 전수 탐색
# ----------------------------
best_combo = None
best_mse = float("inf")
results = []

for r in range(1, len(feature_names) + 1):
    for subset in itertools.combinations(range(len(feature_names)), r):
        X_sub = x_scaled[:, subset]
        beta, mse, _ = gradient_descent(X_sub, y)
        results.append((subset, mse))
        if mse < best_mse:
            best_mse = mse
            best_combo = subset

best_features = [feature_names[i] for i in best_combo]
print("\n✅ 최적 피처 조합:", best_features)
print("최소 MSE:", best_mse)

# ----------------------------
# 6. 최적 조합으로 모델 학습
# ----------------------------
X_best = x_scaled[:, best_combo]
beta_best, mse_best, y_pred_best = gradient_descent(X_best, y)

# ----------------------------
# 7. 3D 곡면 시각화
# ----------------------------
# 공부시간(study_time)과 컨디션(condition)을 격자로 만들고
# rest_time은 평균값으로 고정
study_range = np.linspace(df['study_time'].min(), df['study_time'].max(), 50)
cond_range = np.linspace(df['condition'].min(), df['condition'].max(), 50)
rest_mean = df['rest_time'].mean()

S, C = np.meshgrid(study_range, cond_range)
R = np.full_like(S, rest_mean)

grid = np.stack([S.ravel(), R.ravel(), C.ravel()], axis=1)

# 같은 변환 적용
grid_poly = poly.transform(grid)
grid_poly_scaled = scaler.transform(grid_poly)
grid_best = grid_poly_scaled[:, best_combo]
grid_best = np.c_[np.ones(grid_best.shape[0]), grid_best]

Z = grid_best @ beta_best
Z = Z.reshape(S.shape)

# ----------------------------
# 8. 실제 데이터 + 예측 곡면 출력
# ----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 실제 데이터 점
ax.scatter(df['study_time'], df['condition'], y, color='blue', alpha=0.3, label='실제값')

# 예측 곡면
ax.plot_surface(S, C, Z, cmap='viridis', alpha=0.7)

ax.set_title(f"3D 예측 곡면 (최적 조합: {', '.join(best_features)})")
ax.set_xlabel("공부시간 (study_time)")
ax.set_ylabel("컨디션 (condition)")
ax.set_zlabel("예측 점수 (score)")
ax.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 🔹 잔차 계산
# ----------------------------
X_best = x_scaled[:, best_combo]
_, _, y_pred_best = gradient_descent(X_best, y)
residuals = y - y_pred_best

# ----------------------------
# 🔹 hexbin 기반 잔차 분포
# ----------------------------
feature_cols = df.columns[:-1]

fig, axes = plt.subplots(1, len(feature_cols), figsize=(15, 4))

for i, col in enumerate(feature_cols):
    axes[i].hexbin(df[col], residuals, gridsize=60, cmap='viridis', mincnt=1)
    axes[i].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[i].set_title(f"{col} vs Residuals (density)")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("잔차 (y - ŷ)")

plt.suptitle("피처별 잔차 분포 (Hexbin: 색상은 데이터 밀도)", fontsize=14)
plt.tight_layout()
plt.show()
