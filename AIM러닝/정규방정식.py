import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)
n = 200

study_time = np.random.uniform(0, 10, n)  # 공부시간
rest_time = np.random.uniform(0, 5, n)    # 휴식시간

# 점수 생성 (공부시간↑ → 성적↑, 휴식은 중간일 때 최고)
scores = 50 + 5*study_time - 3*(rest_time - 2.5)**2 + np.random.normal(0, 5, n)

data = pd.DataFrame({
    'study_time': study_time,
    'rest_time': rest_time,
    'score': scores
})

data.head()

X1 = data['study_time'].values
X2 = data['rest_time'].values
y = data['score'].values

theta0, theta1, theta2 = 0, 0, 0   # 초기값
lr = 0.0005                         # 학습률
epochs = 50000                       # 반복 횟수
n = len(y)

theta0, theta1, theta2 = 0, 0, 0   # 초기값
lr = 0.0005                         # 학습률
epochs = 50000                       # 반복 횟수
n = len(y)


for i in tqdm(range(epochs)):
    pred = theta0 + theta1 * X1 + theta2 * X2
    error = pred - y
    
    # 평균 제곱 오차(MSE) 기반 기울기
    d_theta0 = (1/n) * np.sum(error)
    d_theta1 = (1/n) * np.sum(error * X1)
    d_theta2 = (1/n) * np.sum(error * X2)
    
    # 파라미터 업데이트
    theta0 -= lr * d_theta0
    theta1 -= lr * d_theta1
    theta2 -= lr * d_theta2

print(f"θ0 = {theta0:.3f}, θ1 = {theta1:.3f}, θ2 = {theta2:.3f}")


# 예측
pred = theta0 + theta1 * X1 + theta2 * X2

# -----------------------------
# 정규방정식(Normal Equation)
# -----------------------------

# 1) X 행렬 만들기 (절편항 포함)
X = np.column_stack((np.ones(n), X1, X2))  # [1, study_time, rest_time]

# 2) 정규방정식 계산: θ = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X.T @ X) @ X.T @ y

theta0_ne, theta1_ne, theta2_ne = theta
print(f"[정규방정식 결과]")
print(f"θ0 = {theta0_ne:.3f}, θ1 = {theta1_ne:.3f}, θ2 = {theta2_ne:.3f}")

# 3) 예측값
pred_ne = X @ theta

# 4) 시각화 비교
plt.figure(figsize=(6,6))
sns.scatterplot(x=y, y=pred_ne, color='green', label='정규방정식 예측')
sns.scatterplot(x=y, y=pred, color='blue', alpha=0.5, label='경사하강법 예측')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='y=x 기준선')
plt.xlabel("실제 점수")
plt.ylabel("예측 점수")
plt.title("경사하강법 vs 정규방정식 비교")
plt.legend()
plt.show()

# 5) 비교 출력
print(f"공부시간 계수 θ₁(정규방정식) = {theta1_ne:.2f}")
print(f"휴식시간 계수 θ₂(정규방정식) = {theta2_ne:.2f}")


sns.pairplot(
    data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'target']],
    corner=True
)
plt.suptitle("피처 간 관계 및 Target 분포", y=1.02)
plt.show()
