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

# 실제 vs 예측 시각화
plt.figure(figsize=(6,6))
sns.scatterplot(x=y, y=pred, color='blue')
plt.xlabel("실제 점수")
plt.ylabel("예측 점수")
plt.title("다변수 경사하강법 결과: 실제 vs 예측")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # y=x 기준선
plt.show()


print(f"공부시간 계수 θ₁ = {theta1:.2f} (공부시간 1시간 ↑ → 점수 약 +{theta1:.2f})")
print(f"휴식시간 계수 θ₂ = {theta2:.2f} (휴식시간 1시간 ↑ → 점수 약 +{theta2:.2f})")
