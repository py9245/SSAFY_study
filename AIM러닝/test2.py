import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# 1. 데이터 생성
# -----------------------------
np.random.seed(42)
n = 100
study_time = np.linspace(0, 10, n)

# 휴식시간 그룹 (1, 2, 3시간)
rest_times = np.repeat([1, 2, 3], n)  # 총 300개 (각 그룹 100개)

# 각 휴식시간별로 공부 효율(기울기)이 다르게 설정
slopes = {1: 6, 2: 5, 3: 3}  # 휴식시간이 많을수록 효율↓
scores = []

for r in [1, 2, 3]:
    s = 50 + slopes[r] * study_time + np.random.normal(0, 0.5, n)
    scores.extend(s)

data = pd.DataFrame({
    'study_time': np.tile(study_time, 3),
    'rest_time': rest_times,
    'score': scores
})

print(data.head())

# -----------------------------
# 2. 시각화
# -----------------------------
sns.set(style="whitegrid", font="Malgun Gothic")
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='study_time', y='score', hue='rest_time', palette='viridis')
plt.title("휴식시간별 공부시간-성적 관계 (기울기 다름)")
plt.xlabel("공부시간 (시간)")
plt.ylabel("성적")
plt.legend(title='휴식시간')
plt.show()

# -----------------------------
# 3. 그룹별 기울기(미분값) 계산
# -----------------------------
derivatives = {}

for r in [1, 2, 3]:
    subset = data[data['rest_time'] == r]
    X = subset['study_time'].values
    y = subset['score'].values
    
    # 정규방정식으로 기울기(θ1) 계산
    X_mat = np.column_stack((np.ones_like(X), X))
    theta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y
    theta0, theta1 = theta
    derivatives[r] = theta1

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("\n=== 휴식시간별 미분값(기울기) ===")
for r, slope in derivatives.items():
    print(f"휴식시간 {r}시간 → 기울기 θ₁ = {slope:.3f}")

# -----------------------------
# 5. 기울기 시각화
# -----------------------------
plt.figure(figsize=(7,4))
plt.bar(derivatives.keys(), derivatives.values(), color=['#4c72b0', '#55a868', '#c44e52'])
plt.title("휴식시간별 성적 증가율(기울기)")
plt.xlabel("휴식시간 (시간)")
plt.ylabel("기울기 (dScore/dStudyTime)")
plt.show()
