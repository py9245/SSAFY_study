import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100000

# 1) 데이터 생성
study_time = np.random.uniform(0, 8, n)
rest_time  = np.random.choice([1, 2, 3, 4], n)
condition  = np.random.uniform(1, 8, n)

# 2) 계수(분산을 줄이기 위해 기존보다 완만하게)
intercept = 20.0
a = 3.2    # 공부 효과(기본)
b = 1.6    # 컨디션 효과(기본)
c = -0.8   # 휴식 기본효과(과하면 감점)
d = 0.08   # 공부×컨디션 시너지
e = 0.10   # 공부×휴식 상호작용(과휴식 페널티)

# 3) 원시 점수(raw) 계산
raw = (
    intercept
    + a * study_time
    + b * condition
    + c * rest_time
    + d * study_time * condition
    - e * study_time * rest_time
    # + np.random.normal(0, 1.5, n)  # 노이즈 줄여 과도한 100 초과 방지
)

# (선택) 특수 규칙: 공부 7h+ & 휴식 1h 이하면 소폭 감점
mask = (study_time >= 7) & (rest_time <= 1)
raw[mask] -= np.random.uniform(0, 0.001, mask.sum())

# 4) 두 점으로 선형 보정 (alpha, beta)
#    - 앵커: (8,2,8) → 100
#    - 중앙: median(raw) → 70
raw_anchor = (
    intercept
    + a*8 + b*8 + c*2
    + d*(8*8) - e*(8*2)
)  # 노이즈/특수감점 없음(정의식 그대로)

raw_median = np.median(raw)

alpha = (100 - 70) / (raw_anchor - raw_median + 1e-8)  # 1e-8은 0나눗셈 방지용
beta  = 100 - alpha * raw_anchor

score = alpha * raw + beta

# 5) 안전 클리핑
score = np.clip(score, 0, 100)

# 6) 데이터프레임
data = pd.DataFrame({
    "study_time": study_time,
    "rest_time": rest_time,
    "condition": condition,
    "score": score
})
data.to_csv("study_data.csv", index=False, encoding="utf-8-sig")

# 7) 시각화: 휴식시간별 FacetGrid
sns.set(style="whitegrid", font="Malgun Gothic", context="talk")

g = sns.FacetGrid(
    data,
    col="rest_time",
    col_wrap=2,
    height=5,
    sharex=True,
    sharey=True
)
g.map_dataframe(
    sns.scatterplot,
    x="study_time",
    y="score",
    hue="condition",
    palette="viridis",
    alpha=0.75
)
g.add_legend(title="컨디션 (1~8)")
g.set_axis_labels("공부시간 (시간)", "성적")
g.set_titles("휴식시간 {col_name}시간")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("휴식시간별 공부시간-컨디션-성적 (정밀 보정)")
plt.show()
