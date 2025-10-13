import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter, defaultdict

# =========================
# 0) 준비
# =========================
np.random.seed(42)

CSV_PATH = "lotto.csv"
COL_RANKS = ['1등 당첨자수','2등 당첨자수','3등 당첨자수','4등 당첨자수','5등 당첨자수']
COL_NUMS  = ['1번 번호','2번 번호','3번 번호','4번 번호','5번 번호','6번 번호']
COL_BONUS = '보너스번호'

df = pd.read_csv(CSV_PATH)
nums = df[COL_NUMS].astype(int)

# =========================
# 1) 피처 함수
# =========================
def weighted_scores(df_train):
    """
    1순위: 1·2등 희귀 가중치 (적게 나왔을수록 가중↑)
    점수 = 1 / (1등+2등)
    """
    score = 1.0 / (df_train['1등 당첨자수'] + df_train['2등 당첨자수'])
    w = Counter()
    for i, row in df_train[COL_NUMS].astype(int).iterrows():
        s = float(score.loc[i])
        for n in row:
            w[int(n)] += s
    return w  # 번호 -> 가중치 합

def rare_pool(df_train, top_k_rounds=40):
    """
    2순위: 사람들이 적게 산 번호세트의 대리변수
    => 1등 당첨자 수가 적은 회차 top_k_rounds개에서 자주 나온 번호
    """
    rare_rounds = df_train.nsmallest(top_k_rounds, '1등 당첨자수')
    cnt = Counter(rare_rounds[COL_NUMS].stack().astype(int))
    # 상위 20개 추천
    return [int(x) for x, _ in cnt.most_common(20)]

def stable_pool(df_train, top_m=20):
    """
    3순위: 전체 빈도 상위
    """
    cnt = Counter(df_train[COL_NUMS].stack().astype(int))
    return [int(x) for x, _ in cnt.most_common(top_m)]

def build_pools(df_train, hp):
    """
    하이퍼파라미터에 따라 각 풀 구성
    hp: dict with keys - rare_rounds, stable_top_m
    """
    w = weighted_scores(df_train)  # dict: 번호->가중치
    top_pool = [k for k, _ in sorted(w.items(), key=lambda x: x[1], reverse=True)[:10]]
    rare = rare_pool(df_train, top_k_rounds=hp['rare_rounds'])
    stab = stable_pool(df_train, top_m=hp['stable_top_m'])
    return top_pool, rare, stab, w

# =========================
# 2) 조합 생성 & 평가
# =========================
def sample_combo(top_pool, rare_pool, stable_pool, w, hp):
    """
    6개 번호 세트 샘플링.
    - top_pool에서 a개, rare에서 b개, stable에서 c개 뽑기 (중복 제거 후 부족분은 가중치 샘플로 채움)
    - 남은 칸은 {1..45}에서 w 가중치 비례 확률로 채움(중복 없이)
    """
    a, b, c = hp['pick_top'], hp['pick_rare'], hp['pick_stable']
    chosen = set()

    def choose_from(pool, k):
        picks = []
        pool = [x for x in pool if x not in chosen]
        if len(pool) >= k:
            picks = list(np.random.choice(pool, k, replace=False))
        elif len(pool) > 0:
            picks = pool[:]  # 부족하면 있는 만큼만
        return picks

    chosen.update(choose_from(top_pool, a))
    chosen.update(choose_from(rare_pool, b))
    chosen.update(choose_from(stable_pool, c))

    # 부족분은 가중치 기반으로 채우기
    all_nums = list(range(1, 46))
    weights = np.array([w.get(n, 1.0) for n in all_nums], dtype=float)
    for i, n in enumerate(all_nums):
        if n in chosen:
            weights[i] = 0.0
    need = 6 - len(chosen)
    if need > 0 and weights.sum() > 0:
        picks = np.random.choice(all_nums, need, replace=False, p=weights/weights.sum())
        chosen.update(int(x) for x in picks)

    # 혹시 그래도 부족하면 랜덤 보충
    while len(chosen) < 6:
        chosen.add(int(np.random.randint(1, 46)))

    return tuple(sorted(int(x) for x in chosen))

def score_combo(combo, row):
    """
    검증 스코어: 본번호 일치 개수 + (보너스 맞추면 0.5 가점)
    """
    main = set(int(x) for x in row[COL_NUMS])
    bonus = int(row[COL_BONUS])
    hits = len(main.intersection(combo))
    bonus_hit = 0.5 if bonus in combo else 0.0
    return hits + bonus_hit

# =========================
# 3) 593-fold CV
# =========================
K = 593
n = len(df)
assert K <= n, "K가 표본 수보다 클 수 없습니다."

# fold 경계 만들기 (거의 동일 크기)
fold_sizes = [n // K + (1 if i < n % K else 0) for i in range(K)]
indices = np.arange(n)
folds = []
start = 0
for fs in fold_sizes:
    folds.append(indices[start:start+fs])
    start += fs

# 하이퍼파라미터 후보 (필요하면 trials 키워도 됨)
hp_space = [
    {'pick_top':3, 'pick_rare':2, 'pick_stable':1, 'rare_rounds':30, 'stable_top_m':20, 'trials':20},
    {'pick_top':3, 'pick_rare':1, 'pick_stable':2, 'rare_rounds':40, 'stable_top_m':20, 'trials':20},
    {'pick_top':4, 'pick_rare':1, 'pick_stable':1, 'rare_rounds':25, 'stable_top_m':25, 'trials':20},
    {'pick_top':2, 'pick_rare':2, 'pick_stable':2, 'rare_rounds':35, 'stable_top_m':20, 'trials':20},
]

def cross_val(df, folds, hp):
    scores = []
    for val_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(df)), val_idx)
        df_train = df.iloc[train_idx]
        df_val   = df.iloc[val_idx]

        top_pool, rare_pool, stable_pool, w = build_pools(df_train, hp)

        fold_best = 0.0
        for _ in range(hp['trials']):
            combo = sample_combo(top_pool, rare_pool, stable_pool, w, hp)
            s = 0.0
            for _, row in df_val.iterrows():
                s += score_combo(combo, row)
            s /= len(df_val)
            if s > fold_best:
                fold_best = s
        scores.append(fold_best)
    return float(np.mean(scores)), float(np.std(scores))

# 하이퍼파라미터 선택
cv_results = []
for hp in hp_space:
    mean_s, std_s = cross_val(df, folds, hp)
    cv_results.append((mean_s, std_s, hp))

cv_results.sort(key=lambda x: x[0], reverse=True)
best_mean, best_std, best_hp = cv_results[0]

print("🔧 CV 결과 (상위 3)")
for m, s, h in cv_results[:3]:
    print(f"  mean={m:.4f}  std={s:.4f}  hp={h}")
print("\n✅ 선택된 HP:", best_hp, f"(mean={best_mean:.4f}, std={best_std:.4f})")

# =========================
# 4) 전체 데이터 재학습 & 최종 샘플 분포
# =========================
top_pool, rare_pool, stable_pool, w = build_pools(df, best_hp)

final_candidates = Counter()
SAMPLES = 500  # 필요시 1000 등으로 키워도 됨
for _ in range(SAMPLES):
    c = sample_combo(top_pool, rare_pool, stable_pool, w, best_hp)
    final_candidates[c] += 1

best_sets = [combo for combo, _ in final_candidates.most_common(6)]
print("\n🎯 최종 추천 6세트 (출현 빈도 상위)")
for i, c in enumerate(best_sets, 1):
    print(f"  세트 {i}: {list(map(int, c))}")

# =========================
# 5) 확률적 오차 계산
#    - hist_prob: 전체 회차에서의 번호 등장 확률
#    - obs_prob : 위 샘플링 조합에서의 번호 등장 확률
#    - err = obs - hist, |err| 기준 75퍼센타일 이상 번호만 선택
# =========================
top_hist = 20  # '자주 출현'의 기준 (히스토리 상위 N개 번호)
sample_times = sum(final_candidates.values())
total_slots = sample_times * 6                 # 조합 하나당 6칸

# 5-1) 히스토리 확률
hist_counts = Counter(df[COL_NUMS].stack().astype(int))
total_hist_slots = len(df) * 6
hist_prob = {n: hist_counts.get(n, 0) / total_hist_slots for n in range(1, 46)}

# 5-2) 관측(샘플) 확률
obs_counts = defaultdict(int)
for combo, cnt in final_candidates.items():
    for n in combo:
        obs_counts[int(n)] += cnt
obs_prob = {n: obs_counts.get(n, 0) / total_slots for n in range(1, 46)}

# 5-3) '자주 출현한 번호' 필터
hist_top_nums = [n for n, _ in hist_counts.most_common(top_hist)]

# 5-4) 오차 & 75퍼센타일 임계값
rows = []  # (번호, hist_p, obs_p, err, |err|)
abs_errors = []
for n in hist_top_nums:
    hp = hist_prob.get(n, 0.0)
    op = obs_prob.get(n, 0.0)
    err = op - hp
    ae = abs(err)
    rows.append((n, hp, op, err, ae))
    abs_errors.append(ae)

if len(abs_errors) == 0:
    raise RuntimeError("히스토리 상위 집합이 비어있습니다. top_hist를 확인하세요.")

threshold = float(np.percentile(abs_errors, 75))  # 75% 수준(상위 25%)
rows.sort(key=lambda x: x[4], reverse=True)

selected_nums = [n for (n, hp, op, err, ae) in rows if ae >= threshold]

print("\n📉 '자주 출현'(hist 상위 {}) 중 |오차| Top 10".format(top_hist))
print("번호 | hist_p     | obs_p      | Δ=obs-hist | |Δ|")
for n, hp, op, err, ae in rows[:10]:
    print(f"{n:>3} | {hp:>10.6f} | {op:>10.6f} | {err:>+10.6f} | {ae:>8.6f}")

print("\n🎯 선택 기준: |오차| ≥ 75퍼센타일 (threshold={:.6f})".format(threshold))
print("➡ 선택된 번호(오차 큰 상위 25% 범주, hist 상위 안에서):", sorted(selected_nums))

# =========================
# 6) '오차 큰 번호' 중심의 역발상 세트 생성 (정확히 6개 × 5세트)
# =========================
def sample_counter_combo(selected, weights, prefer=4):
    """
    selected: |오차| 상위 집합
    prefer  : selected에서 우선적으로 뽑을 개수(4 추천)
    """
    chosen = set()
    pool = [x for x in selected]
    k = min(prefer, 6, len(pool))
    if k > 0:
        chosen.update(np.random.choice(pool, k, replace=False))

    # 남은 칸은 가중치 기반으로 보충(중복 금지)
    all_nums = list(range(1, 46))
    w_arr = np.array([weights.get(n, 1.0) for n in all_nums], dtype=float)
    for i, n in enumerate(all_nums):
        if n in chosen:
            w_arr[i] = 0.0
    need = 6 - len(chosen)
    if need > 0 and w_arr.sum() > 0:
        picks = np.random.choice(all_nums, need, replace=False, p=w_arr/w_arr.sum())
        chosen.update(int(x) for x in picks)
    while len(chosen) < 6:
        chosen.add(int(np.random.randint(1, 46)))
    return tuple(sorted(chosen))

TARGET_SETS = 5  # ← 정확히 5세트
if len(selected_nums) > 0:
    print(f"\n🧪 역발상 세트(오차 큰 범주 중심) {TARGET_SETS}종")
    counter_sets = []
    tried = set()
    # 살짝 다양성 확보: 중복 세트 피하려고 시도 횟수 제한
    while len(counter_sets) < TARGET_SETS and len(tried) < 200:
        cc = sample_counter_combo(selected_nums, w, prefer=4)
        if cc not in tried:
            tried.add(cc)
            if cc not in counter_sets:
                counter_sets.append(cc)
    for i, cc in enumerate(counter_sets, 1):
        print(f"  역세트 {i}: {list(cc)}")
else:
    print("\n(참고) 75% 기준을 만족하는 번호가 없어 역발상 세트를 생성하지 않았습니다. top_hist 또는 샘플 수를 조정해 보세요.")
