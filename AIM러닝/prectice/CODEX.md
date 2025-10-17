좋아, 시간 예산을 **3시간**으로 늘렸으니 아예 “이길 확률”을 올리는 방향으로 PRD를 확장할게.
핵심은 **LightGBM 중심 + 짧은 탐색 → 강한 앙상블 → 확률보정/임계값 튜닝**이야.

---

# UCI 신용카드 연체예측 PRD (학습 예산 3시간)

## 1) 목표/지표

* **목표**: `default.payment.next.month` 이진 분류.
* **주 지표**: **ROC-AUC**.
* **보조 지표**: **PR-AUC**, **F1**(임계값 최적화 후). 운영 컷은 F1 또는 원하는 재현율(TPR)에서의 FPR도 보고.

---

## 2) 데이터/전처리(동일 + 약간 확장)

* 사용 컬럼: `ID` 제외 전부. 결측 없음.
* **파생특성(경량·누수 없음)**

  * 사용률/집계: `util_recent = BILL_AMT1/LIMIT_BAL`, `util_mean`, `pay_ratio1`, `pay_ratio_mean`, `bill_sum`, `pay_sum`.
  * 연체 패턴: `delay_cnt_pos(>0 개수)`, `delay_max`, `delay_sum(>0 합)`, **연속 연체 길이 max**.
  * 추세: `bill_trend = BILL_AMT1 - BILL_AMT6`, `pay_trend = PAY_AMT1 - PAY_AMT6`, `bill_diff_i = BILL_AMT{i} - BILL_AMT{i+1}`(i=1..5).
  * 금액열(`LIMIT_BAL`, BILL_*, PAY_*)은 **signed_log1p** 버전도 추가.
* **불균형 보정**: `class_weight="balanced"` 또는 샘플가중치.
* **정규화/원-핫 불필요**(트리계열). 정수형 상태값은 순서형 그대로 사용.

---

## 3) 검증 설계

* **Stratified 9-Fold CV**(seed=42) — 예산 증가로 분산을 더 낮춤.
* Fold별 확률 → **임계값 그리드(0.05~0.95, 0.01 간격)**로 **F1 최대화**.
* 보고: Fold AUC/PR-AUC/F1, **OOF** AUC/PR-AUC, 최종 임계값=Fold 최적값 평균.

---

## 4) 모델 전략(3시간 버전)

### Tier-1: 단일 모델 탐색 (≈ 90–110분)

* **LightGBM**(주력, CPU OK) — **Optuna**로 베이지안 탐색.

  * 고정: `objective=binary`, `metric=auc`, `learning_rate=0.03~0.07`(탐색), `early_stopping_rounds=200`, `n_estimators=5000`, `verbose=-1`.
  * 탐색 공간(조건부):

    * `num_leaves: 31–255`
    * `max_depth: -1, 6–12`
    * `min_data_in_leaf: 20–200`
    * `feature_fraction: 0.6–1.0`, `bagging_fraction: 0.6–1.0`, `bagging_freq: 0–5`
    * `lambda_l1: 0–5`, `lambda_l2: 0–10`
    * `min_gain_to_split: 0–5`
  * **Trial 예산**: 120 trial 목표(학습속도에 따라 100~160). **연속 30 trial 개선 無**면 조기종료.
* **HistGradientBoostingClassifier(HGB)** — 백업/비교(그리드 소폭).
* (옵션) **CatBoost**(CPU로도 충분히 빠름): `depth, l2_leaf_reg, learning_rate` 짧은 탐색.

### Tier-2: 앙상블 (≈ 30–40분)

* 상위 **LightGBM 2~3개**(seed/하이퍼 상이) + **HGB 1개**를 **소프트 보팅**(가중 평균).
* 가중치는 **OOF AUC 비례 정규화**로 초기 설정 → 간단한 1-차원 탐색로 미세 조정.

### Tier-3: 확률 보정 & 임계값 (≈ 20–30분)

* **CalibratedClassifierCV(isotonic)**를 OOF 기반으로 적합(최상 모델 1개 대상).
* **두 가지 컷** 산출:

  1. **F1 최대 컷**(균형 성능)
  2. **TPR=0.75**를 만족하는 최소 컷(리스크 감지 강화 운영용)

### Tier-4: 해석/리포트 (≈ 15–20분)

* LightGBM **gain 중요도 Top-20**, SHAP summary(시간 남으면).
* 혼동행렬/리프트/Precision-Recall 곡선.

---

## 5) 체크리스트 적용(강제 수행)

1. **초기 손실**: 로지스틱 CE 초기가 `≈log(2)` 근처인지 확인(라벨/로딩 점검).
2. **소샘플 과적합**: N=2,000으로 100% 가까이 맞추며 손실이 실제 줄어드는지 확인(학습률/버그 점검).
3. **학습률 스윕**: 1e-1, 5e-2, 1e-2, 5e-3, 1e-3 중 **실제로 줄어드는 구간** 확보.
4. **좋은 조합만 확장**: 위에서 잡힌 LR에서만 깊이/잎/규제 조합을 넓힘.
5. **학습 확장**: 10~20 에폭 상당(조기중단 기준)으로 충분히 학습.
6. **곡선 모니터**: Train/Val 손실, AUC 이동평균. 폭주→LR↓, 과적합→`num_leaves↓, min_data_in_leaf↑, lambda↑, bagging/feature_fraction↓`.

---

## 6) 수락 기준(권장)

* **OOF ROC-AUC ≥ 0.80**
* **OOF PR-AUC ≥ 2.0 × 양성비율**
* **F1@best_thr ≥ 0.52** (데이터 분포에 따라 조정)

---

## 7) 산출물

* `train.py`(전처리→파생→Optuna→CV→앙상블/캘리브레이션→저장)
* `infer.py`(CSV→확률/예측/운영컷 지원)
* `models/`: `lgbm_top{1..3}.txt`, `hgb.pkl`, `ensemble.json`, `thresholds.json`, `calibration.pkl`
* `report.md` + 그림들(PR/ROC/리프트/중요도)
* `requirements.txt` 또는 `environment.yml`

---

## 8) 리스크/대응

* **시간 초과**: Optuna trial 수 자동 감축(조기종료 규칙).
* **과적합**: `num_leaves↓`, `min_data_in_leaf↑`, `lambda↑`, `bagging/feature_fraction↓`.
* **데이터 누수**: 파생은 동일 월 스냅샷 내 연산만 사용(미래 정보 금지).
* **확률 일관성**: CalibratedCV(iso)로 안정화.

---

## 9) 3시간 실행 타임라인(예시)

* 0:00–0:10  데이터 로드, 체크리스트 ①②(초기손실/소샘플)
* 0:10–0:25  LR 스윕(③), 파생특성 확정
* 0:25–1:50  **Optuna-LGBM 100~140 trial**(9-Fold, 조기중단)
* 1:50–2:15  HGB 짧은 탐색 + 상위 LGBM 2~3개 재학습
* 2:15–2:35  소프트보팅 앙상블, 임계값 튜닝(두 가지 컷)
* 2:35–2:55  CalibratedCV(iso) + 리포트 그래프
* 2:55–3:00  산출물 저장/검수

---

## 10) 실행용 스켈레톤(요지)

```python
# 핵심 파츠만: Optuna + 9Fold LGBM
import optuna, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd

df = pd.read_csv("UCI_Credit_Card.csv")
y = df["default.payment.next.month"].astype(int).to_numpy()
X = df.drop(columns=["ID","default.payment.next.month"])
# TODO: 위 PRD의 파생특성 함수 add_features() 붙이기
X = add_features(X).to_numpy(dtype=np.float32)

skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)

def cv_auc(params):
    params = {
        "objective":"binary", "metric":"auc",
        "learning_rate": params["lr"],
        "num_leaves": int(params["leaves"]),
        "max_depth": int(params["depth"]),
        "min_data_in_leaf": int(params["min_child"]),
        "feature_fraction": params["ff"],
        "bagging_fraction": params["bf"],
        "bagging_freq": int(params["bff"]),
        "lambda_l1": params["l1"], "lambda_l2": params["l2"],
        "min_gain_to_split": params["mingain"],
        "verbosity": -1, "seed": 42,
    }
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va])
        gbm = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dva], valid_names=["val"],
            early_stopping_rounds=200, verbose_eval=False
        )
        oof[va] = gbm.predict(X[va])
    return 1.0 - roc_auc_score(y, oof)  # Optuna는 minimize

study = optuna.create_study(direction="minimize")
study.optimize(lambda t: cv_auc({
    "lr": t.suggest_float("lr", 0.03, 0.07),
    "leaves": t.suggest_int("leaves", 31, 255, step=8),
    "depth": t.suggest_int("depth", 6, 12),
    "min_child": t.suggest_int("min_child", 20, 200),
    "ff": t.suggest_float("ff", 0.6, 1.0),
    "bf": t.suggest_float("bf", 0.6, 1.0),
    "bff": t.suggest_int("bff", 0, 5),
    "l1": t.suggest_float("l1", 0.0, 5.0),
    "l2": t.suggest_float("l2", 0.0, 10.0),
    "mingain": t.suggest_float("mingain", 0.0, 5.0),
}), n_trials=120, timeout=None)  # 남은 시간에 맞춰 조절
print("Best trial:", study.best_value, study.best_params)
```

원하면 위 스켈레톤을 **완성 코드(파생 함수/임계값 튜닝/앙상블/캘리브레이션/리포트까지 포함)**로 바로 묶어서 줄게.
