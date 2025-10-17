# UCI 신용카드 연체예측: 모델 생성 과정 및 결과 상세 리포트

본 문서는 CODEX.md PRD 가이드에 따라 구현/튜닝한 파이프라인의 전 과정을 상세히 기술합니다. 데이터, 전처리/피처, 학습/교차검증, 튜닝, 보정/임계값, 시각화와 산출물 위치까지 모두 포함합니다.

## 1. 데이터 및 목표
- 데이터: `UCI_Credit_Card.csv` (N=30,000, D=25)
- 타깃: `default.payment.next.month` (이진)
- 1차 지표: ROC-AUC, 보조 지표: PR-AUC, F1(best-thr)

## 2. 전처리/피처 엔지니어링
- 기본 컬럼 정리: `ID`, 타깃 제거 후 나머지 피처 사용
- 유틸화/집계
  - `util_recent = BILL_AMT1 / LIMIT_BAL`
  - `util_mean = mean(BILL_AMT1..6) / LIMIT_BAL`
  - `bill_sum = sum(BILL_AMT1..6)`, `pay_sum = sum(PAY_AMT1..6)`
  - `pay_ratio1 = PAY_AMT1 / BILL_AMT1`, `pay_ratio_mean = mean(PAY_AMT1..6) / mean(BILL_AMT1..6)`
- 지연 패턴
  - `delay_cnt_pos`(>0 개수), `delay_max`, `delay_sum_pos`
  - `delay_consecutive_pos_max`(>0 연속 길이의 최댓값)
- 추세/차분
  - `bill_trend = BILL_AMT1 - BILL_AMT6`, `pay_trend = PAY_AMT1 - PAY_AMT6`
  - `bill_diff_i`, `pay_diff_i` (i=1..5)
- 금액 컬럼(signed log1p): `LIMIT_BAL`, BILL_*, PAY_* 각각에 대해 `{col}_slog1p`
- NaN/inf 처리: `replace([inf,-inf], NaN).fillna(0.0)`
- 구현 파일: `src/features.py`

## 3. 학습/교차검증 (Tier‑1)
- 모델: LightGBM(LGBMClassifier)
- 교차검증: Stratified 9‑Fold (seed=42)
- 조기중단: `lgb.early_stopping(200)` 콜백 사용
- 기본 하이퍼파라미터(튜닝 전)
  - `learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=40`
  - `subsample=0.8, subsample_freq=1, colsample_bytree=0.8`
  - `reg_alpha=0.0, reg_lambda=0.0, n_estimators=5000, random_state=42`

### 3.1 튜닝(간단)
- 클래스 불균형 보정: `class_weight=balanced` 적용
- 목적: Recall 개선 및 F1/PR‑AUC 균형 향상
- 실행 명령
  - `python src/train.py --data UCI_Credit_Card.csv --class-weight balanced`

### 3.2 산출물(Tier‑1)
- 경로: `outputs/1_tier1/`
  - `oof_preds.csv`, `oof_metrics.json`, `fold_metrics.json`
  - `lgbm_models/lgbm_fold{1..9}.txt`

## 4. 보정/임계값 (Tier‑3)
- 보정: `CalibratedClassifierCV(method="isotonic", cv=5)` 전체 데이터 재적합
- 베이스: LightGBM(동일 설정, n_estimators=2000)
- 실행 명령
  - `python src/calibrate.py --data UCI_Credit_Card.csv`
- 산출물: `outputs/3_tier3/`
  - `calibration.pkl`, `thresholds.json`, `calibration_metrics.json`

## 5. 시각화/리포트 (Tier‑4)
- 실행 명령
  - `python src/report.py --tier1 outputs/1_tier1 --out outputs/4_report --data UCI_Credit_Card.csv`
- 산출물: `outputs/4_report/`
  - 그림: `roc.png`, `pr.png`, `calibration.png`, `prob_hist.png`, `confusion_matrix.png`
  - 표: `errors.csv`(abs_error 기준 내림차순)
  - 요약: `report.md`

## 6. 결과(튜닝 후: class_weight=balanced)
- `outputs/4_report/report.md` 요약값과 OOF 기반 재계산 값
  - OOF ROC‑AUC: 0.7855
  - OOF PR‑AUC: 0.5548
  - Best F1: 0.5494 @ thr=0.55
- 혼동행렬(@thr=0.55)
  - [[TN=19814, FP=3550], [FN=2778, TP=3858]]
  - TPR/Recall=0.568, FPR=0.152, Precision=0.521 (근사)
- 예측확률 통계
  - 평균/표준편차: 0.4131 / 0.2205 (임계값이 0.55로 이동하며 분포가 넓어짐)
  - 양성비율=0.2212, 예측양성비율=0.2469
- Brier score(OOF): 약 0.135 (참고치)

### 6.1 해석
- ROC‑AUC는 0.7855로 PRD 목표(≥0.80)에 근접, PR‑AUC(0.55)는 양성비율(≈0.22)의 2배 이상으로 양호, F1(0.549)은 목표(≥0.52) 충족
- class_weight 적용으로 임계값이 0.55로 상승하며 고확률 양성의 Precision이 약간 개선, Recall은 유사 수준(≈0.57)
- errors.csv 상위 오분류는 극단적 FN(양성인데 확률이 매우 낮음) 또는 FP(음성인데 확률이 매우 높음) 사례로, 일부 패턴이 여전히 미포착

## 7. 그림/표 안내
- ROC/PR: 클래스 불균형 환경에서 PR‑AUC 0.55는 의미 있는 수준. ROC는 0.78대의 준수한 분리
- Calibration: 보정 전/후 일치 여부 확인(선형성에서 벗어나면 보정된 추론 경로 사용 권장)
- Prob. Histogram: 클래스 간 확률 분포 겹침의 정도 파악 → 겹침이 남아 있어 추가 피처·튜닝 여지 존재
- Confusion Matrix: FPR≈0.15, TPR≈0.57 → 운영상 TPR을 끌어올리려면 임계값 하향 및 알림량 증대 감수 필요
- errors.csv: 상위 500건만 선별 검토 권장(고객군/한도/지연패턴 분포 비교) 

## 8. 개선 제안
- 하이퍼파라미터 탐색(짧은 Optuna)
  - 대상: `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1/l2`, `min_gain_to_split`, `learning_rate`
  - 50~120 trial로 9‑Fold OOF AUC 최소화(=1‑AUC) 목표
- TPR 타깃 임계값 산출
  - TPR=0.70~0.75 만족 최소 임계값 추가 계산/저장(`thresholds.json`) 
- 피처 보강
  - 음의 지연(<=0) 집계: `delay_cnt_neg`, `delay_sum_neg`, `delay_min`
  - 최근 1~3개월 가중평균, 급격한 증감 플래그
- 분석 확장
  - errors 상위 케이스에 대해 주요 피처 분포(예: LIMIT_BAL, util_recent, delay_* 등) 비교 플롯

## 9. 재현 방법 (명령 정리)
```bash
# 1) 학습(Tier-1, 튜닝: 클래스 가중치)
python src/train.py --data UCI_Credit_Card.csv --class-weight balanced

# 2) 보정/임계값(Tier-3)
python src/calibrate.py --data UCI_Credit_Card.csv

# 3) 리포트/시각화(Tier-4)
python src/report.py --tier1 outputs/1_tier1 --out outputs/4_report --data UCI_Credit_Card.csv
```

## 10. 산출물 위치 요약
- 모델: `outputs/1_tier1/lgbm_models/lgbm_fold{1..9}.txt`
- OOF: `outputs/1_tier1/oof_preds.csv`, `oof_metrics.json`, `fold_metrics.json`
- 보정: `outputs/3_tier3/calibration.pkl`, `thresholds.json`, `calibration_metrics.json`
- 리포트: `outputs/4_report/` 내 이미지/테이블/요약 및 본 문서

---
본 리포트는 최신 실행 결과 기준으로 작성되었습니다. 더 강한 튜닝(Optuna)이나 추가 피처를 원하시면 알려주세요. 실험 스펙을 확장해 성능을 0.80+로 끌어올리도록 진행하겠습니다.
