프로젝트 개요

- 데이터: `UCI_Credit_Card.csv`
- 목표: `default.payment.next.month` 이진 분류, 주 지표는 ROC-AUC
- 설계 기준: CODEX.md의 3시간 PRD 가이드에 맞춰 단계별 산출물을 디렉토리로 분리

디렉토리 구조(단계별 산출물)

- `outputs/1_tier1/` — 단일 모델 탐색/교차검증 결과
  - `oof_preds.csv`, `fold_metrics.json`, `oof_metrics.json`, `lgbm_models/lgbm_fold{1..9}.txt`
- `outputs/2_tier2/` — 앙상블(가중 평균) 결과
  - `ensemble.json`, `oof_preds_ensemble.csv`, `oof_metrics_ensemble.json`
- `outputs/3_tier3/` — 확률 보정/임계치 선정 결과
  - `calibration.pkl`, `thresholds.json`
- `outputs/4_report/` — 간단 리포트/그래프(옵션)
  - `report.md`, `roc.png`, `pr.png`

주요 스크립트

- `src/train.py` — 전처리/피처엔지니어링/9-Fold CV 학습 및 산출물 저장
- `src/calibrate.py` — Isotonic Calibration 및 임계치 스윕으로 F1 최적화
- `src/infer.py` — 저장된 모델(+옵션: 보정/임계치)로 추론
- `src/features.py` — CODEX 명세 기반 경량 피처 생성
- `src/metrics.py` — AUC/PR-AUC/F1 및 임계치 스윕

빠른 시작

1) 가상환경에 의존성 설치
   - `pip install -r requirements.txt`
2) 학습(Tier-1)
   - `python src/train.py --data UCI_Credit_Card.csv`
3) 보정/임계치(Tier-3)
   - `python src/calibrate.py --data UCI_Credit_Card.csv`
4) 추론
   - `python src/infer.py --data UCI_Credit_Card.csv --output preds.csv`

비고

- Optuna 하이퍼파라미터 탐색은 옵션입니다(`--optuna-trials`로 제어). 미설치 시 기본 파라미터로 동작합니다.
- 실제 학습/리포트 생성은 실행 환경의 패키지/리소스에 따라 시간이 소요될 수 있습니다.

