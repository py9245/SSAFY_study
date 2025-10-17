프로젝트 실행 가이드 (RUNBOOK)

목표

- UCI Credit Card 데이터(`UCI_Credit_Card.csv`)로 부도 여부 분류
- CODEX.md(PRD) 단계에 맞춰 Tier-1(단일모델) → Tier-2(앙상블) → Tier-3(보정/임계치) → 리포트/추론 순서로 수행

사전 준비

- 파이썬 패키지 설치: `pip install -r requirements.txt`
- 데이터 위치: 프로젝트 루트에 `UCI_Credit_Card.csv` 배치(또는 `--data` 인자로 경로 지정)

실행 순서

1) Tier-1: 학습/교차검증/OOF 산출
   - 명령: `python src/train.py --data UCI_Credit_Card.csv`
   - 출력: `outputs/1_tier1/` 아래 OOF 예측/메트릭/LightGBM 모델 파일(
     `oof_preds.csv`, `fold_metrics.json`, `oof_metrics.json`, `lgbm_models/lgbm_fold{1..9}.txt`)

2) Tier-2: 앙상블(가중 평균) 결과
   - 명령: `python src/ensemble.py --data UCI_Credit_Card.csv`
   - 출력: `outputs/2_tier2/`(`ensemble.json`, `oof_preds_ensemble.csv`, `oof_metrics_ensemble.json`)

3) Tier-3: 확률 보정(Isotonic) + 임계치(F1 최적)
   - 명령: `python src/calibrate.py --data UCI_Credit_Card.csv`
   - 출력: `outputs/3_tier3/`(`calibration.pkl`, `thresholds.json`, `calibration_metrics.json`)

4) 리포트 생성(요약 수치)
   - 명령: `python src/report.py`
   - 출력: `outputs/4_report/report.md`

5) 추론(옵션: 보정/임계치 적용)
   - 명령: `python src/infer.py --data UCI_Credit_Card.csv --output outputs/preds.csv --tier1 outputs/1_tier1 --tier3 outputs/3_tier3`
   - 출력: `outputs/preds.csv`(prob/pred 컬럼)

파일별 역할

- `requirements.txt`: 실행에 필요한 패키지 목록
- `README.md`: 요약 설명과 빠른 시작 가이드
- `RUNBOOK.md`: 본 문서(실행 순서 및 파일 역할 상세)
- `src/data.py`: 데이터 로딩(`load_dataset`), 타깃/식별자 분리(`split_xy`)
- `src/features.py`: 경량 특성 생성(util, sum, delay, trend, signed_log1p 등)
- `src/metrics.py`: ROC-AUC/PR-AUC/F1, 임계치 스윕 유틸
- `src/train.py`: 9-Fold CV 학습, OOF/메트릭 저장, LGBM 모델 저장
- `src/ensemble.py`: Tier-1 OOF 기반 베이스라인 앙상블 산출
- `src/calibrate.py`: Isotonic Calibration 적합, 최적 F1 임계치 저장
- `src/report.py`: Tier-1 결과 요약 리포트 생성
- `src/infer.py`: 모델(옵션: 보정/임계치)로 추론 파일 생성

산출물 디렉토리

- `outputs/1_tier1/`: 단일 모델 학습/OOF/메트릭/모델 파일
- `outputs/2_tier2/`: 앙상블 결과/메트릭
- `outputs/3_tier3/`: 보정 모델 및 임계치, 보정 성능
- `outputs/4_report/`: 리포트(수치/그래프 자리)

유용한 옵션(일부)

- `src/train.py`: `--splits`(기본 9), `--seed`, `--class-weight`(예: balanced)
- `src/infer.py`: `--tier1`(모델 경로), `--tier3`(보정/임계치 경로), `--output`

비고

- CODEX.md의 PRD를 기준으로 최소 구현 스켈레톤입니다. Optuna 탐색/다중 시드/추가 모델(HGB/CatBoost)·그래프 자동 저장 등은 확장 가능합니다.

