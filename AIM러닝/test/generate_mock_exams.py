# -*- coding: utf-8 -*-
"""
모의고사 HTML 생성기
 - 모의고사 5세트 × 50문항(객관식 30, 단답형 15, 서답형 5)
 - 평균 난이도 약 2 수준
 - 정답표 HTML 별도 생성
"""

from __future__ import annotations

import html
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

OUTPUT_DIR = Path("모의ver2")

EXAM_TITLES = [
    "모의고사 1 - 머신러닝 기초",
    "모의고사 2 - 데이터 전처리와 통계",
    "모의고사 3 - 탐색적 데이터 분석",
    "모의고사 4 - 신경망과 딥러닝",
    "모의고사 5 - 순환 신경망과 시계열",
]

EXAM_DESCRIPTIONS = [
    "지도학습과 분류·회귀 전반의 기초 개념을 점검하는 모의고사입니다. 기본 계산 문제와 핵심 개념을 고루 포함했습니다.",
    "데이터 전처리, 통계적 해석, 지표 계산 역량을 확인하기 위한 모의고사입니다. 실무 데이터를 가정한 계산 문제를 다수 포함했습니다.",
    "탐색적 데이터 분석(EDA)과 피처 엔지니어링, 데이터 품질 점검에 초점을 맞춘 모의고사입니다.",
    "심층 신경망, 최적화, 정규화 기법과 실전 적용 전략을 묻는 모의고사입니다.",
    "순환 신경망(RNN), LSTM/GRU, 시계열 분석 기법을 다루는 모의고사입니다.",
]

NUMERIC_SPEC = [
    {"template": "accuracy", "base": 0, "difficulty": 1},
    {"template": "accuracy", "base": 1, "difficulty": 2},
    {"template": "accuracy", "base": 2, "difficulty": 2},
    {"template": "precision", "base": 0, "difficulty": 2},
    {"template": "precision", "base": 1, "difficulty": 2},
    {"template": "precision", "base": 2, "difficulty": 1},
    {"template": "recall", "base": 0, "difficulty": 2},
    {"template": "recall", "base": 1, "difficulty": 2},
    {"template": "recall", "base": 2, "difficulty": 2},
    {"template": "specificity", "base": 0, "difficulty": 2},
    {"template": "f1", "base": 0, "difficulty": 2},
    {"template": "f1", "base": 1, "difficulty": 1},
    {"template": "f1", "base": 2, "difficulty": 1},
    {"template": "entropy", "base": 0, "difficulty": 2},
    {"template": "entropy", "base": 1, "difficulty": 2},
    {"template": "gini", "base": 0, "difficulty": 1},
    {"template": "gini", "base": 1, "difficulty": 2},
    {"template": "gradient", "base": 0, "difficulty": 2},
    {"template": "gradient", "base": 1, "difficulty": 2},
    {"template": "logistic", "base": 0, "difficulty": 2},
]

NUMERIC_CONTEXTS: Dict[int, List[str]] = {
    0: [
        "고객 이탈 예측 모델",
        "의료 진단 보조 모델",
        "신용카드 사기 탐지 모델",
        "온라인 광고 클릭 예측 모델",
        "부정 리뷰 판별 모델",
        "스팸 메일 필터",
        "의료 영상 분류 모델",
        "대출 연체 예측 모델",
        "불량품 탐지 시스템",
        "암 진단 모델",
        "이미지 분류 모델",
        "대출 부도 예측 모델",
        "추천 시스템 클릭 예측",
        "고객 세그먼트 분할",
        "회원 이탈 판단",
        "설문 응답 만족도 분석",
        "보험 청구 이상 탐지",
        "선형 회귀 가중치 업데이트",
        "로지스틱 회귀 경사 업데이트",
        "사용자 이탈 확률 추정",
    ],
    1: [
        "온라인 쇼핑 반품 예측",
        "의료 검사 판독 시스템",
        "제조 설비 고장 감지",
        "구독자 전환 캠페인",
        "보험 사기 탐지 모델",
        "이메일 스팸 분류",
        "회원 유지 캠페인",
        "품질 관리 검사",
        "신용 리스크 평가",
        "건강검진 선별 모델",
        "제품 추천 모델",
        "고객 재방문 예측",
        "이상 거래 탐지",
        "리드 전환 분류",
        "고장 원인 분류",
        "구매 여부 분류",
        "재무 리스크 스코어링",
        "회귀 파라미터 조정",
        "확률적 경사 하강법 업데이트",
        "구매 확률 추정",
    ],
    2: [
        "소매 매출 이상 탐지",
        "제휴사 쿠폰 사용 예측",
        "IoT 센서 고장 감지",
        "고객 만족도 조사 분류",
        "SNS 악성 댓글 판별",
        "문서 분류 시스템",
        "감성 분석 모델",
        "서비스 해지 예측",
        "제품 불량 품질 검사",
        "응급실 선별 모델",
        "사용자 리뷰 감성 분류",
        "VIP 고객 이탈 위험",
        "사기 거래 감시",
        "구간별 매장 성과 분류",
        "채무 불이행 위험 평가",
        "설문 응답 그룹 분석",
        "재고 이상 탐지",
        "배치 경사 하강법 조정",
        "모멘텀 기반 업데이트",
        "신규 상품 구매 확률",
    ],
    3: [
        "이미지 분류 CNN 모델",
        "의료 영상 병변 검출",
        "자율주행 객체 탐지",
        "음성 인식 후처리",
        "텍스트 감성 분류",
        "스팸 탐지 BERT",
        "문장 분류 Transformer",
        "의도 파악 모델",
        "불량품 시각 검사",
        "바이오마커 분류",
        "합성곱 신경망 정확도",
        "전이 학습 결과",
        "자연어 질의응답 평가",
        "다중 레이블 분류",
        "주차 수요 예측",
        "임베딩 품질 평가",
        "강화학습 상태 분류",
        "Adam Optimizer 업데이트",
        "RMSProp 업데이트",
        "딥러닝 출력 확률",
    ],
    4: [
        "시계열 수요 예측 LSTM",
        "음성 감정 분류 모델",
        "주가 이상 탐지 모델",
        "챗봇 발화 의도 분류",
        "뉴스 카테고리 분류",
        "스팸 문자 감지",
        "리뷰 감성 RNN",
        "고객 이탈 시점 예측",
        "장비 고장 시점 분류",
        "환자 상태 경보 시스템",
        "시퀀스 분류 정확도",
        "세션 추천 모델",
        "이상 로그 탐지",
        "재고 예측 분류",
        "신용 점수 변동 탐지",
        "사용 빈도 특성 분석",
        "스트리밍 거래 감시",
        "LSTM 가중치 업데이트",
        "GRU 게이트 조정",
        "다음 이벤트 확률",
    ],
}

BASE_CONFUSION = {
    "accuracy": [
        {"tp": 45, "fp": 5, "fn": 10, "tn": 40},
        {"tp": 68, "fp": 11, "fn": 6, "tn": 75},
        {"tp": 102, "fp": 27, "fn": 18, "tn": 153},
    ],
    "precision": [
        {"tp": 38, "fp": 12, "fn": 9, "tn": 41},
        {"tp": 82, "fp": 20, "fn": 15, "tn": 103},
        {"tp": 56, "fp": 9, "fn": 14, "tn": 71},
    ],
    "recall": [
        {"tp": 60, "fp": 18, "fn": 12, "tn": 90},
        {"tp": 74, "fp": 16, "fn": 20, "tn": 110},
        {"tp": 48, "fp": 8, "fn": 22, "tn": 130},
    ],
    "specificity": [
        {"tp": 84, "fp": 11, "fn": 17, "tn": 132},
    ],
}

BASE_F1 = [
    {"precision": 0.82, "recall": 0.65},
    {"precision": 0.75, "recall": 0.60},
    {"precision": 0.68, "recall": 0.72},
]

BASE_ENTROPY = [
    {"pos": 30, "neg": 70},
    {"pos": 55, "neg": 45},
]

BASE_GINI = [
    {"pos": 65, "neg": 35},
    {"pos": 18, "neg": 42},
]

BASE_GRADIENT = [
    {"weight": 1.20, "learning_rate": 0.05, "gradient": 0.40},
    {"weight": -0.60, "learning_rate": 0.01, "gradient": -1.80},
]

BASE_LOGIT = [0.80]
SHORT_SPEC = [
    {"template": "f1", "base": 0, "difficulty": 2},
    {"template": "weight", "base": 0, "difficulty": 2},
    {"template": "minmax", "base": 0, "difficulty": 1},
    {"template": "pca", "base": 0, "difficulty": 1},
    {"template": "kfold", "base": 0, "difficulty": 1},
    {"template": "sigmoid", "base": 0, "difficulty": 2},
    {"template": "precision", "base": 0, "difficulty": 2},
    {"template": "bootstrap", "base": 0, "difficulty": 2},
    {"template": "accuracy_count", "base": 0, "difficulty": 1},
    {"template": "l2", "base": 0, "difficulty": 2},
    {"template": "mae", "base": 0, "difficulty": 2},
    {"template": "variance", "base": 0, "difficulty": 2},
    {"template": "entropy", "base": 0, "difficulty": 2},
    {"template": "knn", "base": 0, "difficulty": 1},
    {"template": "zscore", "base": 0, "difficulty": 2},
]

SHORT_BASE = {
    "f1": [
        {"precision": 0.80, "recall": 0.50},
        {"precision": 0.74, "recall": 0.62},
        {"precision": 0.68, "recall": 0.71},
    ],
    "weight": [
        {"weight": 1.20, "learning_rate": 0.05, "gradient": 0.40},
        {"weight": -0.45, "learning_rate": 0.02, "gradient": -0.90},
    ],
    "minmax": [
        {"value": 8.0, "min_value": 4.0, "max_value": 10.0},
        {"value": 35.0, "min_value": 20.0, "max_value": 50.0},
    ],
    "pca": [
        {"original_dim": 10, "target_dim": 4},
        {"original_dim": 12, "target_dim": 5},
    ],
    "kfold": [
        {"k": 10},
        {"k": 5},
        {"k": 8},
    ],
    "sigmoid": [
        {"z": 0.70},
        {"z": -0.40},
    ],
    "precision": [
        {"tp": 45, "fp": 5},
        {"tp": 70, "fp": 20},
        {"tp": 32, "fp": 8},
    ],
    "bootstrap": [
        {"dataset_size": 1000},
        {"dataset_size": 750},
        {"dataset_size": 500},
    ],
    "accuracy_count": [
        {"accuracy": 0.92, "total": 250},
        {"accuracy": 0.87, "total": 400},
        {"accuracy": 0.89, "total": 320},
    ],
    "l2": [
        {"lambda": 0.10, "weights": [3.0, 4.0]},
        {"lambda": 0.05, "weights": [0.8, -1.2, 0.5]},
    ],
    "mae": [
        {"predictions": [3, 5, 2], "targets": [2, 7, 2]},
        {"predictions": [15, 20, 13], "targets": [14, 18, 15]},
    ],
    "variance": [
        {"values": [4, 6, 8, 10], "sample": False},
        {"values": [12, 15, 18], "sample": True},
    ],
    "entropy": [
        {"pos": 20, "neg": 80},
        {"pos": 45, "neg": 55},
    ],
    "knn": [
        {"k": 5, "positive": 3},
        {"k": 7, "positive": 4},
    ],
    "zscore": [
        {"value": 58, "mean": 50, "std": 4},
        {"value": 42, "mean": 35, "std": 5},
    ],
}
CONCEPT_DATA: Dict[int, List[Tuple[str, str, str, int, str]]] = {
    0: [
        ("지도학습에서 모델이 관계를 학습하기 위해 필요한 데이터 구성은 무엇인가?",
         "정답 레이블이 없는 데이터만 제공되는 경우|입력 특징과 정답 레이블이 함께 제공되는 데이터|오로지 잡음이 제거된 데이터|동일한 값이 반복되는 데이터",
         "B", 1,
         "지도학습은 입력과 정답 레이블 쌍을 사용해 목표 함수를 학습한다."),
        ("회귀(regression) 문제에서 일반적으로 사용하는 손실 함수 또는 평가 지표는 무엇인가?",
         "정확도(Accuracy)|F1-Score|평균제곱오차(MSE)|ROC-AUC",
         "C", 1,
         "회귀 문제는 연속값 예측이므로 예측 오차의 제곱 평균 등을 활용한다."),
        ("훈련 데이터에 과도하게 적합된 모델이 새로운 데이터에서 보이는 대표적 문제는 무엇인가?",
         "과소적합(Underfitting)|과적합(Overfitting)|정규화(Normalization)|배깅(Bagging)",
         "B", 2,
         "과적합은 훈련 데이터에는 정확하지만 새로운 데이터에서는 오차가 커지는 현상이다."),
        ("배깅(Bagging) 기법의 핵심 아이디어는 무엇인가?",
         "서로 다른 모델을 직렬로 연결한다|동일한 데이터셋을 반복 학습한다|부트스트랩 샘플을 학습해 다수결로 예측을 결합한다|손실 함수를 변경해 모델을 수정한다",
         "C", 2,
         "배깅은 부트스트랩 샘플로 여러 기본 모델을 학습하고 다수결 또는 평균으로 결합한다."),
        ("로지스틱 회귀의 출력 확률 범위는 무엇인가?",
         "(-∞, ∞)|[0, 1]|[-1, 1]|정수 값만 가능하다",
         "B", 1,
         "시그모이드를 사용하므로 출력 확률은 0과 1 사이로 제한된다."),
        ("학습률을 지나치게 크게 설정했을 때 주로 발생하는 문제는 무엇인가?",
         "항상 빠르게 전역 최적값에 수렴한다|최적점 주변에서 발산하거나 진동한다|과적합이 즉시 해결된다|데이터 정규화가 필요 없어진다",
         "B", 2,
         "학습률이 너무 크면 기울기 업데이트가 과격해져 발산하거나 진동하기 쉽다."),
        ("L2 정규화에 대한 설명으로 옳은 것은 무엇인가?",
         "가중치 절대값의 합을 패널티로 추가한다|가중치 제곱합에 패널티를 부여한다|드롭아웃과 동일한 효과를 낸다|데이터 표준화를 수행한다",
         "B", 2,
         "L2 정규화는 가중치 제곱합을 손실에 더해 큰 가중치를 억제한다."),
        ("K-최근접 이웃(KNN) 모델의 가장 중요한 하이퍼파라미터는 무엇인가?",
         "은닉층 개수|학습률|이웃의 수 k|배치 크기",
         "C", 1,
         "KNN은 이웃 수 k에 따라 결정 경계가 바뀌므로 핵심 하이퍼파라미터이다."),
        ("선형 SVM이 선형 분리 불가능한 데이터를 다루기 위해 주로 사용하는 기법은 무엇인가?",
         "드롭아웃|배치 정규화|커널 트릭|데이터 중복 생성",
         "C", 2,
         "커널 트릭을 통해 고차원으로 매핑하면 선형 분리가 가능해진다."),
        ("교차 검증을 수행하는 주된 목적은 무엇인가?",
         "훈련 시간을 줄이기 위해서|모델의 일반화 성능을 안정적으로 추정하기 위해서|데이터 정규화를 대신하기 위해서|손실 함수를 변경하기 위해서",
         "B", 1,
         "교차 검증은 다양한 분할에 대해 검증 성능을 측정해 일반화 능력을 평가한다."),
    ],
    1: [
        ("연속형 변수에 이상치가 많이 존재할 때 결측치를 대체하기 가장 적절한 방법은 무엇인가?",
         "평균으로 대체한다|중앙값으로 대체한다|최빈값으로 대체한다|무작위 값으로 채운다",
         "B", 1,
         "중앙값은 이상치의 영향을 덜 받아 안정적으로 결측치를 대체할 수 있다."),
        ("로그 변환(Log transform)을 적용하기 가장 적절한 데이터 분포는 무엇인가?",
         "완벽한 정규분포|균등분포|강한 양의 편향(오른쪽 꼬리가 긴) 분포|이진 분포",
         "C", 2,
         "로그 변환은 오른쪽으로 치우친 분포를 안정화하고 분산을 줄이는 데 유용하다."),
        ("박스플롯에서 상자 윗변(상자 상단)이 의미하는 값은 무엇인가?",
         "최솟값|중앙값|평균값|3사분위수(Q3)",
         "D", 1,
         "박스플롯의 상자 윗변은 상위 25% 지점을 나타내는 3사분위수이다."),
        ("피어슨 상관계수 r=0.9의 의미로 올바른 것은 무엇인가?",
         "강한 음의 선형 관계|강한 양의 선형 관계|비선형 관계|거의 관계 없음",
         "B", 1,
         "상관계수가 0.9이면 선형적으로 매우 강한 양의 관계가 있음을 의미한다."),
        ("Z-score 표준화의 공식으로 올바른 것은 무엇인가?",
         "(x - 최솟값) / (최댓값 - 최솟값)|(x - 평균) / 표준편차|x / 전체 합계|log(x)",
         "B", 1,
         "Z-score 표준화는 (x-μ)/σ로 데이터를 평균 0, 표준편차 1로 맞춘다."),
        ("카이제곱 독립성 검정을 적용하기 적절한 상황은 무엇인가?",
         "두 범주형 변수의 관련성을 평가할 때|두 집단의 평균 차이를 비교할 때|두 집단의 분산 차이를 비교할 때|데이터가 정규분포인지 확인할 때",
         "A", 2,
         "카이제곱 독립성 검정은 범주형 변수 간의 독립 여부를 판단한다."),
        ("정규성을 시각적으로 점검하기 위한 대표적인 그래프는 무엇인가?",
         "막대그래프|Q-Q Plot|파이차트|워드클라우드",
         "B", 2,
         "Q-Q Plot은 데이터 분위수를 이론적 정규분포와 비교해 정규성을 확인한다."),
        ("IQR(사분위수 범위) 기반 이상치 탐지에서 사용되는 기준은 무엇인가?",
         "Q1 - 0.5*IQR|Q1 - 1.5*IQR|Q1 - 2.5*IQR|Q1 - 3.5*IQR",
         "B", 2,
         "일반적으로 1.5*IQR 범위를 벗어나면 이상치로 간주한다."),
        ("다중공선성을 진단할 때 사용하는 대표적인 지표는 무엇인가?",
         "F1-Score|MAPE|분산팽창계수(VIF)|혼동행렬 정밀도",
         "C", 2,
         "VIF 값이 높을수록 해당 변수와 다른 변수 사이의 공선성이 크다는 뜻이다."),
        ("표준화(Z-score)를 완료한 데이터의 평균과 표준편차는 각각 어떻게 되는가?",
         "평균 0, 표준편차 1|평균 1, 표준편차 1|평균 0, 표준편차 0|평균 1, 표준편차 0",
         "A", 1,
         "Z-score 표준화를 적용하면 평균은 0, 표준편차는 1이 된다."),
    ],
    2: [
        ("산점도 행렬(Pairplot)을 통해 가장 쉽게 확인할 수 있는 정보는 무엇인가?",
         "변수 간 관계와 분포|시간에 따른 추세|모델 손실 감소|하이퍼파라미터 중요도",
         "A", 1,
         "Pairplot은 변수 쌍의 산점도와 대각선상의 분포를 통해 관계를 파악할 수 있다."),
        ("범주형 변수의 비율을 비교하기에 가장 적절한 시각화는 무엇인가?",
         "선 그래프|막대그래프|산점도|히트맵",
         "B", 1,
         "막대그래프는 범주별 빈도나 비율 비교에 적합하다."),
        ("수치형 변수의 분포와 이상치를 동시에 파악할 수 있는 대표적인 도구는 무엇인가?",
         "박스플롯|워드클라우드|파이차트|네트워크 그래프",
         "A", 1,
         "박스플롯은 사분위수와 이상치를 한눈에 보여준다."),
        ("결측치가 변수 간 어떤 패턴으로 나타나는지를 확인하기 위해 가장 먼저 수행하면 좋은 작업은 무엇인가?",
         "결측치 히트맵이나 패턴 분석|하이퍼파라미터 튜닝|모델 앙상블|손실 함수 변경",
         "A", 2,
         "결측치 히트맵 등을 통해 어느 변수에 결측이 집중되는지 파악할 수 있다."),
        ("상관계수가 매우 높은 두 변수를 그대로 회귀 모델에 넣었을 때 주로 발생할 수 있는 문제는 무엇인가?",
         "학습률 증가|회귀계수 불안정|데이터 누수|즉시 과적합",
         "B", 2,
         "높은 상관을 가진 변수는 다중공선성을 유발해 계수 추정이 불안정해진다."),
        ("수치형 변수의 왜도가 크게 오른쪽으로 치우쳐 있을 때 흔히 적용하는 전처리는 무엇인가?",
         "로그 또는 루트 변환|원-핫 인코딩|차원 축소|배치 정규화",
         "A", 2,
         "로그/루트 변환은 오른쪽으로 긴 꼬리를 완화하는 데 도움이 된다."),
        ("EDA 단계에서 파생 변수를 생성하는 주요 목적은 무엇인가?",
         "모델 복잡도 증가|데이터의 양 감소|중요한 관계를 드러내고 설명력을 높인다|정규화를 수행하지 않아도 된다",
         "C", 2,
         "적절한 파생 변수는 모델의 예측력을 높이고 도메인 지식을 반영한다."),
        ("시간 순서가 있는 데이터를 EDA할 때 추세와 계절성을 동시에 확인하기 좋은 방법은 무엇인가?",
         "누적 막대그래프|시계열 분해 그래프|산점도 행렬|K-Means 군집 결과",
         "B", 2,
         "시계열 분해 그래프는 추세와 계절성을 분리해 보여준다."),
        ("EDA 과정에서 데이터 누수(Data Leakage)를 방지하기 위한 올바른 절차는 무엇인가?",
         "훈련/검증 데이터를 분리한 뒤 전처리를 적용한다|전체 데이터를 한 번에 정규화한다|하이퍼파라미터를 먼저 최적화한다|모델을 여러 개 앙상블한다",
         "A", 2,
         "전처리를 포함한 모든 변환은 훈련 데이터 기준으로 학습하고 검증 데이터에 동일하게 적용해야 한다."),
        ("주성분분석(PCA)을 적용하기 전에 선행해야 하는 작업으로 가장 중요한 것은 무엇인가?",
         "특징 스케일을 정규화하거나 표준화한다|드롭아웃 비율을 조정한다|교차 검증 폴드를 선택한다|배치 크기를 늘린다",
         "A", 2,
         "PCA는 공분산 기반 기법이라 스케일이 다른 변수는 반드시 정규화/표준화해야 한다."),
    ],
    3: [
        ("ReLU 활성함수를 사용하는 주요 이유 중 하나는 무엇인가?",
         "출력이 항상 0과 1 사이로 제한된다|기울기 소실을 완화하고 계산이 단순하다|음수 입력을 그대로 전달한다|정규화 과정을 자동으로 수행한다",
         "B", 1,
         "ReLU는 양수 영역에서 기울기를 유지해 기울기 소실을 줄이고 계산도 간단하다."),
        ("심층 신경망에서 기울기 소실(Vanishing Gradient)을 완화하기 위한 대표적인 방법은 무엇인가?",
         "시그모이드 활성함수를 더 많이 사용한다|잔차 연결(Residual Connection)을 추가한다|학습률을 극단적으로 줄인다|가중치를 무작위로 크게 초기화한다",
         "B", 2,
         "Residual 연결이나 LSTM과 같은 구조는 기울기가 전달될 경로를 확보해 기울기 소실을 완화한다."),
        ("Dropout을 적용하는 주된 목적은 무엇인가?",
         "학습 속도를 높이기 위해서|과적합을 줄이기 위해 일부 뉴런을 무작위로 비활성화한다|모델 용량을 크게 늘리기 위해서|손실 함수를 변경하기 위해서",
         "B", 1,
         "Dropout은 무작위로 뉴런을 끄면서 학습해 과적합을 방지하는 정규화 기법이다."),
        ("Batch Normalization을 적용했을 때 기대할 수 있는 효과는 무엇인가?",
         "데이터 정규화를 대신할 수 있다|내부 공변량 변화를 줄여 학습을 안정화한다|가중치 수를 크게 줄인다|학습률을 크게 줄여도 된다",
         "B", 2,
         "Batch Normalization은 각 미니배치의 분포를 정규화해 학습을 안정화한다."),
        ("합성곱 신경망(CNN)에서 커널 크기를 줄이면 주로 어떤 효과가 있는가?",
         "파라미터 수와 계산량이 줄어든다|특징 맵의 공간 해상도가 증가한다|모델이 순환 구조로 변한다|번역 불변성이 사라진다",
         "A", 2,
         "커널이 작아지면 파라미터와 연산량이 감소하고 세밀한 특징을 학습하기 쉽다."),
        ("Adam 옵티마이저는 가중치 업데이트에 어떤 정보를 활용하는가?",
         "현재 기울기만 사용한다|1차 및 2차 모멘트(평균과 분산 추정)를 함께 사용한다|기울기의 부호만 사용한다|훈련 데이터의 표준편차만 사용한다",
         "B", 2,
         "Adam은 1차 모멘트와 2차 모멘트를 추정해 적응형 학습률을 제공한다."),
        ("Softmax 함수는 주로 어떤 상황에서 사용되는가?",
         "회귀 문제에서 연속값을 출력할 때|다중 클래스 분류에서 클래스 확률을 계산할 때|배치 정규화를 대체할 때|옵티마이저를 선택할 때",
         "B", 1,
         "Softmax는 각 클래스에 대한 확률 분포를 출력한다."),
        ("Xavier/He 초기화 기법의 주요 목적은 무엇인가?",
         "모델의 파라미터 수를 줄이기 위해서|기울기 폭발 또는 소실을 줄이기 위해 적절한 분산으로 초기화한다|학습률을 자동으로 조정하기 위해서|배치 크기를 자동으로 결정하기 위해서",
         "B", 2,
         "적절한 초기화는 층을 지나는 신호가 적절한 분산을 유지하도록 도와 기울기 문제를 완화한다."),
        ("Early Stopping을 적용할 때 보통 어떤 값을 모니터링하여 학습을 중단하는가?",
         "훈련 손실|검증 손실 또는 검증 지표|학습률|에폭 수",
         "B", 1,
         "검증 성능이 더 이상 개선되지 않으면 Early Stopping으로 학습을 중단한다."),
        ("Pooling 계층의 주요 역할은 무엇인가?",
         "채널 수를 늘려 모델 용량을 증가시킨다|공간 차원을 축소하고 위치 변화에 대한 불변성을 높인다|학습률을 동적으로 조정한다|정확도를 즉시 높인다",
         "B", 1,
         "Pooling은 특징 맵 크기를 줄이고 위치 변화에 덜 민감하도록 만든다."),
    ],
    4: [
        ("LSTM에서 Forget Gate의 주된 역할은 무엇인가?",
         "이전 상태를 모두 유지한다|이전 상태를 얼마나 잊을지 조절한다|출력 값을 정규화한다|입력 값을 0과 1로 변환한다",
         "B", 2,
         "Forget Gate는 이전 셀 상태 중 유지할 정보와 버릴 정보를 결정한다."),
        ("Teacher Forcing 기법을 가장 잘 설명한 것은 무엇인가?",
         "모델이 스스로 생성한 출력을 다음 입력으로 사용한다|실제 정답 토큰을 다음 시점의 입력으로 공급한다|모델이 학습 데이터를 무작위로 섞는다|손실 함수를 교차 엔트로피로 바꾼다",
         "B", 2,
         "Teacher Forcing은 순환 모델 학습 시 이전 시점의 예측 대신 실제 정답을 입력으로 사용한다."),

        ("Gradient Clipping을 적용하는 주된 이유는 무엇인가?",
         "기울기 소실을 방지하기 위해|기울기 폭발을 제한하기 위해|학습률을 자동으로 조정하기 위해|모델 파라미터 수를 줄이기 위해",
         "B", 2,
         "Gradient Clipping은 기울기의 크기를 제한해 폭발하는 것을 막는다."),
        ("시계열 데이터에서 계절성을 파악하는 데 유용한 분석 방법은 무엇인가?",
         "자동상관함수(ACF) 분석|k-means 군집|원-핫 인코딩|워드클라우드",
         "A", 2,
         "ACF/PACF를 통해 주기적 패턴을 확인할 수 있다."),
        ("시퀀스 모델링에서 Padding Mask의 역할은 무엇인가?",
         "모든 토큰을 동일한 길이로 확장한다|실제 토큰 위치만 학습에 반영하고 패딩을 무시한다|학습률을 조정한다|모델 파라미터를 감소시킨다",
         "B", 1,
         "Padding Mask는 패딩 토큰이 어텐션 계산에 영향을 주지 않도록 한다."),
        ("Seq2Seq 모델에서 어텐션 메커니즘을 사용하는 주요 이점은 무엇인가?",
         "학습 속도를 줄인다|긴 시퀀스에서도 중요한 인코더 상태를 집중적으로 참조할 수 있다|모델 파라미터를 줄인다|드롭아웃을 대체한다",
         "B", 2,
         "어텐션은 각 디코더 단계에서 중요한 인코더 정보를 가중합으로 활용한다."),
        ("Rolling Window(슬라이딩 윈도우) 기법을 사용하는 주된 목적은 무엇인가?",
         "무작위로 데이터를 샘플링하기 위해|최근 구간의 데이터를 이용해 이동하며 모델을 학습하거나 예측하기 위해|정답 레이블을 생성하기 위해|데이터를 정규화하기 위해",
         "B", 2,
         "슬라이딩 윈도우는 일정 길이의 최근 데이터를 사용해 모델을 학습 및 업데이트한다."),
        ("ARIMA 모델을 적용하기 전에 확인해야 하는 중요한 조건은 무엇인가?",
         "데이터가 독립이고 동일 분포인지 여부|시계열이 정상성(Stationarity)을 만족하는지 여부|데이터가 이진 분포인지 여부|정답 레이블이 충분한지 여부",
         "B", 2,
         "ARIMA는 정상성을 가정하므로 차분 등을 통해 정상성 여부를 확인해야 한다."),
        ("시계열 예측 성능을 평가할 때 자주 사용하는 지표는 무엇인가?",
         "BLEU Score|MAPE(평균 절대 백분율 오차)|정확도(Accuracy)|F1-Score",
         "B", 1,
         "MAPE, RMSE 등은 시계열 예측 오차를 평가하는 대표적인 지표이다."),
        ("양방향 RNN(Bidirectional RNN)의 장점으로 옳은 것은 무엇인가?",
         "학습 속도가 항상 빨라진다|과거와 미래 문맥을 모두 활용해 표현력을 높인다|파라미터 수가 줄어든다|교사 강요가 필요 없다",
         "B", 2,
         "양방향 RNN은 정방향과 역방향 정보를 동시에 활용해 문맥을 풍부하게 만든다."),
    ],
}
ESSAY_DATA: Dict[int, List[Tuple[str, str]]] = {
    0: [
        ("훈련과 검증 데이터에서 과적합을 조기에 감지하고 완화하기 위한 방법을 설명하시오. 최소 두 가지 이상의 전략을 제시하고 이유를 쓰시오.",
         "훈련/검증 손실 곡선을 비교해 과적합을 감지하고, 조기 종료, 정규화(L1/L2), 드롭아웃, 데이터 증강 등의 전략을 근거와 함께 서술한다."),
        ("특징 스케일링이 필요한 이유와 Min-Max 정규화, Z-Score 표준화의 장단점을 비교하여 설명하시오.",
         "거리 기반 모델과 경사 하강법의 안정성을 위해 스케일링이 필요한 이유를 제시하고, 두 방식의 적용 시점과 장단점을 비교한다."),
        ("정형 데이터와 비정형 데이터의 차이를 설명하고, 각 데이터 유형에서 자주 사용하는 전처리 방법을 예시와 함께 제시하시오.",
         "정형/비정형 데이터의 구조를 비교하고, 표준화·원핫인코딩·토큰화·데이터 증강 등 전처리 방법을 도메인 사례와 함께 기술한다."),
        ("앙상블 학습이 단일 모델 대비 갖는 장점과 실제 서비스에 적용할 때 고려해야 할 점을 설명하시오.",
         "배깅/부스팅을 통한 분산 감소와 성능 향상 장점을 쓰고, 추론 지연·리소스·운영 복잡성을 고려해야 함을 설명한다."),
        ("머신러닝 프로젝트에서 데이터 준비부터 모델 평가까지의 표준 파이프라인을 단계별로 요약하고, 각 단계에서 주의할 점을 간단히 서술하시오.",
         "데이터 수집, 탐색, 전처리, 특징 엔지니어링, 모델 학습, 튜닝, 평가/배포까지 단계별 핵심 포인트를 정리한다."),
    ],
    1: [
        ("결측치 처리 전략을 세울 때 고려해야 할 요소들을 설명하고, 서로 다른 세 가지 대체 기법을 비교하시오.",
         "결측 패턴, 변수 유형, 비율을 고려해야 함을 설명하고 평균/중앙값 대체, KNN 대체, 다중 대체(MICE) 등의 장단점을 비교한다."),
        ("이상치 탐지와 영향 관측치(Influential Observation)를 구분하고, 이를 다루기 위한 분석 절차를 서술하시오.",
         "이상치와 영향 관측치의 차이를 정의하고 IQR·Z-score·Cook's Distance 등의 절차와 처리 전략을 기술한다."),
        ("정규화(Normalization)와 표준화(Standardization)의 차이와 활용 시점을 비교하시오.",
         "두 방법의 정의와 목적을 구분하고, 거리 기반 모델, 통계 모델, 신경망 등에서의 활용 예시를 제시한다."),
        ("범주형 변수를 인코딩할 때 고려해야 할 사항과 대표 기법을 설명하시오.",
         "희소성, 순서성, 타깃과의 관계를 고려해야 함을 설명하고 원-핫, 타깃, 빈도 인코딩 등의 특성을 비교한다."),
        ("통계적 가설검정 절차를 실제 예시와 함께 단계별로 설명하시오.",
         "귀무/대립가설 설정, 유의수준 선정, 검정 통계량 계산, p-value 해석, 결론 도출까지의 절차를 예시와 함께 정리한다."),
    ],
    2: [
        ("새로운 데이터셋을 접했을 때 수행할 EDA 계획을 단계별로 작성하시오.",
         "데이터 구조 파악, 결측·이상치 탐지, 기초 통계, 변수 간 관계 분석, 시각화 계획을 순차적으로 기술한다."),
        ("다중공선성을 진단하고 해결하기 위한 방법을 설명하시오.",
         "상관행렬, VIF 등의 진단 도구를 언급하고 변수 제거, 차원 축소, 정규화 회귀 등 해결책을 제시한다."),
        ("데이터 시각화를 선택할 때 고려해야 할 요소와 예시를 제시하시오.",
         "변수 유형, 전달 목적, 대상자 수준 등을 고려해야 함을 설명하고 적절한 시각화 예시를 매칭한다."),
        ("데이터 품질을 평가할 때 확인해야 할 체크리스트를 작성하시오.",
         "결측률, 이상치, 중복, 일관성, 최신성 등을 점검해야 함을 정리한다."),
        ("EDA 결과로부터 파생변수나 추가 분석 아이디어를 도출하는 방법을 설명하시오.",
         "EDA에서 발견한 패턴을 기반으로 파생변수를 설계하고 가설을 세우는 방법을 기술한다."),
    ],
    3: [
        ("CNN 기반 이미지 분류 작업에서 모델 구조를 설계할 때 고려해야 할 사항을 설명하시오.",
         "커널 크기, 층 깊이, 풀링, 전이 학습 활용, 데이터 증강 등 구조 선택 기준을 제시한다."),
        ("심층 신경망 학습에서 기울기 소실과 폭발을 완화하기 위한 전략을 비교하시오.",
         "적절한 초기화, 활성함수, 잔차 연결, 정규화 기법 등을 장단점과 함께 설명한다."),
        ("드롭아웃과 배치 정규화를 함께 사용할 때의 주의점을 설명하시오.",
         "적용 순서, 훈련/추론 모드 차이, 하이퍼파라미터 조정에 대해 언급한다."),
        ("옵티마이저 선택 시 고려해야 할 요소를 실제 사례와 함께 설명하시오.",
         "SGD, Adam, RMSProp 등의 특성과 학습률 스케줄링 전략을 비교한다."),
        ("딥러닝 모델의 하이퍼파라미터 튜닝 전략을 제시하시오.",
         "그리드/랜덤/베이지안 탐색, 학습률 탐색, 조기 종료 기반 탐색 전략 등을 서술한다."),
    ],
    4: [
        ("LSTM과 GRU의 구조적 차이와 장단점을 비교하시오.",
         "각 게이트 구성과 파라미터 수 차이를 설명하고, 긴/짧은 시퀀스에서의 장단점을 비교한다."),
        ("시계열 데이터를 학습·검증·테스트로 분할할 때 주의해야 할 점을 설명하시오.",
         "시간 순서를 보존한 분할, 롤링 평가, 누적 학습 전략 등을 다룬다."),
        ("긴 시퀀스를 학습할 때 발생할 수 있는 문제와 해결책을 제시하시오.",
         "기울기 소실과 메모리 한계를 지적하고 어텐션, 트렁케이션, 계층적 모델 등 해결책을 설명한다."),
        ("시계열 예측 모델의 성능을 평가하는 방법을 다양한 지표와 함께 설명하시오.",
         "MAE, RMSE, MAPE, SMAPE 등 지표와 롤링 윈도우 평가 전략을 정리한다."),
        ("제조 설비 이상 감지를 위한 RNN 기반 파이프라인을 설계하고 필요한 데이터를 설명하시오.",
         "센서 시계열 수집, 이상 라벨링, 전처리, 모델 학습(오토인코더/분류기), 경보 기준 등을 제시한다."),
    ],
}

def adjust_confusion(base: Dict[str, int], exam_idx: int, variant: int) -> Dict[str, int]:
    return {
        "tp": base["tp"] + exam_idx * 4 + variant * 2,
        "fp": base["fp"] + exam_idx * 2 + variant,
        "fn": base["fn"] + exam_idx * 2 + (variant % 2),
        "tn": base["tn"] + exam_idx * 4 + variant * 2,
    }


def adjust_prob(value: float, exam_idx: int, variant: int, step: float = 0.02) -> float:
    return max(0.45, min(0.95, value + (exam_idx - 2) * step + variant * 0.01))


def adjust_counts(pos: int, neg: int, exam_idx: int, variant: int) -> Tuple[int, int]:
    return pos + exam_idx * 3 + variant * 2, neg + exam_idx * 3 + variant * 3


def adjust_gradient(base: Dict[str, float], exam_idx: int, variant: int) -> Dict[str, float]:
    return {
        "weight": base["weight"] + (exam_idx - 2) * 0.12 + variant * 0.05,
        "learning_rate": base["learning_rate"] * (1 + 0.08 * exam_idx),
        "gradient": base["gradient"] * (1 + 0.05 * exam_idx),
    }


def adjust_logit(base: float, exam_idx: int, variant: int) -> float:
    return base + (exam_idx - 2) * 0.25 + variant * 0.1


def select_short_params(template: str, base_index: int, exam_idx: int) -> Dict[str, float]:
    items = SHORT_BASE[template]
    data = dict(items[(base_index + exam_idx) % len(items)])
    if template == "weight":
        data["weight"] += (exam_idx - 2) * 0.1
        data["learning_rate"] *= 1 + 0.06 * exam_idx
        data["gradient"] *= 1 + 0.04 * exam_idx
    elif template == "minmax":
        data["value"] += exam_idx * 0.4
        data["min_value"] += exam_idx * 0.4
        data["max_value"] += exam_idx * 0.4
    elif template == "pca":
        data["original_dim"] += exam_idx
        data["target_dim"] = min(data["original_dim"] - 1, data["target_dim"] + exam_idx % 2)
    elif template == "kfold":
        data["k"] = max(3, data["k"] + exam_idx % 3)
    elif template == "sigmoid":
        data["z"] += (exam_idx - 2) * 0.2
    elif template == "precision":
        data["tp"] += exam_idx * 3
        data["fp"] += exam_idx
    elif template == "bootstrap":
        data["dataset_size"] += exam_idx * 80
    elif template == "accuracy_count":
        data["accuracy"] = max(0.6, min(0.97, data["accuracy"] + (exam_idx - 2) * 0.01))
        data["total"] += exam_idx * 40
    elif template == "l2":
        data["lambda"] *= 1 + 0.05 * exam_idx
    elif template == "mae":
        data["predictions"] = [v + exam_idx for v in data["predictions"]]
        data["targets"] = [v + exam_idx for v in data["targets"]]
    elif template == "variance":
        data["values"] = [v + exam_idx for v in data["values"]]
    elif template == "entropy":
        data["pos"] += exam_idx * 5
        data["neg"] += exam_idx * 5
    elif template == "knn":
        data["positive"] = min(data["k"], data["positive"] + exam_idx % 2)
    elif template == "zscore":
        data["value"] += exam_idx * 1.2
        data["mean"] += exam_idx * 1.0
    return data


def format_decimal(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def build_numeric_options(correct: float, offsets: Sequence[float], *, decimals: int = 2, bounds: Tuple[float, float] | None = None) -> Tuple[List[Tuple[str, str]], str]:
    letters = ["A", "B", "C", "D"]
    options: List[Tuple[str, str]] = []
    used: set[str] = set()
    answer_letter = "A"
    for idx, offset in enumerate(offsets):
        value = correct + offset
        if bounds:
            value = max(bounds[0], min(bounds[1], value))
        value = round(value, decimals + 3)
        text = format_decimal(value, decimals)
        while text in used:
            value = round(value + 0.01, decimals + 3)
            text = format_decimal(value, decimals)
        used.add(text)
        options.append((letters[idx], text))
        if offset == 0:
            answer_letter = letters[idx]
    return options, answer_letter


def confusion_html(params: Dict[str, int]) -> str:
    return (
        "<table class=\"confusion\">"
        "<tr><th></th><th>예측 양성</th><th>예측 음성</th></tr>"
        f"<tr><th>실제 양성</th><td>{params['tp']}</td><td>{params['fn']}</td></tr>"
        f"<tr><th>실제 음성</th><td>{params['fp']}</td><td>{params['tn']}</td></tr>"
        "</table>"
    )

def mc_accuracy(context: str, params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    total = params["tp"] + params["fp"] + params["fn"] + params["tn"]
    value = (params["tp"] + params["tn"]) / total
    options, answer = build_numeric_options(value, [0, -0.05, 0.05, -0.1], decimals=2, bounds=(0, 1))
    explanation = f"정확도 = (TP+TN)/전체 = ({params['tp']}+{params['tn']})/{total} = {value:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 모델의 정확도를 소수 둘째 자리까지 구하시오.",
        "extra": confusion_html(params),
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
    }


def mc_precision(context: str, params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    value = params["tp"] / (params["tp"] + params["fp"])
    options, answer = build_numeric_options(value, [0, 0.05, -0.05, 0.1], decimals=2, bounds=(0, 1))
    explanation = f"정밀도 = TP/(TP+FP) = {value:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 모델의 정밀도를 소수 둘째 자리까지 구하시오.",
        "extra": confusion_html(params),
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
    }


def mc_recall(context: str, params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    value = params["tp"] / (params["tp"] + params["fn"])
    options, answer = build_numeric_options(value, [0, 0.05, -0.05, 0.1], decimals=2, bounds=(0, 1))
    explanation = f"재현율 = TP/(TP+FN) = {value:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 모델의 재현율을 소수 둘째 자리까지 구하시오.",
        "extra": confusion_html(params),
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
    }


def mc_specificity(context: str, params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    value = params["tn"] / (params["tn"] + params["fp"])
    options, answer = build_numeric_options(value, [0, 0.05, -0.05, 0.1], decimals=2, bounds=(0, 1))
    explanation = f"특이도 = TN/(TN+FP) = {value:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 모델의 특이도를 소수 둘째 자리까지 구하시오.",
        "extra": confusion_html(params),
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
    }


def mc_f1(context: str, precision: float, recall: float, difficulty: int) -> Dict[str, object]:
    value = 2 * precision * recall / (precision + recall)
    options, answer = build_numeric_options(value, [0, 0.06, -0.06, 0.1], decimals=2, bounds=(0, 1))
    explanation = f"F1 = 2·P·R/(P+R) = {value:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 모델의 정밀도 {precision:.2f}, 재현율 {recall:.2f}일 때 F1-score를 소수 둘째 자리까지 구하시오.",
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
        "note": "소수 둘째 자리까지 반올림하세요.",
    }


def mc_entropy(context: str, pos: int, neg: int, difficulty: int) -> Dict[str, object]:
    total = pos + neg
    p_pos = pos / total
    p_neg = neg / total
    entropy = 0.0
    for p in (p_pos, p_neg):
        if p > 0:
            entropy -= p * math.log2(p)
    options, answer = build_numeric_options(entropy, [0, 0.12, -0.12, 0.2], decimals=2, bounds=(0, 1.5))
    explanation = f"엔트로피 = -Σp log2 p = {entropy:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 분할의 엔트로피를 소수 둘째 자리까지 구하시오.",
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
    }


def mc_gini(context: str, pos: int, neg: int, difficulty: int) -> Dict[str, object]:
    total = pos + neg
    p_pos = pos / total
    p_neg = neg / total
    value = 1 - (p_pos ** 2 + p_neg ** 2)
    options, answer = build_numeric_options(value, [0, 0.1, -0.1, 0.18], decimals=2, bounds=(0, 1))
    explanation = f"지니 불순도 = 1 - Σp^2 = {value:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 분할의 지니(Gini) 불순도를 소수 둘째 자리까지 구하시오.",
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
    }


def mc_gradient(context: str, params: Dict[str, float], difficulty: int) -> Dict[str, object]:
    new_weight = params["weight"] - params["learning_rate"] * params["gradient"]
    options, answer = build_numeric_options(new_weight, [0, 0.15, -0.15, 0.25], decimals=3)
    explanation = f"새 가중치 = {params['weight']:.3f} - {params['learning_rate']:.3f}×{params['gradient']:.3f} = {new_weight:.3f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 모델에서 다음 스텝의 가중치를 소수 셋째 자리까지 구하시오.",
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
        "note": "소수 셋째 자리까지 반올림하세요.",
    }


def mc_logistic(context: str, z_value: float, difficulty: int) -> Dict[str, object]:
    prob = 1 / (1 + math.exp(-z_value))
    options, answer = build_numeric_options(prob, [0, 0.07, -0.07, 0.12], decimals=2, bounds=(0, 1))
    explanation = f"σ(z) = 1/(1+e^-z) = {prob:.2f}"
    return {
        "type": "객관식",
        "section": "객관식",
        "difficulty": difficulty,
        "stem": f"{context} 로지스틱 회귀에서 z={z_value:.2f}일 때 양성 확률을 소수 둘째 자리까지 구하시오.",
        "options": options,
        "answer": answer,
        "answer_detail": explanation,
    }


def short_f1(params: Dict[str, float], difficulty: int) -> Dict[str, object]:
    f1 = 2 * params["precision"] * params["recall"] / (params["precision"] + params["recall"])
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"정밀도 {params['precision']:.2f}, 재현율 {params['recall']:.2f}인 모델의 F1-score를 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(f1, 2),
        "answer_detail": f"F1 = 2PR/(P+R) = {f1:.2f}",
        "note": "소수 둘째 자리까지 반올림하세요.",
    }


def short_weight(params: Dict[str, float], difficulty: int) -> Dict[str, object]:
    new_weight = params["weight"] - params["learning_rate"] * params["gradient"]
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"가중치 {params['weight']:.3f}, 학습률 {params['learning_rate']:.3f}, 기울기 {params['gradient']:.3f}일 때 다음 스텝 가중치를 소수 셋째 자리까지 작성하세요.",
        "answer": format_decimal(new_weight, 3),
        "answer_detail": f"새 가중치 = {new_weight:.3f}",
        "note": "소수 셋째 자리까지 반올림하세요.",
    }


def short_minmax(params: Dict[str, float], difficulty: int) -> Dict[str, object]:
    value = (params["value"] - params["min_value"]) / (params["max_value"] - params["min_value"])
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"값 {params['value']:.2f}를 [{params['min_value']:.2f}, {params['max_value']:.2f}] 범위로 Min-Max 정규화한 값을 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(value, 2),
        "answer_detail": f"(x-min)/(max-min) = {value:.2f}",
    }


def short_pca(params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    removed = params["original_dim"] - params["target_dim"]
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"원본 {params['original_dim']}차원을 {params['target_dim']}차원으로 축소했을 때 제거된 차원의 수를 작성하세요.",
        "answer": str(removed),
        "answer_detail": f"제거된 차원 수 = {removed}",
    }


def short_kfold(params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"{params['k']}-fold 교차검증에서 훈련에 사용되는 폴드 개수를 작성하세요.",
        "answer": str(params["k"] - 1),
        "answer_detail": f"폴드 하나를 검증에 사용하므로 훈련 폴드는 {params['k'] - 1}개입니다.",
    }


def short_sigmoid(params: Dict[str, float], difficulty: int) -> Dict[str, object]:
    prob = 1 / (1 + math.exp(-params["z"]))
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"z = {params['z']:.2f}일 때 시그모이드 확률을 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(prob, 2),
        "answer_detail": f"σ(z) = {prob:.2f}",
    }


def short_precision(params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    precision = params["tp"] / (params["tp"] + params["fp"])
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"TP={params['tp']}, FP={params['fp']}일 때 정밀도를 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(precision, 2),
        "answer_detail": f"정밀도 = {precision:.2f}",
    }


def short_bootstrap(params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    probability = (1 - 1 / params["dataset_size"]) ** params["dataset_size"]
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"샘플 수 {params['dataset_size']}개에서 부트스트랩 시 특정 샘플이 선택되지 않을 확률을 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(probability, 2),
        "answer_detail": f"(1-1/n)^n ≈ {probability:.2f}",
    }


def short_accuracy_count(params: Dict[str, float], difficulty: int) -> Dict[str, object]:
    correct = round(params["accuracy"] * params["total"])  # 가깝게 반올림
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"정확도 {params['accuracy']*100:.1f}%로 {params['total']}개를 분류했을 때 올바르게 분류한 샘플 수를 작성하세요.",
        "answer": str(correct),
        "answer_detail": f"정확 분류 수 = {correct}",
    }


def short_l2(params: Dict[str, object], difficulty: int) -> Dict[str, object]:
    penalty = params["lambda"] * sum(w ** 2 for w in params["weights"])
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"λ={params['lambda']:.3f}, 가중치 {params['weights']}일 때 L2 패널티 값을 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(penalty, 2),
        "answer_detail": f"λ·Σw^2 = {penalty:.2f}",
    }


def short_mae(params: Dict[str, List[int]], difficulty: int) -> Dict[str, object]:
    errors = [abs(p - t) for p, t in zip(params["predictions"], params["targets"])]
    mae = sum(errors) / len(errors)
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"예측값 {params['predictions']}와 실제값 {params['targets']}의 MAE를 작성하세요.",
        "answer": format_decimal(mae, 2),
        "answer_detail": f"MAE = {mae:.2f}",
    }


def short_variance(params: Dict[str, object], difficulty: int) -> Dict[str, object]:
    values = params["values"]
    mean = sum(values) / len(values)
    if params.get("sample", False):
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        label = "표본 분산"
    else:
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        label = "모집단 분산"
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"값 {values}의 {label}을 작성하세요.",
        "answer": format_decimal(variance, 2),
        "answer_detail": f"{label} = {variance:.2f}",
    }


def short_entropy(params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    total = params["pos"] + params["neg"]
    p_pos = params["pos"] / total
    p_neg = params["neg"] / total
    entropy = 0.0
    for p in (p_pos, p_neg):
        if p > 0:
            entropy -= p * math.log2(p)
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"양성 {params['pos']}개, 음성 {params['neg']}개의 엔트로피를 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(entropy, 2),
        "answer_detail": f"엔트로피 = {entropy:.2f}",
    }


def short_knn(params: Dict[str, int], difficulty: int) -> Dict[str, object]:
    result = "양성" if params["positive"] > params["k"] / 2 else "음성"
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"k={params['k']}인 KNN에서 양성 투표가 {params['positive']}표일 때 최종 분류를 작성하세요.",
        "answer": result,
        "answer_detail": f"양성 득표수가 {params['positive']}이므로 결과는 {result}입니다.",
    }


def short_zscore(params: Dict[str, float], difficulty: int) -> Dict[str, object]:
    zscore = (params["value"] - params["mean"]) / params["std"]
    return {
        "type": "단답형",
        "section": "단답형",
        "difficulty": difficulty,
        "stem": f"평균 {params['mean']:.1f}, 표준편차 {params['std']:.1f}에서 값 {params['value']:.1f}의 Z-score를 소수 둘째 자리까지 작성하세요.",
        "answer": format_decimal(zscore, 2),
        "answer_detail": f"Z = (x-μ)/σ = {zscore:.2f}",
    }

NUMERIC_BUILDERS = {
    "accuracy": mc_accuracy,
    "precision": mc_precision,
    "recall": mc_recall,
    "specificity": mc_specificity,
    "f1": mc_f1,
    "entropy": mc_entropy,
    "gini": mc_gini,
    "gradient": mc_gradient,
    "logistic": mc_logistic,
}

SHORT_BUILDERS = {
    "f1": short_f1,
    "weight": short_weight,
    "minmax": short_minmax,
    "pca": short_pca,
    "kfold": short_kfold,
    "sigmoid": short_sigmoid,
    "precision": short_precision,
    "bootstrap": short_bootstrap,
    "accuracy_count": short_accuracy_count,
    "l2": short_l2,
    "mae": short_mae,
    "variance": short_variance,
    "entropy": short_entropy,
    "knn": short_knn,
    "zscore": short_zscore,
}


def build_concept_questions(exam_idx: int) -> List[Dict[str, object]]:
    questions = []
    for stem, option_str, answer, difficulty, explanation in CONCEPT_DATA[exam_idx]:
        options = [(chr(65 + i), text) for i, text in enumerate(option_str.split("|"))]
        questions.append({
            "type": "객관식",
            "section": "객관식",
            "difficulty": difficulty,
            "stem": stem,
            "options": options,
            "answer": answer,
            "answer_detail": explanation,
        })
    return questions


def build_numeric_questions(exam_idx: int) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    for idx, spec in enumerate(NUMERIC_SPEC):
        template = spec["template"]
        difficulty = spec["difficulty"]
        base_idx = spec["base"]
        context = NUMERIC_CONTEXTS[exam_idx][idx]
        if template in ("accuracy", "precision", "recall", "specificity"):
            base = BASE_CONFUSION[template if template != "specificity" else "accuracy"][base_idx]
            params = adjust_confusion(base, exam_idx, base_idx)
            questions.append(NUMERIC_BUILDERS[template](context, params, difficulty))
        elif template == "f1":
            base = BASE_F1[base_idx]
            precision = adjust_prob(base["precision"], exam_idx, base_idx, 0.02)
            recall = adjust_prob(base["recall"], exam_idx, base_idx, 0.015)
            questions.append(mc_f1(context, precision, recall, difficulty))
        elif template == "entropy":
            base = BASE_ENTROPY[base_idx]
            pos, neg = adjust_counts(base["pos"], base["neg"], exam_idx, base_idx)
            questions.append(mc_entropy(context, pos, neg, difficulty))
        elif template == "gini":
            base = BASE_GINI[base_idx]
            pos, neg = adjust_counts(base["pos"], base["neg"], exam_idx, base_idx)
            questions.append(mc_gini(context, pos, neg, difficulty))
        elif template == "gradient":
            base = BASE_GRADIENT[base_idx]
            params = adjust_gradient(base, exam_idx, base_idx)
            questions.append(mc_gradient(context, params, difficulty))
        elif template == "logistic":
            z_value = adjust_logit(BASE_LOGIT[base_idx], exam_idx, base_idx)
            questions.append(mc_logistic(context, z_value, difficulty))
    return questions


def build_short_questions(exam_idx: int) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    for spec in SHORT_SPEC:
        params = select_short_params(spec["template"], spec["base"], exam_idx)
        builder = SHORT_BUILDERS[spec["template"]]
        questions.append(builder(params, spec["difficulty"]))
    return questions


def build_essay_questions(exam_idx: int) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    for stem, guide in ESSAY_DATA[exam_idx]:
        questions.append({
            "type": "서답형",
            "section": "서답형",
            "difficulty": 3,
            "stem": stem,
            "answer": "서술형",
            "answer_detail": guide,
        })
    return questions


def generate_exam(exam_idx: int) -> Dict[str, object]:
    exam_id = exam_idx + 1
    title = EXAM_TITLES[exam_idx]
    description = EXAM_DESCRIPTIONS[exam_idx]

    numeric_questions = build_numeric_questions(exam_idx)
    concept_questions = build_concept_questions(exam_idx)
    short_questions = build_short_questions(exam_idx)
    essay_questions = build_essay_questions(exam_idx)

    questions = numeric_questions + concept_questions + short_questions + essay_questions
    answers_for_table: List[Dict[str, object]] = []
    for number, question in enumerate(questions, start=1):
        question["number"] = number
        answers_for_table.append({
            "exam_id": exam_id,
            "exam_title": title,
            "number": number,
            "type": question["type"],
            "difficulty": question["difficulty"],
            "answer": question.get("answer", ""),
            "detail": question.get("answer_detail", ""),
        })

    avg_difficulty = sum(q["difficulty"] for q in questions) / len(questions)

    return {
        "id": exam_id,
        "title": title,
        "description": description,
        "questions": questions,
        "avg_difficulty": avg_difficulty,
        "answers_for_table": answers_for_table,
    }

EXAM_STYLE = """
body { font-family: 'Segoe UI', 'Noto Sans KR', sans-serif; margin: 32px; background:#f6f8fb; color:#222; line-height:1.6; }
h1 { font-size: 1.9rem; margin-bottom: 0.3rem; }
h2 { margin-top: 2.2rem; border-bottom: 2px solid #d0d7e3; padding-bottom: 0.4rem; }
section.meta { background:#fff; border:1px solid #d0d7e3; border-radius:10px; padding:1.2rem; margin-bottom:1.6rem; }
.question { background:#fff; border:1px solid #d7deeb; border-left:6px solid #4a6fa5; padding:1.1rem; border-radius:8px; margin-bottom:1.1rem; }
.question-meta { font-size:0.85rem; color:#52627a; display:flex; gap:0.8rem; flex-wrap:wrap; margin-bottom:0.5rem; }
ol.options { margin-top:0.5rem; padding-left:1.6rem; }
ol.options li { margin-bottom:0.45rem; }
table.confusion { border-collapse:collapse; margin-top:0.6rem; }
table.confusion th, table.confusion td { border:1px solid #aeb9cf; padding:0.45rem 0.8rem; text-align:center; background:#fff; }
.note { margin-top:0.5rem; font-size:0.85rem; color:#5a6a82; }
.answer-line, .essay-area { margin-top:0.8rem; border:1px dashed #90a1bc; background:#fdfdff; min-height:2.4rem; padding:0.6rem; }
.essay-area { min-height:6rem; }
"""

ANSWER_STYLE = """
body { font-family:'Segoe UI','Noto Sans KR',sans-serif; margin:32px; background:#f5f7fb; color:#233046; }
h1 { font-size:1.8rem; margin-bottom:1rem; }
table { width:100%; border-collapse:collapse; background:#fff; }
thead { background:#4a6fa5; color:#fff; }
th, td { border:1px solid #d0d7e3; padding:0.6rem 0.75rem; text-align:left; vertical-align:top; }
tbody tr:nth-child(even) { background:#f2f5fb; }
.badge { display:inline-block; padding:0.1rem 0.45rem; border-radius:12px; background:#e3e9ff; color:#2d3f73; font-size:0.78rem; margin-right:0.35rem; }
"""


def render_question(question: Dict[str, object]) -> str:
    parts = [f"<div class='question' id='q{question['number']}'>"]
    parts.append(
        f"<div class='question-meta'><span>문제 {question['number']}</span><span>{question['type']}</span><span>난이도 {question['difficulty']}</span></div>"
    )
    parts.append(f"<p>{html.escape(question['stem'])}</p>")
    if question.get("extra"):
        parts.append(question["extra"])
    if question["type"] == "객관식":
        parts.append("<ol class='options' type='A'>")
        for letter, text in question["options"]:
            parts.append(f"<li><strong>{letter}.</strong> {html.escape(text)}</li>")
        parts.append("</ol>")
    elif question["type"] == "단답형":
        parts.append("<div class='answer-line'></div>")
    else:
        parts.append("<div class='essay-area'></div>")
    if question.get("note"):
        parts.append(f"<div class='note'>{html.escape(question['note'])}</div>")
    parts.append("</div>")
    return "".join(parts)


def render_exam_html(exam: Dict[str, object], path: Path) -> None:
    questions = exam["questions"]
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='ko'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(exam['title'])}</title>",
        f"<style>{EXAM_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(exam['title'])}</h1>",
        "<section class='meta'>",
        f"<p>{html.escape(exam['description'])}</p>",
        f"<p>총 {len(questions)}문항 · 평균 난이도 {exam['avg_difficulty']:.2f}</p>",
        "</section>",
    ]
    current_section = None
    counts = Counter(q["section"] for q in questions)
    for question in questions:
        if question["section"] != current_section:
            current_section = question["section"]
            html_parts.append(f"<h2>{current_section} ({counts[current_section]}문항)</h2>")
        html_parts.append(render_question(question))
    html_parts.append("</body></html>")
    path.write_text("\n".join(html_parts), encoding="utf-8")


def render_answers_html(rows: List[Dict[str, object]], path: Path) -> None:
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='ko'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>모의고사 정답표</title>",
        f"<style>{ANSWER_STYLE}</style>",
        "</head>",
        "<body>",
        "<h1>모의고사 정답표</h1>",
        "<table>",
        "<thead><tr><th>모의고사</th><th>번호</th><th>유형</th><th>난이도</th><th>정답 및 기준</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        detail = html.escape(str(row.get("detail", "")))
        answer = html.escape(str(row.get("answer", "")))
        html_parts.append(
            "<tr>"
            f"<td>{html.escape(row['exam_title'])}</td>"
            f"<td>{row['number']}</td>"
            f"<td>{row['type']}</td>"
            f"<td>{row['difficulty']}</td>"
            f"<td><span class='badge'>정답</span>{answer}" + (f"<br>{detail}" if detail else "") + "</td>"
            "</tr>"
        )
    html_parts.extend(["</tbody>", "</table>", "</body></html>"])
    path.write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    answers: List[Dict[str, object]] = []
    for exam_idx in range(len(EXAM_TITLES)):
        exam = generate_exam(exam_idx)
        render_exam_html(exam, OUTPUT_DIR / f"모의고사{exam['id']}.html")
        answers.extend(exam["answers_for_table"])
        print(f"{exam['title']} 생성 완료 · 평균 난이도 {exam['avg_difficulty']:.2f}")
    render_answers_html(answers, OUTPUT_DIR / "모의고사_정답.html")
    print("정답표 생성 완료")


if __name__ == "__main__":
    main()
