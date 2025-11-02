#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모의고사 HTML 생성 스크립트
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Dict, List


def generate_python_mc(exam_no: int) -> List[Dict]:
    max_val = 6 + exam_no
    step = 2 + (exam_no % 3)
    dataset_size = 50 + exam_no * 5
    mc = []
    mc.append(
        {
            "section": "Python 기초",
            "type": "multiple_choice",
            "prompt": f"리스트 컴프리헨션으로 1부터 {max_val}까지의 정수 중 짝수만 제곱한 리스트를 만드는 올바른 코드는?",
            "options": [
                f"A. [x**2 for x in range(1, {max_val}) if x % 2 == 0]",
                f"B. [x**2 for x in range(1, {max_val + 1}) if x % 2 == 0]",
                f"C. [x**2 for x in range({max_val + 1}) if x % 2 == 0]",
                f"D. [x**2 if x % 2 == 0 for x in range(1, {max_val + 1})]",
            ],
            "answer": "B",
            "explanation": "range(1, n+1)으로 끝값을 포함하고 조건을 뒤에 적는다.",
        }
    )
    list_sample = [i for i in range(1, 6)]
    mc.append(
        {
            "section": "Python 기초",
            "type": "multiple_choice",
            "prompt": f"다음 리스트 {list_sample}에서 {step}번째 원소부터 슬라이싱한 결과는?",
            "options": [
                f"A. {list_sample[step-1:]}",
                f"B. {list_sample[:step]}",
                f"C. {list_sample[::step]}",
                f"D. {list_sample[step:]}",
            ],
            "answer": "A",
            "explanation": "슬라이싱은 0 기반 인덱스이므로 step-1부터 끝까지 반환한다.",
        }
    )
    mc.append(
        {
            "section": "Python 기초",
            "type": "multiple_choice",
            "prompt": f"NumPy에서 shape가 ({step}, {step+1})인 배열 A와 shape가 ({step+1}, {step})인 배열 B를 행렬곱할 때 결과 shape는?",
            "options": [
                f"A. ({step}, {step})",
                f"B. ({step+1}, {step+1})",
                f"C. ({step+1}, {step})",
                f"D. ({step}, {step+1})",
            ],
            "answer": "A",
            "explanation": "행렬곱 결과 shape는 (m, n) x (n, p) -> (m, p)이다.",
        }
    )
    vector = list(range(1, step + 1))
    dot = sum(i * i for i in vector)
    mc.append(
        {
            "section": "Python 기초",
            "type": "multiple_choice",
            "prompt": f"벡터 v = {vector}의 자기 내적(np.dot(v, v)) 값은?",
            "options": [
                f"A. {dot - 2}",
                f"B. {dot - 1}",
                f"C. {dot}",
                f"D. {dot + 1}",
            ],
            "answer": "C",
            "explanation": "자기 내적은 각 원소 제곱의 합이다.",
        }
    )
    mc.append(
        {
            "section": "Python 기초",
            "type": "multiple_choice",
            "prompt": f"numpy.arange(0, {dataset_size}, {step})로 생성한 배열의 길이는?",
            "options": [
                f"A. {dataset_size // step - 1}",
                f"B. {dataset_size // step}",
                f"C. {dataset_size // step + 1}",
                f"D. {dataset_size // step + 2}",
            ],
            "answer": "B",
            "explanation": "arange는 stop을 포함하지 않으므로 총 원소 수는 stop/step이다.",
        }
    )
    mc.append(
        {
            "section": "Python 기초",
            "type": "multiple_choice",
            "prompt": "다음 중 불변(immutable) 자료형만을 고른 것은?",
            "options": [
                "A. list, tuple",
                "B. tuple, str",
                "C. dict, tuple",
                "D. set, tuple",
            ],
            "answer": "B",
            "explanation": "tuple과 str은 불변 자료형이다.",
        }
    )
    mc.append(
        {
            "section": "Python 기초",
            "type": "multiple_choice",
            "prompt": "파이썬에서 enumerate() 함수의 기본 동작 설명으로 옳은 것은?",
            "options": [
                "A. 인덱스 1부터 시작하여 요소와 함께 반환한다",
                "B. 인덱스 0부터 시작하여 (인덱스, 값) 튜플을 반환한다",
                "C. 리스트를 사전으로 변환한다",
                "D. 반복 가능한 객체를 역순으로 정렬한다",
            ],
            "answer": "B",
            "explanation": "enumerate(iterable)는 기본으로 0부터 인덱스를 부여한다.",
        }
    )
    return mc


def generate_python_short(exam_no: int) -> List[Dict]:
    end = 4 + exam_no
    reshape_rows = 2 + (exam_no % 2)
    total = sum(range(1, end + 1))
    return [
        {
            "section": "Python 기초",
            "type": "short_answer",
            "prompt": f"리스트 컴프리헨션 [n for n in range(1, {end + 1})]의 합은?",
            "answer": str(total),
            "explanation": "등차수열 합 공식 n(n+1)/2.",
        },
        {
            "section": "Python 기초",
            "type": "short_answer",
            "prompt": f"NumPy 배열 np.arange(0, {reshape_rows * 4}).reshape({reshape_rows}, -1)의 열(column) 수는?",
            "answer": str((reshape_rows * 4) // reshape_rows),
            "explanation": "총 원소 수를 행 수로 나누면 열 수가 된다.",
        },
    ]


def generate_lr_mc(exam_no: int) -> List[Dict]:
    n = 80 + exam_no * 10
    lr = 0.01 * (1 + exam_no / 10)
    mc = []
    mc.append(
        {
            "section": "선형 회귀",
            "type": "multiple_choice",
            "prompt": "평균제곱오차(MSE)에 대한 설명으로 옳은 것은?",
            "options": [
                "A. 절댓값 오차의 평균을 의미한다",
                "B. 오차 제곱의 평균으로 이상치에 민감하다",
                "C. 분류 문제에서 사용하는 전형적 손실이다",
                "D. 학습률을 자동 조절하는 방법이다",
            ],
            "answer": "B",
            "explanation": "MSE는 제곱 오차 평균으로 이상치에 민감하다.",
        }
    )
    mc.append(
        {
            "section": "선형 회귀",
            "type": "multiple_choice",
            "prompt": f"샘플 {n}개, 특성 3개인 데이터를 LinearRegression으로 학습할 때 편향(bias) 포함 파라미터 개수는?",
            "options": ["A. 3", "B. 4", "C. 6", "D. 12"],
            "answer": "B",
            "explanation": "가중치 3개와 편향 1개가 필요하다.",
        }
    )
    mc.append(
        {
            "section": "선형 회귀",
            "type": "multiple_choice",
            "prompt": f"경사하강법에서 학습률을 {lr:.3f}로 설정했더니 발산한다. 가장 우선 고려할 조치는?",
            "options": [
                "A. 학습률을 더 크게 설정한다",
                "B. 학습률을 더 작게 설정한다",
                "C. 에폭 수를 줄인다",
                "D. 드롭아웃을 추가한다",
            ],
            "answer": "B",
            "explanation": "발산 시에는 학습률을 낮추어 안정화한다.",
        }
    )
    mc.append(
        {
            "section": "선형 회귀",
            "type": "multiple_choice",
            "prompt": "L2 정규화(Ridge)의 기대 효과는?",
            "options": [
                "A. 희소한 해를 얻는다",
                "B. 큰 가중치를 억제해 과적합을 완화한다",
                "C. 비선형성을 모델링한다",
                "D. 데이터 누락을 보완한다",
            ],
            "answer": "B",
            "explanation": "L2 패널티는 큰 가중치를 억제해 과적합을 줄인다.",
        }
    )
    mc.append(
        {
            "section": "선형 회귀",
            "type": "multiple_choice",
            "prompt": "다중공선성(multicollinearity)의 주요 문제점은?",
            "options": [
                "A. 데이터 차원이 줄어든다",
                "B. 가중치 추정이 불안정해진다",
                "C. 데이터가 희소해진다",
                "D. 손실 함수가 볼록하지 않게 된다",
            ],
            "answer": "B",
            "explanation": "높은 상관으로 계수 추정이 불안정해진다.",
        }
    )
    mc.append(
        {
            "section": "선형 회귀",
            "type": "multiple_choice",
            "prompt": "결정계수(R^2)에 대한 설명으로 옳은 것은?",
            "options": [
                "A. 값이 1에 가까울수록 설명력이 높다",
                "B. 항상 0과 1 사이의 값을 갖는다",
                "C. 음수가 될 수 없다",
                "D. 분류 정확도를 의미한다",
            ],
            "answer": "A",
            "explanation": "R^2은 1에 가까울수록 좋은 설명력을 가진다.",
        }
    )
    mc.append(
        {
            "section": "선형 회귀",
            "type": "multiple_choice",
            "prompt": "표준화(Standardization)의 주된 목적은?",
            "options": [
                "A. 이상치를 제거한다",
                "B. 모델을 선형으로 만든다",
                "C. 특성 스케일 차이를 줄여 학습을 안정화한다",
                "D. 데이터 개수를 증가시킨다",
            ],
            "answer": "C",
            "explanation": "스케일을 맞추면 가중치 학습이 안정적이다.",
        }
    )
    return mc


def generate_lr_short(exam_no: int) -> List[Dict]:
    x_val = 2 + exam_no
    y_true = 10 + 1.2 * exam_no
    y_hat = 12 + exam_no
    gradient = round(2 * (y_hat - y_true) * x_val, 2)
    return [
        {
            "section": "선형 회귀",
            "type": "short_answer",
            "prompt": f"단일 샘플에 대해 예측값 {y_hat:.1f}, 실제값 {y_true:.1f}, 입력 특성 값 {x_val}일 때 MSE 기준 가중치 w에 대한 경사 2*(y_hat - y)*x 값은?",
            "answer": f"{gradient}",
            "explanation": "단일 샘플 MSE 경사는 2*(y_hat - y)*x.",
        },
        {
            "section": "선형 회귀",
            "type": "short_answer",
            "prompt": "train_test_split(data, test_size=0.2)로 분할하면 학습 세트 비율은?",
            "answer": "0.8",
            "explanation": "학습 비율은 1 - test_size이다.",
        },
    ]


def generate_eda_mc(exam_no: int) -> List[Dict]:
    corr_threshold = 0.7 + exam_no * 0.02
    bins = 10 + exam_no
    mc = []
    mc.append(
        {
            "section": "EDA",
            "type": "multiple_choice",
            "prompt": "상관계수 heatmap을 통해 확인할 수 있는 것은?",
            "options": [
                "A. 변수 사이 선형 관계의 방향과 강도",
                "B. 모델의 학습률",
                "C. 데이터의 정규분포 여부",
                "D. 결측치 개수",
            ],
            "answer": "A",
            "explanation": "상관계수는 변수 간 선형 관계를 보여준다.",
        }
    )
    mc.append(
        {
            "section": "EDA",
            "type": "multiple_choice",
            "prompt": "박스플롯(box plot)이 제공하지 않는 정보는?",
            "options": [
                "A. 중앙값",
                "B. 사분위 범위",
                "C. 이상치 후보",
                "D. 변수간 상관관계",
            ],
            "answer": "D",
            "explanation": "박스플롯은 단일 변수 분포만 보여준다.",
        }
    )
    mc.append(
        {
            "section": "EDA",
            "type": "multiple_choice",
            "prompt": f"히스토그램 bin 수를 {bins}으로 설정하는 주된 이유는?",
            "options": [
                "A. bin 수가 많을수록 좋기 때문이다",
                "B. 데이터 규모와 분포를 고려해 적절한 구간으로 나누기 위해서다",
                "C. 이상치를 제거하기 위해서다",
                "D. 정규화를 자동으로 수행하기 위해서다",
            ],
            "answer": "B",
            "explanation": "bin 수는 분포 파악을 위한 설계 요소다.",
        }
    )
    mc.append(
        {
            "section": "EDA",
            "type": "multiple_choice",
            "prompt": "pairplot을 사용했을 때 얻을 수 있는 이점은?",
            "options": [
                "A. 각 변수의 시간 추세를 본다",
                "B. 변수 쌍의 산점도와 분포를 동시에 확인한다",
                "C. 문자열 변수 자동 인코딩을 수행한다",
                "D. 모델 정확도를 계산한다",
            ],
            "answer": "B",
            "explanation": "pairplot은 산점도 행렬과 변수별 분포를 제공한다.",
        }
    )
    mc.append(
        {
            "section": "EDA",
            "type": "multiple_choice",
            "prompt": f"상관계수 절대값이 {corr_threshold:.2f} 이상인 변수쌍을 제거하는 목적은?",
            "options": [
                "A. 데이터 차원을 늘리기 위해서다",
                "B. 다중공선성을 완화하기 위해서다",
                "C. 결측치를 대체하기 위해서다",
                "D. 정규분포를 만들기 위해서다",
            ],
            "answer": "B",
            "explanation": "높은 상관 변수 제거는 다중공선성을 완화한다.",
        }
    )
    mc.append(
        {
            "section": "EDA",
            "type": "multiple_choice",
            "prompt": "결측치 처리 방법 중 평균 대체(mean imputation)의 단점은?",
            "options": [
                "A. 계산량이 많다",
                "B. 분산을 과소추정할 수 있다",
                "C. 이상치를 늘린다",
                "D. 범주형 변수에만 적용 가능하다",
            ],
            "answer": "B",
            "explanation": "평균 대체는 분산을 줄여 데이터 변동성을 왜곡할 수 있다.",
        }
    )
    mc.append(
        {
            "section": "EDA",
            "type": "multiple_choice",
            "prompt": "Seaborn jointplot으로 시각화할 수 있는 것은?",
            "options": [
                "A. 3차원 산점도",
                "B. 두 변수의 산점도와 주변 분포",
                "C. 연속형 변수의 누적 분포",
                "D. 카테고리형 변수의 트리맵",
            ],
            "answer": "B",
            "explanation": "jointplot은 산점도와 히스토그램을 결합해 보여준다.",
        }
    )
    return mc


def generate_eda_short(exam_no: int) -> List[Dict]:
    missing = 5 + exam_no
    total = 200 + exam_no * 20
    rate = round(missing / total, 3)
    value = 50 + exam_no * 3
    mean_val = 40 + exam_no * 2
    std = 5 + exam_no
    z_score = round((value - mean_val) / std, 3)
    return [
        {
            "section": "EDA",
            "type": "short_answer",
            "prompt": f"총 {total}개 관측치 중 결측치 {missing}개의 결측률은? (소수 셋째 자리)",
            "answer": f"{rate}",
            "explanation": "결측률은 missing / total로 계산한다.",
        },
        {
            "section": "EDA",
            "type": "short_answer",
            "prompt": f"값 {value}, 평균 {mean_val}, 표준편차 {std}일 때 z-score는?",
            "answer": f"{z_score}",
            "explanation": "z = (x - mean) / std.",
        },
    ]


def generate_mlp_mc(exam_no: int) -> List[Dict]:
    hidden = 2 + exam_no
    dropout = 0.1 * (exam_no % 3 + 2)
    mc = []
    mc.append(
        {
            "section": "MLP",
            "type": "multiple_choice",
            "prompt": "MLP에서 ReLU 활성화 함수를 사용하는 주된 이유는?",
            "options": [
                "A. 모든 입력에서 미분값이 0이어서 계산이 단순하다",
                "B. 기울기 소실 문제를 완화한다",
                "C. 출력이 항상 0과 1 사이가 된다",
                "D. 확률 분포를 바로 반환한다",
            ],
            "answer": "B",
            "explanation": "ReLU는 양수 구간에서 기울기를 유지한다.",
        }
    )
    mc.append(
        {
            "section": "MLP",
            "type": "multiple_choice",
            "prompt": f"은닉층이 {hidden}층인 MLP에서 파라미터 수가 급격히 증가하는 이유는?",
            "options": [
                "A. 편향 항이 없어지기 때문",
                "B. 층마다 완전연결 가중치 행렬이 추가되기 때문",
                "C. 손실 함수가 비선형이기 때문",
                "D. 학습률이 작기 때문",
            ],
            "answer": "B",
            "explanation": "완전연결층은 입력과 출력을 모두 연결하므로 가중치가 많다.",
        }
    )
    mc.append(
        {
            "section": "MLP",
            "type": "multiple_choice",
            "prompt": "역전파(backpropagation)의 핵심 아이디어는?",
            "options": [
                "A. 출력을 정규화한다",
                "B. 손실 미분을 체인룰로 각 층에 전달한다",
                "C. 입력 데이터를 무작위로 섞는다",
                "D. 모델을 얕게 만든다",
            ],
            "answer": "B",
            "explanation": "체인룰을 사용해 기울기를 전파한다.",
        }
    )
    mc.append(
        {
            "section": "MLP",
            "type": "multiple_choice",
            "prompt": "소프트맥스(softmax) 함수의 특징으로 옳은 것은?",
            "options": [
                "A. 각 원소에 독립적으로 적용되어 합이 1보다 크다",
                "B. 출력 합이 1이 되는 확률 분포를 만든다",
                "C. 음수 입력을 모두 0으로 만든다",
                "D. 활성화 없이도 적용 불필요하다",
            ],
            "answer": "B",
            "explanation": "softmax는 확률 분포를 반환한다.",
        }
    )
    mc.append(
        {
            "section": "MLP",
            "type": "multiple_choice",
            "prompt": f"드롭아웃(dropout) 비율을 {dropout:.1f}로 설정하는 목적은?",
            "options": [
                "A. 항상 같은 뉴런만 활성화하기 위해서다",
                "B. 훈련 시 일부 뉴런을 무작위 비활성화해 과적합을 줄인다",
                "C. 학습률을 조절한다",
                "D. 가중치를 0으로 초기화한다",
            ],
            "answer": "B",
            "explanation": "드롭아웃은 무작위 비활성화로 일반화 성능을 높인다.",
        }
    )
    mc.append(
        {
            "section": "MLP",
            "type": "multiple_choice",
            "prompt": "배치 정규화(batch normalization)의 효과가 아닌 것은?",
            "options": [
                "A. 학습 안정화",
                "B. 내부 공변량 변화 감소",
                "C. 과적합 완화에 기여",
                "D. 모델 깊이를 줄인다",
            ],
            "answer": "D",
            "explanation": "배치 정규화는 층 깊이를 줄이지 않는다.",
        }
    )
    mc.append(
        {
            "section": "MLP",
            "type": "multiple_choice",
            "prompt": "조기 종료(early stopping)의 동작 원리는?",
            "options": [
                "A. 학습률이 0이 되면 중단한다",
                "B. 검증 손실이 개선되지 않으면 학습을 멈춘다",
                "C. 모든 가중치가 0이 되면 중단한다",
                "D. 에폭 수를 무한히 늘린다",
            ],
            "answer": "B",
            "explanation": "검증 손실이 나빠지면 학습을 중단한다.",
        }
    )
    return mc


def generate_mlp_short(exam_no: int) -> List[Dict]:
    fan_in = 32 + exam_no * 4
    he_std = round((2 / fan_in) ** 0.5, 4)
    input_dim = 10 + exam_no
    hidden_units = 8 + exam_no
    params = (input_dim * hidden_units) + hidden_units
    return [
        {
            "section": "MLP",
            "type": "short_answer",
            "prompt": f"ReLU용 He 초기화에서 fan_in={fan_in}일 때 표준편차 sqrt(2/fan_in) 값은? (소수 넷째 자리)",
            "answer": f"{he_std}",
            "explanation": "He 초기화 표준편차는 sqrt(2/fan_in)이다.",
        },
        {
            "section": "MLP",
            "type": "short_answer",
            "prompt": f"입력 차원 {input_dim}, 은닉 유닛 {hidden_units}개인 완전연결층의 파라미터(가중치+편향) 수는?",
            "answer": f"{params}",
            "explanation": "가중치는 input*hidden, 편향은 hidden이다.",
        },
    ]


def generate_nlp_mc(exam_no: int) -> List[Dict]:
    vocab = 30000 + exam_no * 500
    embed_dim = 128 + 16 * exam_no
    time_steps = 10 + exam_no
    mc = []
    mc.append(
        {
            "section": "NLP & 순환신경망",
            "type": "multiple_choice",
            "prompt": "WordPiece 토큰화의 특징으로 옳은 것은?",
            "options": [
                "A. 항상 완전한 단어 단위로만 분해한다",
                "B. 자주 등장하는 서브워드를 우선 포함한다",
                "C. 문장 길이를 고정하는 패딩 기법이다",
                "D. 띄어쓰기를 제거한다",
            ],
            "answer": "B",
            "explanation": "WordPiece는 빈도 기반 서브워드를 학습한다.",
        }
    )
    mc.append(
        {
            "section": "NLP & 순환신경망",
            "type": "multiple_choice",
            "prompt": f"허깅페이스 토크나이저에서 vocab size를 {vocab}으로 설정할 때 고려해야 할 사항은?",
            "options": [
                "A. vocab이 작을수록 표현력이 증가한다",
                "B. vocab이 클수록 메모리와 연산 비용이 증가한다",
                "C. vocab은 항상 32000으로 고정된다",
                "D. vocab은 모델 구조와 무관하다",
            ],
            "answer": "B",
            "explanation": "큰 vocab은 메모리와 연산 비용을 증가시킨다.",
        }
    )
    mc.append(
        {
            "section": "NLP & 순환신경망",
            "type": "multiple_choice",
            "prompt": f"임베딩 차원을 {embed_dim}으로 설정했을 때 장점은?",
            "options": [
                "A. 항상 모델이 과적합된다",
                "B. 적절한 표현력과 효율의 균형을 잡을 수 있다",
                "C. 파라미터 수가 입력 길이에 따라 감소한다",
                "D. 토큰이 사전순으로 정렬된다",
            ],
            "answer": "B",
            "explanation": "임베딩 차원은 표현력과 효율 사이에서 균형을 맞춘다.",
        }
    )
    mc.append(
        {
            "section": "NLP & 순환신경망",
            "type": "multiple_choice",
            "prompt": "순환신경망(RNN)에서 기울기 소실 문제가 발생하는 주된 원인은?",
            "options": [
                "A. 활성화 함수가 선형이기 때문이다",
                "B. 긴 시퀀스를 따라 미분을 반복하며 값이 0에 수렴하기 때문이다",
                "C. 출력층이 없기 때문이다",
                "D. 가중치를 업데이트하지 않기 때문이다",
            ],
            "answer": "B",
            "explanation": "길어진 시퀀스에서 기울기가 0에 수렴한다.",
        }
    )
    mc.append(
        {
            "section": "NLP & 순환신경망",
            "type": "multiple_choice",
            "prompt": "LSTM에서 입력 게이트의 역할은?",
            "options": [
                "A. 이전 셀 상태를 잊게 한다",
                "B. 후보 메모리를 얼마나 저장할지 결정한다",
                "C. 출력을 스케일링한다",
                "D. 시간축을 뒤집는다",
            ],
            "answer": "B",
            "explanation": "입력 게이트는 후보 상태를 얼마나 반영할지 결정한다.",
        }
    )
    mc.append(
        {
            "section": "NLP & 순환신경망",
            "type": "multiple_choice",
            "prompt": f"시퀀스 길이를 {time_steps}으로 패딩할 때 주의할 점은?",
            "options": [
                "A. 패딩 토큰도 손실 계산에 포함되면 성능이 저하될 수 있다",
                "B. 패딩을 사용하면 역전파가 불가능하다",
                "C. 패딩은 GPU에서만 작동한다",
                "D. 패딩은 단어 순서를 바꾼다",
            ],
            "answer": "A",
            "explanation": "패딩 토큰은 손실 계산에서 제외되도록 마스킹해야 한다.",
        }
    )
    mc.append(
        {
            "section": "NLP & 순환신경망",
            "type": "multiple_choice",
            "prompt": "교사 강요(teacher forcing)의 단점은?",
            "options": [
                "A. 항상 학습 속도가 느려진다",
                "B. 추론 시 이전 예측에 덜 적응할 수 있다",
                "C. 모델이 순전파를 못 한다",
                "D. 손실 함수가 정의되지 않는다",
            ],
            "answer": "B",
            "explanation": "teacher forcing은 노출 편향을 유발할 수 있다.",
        }
    )
    return mc


def generate_nlp_short(exam_no: int) -> List[Dict]:
    seq_len = 12 + exam_no
    target_len = 16 + exam_no
    padding_needed = target_len - seq_len
    forget_gate = round(0.6 + 0.05 * exam_no, 2)
    prev_cell = round(1.2 + 0.3 * exam_no, 1)
    input_gate = round(0.5 + 0.04 * exam_no, 2)
    candidate = round(0.3 + 0.05 * exam_no, 2)
    new_cell = round(forget_gate * prev_cell + input_gate * candidate, 3)
    return [
        {
            "section": "NLP & 순환신경망",
            "type": "short_answer",
            "prompt": f"입력 시퀀스 길이 {seq_len}을 최대 길이 {target_len}으로 패딩할 때 필요한 패딩 토큰 수는?",
            "answer": f"{padding_needed}",
            "explanation": "필요한 패딩 수는 target_len - seq_len.",
        },
        {
            "section": "NLP & 순환신경망",
            "type": "short_answer",
            "prompt": f"LSTM에서 forget 게이트 {forget_gate}, 이전 셀 상태 {prev_cell}, input 게이트 {input_gate}, 후보 상태 {candidate}일 때 새로운 셀 상태 값은? (소수 셋째 자리)",
            "answer": f"{new_cell}",
            "explanation": "c_t = f_t * c_{t-1} + i_t * g_t.",
        },
    ]


def generate_descriptive(exam_no: int) -> List[Dict]:
    dataset_name = ["스마트홈 센서", "온라인 쇼핑", "헬스케어 웨어러블", "금융 거래", "자율주행 로그"][exam_no - 1]
    features = 12 + exam_no * 2
    seq_len = 50 + exam_no * 10
    dropout = 0.2 + 0.05 * (exam_no - 1)
    vocab = 20000 + exam_no * 4000
    descriptive = []
    descriptive.append(
        {
            "section": "서술형",
            "type": "descriptive",
            "prompt": f"[{dataset_name}] 데이터({features}개 특성)를 선형 회귀로 예측하려 한다. 데이터 확보부터 모델 평가까지 단계별 전략을 상세히 기술하라. (전처리, EDA, 모델 학습, 검증 포함)",
            "answer": "예시 답안: 데이터 수집 및 검증 -> 결측치/이상치 처리 -> 특징 스케일링 -> 훈련/검증 분할 -> 선형 회귀 학습 -> 평가 지표 보고 및 개선 방안 제시.",
            "explanation": "각 단계의 목적과 실행 방안을 포함해야 한다.",
        }
    )
    descriptive.append(
        {
            "section": "서술형",
            "type": "descriptive",
            "prompt": f"EDA 단계에서 {features}개 특성을 대상으로 상관분석과 시각화를 수행할 때 사용할 도구와 절차를 설명하고, 결과로 확인해야 할 핵심 항목을 열거하라.",
            "answer": "예시 답안: pairplot/heatmap 활용, 상관계수 기준 특징 선택, 이상치/분포 확인, 시각화별 해석.",
            "explanation": "EDA의 목적과 도구 선택 근거를 제시해야 한다.",
        }
    )
    descriptive.append(
        {
            "section": "서술형",
            "type": "descriptive",
            "prompt": f"MLP로 {dataset_name} 데이터를 학습할 때 은닉층 구성, 활성화 함수, 정규화 기법(dropout 약 {dropout:.2f})을 포함한 모델 설계안을 제시하고 이유를 설명하라.",
            "answer": f"예시 답안: 입력 -> 은닉층 2~3개(ReLU), 드롭아웃 {dropout:.2f}, 배치 정규화, 최적화 기법 선택 근거.",
            "explanation": "모델 설계 선택을 데이터 특성과 연결해야 한다.",
        }
    )
    descriptive.append(
        {
            "section": "서술형",
            "type": "descriptive",
            "prompt": f"텍스트 길이 약 {seq_len} 토큰, vocab {vocab}인 감성 분석 문제를 RNN/LSTM으로 해결할 때 토크나이징, 임베딩, 학습 전략을 포함한 전체 파이프라인을 설계하라.",
            "answer": "예시 답안: 서브워드 토크나이저 선택, 패딩/마스킹 처리, 임베딩 층, LSTM/GRU 구성, 정규화 및 검증 전략 제시.",
            "explanation": "NLP 파이프라인의 단계별 설정을 구체화해야 한다.",
        }
    )
    descriptive.append(
        {
            "section": "서술형",
            "type": "descriptive",
            "prompt": "모델 성능을 향상시키기 위한 실험 설계안을 제시하라. (하이퍼파라미터 튜닝, 교차검증, 특징 공학 등 최소 3가지 전략 포함)",
            "answer": "예시 답안: 학습률/은닉 유닛 탐색, 교차검증, 데이터 증강, 조기 종료, 정규화 기법 비교 등.",
            "explanation": "향상 전략을 근거와 함께 기술해야 한다.",
        }
    )
    return descriptive


def build_exam(exam_no: int) -> Dict:
    questions: List[Dict] = []
    questions += generate_python_mc(exam_no)
    questions += generate_python_short(exam_no)
    questions += generate_lr_mc(exam_no)
    questions += generate_lr_short(exam_no)
    questions += generate_eda_mc(exam_no)
    questions += generate_eda_short(exam_no)
    questions += generate_mlp_mc(exam_no)
    questions += generate_mlp_short(exam_no)
    questions += generate_nlp_mc(exam_no)
    questions += generate_nlp_short(exam_no)
    questions += generate_descriptive(exam_no)
    for idx, question in enumerate(questions, start=1):
        question["number"] = idx
    return {
        "exam_no": exam_no,
        "title": f"모의고사 {exam_no}",
        "questions": questions,
    }


def render_exam_html(exam: Dict) -> str:
    head = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: 'Segoe UI', 'Nanum Gothic', sans-serif; margin: 24px; background: #f6f7fb; color: #222; }}
    header {{ text-align: center; margin-bottom: 32px; }}
    h1 {{ margin-bottom: 8px; }}
    .info {{ font-size: 0.95rem; color: #555; }}
    .question-list {{ list-style: none; padding: 0; margin: 0; }}
    .question {{ background: #fff; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(15, 34, 58, 0.12); }}
    .q-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
    .q-number {{ font-weight: 700; font-size: 1.1rem; color: #2c4a78; }}
    .q-tags {{ font-size: 0.85rem; color: #446089; }}
    .prompt {{ margin-bottom: 12px; line-height: 1.6; }}
    .options {{ margin: 0; padding-left: 18px; }}
    .options li {{ margin-bottom: 6px; }}
    .options input[type="radio"] {{ margin-right: 6px; transform: scale(1.1); }}
    textarea.answer-area {{ margin-top: 12px; border: 1px dashed #b6c2d9; border-radius: 8px; background: #fdfefe; width: 100%; box-sizing: border-box; padding: 10px; font-size: 0.95rem; color: #1f2d3d; resize: vertical; }}
    textarea.answer-area.short {{ height: 60px; }}
    textarea.answer-area.descriptive {{ height: 160px; }}
  </style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="info">
    총 50문항 (객관식·주관식 45문항, 서술형 5문항) | 평균 난이도 2/5
  </div>
  <div class="info">
    응시 시간 권장: 90분 | 필요시 페이지를 출력해 활용하세요.
  </div>
</header>
<ol class="question-list">
""".format(
        title=html.escape(exam["title"])
    )
    items: List[str] = []
    for q in exam["questions"]:
        question_type = q["type"]
        section = q["section"]
        prompt = html.escape(q["prompt"])
        tags = f"[{section}] {('객관식' if question_type == 'multiple_choice' else '주관식' if question_type == 'short_answer' else '서술형')}"
        block = [
            '  <li class="question">',
            '    <div class="q-header">',
            f'      <span class="q-number">Q{q["number"]:02d}</span>',
            f'      <span class="q-tags">{html.escape(tags)}</span>',
            "    </div>",
            f'    <div class="prompt">{prompt}</div>',
        ]
        if question_type == "multiple_choice":
            block.append('    <ul class="options">')
            for option in q["options"]:
                option_key = option.split(".", 1)[0].strip()
                option_value = html.escape(option_key)
                option_text = html.escape(option)
                block.append(
                    f'      <li><label><input type="radio" name="q{q["number"]}" value="{option_value}"> {option_text}</label></li>'
                )
            block.append("    </ul>")
        else:
            css_class = "descriptive" if question_type == "descriptive" else "short"
            block.append(f'    <textarea class="answer-area {css_class}" placeholder="답을 입력하세요"></textarea>')
        block.append("  </li>")
        items.append("\n".join(block))
    tail = """</ol>
</body>
</html>
"""
    return head + "\n".join(items) + "\n" + tail


def render_answers_html(exams: List[Dict]) -> str:
    head = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>모의고사 정답 모음</title>
  <style>
    body { font-family: 'Segoe UI', 'Nanum Gothic', sans-serif; margin: 24px; background: #fefefe; color: #1f2d3d; }
    h1 { text-align: center; margin-bottom: 32px; }
    h2 { margin-top: 32px; color: #274472; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.92rem; }
    th, td { border: 1px solid #d5dee9; padding: 8px 10px; vertical-align: top; }
    th { background: #f0f5fc; }
    td.answer { width: 120px; font-weight: 600; color: #1a4d8f; }
    td.explanation { color: #44556b; }
  </style>
</head>
<body>
  <h1>모의고사 정답 및 해설</h1>
  <p>각 문항의 정답과 핵심 해설을 정리했습니다. 서술형 문항은 포인트 위주 예시 답안을 제공합니다.</p>
"""
    sections: List[str] = []
    for exam in exams:
        rows: List[str] = []
        rows.append(
            "    <tr><th>번호</th><th>유형</th><th>정답</th><th>해설/채점 포인트</th></tr>"
        )
        for q in exam["questions"]:
            qtype = (
                "객관식"
                if q["type"] == "multiple_choice"
                else "주관식"
                if q["type"] == "short_answer"
                else "서술형"
            )
            answer = q.get("answer", "")
            explanation = q.get("explanation", "")
            rows.append(
                "    <tr>"
                f"<td>{q['number']:02d}</td>"
                f"<td>{html.escape(q['section'])} / {qtype}</td>"
                f"<td class=\"answer\">{html.escape(answer)}</td>"
                f"<td class=\"explanation\">{html.escape(explanation)}</td>"
                "</tr>"
            )
        table_html = "\n".join(rows)
        sections.append(
            f"  <h2>{html.escape(exam['title'])}</h2>\n"
            "  <table>\n"
            f"{table_html}\n"
            "  </table>"
        )
    tail = """
</body>
</html>
"""
    return head + "\n\n".join(sections) + tail


def main() -> None:
    output_dir = Path("모의ver1")
    output_dir.mkdir(parents=True, exist_ok=True)
    exams = [build_exam(i) for i in range(1, 6)]
    for exam in exams:
        file_path = output_dir / f"mock_exam{exam['exam_no']}.html"
        file_path.write_text(render_exam_html(exam), encoding="utf-8")
    answer_path = output_dir / "answers.html"
    answer_path.write_text(render_answers_html(exams), encoding="utf-8")
    generated = ", ".join(sorted(p.name for p in output_dir.iterdir() if p.suffix == ".html"))
    print(f"생성 완료: {generated}")


if __name__ == "__main__":
    main()
