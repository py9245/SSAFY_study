
# -*- coding: utf-8 -*-
"""
모의고사 3 세트 생성기

- 각 모의고사: 50문항 (객관식 20, 주관식 25, 서답형 5)
- 난이도 평균: 약 2 수준
- 파이썬 문법 제외, 수식 기반 문항 비중 최소화
- HTML 문제지 5개 + 정답 표 HTML 1개 생성
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Sequence

OUTPUT_DIR = Path("모의ver3")

EXAM_INFOS: Sequence[Dict[str, str]] = [
    {
        "title": "모의고사 1 - 데이터 이해와 준비",
        "description": "데이터 수집, 품질 진단, 기본적인 전처리 전략을 점검하는 모의고사입니다. 실무 초기 단계에서 필요한 기본 절차를 폭넓게 확인합니다.",
    },
    {
        "title": "모의고사 2 - 특징 공학과 데이터 품질",
        "description": "특징 공학, 결측치 처리, 데이터 거버넌스 등을 중심으로 안정적인 데이터 파이프라인을 설계하는 능력을 평가합니다.",
    },
    {
        "title": "모의고사 3 - 모델링 전략과 평가",
        "description": "모델 선택, 실험 설계, 평가 지표 선택과 관련된 전략적 판단력을 확인하는 모의고사입니다.",
    },
    {
        "title": "모의고사 4 - 딥러닝 시스템 운영",
        "description": "딥러닝 학습 과정, 재현성, 배포 전 검증과 모니터링 준비 등 운영 단계의 핵심 쟁점을 다룹니다.",
    },
    {
        "title": "모의고사 5 - ML 서비스 배포와 거버넌스",
        "description": "서비스 운영, 거버넌스, 공정성과 윤리, 재학습 체계 수립 등을 종합적으로 점검합니다.",
    },
]

DOMAINS: Sequence[str] = [
    "고객 이탈 예측",
    "신용 위험 평가",
    "의료 영상 판독",
    "제조 설비 이상 탐지",
    "온라인 광고 클릭 예측",
    "음성 감정 분석",
    "에너지 수요 예측",
    "배송 경로 최적화",
    "소셜 미디어 여론 분석",
    "보험 사기 탐지",
    "스마트 팩토리 품질 검사",
    "자율주행 객체 인식",
    "챗봇 의도 분류",
    "주가 변동 예측",
    "추천 시스템 개인화",
    "실시간 번역 서비스",
    "지능형 고객 응대",
    "반도체 공정 모니터링",
    "위성 영상 분류",
    "물류 수요 예측",
    "수질 오염 감시",
    "스포츠 퍼포먼스 분석",
    "센서 이상 감지",
    "헬스케어 웨어러블 경보",
    "도시 교통량 예측",
    "자원 배분 최적화",
    "금융 이상 거래 탐지",
    "스마트 홈 제어",
    "문서 분류 자동화",
    "관광 수요 예측",
    "농업 수확량 예측",
    "교육 학습 추천",
    "콜센터 상담 분석",
    "사물인터넷 장애 진단",
    "미디어 콘텐츠 큐레이션",
    "의약품 수요 예측",
    "기상 재난 조기 경보",
    "산업용 로봇 이상 탐지",
    "헬스케어 진단 챗봇",
    "스마트 시티 범죄 예방",
    "위험 물질 유출 감시",
    "해양 생태 관측",
    "에너지 설비 효율 개선",
    "커머스 추천 최적화",
    "스마트 농업 질병 예측",
    "금융 상품 추천",
    "문서 요약 자동화",
    "음악 취향 분석",
    "항공 수요 예측",
    "지리 정보 기반 수요 예측",
]

MC_TEMPLATES: Sequence[Dict[str, object]] = [
    {
        "issue": "클래스 불균형으로 소수 클래스를 제대로 예측하지 못한다",
        "stem_template": "{domain} 프로젝트에서 {issue} 상황에서 가장 먼저 고려해야 할 대응은?",
        "options": [
            ("A", "학습률을 크게 높여 소수 클래스의 확률을 강제로 올린다."),
            ("B", "클래스 가중치나 재표본 기법을 적용해 데이터 균형을 맞춘다."),
            ("C", "라벨이 많은 클래스를 삭제한다."),
            ("D", "출력층 활성 함수를 임의로 변경한다."),
        ],
        "answer": "B",
        "explanation": "클래스 불균형을 완화하려면 소수 클래스를 강조할 수 있는 가중치 조정이나 오버·언더샘플링 같은 재표본 기법을 적용해야 한다.",
        "difficulty": 2,
    },
    {
        "issue": "결측치가 많아 모델이 불안정하게 학습된다",
        "stem_template": "{domain} 데이터에서 {issue} 경우 가장 타당한 접근은?",
        "options": [
            ("A", "결측치가 있는 행을 전부 삭제한다."),
            ("B", "결측치를 모두 0으로 채운다."),
            ("C", "결측 원인을 분석하고 적절한 대체 전략을 설계한다."),
            ("D", "결측이 있는 특성을 제거한다."),
        ],
        "answer": "C",
        "explanation": "결측치의 발생 원인에 따라 적합한 대체 전략을 세우고, 통계적·도메인 지식을 사용해 복원하는 것이 가장 안정적인 방법이다.",
        "difficulty": 2,
    },
    {
        "issue": "모델이 과적합되어 평가 지표가 불안정하다",
        "stem_template": "{domain} 프로젝트에서 {issue}면 무엇을 우선 적용해야 하는가?",
        "options": [
            ("A", "평가 데이터를 훈련 세트에 포함한다."),
            ("B", "모델 깊이를 계속 늘린다."),
            ("C", "교차 검증과 정규화·드롭아웃 등 규제를 도입한다."),
            ("D", "학습률을 0에 가깝게 둔다."),
        ],
        "answer": "C",
        "explanation": "과적합을 줄이기 위해서는 교차 검증으로 일반화 성능을 확인하고 정규화, 드롭아웃, 조기 종료 등 규제 기법을 적용해야 한다.",
        "difficulty": 2,
    },
    {
        "issue": "의사결정 근거를 이해관계자에게 설명해야 한다",
        "stem_template": "{domain} 프로젝트에서 {issue} 상황일 때 가장 적절한 방법은?",
        "options": [
            ("A", "가중치를 무작위 초기화 상태로 유지한다."),
            ("B", "모델 구조를 더욱 복잡하게 만든다."),
            ("C", "SHAP이나 LIME 같은 해석 도구로 중요 특성과 기여도를 제시한다."),
            ("D", "모델 사용을 중단한다."),
        ],
        "answer": "C",
        "explanation": "해석 가능성을 높이기 위해 SHAP, LIME, 부분의존도 같은 도구로 예측 근거를 정량적으로 제시하는 것이 바람직하다.",
        "difficulty": 2,
    },
    {
        "issue": "운영 중 자료 분포가 변할 수 있어 성능 저하가 우려된다",
        "stem_template": "{domain} 서비스를 운영할 때 {issue}면 어떤 대응이 필요한가?",
        "options": [
            ("A", "운영 데이터를 저장하지 않는다."),
            ("B", "성능과 입력 분포를 모니터링하고 이상 징후 알림을 설정한다."),
            ("C", "모델을 무작위로 재학습한다."),
            ("D", "평가 지표를 제거한다."),
        ],
        "answer": "B",
        "explanation": "운영 단계에서는 데이터·모델 드리프트를 감지할 수 있도록 분포와 성능 변화를 모니터링하고 임계값 기반 알림을 구성해야 한다.",
        "difficulty": 2,
    },
    {
        "issue": "성능 비교를 위한 기준이 없어 개선 효과를 판단하기 어렵다",
        "stem_template": "{domain} 프로젝트에서 {issue} 때 가장 적절한 절차는?",
        "options": [
            ("A", "바로 복잡한 모델을 배포한다."),
            ("B", "간단한 기준 모델을 마련해 현재 데이터를 점검한다."),
            ("C", "라벨을 무작위로 섞는다."),
            ("D", "평가 지표를 제거한다."),
        ],
        "answer": "B",
        "explanation": "기준 모델을 먼저 구축하면 데이터 품질과 성능 기준을 확인할 수 있어 이후 모델 개선 효과를 명확히 비교할 수 있다.",
        "difficulty": 1,
    },
    {
        "issue": "특성마다 단위와 규모가 크게 달라 학습이 불안정하다",
        "stem_template": "{domain} 데이터에서 {issue}면 어떤 조치가 필요한가?",
        "options": [
            ("A", "수치형 변수를 모두 삭제한다."),
            ("B", "표준화나 정규화 등 스케일링을 적용한다."),
            ("C", "범주형을 숫자로 임의 변환한다."),
            ("D", "모델 층 수를 줄인다."),
        ],
        "answer": "B",
        "explanation": "스케일링으로 변수 간 규모 차이를 줄이면 최적화가 안정되고 거리 기반 계산의 왜곡을 줄일 수 있다.",
        "difficulty": 2,
    },
    {
        "issue": "실험 결과가 팀 내에 제대로 공유되지 않는다",
        "stem_template": "{domain} 프로젝트에서 {issue} 문제를 해결하려면?",
        "options": [
            ("A", "실험 결과를 각자 메모로만 기록한다."),
            ("B", "실험 관리 도구로 하이퍼파라미터와 성능을 추적한다."),
            ("C", "모델 파일을 임의로 덮어쓴다."),
            ("D", "협업을 중단한다."),
        ],
        "answer": "B",
        "explanation": "실험 관리 도구(Mlflow, Weights & Biases 등)를 사용하면 실험 결과를 체계적으로 추적하고 협업 효율을 높일 수 있다.",
        "difficulty": 2,
    },
    {
        "issue": "모델 편향이 사회적 문제를 일으킬 수 있다",
        "stem_template": "{domain} 서비스에서 {issue} 때 우선 수행해야 할 일은?",
        "options": [
            ("A", "민감 속성을 전부 제거한다."),
            ("B", "편향 지표와 집단별 성능을 측정해 보정한다."),
            ("C", "성능이 높다면 문제 없다."),
            ("D", "규제 보고를 생략한다."),
        ],
        "answer": "B",
        "explanation": "집단별 성능과 공정성 지표를 측정하고, 필요 시 데이터 보정이나 제약 기반 학습으로 편향을 완화해야 한다.",
        "difficulty": 2,
    },
    {
        "issue": "수집된 라벨의 품질이 일정하지 않다",
        "stem_template": "{domain} 프로젝트에서 {issue} 경우 어떤 조치를 취해야 하는가?",
        "options": [
            ("A", "라벨 오류를 무시한다."),
            ("B", "라벨을 모두 자동으로 생성한다."),
            ("C", "이중 라벨링과 샘플 검수 프로세스를 도입한다."),
            ("D", "데이터를 삭제한다."),
        ],
        "answer": "C",
        "explanation": "품질 검증을 위해 다중 라벨링, 가이드 문서, 샘플 검수를 시행하면 라벨 신뢰도를 확보할 수 있다.",
        "difficulty": 2,
    },
    {
        "issue": "데이터 버전이 섞여 재현성이 떨어진다",
        "stem_template": "{domain} 프로젝트에서 {issue} 문제를 막으려면?",
        "options": [
            ("A", "데이터 버전을 체계적으로 관리하고 변경 이력을 기록한다."),
            ("B", "임시 파일을 계속 덮어쓴다."),
            ("C", "이전 데이터를 삭제한다."),
            ("D", "데이터 위치를 공유하지 않는다."),
        ],
        "answer": "A",
        "explanation": "데이터 버전 관리와 변경 이력 문서화를 통해 실험 재현성과 감사 가능성을 확보해야 한다.",
        "difficulty": 2,
    },
    {
        "issue": "클래스 불균형과 비용 구조가 다른 평가 지표를 요구한다",
        "stem_template": "{domain} 프로젝트에서 {issue}면 어떤 기준이 필요한가?",
        "options": [
            ("A", "정확도만 확인한다."),
            ("B", "훈련 손실만 모니터링한다."),
            ("C", "재현율·정밀도·AUC 등 문제에 맞는 지표를 설정한다."),
            ("D", "평가 지표를 수시로 바꾼다."),
        ],
        "answer": "C",
        "explanation": "문제 특성에 맞는 지표를 우선 정의해야 비용 구조를 반영한 성능 관리가 가능하다.",
        "difficulty": 2,
    },
    {
        "issue": "개인정보 규제를 준수하면서 모델을 운영해야 한다",
        "stem_template": "{domain} 서비스에서 {issue} 때 가장 적절한 조치는?",
        "options": [
            ("A", "외부 파트너에게 원본 데이터를 제공한다."),
            ("B", "접근 제어를 제거한다."),
            ("C", "암호화를 해제한다."),
            ("D", "민감 정보를 최소화하고 익명화·접근 통제를 적용한다."),
        ],
        "answer": "D",
        "explanation": "민감 정보 최소화, 익명화, 접근 통제, 감사 로그 확보 등 개인정보 보호 조치를 강화해야 한다.",
        "difficulty": 2,
    },
    {
        "issue": "새 모델 배포가 실패했을 때 즉시 복구가 가능해야 한다",
        "stem_template": "{domain} 프로젝트에서 {issue} 준비를 하려면?",
        "options": [
            ("A", "자동 롤백과 버전 고정 전략을 설계한다."),
            ("B", "신규 모델로 바로 운영 데이터를 덮어쓴다."),
            ("C", "문제 발생 시 운영을 중단한다."),
            ("D", "문서화를 생략한다."),
        ],
        "answer": "A",
        "explanation": "배포 실패에 대비해 롤백 전략, 안전 장치, 단계적 배포 계획을 마련해야 한다.",
        "difficulty": 3,
    },
    {
        "issue": "사용자 피드백이 모델에 다시 반영되는 순환 구조가 있다",
        "stem_template": "{domain} 서비스에서 {issue}면 어떤 관리가 필요한가?",
        "options": [
            ("A", "피드백을 차단하고 무시한다."),
            ("B", "피드백 데이터를 추적하고 편향 여부를 주기적으로 점검한다."),
            ("C", "모델을 즉시 재학습한다."),
            ("D", "라벨을 모두 삭제한다."),
        ],
        "answer": "B",
        "explanation": "피드백 루프를 관리해 편향이 누적되지 않도록 추적 지표와 검증 절차를 마련해야 한다.",
        "difficulty": 2,
    },
]

SHORT_TEMPLATES: Sequence[Dict[str, object]] = [
    {
        "focus": "초기 데이터 프로파일링",
        "question_template": "{domain} 프로젝트에서 초기 데이터 프로파일링을 수행할 때 확인해야 할 핵심 항목을 두 가지 이상 서술하라.",
        "answer_template": "{domain} 데이터의 기본 통계(분포, 범위, 결측치)와 수집 경로 및 품질 이상 징후를 먼저 파악해야 이후 전처리와 검증 전략을 설계할 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "정규화 전략",
        "question_template": "{domain} 데이터의 수치형 특성을 정규화해야 하는 이유를 설명하라.",
        "answer_template": "{domain}처럼 변수 규모 차이가 큰 데이터에서는 정규화로 학습 안정성과 수렴 속도를 확보하고 특정 특성에 대한 편향을 줄일 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "결측치 분석",
        "question_template": "{domain} 프로젝트에서 결측치 발생 원인을 조사할 때 고려해야 할 요소를 제시하라.",
        "answer_template": "수집 시스템 로그, 설문/센서 오류, 패턴별 결측 비율 등 구조적 원인을 확인해야 {domain} 데이터의 결측을 적절히 대체할 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "범주형 인코딩",
        "question_template": "{domain} 데이터에서 범주형 변수를 인코딩할 때 선택 기준을 설명하라.",
        "answer_template": "카디널리티, 순서 정보 유무, 모델 유형을 고려해 원-핫, 타깃 인코딩 등 {domain} 도메인에 맞는 방식을 선택해야 과적합을 막을 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "데이터 분할 전략",
        "question_template": "{domain} 프로젝트의 데이터 분할(train/validation/test) 전략을 설계할 때 유의해야 할 점은 무엇인가?",
        "answer_template": "시간 순서, 사용자 단위 누락 금지 등 누수 위험을 차단하고 {domain} 운영 환경과 유사한 분포를 유지하도록 분할해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "교차 검증 계획",
        "question_template": "{domain} 프로젝트에서 교차 검증을 도입해야 하는 이유와 설정 시 주의점을 정리하라.",
        "answer_template": "교차 검증은 샘플 편향을 줄여 일반화 성능을 추정하게 해주며, {domain} 데이터의 시계열·그룹 특성에 맞는 폴드 전략을 선택해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "하이퍼파라미터 탐색",
        "question_template": "{domain} 모델의 하이퍼파라미터 탐색 전략을 어떻게 세울지 서술하라.",
        "answer_template": "탐색 공간과 예산을 정하고, 랜덤/베이지안 탐색 등으로 {domain} 성능 목표와 연동된 평가 지표를 최적화해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "랜덤 시드 관리",
        "question_template": "{domain} 프로젝트에서 랜덤 시드를 관리해야 하는 이유와 실무적인 방법을 제시하라.",
        "answer_template": "재현성과 비교 가능성을 확보하기 위해 모든 난수 발생 지점을 문서화하고 공통 시드를 지정해야 {domain} 실험 결과를 재검증할 수 있다.",
        "difficulty": 1,
    },
    {
        "focus": "피처 중요도 보고",
        "question_template": "{domain} 모델의 피처 중요도를 보고할 때 포함해야 할 내용을 정리하라.",
        "answer_template": "{domain} 맥락에서 중요도 계산 방법, 데이터 편향 가능성, 이해관계자가 해석할 수 있는 설명을 함께 제시해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "모델 재현성 문서",
        "question_template": "{domain} 프로젝트의 재현성 문서를 작성할 때 필수로 담아야 할 항목을 서술하라.",
        "answer_template": "데이터 버전, 전처리 절차, 모델·라이브러리 버전, 하이퍼파라미터와 실험 환경을 기록해야 {domain} 결과를 재현할 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "실험 추적 체계",
        "question_template": "{domain} 프로젝트에서 실험 추적 체계를 갖출 때 고려해야 할 지표와 기록 항목을 설명하라.",
        "answer_template": "실험 목적, 하이퍼파라미터, 지표, 사용 데이터 버전, 노트 등을 로그로 남겨야 {domain} 팀이 효율적으로 의사결정할 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "모델 버전 관리",
        "question_template": "{domain} 서비스에서 모델 버전을 관리할 때 필요한 절차를 제시하라.",
        "answer_template": "모델 아티팩트 이름 규칙, 메타데이터 기록, 배포 이력 관리 체계를 구축해야 {domain} 모델을 안전하게 교체할 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "배포 전 체크리스트",
        "question_template": "{domain} 모델을 배포하기 전에 수행해야 할 체크리스트 항목을 정리하라.",
        "answer_template": "성능 검증, 데이터/환경 일치 여부, 롤백 계획, 모니터링 구성 등을 확인해야 {domain} 배포 위험을 줄일 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "재학습 스케줄",
        "question_template": "{domain} 모델의 재학습 주기를 결정할 때 고려해야 할 요인을 서술하라.",
        "answer_template": "데이터 유입 주기, 드리프트 민감도, 운영 비용을 고려해 {domain} 서비스 요구에 맞는 재학습 스케줄을 설정해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "드리프트 알림",
        "question_template": "{domain} 운영 중 드리프트 알림을 설정할 때 필요한 요소를 설명하라.",
        "answer_template": "분포/성능 지표 선택, 임계값, 알림 채널, 대응 프로세스를 정의해 {domain} 시스템의 이상 징후를 빠르게 감지해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "사용자 영향 분석",
        "question_template": "{domain} 모델 변경이 사용자에게 미치는 영향을 분석할 때 필요한 절차를 정리하라.",
        "answer_template": "영향 사용자 그룹 식별, 지표 변동 분석, 위험 완화 방안을 마련해 {domain} 사용자 경험을 보호해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "A/B 테스트 설계",
        "question_template": "{domain} 서비스에서 A/B 테스트를 설계할 때 주의해야 할 포인트를 설명하라.",
        "answer_template": "대상 분할, 통계 검정 방법, 기간, 윤리적 고려를 명확히 정의해 {domain} 실험 결과가 유효하도록 해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "로깅 전략",
        "question_template": "{domain} 모델을 운영할 때 어떤 로그를 수집해야 하는지 설명하라.",
        "answer_template": "입력 특징, 예측 결과, 의사결정 ID, 오류 로그 등을 수집해야 {domain} 운영 이슈 분석과 감사 요구에 대응할 수 있다.",
        "difficulty": 2,
    },
    {
        "focus": "모니터링 지표",
        "question_template": "{domain} 모델 운영에서 정의해야 할 핵심 모니터링 지표를 제시하라.",
        "answer_template": "성능 지표, 입력 분포 지표, 지연 시간, 자원 사용량 등을 추적해 {domain} 서비스 품질과 비용을 균형 있게 관리해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "에스컬레이션 절차",
        "question_template": "{domain} 모델에서 이상 상황이 발생했을 때의 에스컬레이션 절차를 설명하라.",
        "answer_template": "알림 수신자, 대응 우선순위, 임시 조치, 근본 원인 분석 단계 등을 정의해 {domain} 장애를 신속히 해결해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "공정성 검토",
        "question_template": "{domain} 모델의 공정성을 점검할 때 확인해야 할 항목을 정리하라.",
        "answer_template": "집단별 지표 비교, 데이터 대표성, 영향 평가, 개선 계획을 문서화해 {domain} 모델이 특정 집단에 불리하지 않도록 해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "개인정보 보호",
        "question_template": "{domain} 데이터로 모델을 구축할 때 개인정보 보호 계획에 포함해야 할 조치를 서술하라.",
        "answer_template": "최소 수집 원칙, 익명화/가명화, 접근 통제, 감사 로그와 같은 보호 체계를 마련해야 {domain} 프로젝트가 규제를 준수한다.",
        "difficulty": 2,
    },
    {
        "focus": "윤리 심사 준비",
        "question_template": "{domain} 프로젝트의 윤리 심사를 준비할 때 필요한 자료를 설명하라.",
        "answer_template": "목적, 데이터 출처, 편향 완화 계획, 이해관계자 영향 분석을 정리해 {domain} 프로젝트가 윤리 기준을 충족함을 입증해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "도메인 전문가 협업",
        "question_template": "{domain} 프로젝트에서 도메인 전문가와 협업할 때 준비해야 할 내용을 정리하라.",
        "answer_template": "모델 가정, 데이터 한계, 의사결정 포인트를 공유하고 피드백 반영 프로세스를 마련해 {domain} 전문 지식을 활용해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "데이터 거버넌스 회의",
        "question_template": "{domain} 데이터를 다루는 거버넌스 회의에서 다뤄야 할 안건을 제시하라.",
        "answer_template": "데이터 소유권, 접근 권한, 품질 지표, 변경 관리 등 {domain} 거버넌스 핵심 의제를 논의해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "SLA 설계",
        "question_template": "{domain} 모델의 서비스 수준 협약(SLA)을 정의할 때 포함해야 할 항목을 설명하라.",
        "answer_template": "응답 시간, 가용성, 성능 임계값, 지원 절차를 명시해 {domain} 서비스 기대치를 이해관계자와 합의해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "레이블 갱신 프로세스",
        "question_template": "{domain} 모델의 레이블을 주기적으로 갱신할 때 필요한 프로세스를 서술하라.",
        "answer_template": "데이터 수집 주기, 검수 기준, 승인 절차, 버전 관리 등을 정의해 {domain} 레이블 품질을 지속적으로 유지해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "피드백 루프 관리",
        "question_template": "{domain} 서비스에서 사용자 피드백이 모델에 반영될 때 주의해야 할 점을 설명하라.",
        "answer_template": "피드백 수집·필터링 기준과 편향 모니터링을 준비해 {domain} 모델이 자기 강화 편향에 빠지지 않도록 해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "테스트 데이터 관리",
        "question_template": "{domain} 프로젝트에서 테스트 데이터를 관리할 때 지켜야 할 원칙을 서술하라.",
        "answer_template": "운영 데이터와 분리 보관, 사용 이력 추적, 무단 사용 금지를 지켜 {domain} 결과 검증의 독립성을 보장해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "KPI 협의",
        "question_template": "{domain} 모델의 KPI를 이해관계자와 협의할 때 포함해야 할 논의 항목을 정리하라.",
        "answer_template": "사업 목표, 사용자 영향, 기술 지표, 측정 방법을 명확히 정의해 {domain} 프로젝트의 성공 기준을 합의해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "운영팀 커뮤니케이션",
        "question_template": "{domain} 모델 관련 정보를 운영팀과 공유할 때 포함해야 할 내용을 설명하라.",
        "answer_template": "모델 기능, 한계, 장애 대응 절차, 연락 창구를 정리해 {domain} 운영팀이 이슈에 빠르게 대응할 수 있도록 해야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "장애 대응 훈련",
        "question_template": "{domain} 모델 장애 대응 훈련을 설계할 때 필요한 단계를 제시하라.",
        "answer_template": "가상 시나리오 정의, 역할 분담, 복구 절차 검증, 사후 회고를 포함해 {domain} 팀의 대응 역량을 높여야 한다.",
        "difficulty": 2,
    },
    {
        "focus": "섀도 배포 준비",
        "question_template": "{domain} 모델을 섀도 배포할 때 준비해야 할 요소를 설명하라.",
        "answer_template": "실시간 트래픽 분기, 모니터링, 성능 비교, 자동화된 롤백 조건을 갖춰 {domain} 모델의 안정성을 검증해야 한다.",
        "difficulty": 3,
    },
    {
        "focus": "관측 가능성 대시보드",
        "question_template": "{domain} 시스템의 관측 가능성(Observability) 대시보드를 구성할 때 포함해야 할 지표를 설명하라.",
        "answer_template": "로그, 메트릭, 트레이스 지표와 사용자 행동 데이터를 수집해 {domain} 시스템의 상태를 실시간으로 파악할 수 있어야 한다.",
        "difficulty": 2,
    },
]

ESSAY_TOPICS: Sequence[Dict[str, object]] = [
    {
        "theme": "데이터 파이프라인 자동화 계획",
        "prompt_template": "{domain} 프로젝트에서 {theme}을 수립하기 위한 단계별 계획을 제시하라.",
        "answer_template": (
            "1) 수집·전처리·검증 단계의 표준 작업을 정의한다.\n"
            "2) 워크플로 도구로 스케줄링과 모니터링 체계를 구축한다.\n"
            "3) 실패 대응과 롤백 절차를 문서화해 안정적으로 운영한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "재학습 및 검증 일정 수립",
        "prompt_template": "{domain} 모델의 {theme}을 세울 때 고려해야 할 요소를 서술하라.",
        "answer_template": (
            "1) 데이터 유입 패턴과 품질 지표를 기반으로 재학습 주기를 정의한다.\n"
            "2) 검증 데이터와 기준 지표를 미리 합의한다.\n"
            "3) 재배포 승인 절차와 책임자를 지정한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "드리프트 대응 로드맵",
        "prompt_template": "{domain} 서비스에서 {theme}을 설계하는 방법을 설명하라.",
        "answer_template": (
            "1) 입력/출력 드리프트 지표와 임계값을 설정한다.\n"
            "2) 알림과 임시 완화 방안을 마련한다.\n"
            "3) 근본 원인 분석과 재학습 프로세스를 연결한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "공정성 개선 프로젝트",
        "prompt_template": "{domain} 모델의 {theme}을 실행하기 위한 순서를 작성하라.",
        "answer_template": (
            "1) 집단별 성능과 편향 지표를 측정한다.\n"
            "2) 데이터 수집·학습 단계에서 조정 가능한 대안을 설계한다.\n"
            "3) 개선 효과를 검증하고 이해관계자와 공유한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "실험 추적 체계 구축",
        "prompt_template": "{domain} 프로젝트에서 {theme} 단계별 실행 계획을 서술하라.",
        "answer_template": (
            "1) 실험 메타데이터 스키마와 기록 기준을 정의한다.\n"
            "2) 추적 도구와 저장소를 선정해 통합한다.\n"
            "3) 리뷰·승인 절차를 만들어 협업 품질을 높인다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "모델 설명 책임 체계",
        "prompt_template": "{domain} 서비스에서 {theme}를 마련하기 위한 절차를 제시하라.",
        "answer_template": (
            "1) 설명 요구사항을 수집하고 대상별 보고 템플릿을 만든다.\n"
            "2) 해석 기법과 검토 주기를 정의한다.\n"
            "3) 기록과 전달 프로세스를 운영팀·법무팀과 연결한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "배포 전 위험 점검 프로세스",
        "prompt_template": "{domain} 모델의 {theme}를 설계하라.",
        "answer_template": (
            "1) 성능, 보안, 규제 항목을 포함한 체크리스트를 구성한다.\n"
            "2) 자동·수동 검증 절차와 승인 게이트를 정의한다.\n"
            "3) 점검 결과를 기록하고 배포 결정에 반영한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "피드백 루프 제어 정책",
        "prompt_template": "{domain} 모델에서 {theme}을 수립하는 방법을 설명하라.",
        "answer_template": (
            "1) 피드백 경로와 영향 범위를 분석한다.\n"
            "2) 수집·필터링 기준과 모니터링 지표를 정의한다.\n"
            "3) 편향 발견 시 조치 절차와 책임자를 지정한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "모델 거버넌스 프레임워크",
        "prompt_template": "{domain} 조직에서 {theme}를 구성하는 단계를 서술하라.",
        "answer_template": (
            "1) 역할과 의사결정 위원회를 정의한다.\n"
            "2) 정책·표준·지침을 정리하고 교육한다.\n"
            "3) 모니터링과 감사 절차를 통해 지속적으로 개선한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "데이터 품질 개선 캠페인",
        "prompt_template": "{domain} 데이터의 {theme}을 운영하기 위한 계획을 작성하라.",
        "answer_template": (
            "1) 문제 유형과 우선순위를 정의한다.\n"
            "2) 책임자와 일정, 개선 지표를 설정한다.\n"
            "3) 개선 결과를 모니터링하고 회고한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "윤리 위원회 보고 절차",
        "prompt_template": "{domain} 프로젝트에서 {theme}를 마련하는 과정을 정리하라.",
        "answer_template": (
            "1) 보고 대상과 제출 주기를 합의한다.\n"
            "2) 데이터 출처, 편향 완화, 사용자 영향 자료를 준비한다.\n"
            "3) 질의응답과 후속 조치 프로세스를 운영한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "재현성 보증 전략",
        "prompt_template": "{domain} 모델의 {theme}을 수립하는 절차를 서술하라.",
        "answer_template": (
            "1) 데이터·코드·환경 버전 관리 기준을 정한다.\n"
            "2) 자동화된 재실행 파이프라인을 구축한다.\n"
            "3) 검증 로그와 리뷰 프로세스를 운영해 품질을 보증한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "관측 가능성 체계 구축",
        "prompt_template": "{domain} 서비스의 {theme}을 단계별로 제시하라.",
        "answer_template": (
            "1) 로그·메트릭·트레이스 지표를 정의한다.\n"
            "2) 수집·보관 도구를 연동하고 대시보드를 만든다.\n"
            "3) 경보와 대응 절차를 운영팀과 연계한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "비상 대응 훈련 계획",
        "prompt_template": "{domain} 프로젝트의 {theme}을 설계하는 방법을 설명하라.",
        "answer_template": (
            "1) 주요 장애 시나리오와 영향 범위를 정의한다.\n"
            "2) 역할·연락 체계를 지정한다.\n"
            "3) 모의 훈련과 회고를 반복해 대응력을 높인다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "지표 관리 체계",
        "prompt_template": "{domain} 모델을 위한 {theme}를 구축하는 단계를 제시하라.",
        "answer_template": (
            "1) 사업·기술 지표를 구분해 정의한다.\n"
            "2) 수집·보고 주기와 책임자를 명확히 한다.\n"
            "3) 지표 변동 시 대응 계획을 연동한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "사용자 커뮤니케이션 가이드",
        "prompt_template": "{domain} 서비스의 {theme}를 작성하는 절차를 설명하라.",
        "answer_template": (
            "1) 사용자 질문과 우려 사항을 조사한다.\n"
            "2) 설명 가능한 메시지와 FAQ를 준비한다.\n"
            "3) 변경 시점과 영향 범위를 공지하는 체계를 만든다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "모델 감사 준비",
        "prompt_template": "{domain} 조직에서 {theme}를 진행할 때의 준비 절차를 서술하라.",
        "answer_template": (
            "1) 감사 범위와 필요 자료 목록을 정의한다.\n"
            "2) 데이터·모델·운영 로그를 사전에 정리한다.\n"
            "3) 개선 과제와 추적 계획을 수립한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "데이터 계약 관리",
        "prompt_template": "{domain} 프로젝트의 {theme}를 위해 필요한 단계를 설명하라.",
        "answer_template": (
            "1) 데이터 출처와 이용 조건을 파악한다.\n"
            "2) 계약 조항을 모델 운영 정책과 정합화한다.\n"
            "3) 준수 여부 모니터링과 갱신 프로세스를 마련한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "AI 윤리 교육 프로그램",
        "prompt_template": "{domain} 조직에서 {theme}을 설계하는 과정을 제시하라.",
        "answer_template": (
            "1) 교육 대상과 학습 목표를 정의한다.\n"
            "2) 실제 사례와 우수 실천 기준을 반영한다.\n"
            "3) 평가와 피드백 절차로 지속 개선한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "협업 워크플로 정비",
        "prompt_template": "{domain} 팀의 {theme}를 구축하는 방법을 설명하라.",
        "answer_template": (
            "1) 역할과 승인 흐름을 명확히 한다.\n"
            "2) 공동 작업 도구와 표준 템플릿을 마련한다.\n"
            "3) 회고와 업데이트 주기를 설정한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "모델 서명 및 승인 절차",
        "prompt_template": "{domain} 서비스에서 {theme}를 마련하는 단계를 정리하라.",
        "answer_template": (
            "1) 승인 기준과 책임자를 지정한다.\n"
            "2) 테스트 결과와 리스크 평가서를 제출하도록 한다.\n"
            "3) 승인 기록과 추적 장치를 운영한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "데이터 아카이브 전략",
        "prompt_template": "{domain} 프로젝트에서 {theme}을 수립하는 방법을 설명하라.",
        "answer_template": (
            "1) 보관 대상과 만료 정책을 정의한다.\n"
            "2) 보안·접근 통제와 백업 절차를 설정한다.\n"
            "3) 폐기·재활용 시나리오를 문서화한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "서비스 품질 개선 로드맵",
        "prompt_template": "{domain} 모델의 {theme}을 작성하는 과정을 서술하라.",
        "answer_template": (
            "1) 품질 지표와 현황을 진단한다.\n"
            "2) 개선 우선순위와 일정, 자원을 배분한다.\n"
            "3) 실행 결과와 학습 내용을 공유한다."
        ),
        "difficulty": 2,
    },
    {
        "theme": "규제 대응 체계",
        "prompt_template": "{domain} 서비스의 {theme}를 구축하는 방법을 설명하라.",
        "answer_template": (
            "1) 적용 법규와 요구사항을 목록화한다.\n"
            "2) 준수 점검 절차와 책임 부서를 정한다.\n"
            "3) 규제 변경 시 대응 프로세스를 마련한다."
        ),
        "difficulty": 3,
    },
    {
        "theme": "사용자 피드백 운영 정책",
        "prompt_template": "{domain} 서비스의 {theme}을 설계하는 단계를 제시하라.",
        "answer_template": (
            "1) 피드백 수집 채널과 우선순위 기준을 정의한다.\n"
            "2) 분석·반영 프로세스를 책임자와 연결한다.\n"
            "3) 결과 공유와 사용자 커뮤니케이션을 체계화한다."
        ),
        "difficulty": 2,
    },
]


def select_domains(exam_idx: int, count: int, offset: int = 0) -> List[str]:
    start = exam_idx * 10 + offset
    return [DOMAINS[(start + i) % len(DOMAINS)] for i in range(count)]


def build_multiple_choice(exam_idx: int) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    domains = select_domains(exam_idx, 20, offset=0)
    template_count = len(MC_TEMPLATES)
    for i, domain in enumerate(domains):
        template = MC_TEMPLATES[(exam_idx * 5 + i) % template_count]
        explanation = f"{domain} 프로젝트에서도 {template['explanation']}"
        options = [(label, option) for label, option in template["options"]]
        questions.append(
            {
                "type": "객관식",
                "section": "객관식",
                "stem": template["stem_template"].format(domain=domain, issue=template["issue"]),
                "options": options,
                "answer": template["answer"],
                "answer_detail": explanation,
                "difficulty": template["difficulty"],
            }
        )
    return questions


def build_short_questions(exam_idx: int) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    domains = select_domains(exam_idx, 25, offset=6)
    template_count = len(SHORT_TEMPLATES)
    for i, domain in enumerate(domains):
        template = SHORT_TEMPLATES[(exam_idx * 7 + i) % template_count]
        questions.append(
            {
                "type": "주관식",
                "section": "주관식",
                "stem": template["question_template"].format(domain=domain),
                "answer": template["answer_template"].format(domain=domain),
                "answer_detail": template["answer_template"].format(domain=domain),
                "difficulty": template["difficulty"],
            }
        )
    return questions


def build_essay_questions(exam_idx: int) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    domains = select_domains(exam_idx, 5, offset=18)
    template_count = len(ESSAY_TOPICS)
    for i, domain in enumerate(domains):
        template = ESSAY_TOPICS[(exam_idx * 5 + i) % template_count]
        stem = template["prompt_template"].format(domain=domain, theme=template["theme"])
        answer = template["answer_template"]
        questions.append(
            {
                "type": "서답형",
                "section": "서답형",
                "stem": stem,
                "answer": answer,
                "answer_detail": answer,
                "difficulty": template["difficulty"],
            }
        )
    return questions


def generate_exam(exam_idx: int) -> Dict[str, object]:
    info = EXAM_INFOS[exam_idx]
    exam_id = exam_idx + 1
    mc_questions = build_multiple_choice(exam_idx)
    short_questions = build_short_questions(exam_idx)
    essay_questions = build_essay_questions(exam_idx)
    questions = mc_questions + short_questions + essay_questions

    answers_for_table: List[Dict[str, object]] = []
    for number, question in enumerate(questions, start=1):
        question["number"] = number
        answers_for_table.append(
            {
                "exam_id": exam_id,
                "exam_title": info["title"],
                "number": number,
                "type": question["type"],
                "difficulty": question["difficulty"],
                "answer": question.get("answer", ""),
                "detail": question.get("answer_detail", ""),
            }
        )

    avg_difficulty = sum(q["difficulty"] for q in questions) / len(questions)

    return {
        "id": exam_id,
        "title": info["title"],
        "description": info["description"],
        "questions": questions,
        "avg_difficulty": avg_difficulty,
        "answers_for_table": answers_for_table,
    }


def render_question(question: Dict[str, object]) -> str:
    number = question["number"]
    stem = html.escape(question["stem"])
    difficulty = question["difficulty"]
    q_type = question["type"]
    section = question["section"]
    body_parts: List[str] = [
        f"<div class='question' id='q{number}'>",
        f"<div class='question-meta'><span>문항 {number}</span><span>{section}</span><span>난이도 {difficulty}</span></div>",
        f"<p>{stem}</p>",
    ]
    if q_type == "객관식":
        options_html = []
        for label, option_text in question["options"]:
            option = html.escape(option_text)
            options_html.append(
                f"<li><label><input type='radio' name='q{number}' value='{label}'> "
                f"<strong>{label}.</strong> {option}</label></li>"
            )
        body_parts.append("<ol class='options' type='A'>")
        body_parts.extend(options_html)
        body_parts.append("</ol>")
    elif q_type == "주관식":
        body_parts.append("<textarea class='short-answer' placeholder='답안을 입력하세요'></textarea>")
    else:
        body_parts.append("<textarea class='essay-answer' placeholder='핵심 근거와 단계를 정리하세요'></textarea>")
    body_parts.append("</div>")
    return "\n".join(body_parts)


def render_exam_html(exam: Dict[str, object]) -> str:
    header = f"""<!DOCTYPE html>
<html lang='ko'>
<head>
<meta charset='utf-8'>
<title>{html.escape(exam['title'])}</title>
<style>
body {{ font-family: 'Segoe UI', 'Noto Sans KR', sans-serif; margin: 32px; background:#f6f8fb; color:#222; line-height:1.6; }}
h1 {{ font-size: 1.9rem; margin-bottom: 0.3rem; }}
section.meta {{ background:#fff; border:1px solid #d0d7e3; border-radius:10px; padding:1.3rem; margin-bottom:1.6rem; }}
.question {{ background:#fff; border:1px solid #d7deeb; border-left:6px solid #4a6fa5; padding:1.1rem; border-radius:8px; margin-bottom:1.1rem; }}
.question-meta {{ font-size:0.85rem; color:#52627a; display:flex; gap:0.8rem; flex-wrap:wrap; margin-bottom:0.5rem; }}
ol.options {{ margin-top:0.5rem; padding-left:1.6rem; }}
ol.options li {{ margin-bottom:0.5rem; }}
textarea.short-answer {{ width:100%; min-height:3.4rem; border:1px dashed #9aa5bd; border-radius:6px; padding:0.6rem; background:#fbfcff; }}
textarea.essay-answer {{ width:100%; min-height:7rem; border:1px dashed #9aa5bd; border-radius:6px; padding:0.6rem; background:#fbfcff; }}
</style>
</head>
<body>
<h1>{html.escape(exam['title'])}</h1>
<section class='meta'>
<p>{html.escape(exam['description'])}</p>
<p>총 50문항 · 평균 난이도 {exam['avg_difficulty']:.2f}</p>
</section>
"""
    sections: Dict[str, List[Dict[str, object]]] = {"객관식": [], "주관식": [], "서답형": []}
    for question in exam["questions"]:
        sections[question["section"]].append(question)

    body_parts = [header]
    for section_name in ("객관식", "주관식", "서답형"):
        question_list = sections[section_name]
        if not question_list:
            continue
        body_parts.append(f"<h2>{section_name}</h2>")
        for question in question_list:
            body_parts.append(render_question(question))
    body_parts.append("</body></html>")
    return "\n".join(body_parts)


def render_answers_html(exams: Sequence[Dict[str, object]]) -> str:
    rows: List[str] = []
    for exam in exams:
        for answer in exam["answers_for_table"]:
            detail = html.escape(answer["detail"]).replace("\n", "<br>")
            rows.append(
                "<tr>"
                f"<td>{answer['exam_title']}</td>"
                f"<td>{answer['number']}</td>"
                f"<td>{answer['type']}</td>"
                f"<td>{answer['difficulty']}</td>"
                f"<td>{html.escape(str(answer['answer']))}</td>"
                f"<td>{detail}</td>"
                "</tr>"
            )
    return """<!DOCTYPE html>
<html lang='ko'>
<head>
<meta charset='utf-8'>
<title>모의고사 3 정답표</title>
<style>
body { font-family: 'Segoe UI', 'Noto Sans KR', sans-serif; margin: 32px; background:#f6f8fb; color:#222; line-height:1.6; }
table { width:100%; border-collapse:collapse; background:#fff; border:1px solid #ccd4e0; }
th, td { border:1px solid #ccd4e0; padding:0.6rem 0.8rem; text-align:left; vertical-align:top; }
th { background:#e8eef7; }
tbody tr:nth-child(even) { background:#f7f9fd; }
</style>
</head>
<body>
<h1>모의고사 3 정답표</h1>
<table>
<thead>
<tr><th>시험</th><th>문항</th><th>유형</th><th>난이도</th><th>정답</th><th>정답 해설</th></tr>
</thead>
<tbody>
""" + "\n".join(rows) + """
</tbody>
</table>
</body>
</html>"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exams: List[Dict[str, object]] = []
    for idx, _ in enumerate(EXAM_INFOS):
        exam = generate_exam(idx)
        exams.append(exam)
        html_text = render_exam_html(exam)
        (OUTPUT_DIR / f"mock_exam{exam['id']}.html").write_text(html_text, encoding="utf-8")
    answers_html = render_answers_html(exams)
    (OUTPUT_DIR / "answers.html").write_text(answers_html, encoding="utf-8")


if __name__ == "__main__":
    main()
