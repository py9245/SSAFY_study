# Django
- 다양성 : 웹, 모바일 앱, 백엔드, API서버 및 빅데이터 관리 등 광범위 서비스 개발 적합
- 확장성 : 대량의 데이터에 대해 빠르고 유연하게 확장할 수 있는 기능을 제공
- 보안 : 취약점으로부터 보호하는 보안기능이 기본적으로 내장되어 있음
- 커뮤니티 지원 : 개발자를 위한 지원, 문서 및 업데이트를 제공하는 활성화 된 커뮤니티

## 파이썬 프레임 워크 - Django, Flask, FastAPI 강사님 Flask 추천해주심
    - Flask 어려움 안만들어져 있음 자유도가 높음

## 아키텍쳐 디자인 패턴
- 자바는 MVC디자인패턴 model view controller 데이터, 사용자 인터페이스, 비즈니스 로직 분리
- django는 MTV 디자인 패턴 Model Template View
- 각 역할들은 비슷하지만 약자만 바꾼것 장고 개발자 마음

## 프로젝트와 앱
- 프로젝트 안에 여러가지 앱을 만들어야함
- django project 는 DB설정, URL연결 전체 앱 설정 등
- django application 은 기능들을 구현

- 프로젝트 생성 - 앱 생성 articles
- 프로젝트에선 setting urls만 거의 다룸
- 앱
    - admin - 관리자용 설정
    - models.py - DB와 관련된 모델들의 정의 MTV에서 M
    - views.py - 비즈니스 코드들 : MTV의 V HTTP 요청을 처리

### django 제대로 시작
- 요청 - urls.py 를 통해 요청을 받음

- 일단 지금 초반부의 기본 구조는 urls -> 뷰 -> 템플릿//// 
- 후행쉼표는 잘 해주자, urls.py 에선 문자열 경로 끝에 / 붙이자
- views.py 에서 모든 view함수는 첫번째 인자로 요청 객체를 받음
    - 매개변수 이름은 반드시 request로
    