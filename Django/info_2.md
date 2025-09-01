# 오늘 빡세다고 하심

## template system
### DTL
- 변수
    - render에 세번째 인자로 딕셔너리 데이터를 사용
    - 딕셔너리 key.key 이렇게 접근한다.
    - 변수 + | + filter 적용 필터는 여러가지

## 상속
- 페이지의 공통 요소를 포함하고
- 하위 템플릿이 사용할 스켈레톤 코드를 작성해준다.
- 일반적으로 base.html 로 작성한다.
- {%extend 'path'%} 하나의 템플릿만 상속 받을 수있다.
- {%block cnotent%} 여기에 사용함 {%endblock cnotent%} 

## form 받아오기
- methods 는 GET(조회), POST(데이터 수정하는 경우) 를 지정
- action 입력 데이터가 전송될 URL 지정 (목적지)
- input의 핵심 속성 name!!!
- 사용자가 입력한 데이터에 붙이는 이름(key)
- query string parameters -> URL에 드러나기 때문에 가급적 조회할 떄 사용한다.
- ? 이전까지는 경로 그 이후엔 보내고 싶은 데이터를 보낸다

## Django URLs
- urls 패턴을 정하고 