# box

## normal flow
- 일반적인 흐름을 바꾸지 않고 순리대로 진행하는 경우
- 하나씩 깨면서 이쁘게 만들어야함

## display 타입
- block 타입
    * 책의 각 문단과 같이 독립된 덩어리처럼 동작
    * 항상 새로운 줄에서 시작, 다른 문단이 옆에 끼어들 수 없음.
    * block 타입은 웹 페이지의 큰 구조와 단락을 만듦
    * 항상 새로운 행으로 나뉨(너비 100%)
    * 가로,세로 여백 등 속성 모두 사용 가능
    * 여백으로 인해 다른 요소를 상자로부터 밀어냄
    * width속성 지정 안하면 inline 방향으로 사용 가능한 공간 모두 차지
    * h1~6,p,div 등
    * 타입
        * 대표 div
            * 다양한 섹션을 구조화하는데 가장 많이 쓰이는 요소
- inline
    * 형광펜으로 칠하는 느낌
    * 줄바꿈 일어나지 않음(콘텐츠의 크기만큼만 영역 차지)
    * width와 height 속성을 사용x
    * 수직 방향
        * padding,margin,border 적용 가능 하지만 다른 요소를 밀어낼 수 없음
    * 수평 방향
        * 다른 요소 밀어낼 수 있음
    * a, img, span, strong 등
        * span
            * 스타일 적용전까지는 특별한 변화 없음
            * 일부 조작에 유용
            * 줄바꿈 일으키지 않음, 문서의 구조에 큰 변화x

- inline-block
    * 줄 서있는 사람들
    * 각 사람들은 한 줄로 나란히 서 있지만
    * 각자의 덩치에 따라 각자 공간을 가짐
    * 너비, 높이 속성 가능, 패딩, 마진, 보더로 밀어냄

- none 타입
    * 축구팀의 후보선수 느낌
    * 요소를 화면표시 X, 공간조차 부여X

## CSS Position
### layout
- 각 요소의 위치와 크기를 조정하여 웹 디자인을 결정하는 것
- 핵심 : display

### Position
- 요소를 Normal Flow에서 제거하여 다른 위치로 배치하는 것
- X, y, z축으로 이동 가능
- static : 기본값
- relative : 상대적임
    - Normal Flow에 따라 배치
    - 자신의 원래 위치(static)를 기주으로 이동
    - t, r, b, l 속성으로 위치 조정
    - 다른 요소의 레이아웃 영향을 주지 않음
- absolute : 절대위치
    - Normal flow에서 제거
    - 가까운 relative
- fixd : viewport 기준
    - Normal flow에서 제거
    - 가까운 relative
- sticky : relative와 fixed의 특성을 결합한 속성
    - 스크롤 함에 따라

- z-index
    - z축, 화면에 나타나는 우선순위
    - 부모의 z인덱스에 따라 상속받음
    - 부모

## Flex box
### display type이 flex
- outer display 타입 : block, inline
- inner display 타입 : flex
- 부모의 관점에서 내 안에 정렬들
- 수평이 메인축 수직이 교차축
- 메인은 좌우 교차는 위 아래
- div 하나를 부모로 flex 컨테이너 설정해놓고
- 그 안에 flex item들을 만들어서 정리하는 방법

- justify 주축 좌우 정렬
- align 교차축 세로 정렬
- content 모든 값
- items - 한줄(align에서만 가능)
- grow 개별 아이템에 적용 - 여백을 각 각 나눠줌
- flex-basis - 기본 너비