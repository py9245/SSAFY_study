# BattleSSAFY GUI 시뮬레이터

기존 BattleSSAFY 봇 여섯 개를 수정 없이 실행하면서, 가벼운 로컬 GUI로 전투 상황을 확인할 수 있도록 하는 프로젝트입니다. 기존 메인 프로그램 대신 가짜 소켓 계층을 제공하여 각 봇이 사용하는 `INIT`, `send`, `recv` 프로토콜을 그대로 유지합니다.

## 요구 사항
- Windows에서 동작하는 Python 3.10 이상 (개발 시 Python 3.11 사용)
- Tkinter (표준 Python 배포판에 포함)
- 봇 스크립트 6종: `Attacker1.py`, `Attacker2.py`, `Attacker3.py`, `Player1_서울18_신건하.py`, `Player2_서울18_박유신.py`, `Player3_서울18_이명재.py`
- 맵 텍스트 파일 (예: 저장소의 `map.txt`)

## 빠른 실행 방법
1. Python을 설치한 뒤 프로젝트 루트에서 아래 명령을 실행합니다.
   ```bash
   python main.py
   ```
2. GUI에서 다음을 순서대로 수행합니다.
   - **Load Map** 버튼으로 맵 파일을 선택합니다.
   - 각 봇 경로가 올바른 파일을 가리키는지 확인합니다 (기본 경로가 미리 입력되어 있음).
   - **Start** 버튼을 누르면 가짜 소켓 서버와 봇 프로세스가 동시에 실행됩니다.
   - **Step**으로 한 턴씩 진행하거나 **Run/Pause/Stop**을 이용해 연속 실행을 제어합니다.

격자에는 전장 상태가 표시되며, 우측 패널에서 모든 유닛의 HP·탄약·현재 좌표를 확인할 수 있습니다.

## 주요 구성 요소
- `battlessafygui.engine.GameEngine`
  - 맵 파싱 결과를 기반으로 이동/사격/보급 명령을 처리하고, HP·탄약·나무(T) 파괴·코드 소모를 관리합니다.
- `battlessafygui.compat_socket.install_fake_socket`
  - 각 봇 프로세스 안에서 `socket.socket`을 가짜 구현으로 덮어쓰고, 파이프를 통해 `send`/`recv` 데이터를 주고받습니다.
- `battlessafygui.orchestrator.TurnOrchestrator`
  - Attacker1 → Player1 → Attacker2 → Player2 → Attacker3 → Player3 순서를 지키며 명령을 수집하고, 필요한 봇에 최신 game_data를 브로드캐스트합니다.
- `battlessafygui.gui`
  - Tkinter 기반 GUI로 맵 로드, 봇 경로 설정, 실행 제어(단발/연속)와 실시간 상태 표시를 제공합니다.

## 테스트
개발 과정에서 GUI 없이도 전체 파이프라인이 동작하는지 확인하기 위해 임시 스크립트 `test_orchestrator_run.py`를 작성해 6턴을 실행해 보았습니다. 실행은 다음과 같이 진행했습니다.
```bash
python test_orchestrator_run.py
```
정상 동작을 확인한 뒤 스크립트는 제거했습니다.

## 제한 사항
- PRD에 명시된 고급 기능(리플레이 저장/불러오기, 고속 모드, 배치 실행, 맵 에디터 등)은 아직 구현되지 않았습니다.
- GUI는 기본 Tk 라벨을 사용하기 때문에 매우 큰 맵에서는 가독성이 떨어질 수 있습니다.
- 잘못된 명령은 현재 `S`(대기)로 처리하며 GUI에는 별도의 경고가 표시되지 않습니다.

## 트러블슈팅
- **multiprocessing 관련 오류**: 오케스트레이터를 직접 실행하는 스크립트는 반드시 `if __name__ == "__main__":` 가드를 사용하세요. Windows에서는 `spawn` 시작 방식을 강제해야 합니다.
- **봇 경로 오류**: GUI에서 각 봇의 경로를 실제 파일 위치로 수정한 뒤 다시 Start를 누르세요.
- **실행 중 멈춤**: **Pause** 버튼으로 일시 정지 후 상태를 확인하거나, **Stop** 버튼으로 전체 프로세스를 종료할 수 있습니다.
