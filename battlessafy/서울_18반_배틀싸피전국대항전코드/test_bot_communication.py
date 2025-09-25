#!/usr/bin/env python3
"""
봇 통신 테스트 스크립트
단일 봇이 compat_socket을 통해 제대로 통신하는지 테스트
"""
import os
import sys
import tempfile
import time
import json
from pathlib import Path

# GUI 모듈 경로 추가
sys.path.insert(0, './battlessafygui')

from battlessafygui.orchestrator import BotProcess
from battlessafygui.maps import parse_map_text

def test_single_bot():
    """단일 봇 테스트"""

    # 통신 디렉토리 생성
    comm_dir = Path(tempfile.mkdtemp(prefix="battlessafy_test_"))
    print(f"통신 디렉토리: {comm_dir}")

    # 봇 프로세스 생성
    bot_id = "Attacker1"
    bot_file = os.path.abspath("./Attacker1.py")

    if not os.path.exists(bot_file):
        print(f"봇 파일을 찾을 수 없습니다: {bot_file}")
        return

    bot = BotProcess(bot_id, bot_file, comm_dir)

    # 맵 데이터 로드
    with open("./map.txt", 'r', encoding='utf-8') as f:
        map_data = f.read()

    print("맵 데이터 파싱...")
    try:
        game_state = parse_map_text(map_data)
        print(f"맵 크기: {game_state.height}x{game_state.width}")
        print(f"탱크 수: {len(game_state.tanks)}")
        print(f"터렛 수: {len(game_state.turrets)}")
        for turret_id, turret in game_state.turrets.items():
            print(f"  터렛 {turret_id}: HP={turret.hp}, 위치=({turret.r},{turret.c})")
    except Exception as e:
        print(f"맵 파싱 오류: {e}")
        return

    print(f"봇 시작: {bot_id}")
    bot.start()

    # 봇이 시작될 때까지 대기
    time.sleep(2.0)

    # 연결 신호 확인
    if bot.outgoing_file.exists():
        try:
            with open(bot.outgoing_file, 'r', encoding='utf-8') as f:
                message = json.load(f)
            print(f"봇 연결 신호: {message}")
            bot.outgoing_file.unlink()
        except Exception as e:
            print(f"연결 신호 읽기 오류: {e}")

    # 게임 상태 전송
    from battlessafygui.maps import serialize_game_state
    state_text = serialize_game_state(game_state, bot_id)
    print(f"게임 상태 전송 (길이: {len(state_text)})")
    print("전체 데이터:")
    for i, line in enumerate(state_text.split('\n')):
        print(f"  {i+1}: {line}")

    bot.send_state(state_text)

    # 봇 응답 대기
    print("봇 명령어 수신 중...")
    command = bot.get_command(timeout=10.0)
    print(f"봇 명령어: {command}")

    # 봇 프로세스 출력 확인
    if bot.process and bot.process.poll() is not None:
        print("\n봇 프로세스 출력:")
        stdout, _ = bot.process.communicate(timeout=1.0)
        if stdout:
            print(stdout)
    elif bot.process:
        print(f"봇 프로세스 상태: 실행 중 (PID: {bot.process.pid})")
        # 실행 중인 프로세스의 출력 일부 확인
        try:
            output = bot.process.stdout.read(1000)
            if output:
                print(f"현재 출력: {output}")
        except:
            pass

    # 정리
    bot.stop()

    # 통신 디렉토리 정리
    import shutil
    shutil.rmtree(comm_dir)

    print("테스트 완료")

if __name__ == "__main__":
    test_single_bot()