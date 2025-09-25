#!/usr/bin/env python3
"""
새로운 맵과 암호로 간단한 테스트
"""
import sys
sys.path.insert(0, './battlessafygui')

from battlessafygui.maps import parse_map_text, serialize_game_state

def test_new_map():
    """새로운 맵 테스트"""

    # 맵 데이터 로드
    with open("./map.txt", 'r', encoding='utf-8') as f:
        map_data = f.read()

    print("맵 데이터 파싱...")
    print(f"원본 데이터 첫 5줄:")
    for i, line in enumerate(map_data.split('\n')[:5]):
        print(f"  {i+1}: {line}")

    game_state = parse_map_text(map_data)
    print(f"맵 크기: {game_state.height}x{game_state.width}")
    print(f"탱크 수: {len(game_state.tanks)}")
    print(f"터렛 수: {len(game_state.turrets)}")
    print(f"암호 수: {len(game_state.codes)}")

    for code in game_state.codes:
        print(f"  암호: {code.plaintext}")

    # Attacker1 관점에서 직렬화 테스트
    print("\n=== Attacker1 관점 ===")
    state_text = serialize_game_state(game_state, "Attacker1")
    print("직렬화된 데이터:")
    for i, line in enumerate(state_text.split('\n')):
        print(f"  {i+1}: {line}")

if __name__ == "__main__":
    test_new_map()