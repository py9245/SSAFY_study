import sys
import socket
import heapq
from collections import deque, defaultdict

##############################
# 메인 프로그램 통신 변수 정의
##############################
HOST = '127.0.0.1'
PORT = 8747
ARGS = sys.argv[1] if len(sys.argv) > 1 else ''
sock = socket.socket()


##############################
# 메인 프로그램 통신 함수 정의
##############################

# 메인 프로그램 연결 및 초기화
def init(nickname):
    try:
        print(f'[STATUS] Trying to connect to {HOST}:{PORT}...')
        sock.connect((HOST, PORT))
        print('[STATUS] Connected')
        init_command = f'INIT {nickname}'

        return submit(init_command)

    except Exception as e:
        print('[ERROR] Failed to connect. Please check if the main program is waiting for connection.')
        print(e)


# 메인 프로그램으로 데이터(명령어) 전송
def submit(string_to_send):
    try:
        send_data = ARGS + string_to_send + ' '
        sock.send(send_data.encode('utf-8'))

        return receive()

    except Exception as e:
        print('[ERROR] Failed to send data. Please check if connection to the main program is valid.')

    return None


# 메인 프로그램으로부터 데이터 수신
def receive():
    try:
        game_data = (sock.recv(1024)).decode()

        if game_data and game_data[0].isdigit() and int(game_data[0]) > 0:
            return game_data

        print('[STATUS] No receive data from the main program.')
        close()

    except Exception as e:
        print('[ERROR] Failed to receive data. Please check if connection to the main program is valid.')


# 연결 해제
def close():
    try:
        if sock is not None:
            sock.close()
        print('[STATUS] Connection closed')

    except Exception as e:
        print('[ERROR] Network connection has been corrupted.')


##############################
# 입력 데이터 변수 정의
##############################
map_data = [[]]  # 맵 정보. 예) map_data[0][1] - [0, 1]의 지형/지물
my_allies = {}  # 아군 정보. 예) my_allies['M'] - 플레이어 본인의 정보
enemies = {}  # 적군 정보. 예) enemies['X'] - 적 포탑의 정보
codes = []  # 주어진 암호문. 예) codes[0] - 첫 번째 암호문


##############################
# 입력 데이터 파싱
##############################

# 입력 데이터를 파싱하여 각각의 리스트/딕셔너리에 저장
def parse_data(game_data):
    # 입력 데이터를 행으로 나누기
    game_data_rows = game_data.split('\n')
    row_index = 0

    # 첫 번째 행 데이터 읽기
    header = game_data_rows[row_index].split(' ')
    map_height = int(header[0]) if len(header) >= 1 else 0  # 맵의 세로 크기
    map_width = int(header[1]) if len(header) >= 2 else 0  # 맵의 가로 크기
    num_of_allies = int(header[2]) if len(header) >= 3 else 0  # 아군의 수
    num_of_enemies = int(header[3]) if len(header) >= 4 else 0  # 적군의 수
    num_of_codes = int(header[4]) if len(header) >= 5 else 0  # 암호문의 수
    row_index += 1

    # 기존의 맵 정보를 초기화하고 다시 읽어오기
    map_data.clear()
    map_data.extend([['' for c in range(map_width)] for r in range(map_height)])
    for i in range(0, map_height):
        col = game_data_rows[row_index + i].split(' ')
        for j in range(0, len(col)):
            map_data[i][j] = col[j]
    row_index += map_height

    # 기존의 아군 정보를 초기화하고 다시 읽어오기
    my_allies.clear()
    for i in range(row_index, row_index + num_of_allies):
        ally = game_data_rows[i].split(' ')
        ally_name = ally.pop(0) if len(ally) >= 1 else '-'
        my_allies[ally_name] = ally
    row_index += num_of_allies

    # 기존의 적군 정보를 초기화하고 다시 읽어오기
    enemies.clear()
    for i in range(row_index, row_index + num_of_enemies):
        enemy = game_data_rows[i].split(' ')
        enemy_name = enemy.pop(0) if len(enemy) >= 1 else '-'
        enemies[enemy_name] = enemy
    row_index += num_of_enemies

    # 기존의 암호문 정보를 초기화하고 다시 읽어오기
    codes.clear()
    for i in range(row_index, row_index + num_of_codes):
        codes.append(game_data_rows[i])


# 파싱한 데이터를 화면에 출력
def print_data():
    print(f'\n----------입력 데이터----------\n{game_data}\n----------------------------')

    print(f'\n[맵 정보] ({len(map_data)} x {len(map_data[0])})')
    for i in range(len(map_data)):
        for j in range(len(map_data[i])):
            print(f'{map_data[i][j]} ', end='')
        print()

    print(f'\n[아군 정보] (아군 수: {len(my_allies)})')
    for k, v in my_allies.items():
        if k == 'M':
            print(f'M (내 탱크) - 체력: {v[0]}, 방향: {v[1]}, 보유한 일반 포탄: {v[2]}개, 보유한 메가 포탄: {v[3]}개')
        elif k == 'H':
            print(f'H (아군 포탑) - 체력: {v[0]}')
        else:
            print(f'{k} (아군 탱크) - 체력: {v[0]}')

    print(f'\n[적군 정보] (적군 수: {len(enemies)})')
    for k, v in enemies.items():
        if k == 'X':
            print(f'X (적군 포탑) - 체력: {v[0]}')
        else:
            print(f'{k} (적군 탱크) - 체력: {v[0]}')

    print(f'\n[암호문 정보] (암호문 수: {len(codes)})')
    for i in range(len(codes)):
        print(codes[i])


##############################
# 닉네임 설정 및 최초 연결
##############################
NICKNAME = '기본코드'
game_data = init(NICKNAME)


###################################
# 알고리즘 함수/메서드 부분 구현 시작
###################################
def get_boom_count():
    if 'M' in my_allies and len(my_allies['M']) >= 4:
        normal = int(my_allies['M'][2])
        mega = int(my_allies['M'][3])
        return normal, mega
    return 0, 0


DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상
M_CMD = ["R A", "D A", "L A", "U A"]
S_CMD = ["R F", "D F", "L F", "U F"]
START_SYMBOL = "M"
TARGET_SYMBOL = "X"


def get_map_size():
    return len(map_data), len(map_data[0]) if map_data and map_data[0] else (0, 0)


def find_symbol(grid, symbol):
    """맵에서 특정 기호의 위치를 찾아 반환 (row, col)"""
    if not grid or not grid[0]:
        return None
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] == symbol:
                return (row, col)
    return None


def find_firing_positions(target):
    """포탑을 공격할 수 있는 위치 찾기 (사격 경로상 나무 개수 포함)"""
    tx, ty = target
    H, W = get_map_size()
    positions = {}

    for dir_idx, (dr, dc) in enumerate(DIRS):
        for distance in range(1, 4):  # 1~3 거리
            fx = tx - dr * distance  # 목표에서 역방향
            fy = ty - dc * distance

            if not (0 <= fx < H and 0 <= fy < W):
                break

            if map_data[fx][fy] not in ['G', 'T', 'S']:
                continue

            # 사격 경로 확인 및 나무 개수 세기
            path_clear = True
            trees_in_firing_path = 0

            for step in range(1, distance):
                check_x = fx + dr * step
                check_y = fy + dc * step
                terrain = map_data[check_x][check_y]

                # 포탄이 통과할 수 없는 지형이면 불가능
                if terrain not in ['G', 'T', 'W', 'S']:
                    path_clear = False
                    break

                # 사격 경로상 나무 개수 세기
                if terrain == 'T':
                    trees_in_firing_path += 1

            if path_clear:
                if (fx, fy) not in positions:
                    positions[(fx, fy)] = [dir_idx, trees_in_firing_path]

    return positions


def integrated_dijkstra(start, target):
    """
    위치와 포탄 수를 상태로 관리하는 통합 다익스트라
    사격 경로상 나무도 고려하여 포탄 필요량 계산
    """
    H, W = get_map_size()
    sx, sy = start
    tx, ty = target

    # 초기 포탄 수
    init_normal, init_mega = get_boom_count()

    # 사격 가능 위치 찾기
    firing_positions = find_firing_positions(target)
    if not firing_positions:
        return []

    # visited[상태] = 최소 비용
    INF = float('inf')
    visited = {}
    parent = {}

    # 우선순위 큐: (비용, x, y, 일반포탄, 메가포탄)
    pq = []
    start_state = (sx, sy, init_normal, init_mega)
    heapq.heappush(pq, (0, sx, sy, init_normal, init_mega))
    visited[start_state] = 0

    best_result = None
    best_cost = INF

    while pq:
        cost, x, y, normal, mega = heapq.heappop(pq)

        current_state = (x, y, normal, mega)

        # 이미 더 좋은 경로로 방문했으면 스킵
        if current_state in visited and visited[current_state] < cost:
            continue

        # 사격 위치에 도달했는지 확인
        if (x, y) in firing_positions:
            fire_dir, trees_in_firing_path = firing_positions[(x, y)]

            # 사격 경로상 나무 파괴에 필요한 포탄
            bombs_for_trees = trees_in_firing_path
            # 포탑 파괴에 필요한 포탄
            bombs_for_turret = 1
            # 총 필요 포탄
            total_bombs_needed = bombs_for_trees + bombs_for_turret

            # 현재 보유 포탄으로 가능한지 확인
            if normal + mega >= total_bombs_needed:
                # 실제 비용 = 이동 비용 + 사격 경로 나무 파괴 비용
                actual_cost = cost + (bombs_for_trees * 1)

                if actual_cost < best_cost:
                    best_cost = actual_cost
                    best_result = (current_state, parent, fire_dir, trees_in_firing_path)

        # 4방향 탐색
        for dir_idx, (dr, dc) in enumerate(DIRS):
            nx, ny = x + dr, y + dc

            # 맵 범위 체크
            if not (0 <= nx < H and 0 <= ny < W):
                continue

            terrain = map_data[nx][ny]

            # 통과 불가능한 지형
            if terrain in ['R', 'W', 'F', 'H', 'X']:
                continue

            # 나무인 경우 - 여러 선택지
            if terrain == 'T':
                # 옵션 1: 일반 포탄으로 파괴
                if normal > 0:
                    new_state = (nx, ny, normal - 1, mega)
                    new_cost = cost + 2  # 파괴 + 이동

                    if new_state not in visited or visited[new_state] > new_cost:
                        visited[new_state] = new_cost
                        parent[new_state] = (current_state, dir_idx, 'FIRE_NORMAL')
                        heapq.heappush(pq, (new_cost, nx, ny, normal - 1, mega))

                # 옵션 2: 메가 포탄으로 파괴
                if mega > 0:
                    new_state = (nx, ny, normal, mega - 1)
                    new_cost = cost + 2

                    if new_state not in visited or visited[new_state] > new_cost:
                        visited[new_state] = new_cost
                        parent[new_state] = (current_state, dir_idx, 'FIRE_MEGA')
                        heapq.heappush(pq, (new_cost, nx, ny, normal, mega - 1))


            # 잔디인 경우 - 그냥 이동
            elif terrain == 'G':
                new_state = (nx, ny, normal, mega)
                new_cost = cost + 1

                if new_state not in visited or visited[new_state] > new_cost:
                    visited[new_state] = new_cost
                    parent[new_state] = (current_state, dir_idx, 'MOVE')
                    heapq.heappush(pq, (new_cost, nx, ny, normal, mega))

    # 경로 재구성
    if best_result:
        end_state, parent_dict, fire_dir, trees_in_firing_path = best_result
        return reconstruct_actions(parent_dict, start_state, end_state, fire_dir, trees_in_firing_path)

    return []


def reconstruct_actions(parent_dict, start_state, end_state, fire_dir, trees_in_firing_path):
    """상태 기반 경로를 액션 리스트로 변환"""
    actions = []
    path = []
    current = end_state

    # 경로 역추적
    while current != start_state:
        if current not in parent_dict:
            return []

        prev_state, direction, action_type = parent_dict[current]
        path.append((direction, action_type))
        current = prev_state

    path.reverse()

    # 액션 리스트 생성 (경로 이동)
    for direction, action_type in path:
        if action_type == 'MOVE':
            actions.append(M_CMD[direction])
        elif action_type == 'FIRE_NORMAL':
            actions.append(S_CMD[direction])
            actions.append(M_CMD[direction])  # 파괴 후 이동
        elif action_type == 'FIRE_MEGA':
            actions.append(S_CMD[direction] + " M")
            actions.append(M_CMD[direction])  # 파괴 후 이동

    # 사격 경로상 나무 파괴 및 포탑 공격
    x, y, normal, mega = end_state

    # 사격 경로에 나무가 있으면 먼저 파괴
    for i in range(trees_in_firing_path):
        if normal > 0:
            actions.append(S_CMD[fire_dir])
            normal -= 1
        elif mega > 0:
            actions.append(S_CMD[fire_dir] + " M")
            mega -= 1

    # 마지막으로 포탑 공격
    if mega > 0:
        actions.append(S_CMD[fire_dir] + " M")
    elif normal > 0:
        actions.append(S_CMD[fire_dir])

    return actions


###################################
# 알고리즘 함수/메서드 부분 구현 끝
###################################
parse_data(game_data)
actions = []

# 반복문: 메인 프로그램 <-> 클라이언트(이 코드) 간 순차로 데이터 송수신(동기 처리)
while game_data is not None:

    ##############################
    # 알고리즘 메인 부분 구현 시작
    ##############################

    # 파싱한 데이터를 화면에 출력하여 확인
    # print_data()  # 디버깅 시 주석 해제

    # 이전 경로 탐색 결과가 존재하지 않을 경우 다시 탐색
    if not actions:
        start = find_symbol(map_data, 'M')
        target = find_symbol(map_data, 'X')

        if start and target:
            # 통합 다익스트라로 최적 경로 탐색
            actions = integrated_dijkstra(start, target)

    # 탱크를 제어할 명령어를 output의 값으로 지정(type: string)
    output = actions.pop(0) if actions else 'A'

    # 메인 프로그램에서 명령을 처리할 수 있도록 명령어를 submit()의 인자로 전달
    game_data = submit(output)

    # submit()의 리턴으로 받은 갱신된 데이터를 다시 파싱
    if game_data:
        parse_data(game_data)

    ##############################
    # 알고리즘 메인 구현 끝
    ##############################

# 반복문을 빠져나왔을 때 메인 프로그램과의 연결을 완전히 해제하기 위해 close() 호출
close()