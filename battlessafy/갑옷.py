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
NICKNAME = '갑옷'
game_data = init(NICKNAME)


###################################
# 알고리즘 함수/메서드 부분 구현 시작
###################################

# 전역 변수 - 메가 폭탄 획득 여부 추적
mega_bombs_acquired = False

# 방향 벡터 정의
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상
M_CMD = ["R A", "D A", "L A", "U A"]
S_CMD = ["R F", "D F", "L F", "U F"]
S_MEGA_CMD = ["R F M", "D F M", "L F M", "U F M"]


# ★ get_map_size 괄호 보정 (우선순위 오류 방지)
def get_map_size():
    if map_data and map_data[0]:
        return len(map_data), len(map_data[0])
    return (0, 0)


def find_symbol(grid, symbol):
    """맵에서 특정 기호의 위치를 찾아 반환"""
    if not grid or not grid[0]:
        return None
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] == symbol:
                return (row, col)
    return None


def get_my_position():
    """내 탱크(M)의 위치 반환"""
    return find_symbol(map_data, 'M')


def get_mega_bomb_count():
    """내가 보유한 메가 폭탄 수 반환"""
    if 'M' in my_allies and len(my_allies['M']) >= 4:
        return int(my_allies['M'][3])
    return 0


def caesar_decode(ciphertext, shift):
    """카이사르 암호 해독"""
    result = ""
    for char in ciphertext:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base - shift) % 26 + base)
        else:
            result += char
    return result


def find_valid_caesar_decode(ciphertext):
    """유효한 카이사르 해독 찾기 (키워드 매칭)"""
    keywords = [
              "YOUWILLNEVERKNOWUNTILYOUTRY",
              "THEREISNOROYALROADTOLEARNING",
              "BETTERLATETHANNEVER",
              "THISTOOSHALLPASSAWAY",
              "FAITHWITHOUTDEEDSISUSELESS",
              "FORGIVENESSISBETTERTHANREVENGE",
              "LIFEISNOTALLBEERANDSKITTLES",
              "UNTILDEATHITISALLLIFE",
              "WHATEVERYOUDOMAKEITPAY",
              "TIMEISGOLD",
              "THEONLYCUREFORGRIEFISACTION",
              "GIVEMELIBERTYORGIVEMEDEATH",
              "APOETISTHEPAINTEROFTHESOUL",
              "BELIEVEINYOURSELF",
              "NOSWEATNOSWEET",
              "EARLYBIRDCATCHESTHEWORM",
              "SEEINGISBELIEVING",
              "ASKINGCOSTSNOTHING",
              "GOODFENCESMAKESGOODNEIGHBORS",
              "AROLLINGSTONEGATHERSNOMOSS",
              "ONESUTMOSTMOVESTHEHEAVENS",
              "LITTLEBYLITTLEDOESTHETRICK",
              "LIVEASIFYOUWERETODIETOMORROW",
              "LETBYGONESBEBYGONES",
              "THEBEGINNINGISHALFOFTHEWHOLE",
              "NOPAINNOGAIN",
              "STEPBYSTEPGOESALONGWAY",
              "THEDIFFICULTYINLIFEISTHECHOICE",
              "LIFEISFULLOFUPSANDDOWNS",
              "ROMEWASNOTBUILTINADAY",
              "IFYOUCANTBEATTHEMJOINTHEM",
              "NOTHINGVENTUREDNOTHINGGAINED",
              "KNOWLEDGEINYOUTHISWISDOMINAGE",
              "NOBEESNOHONEY",
              "WHERETHEREISAWILLTHEREISAWAY",
              "HABITISSECONDNATURE",
              "DON",
              "DAY",
              "SUTMOSTMOVESTHEHEAVENS",
              "ONE",
              "THE",
              "ONLY",
              "ACTION",
              "CUREFORGRIEFIS",
              "SSAFY",
              "BATTLE",
              "ALGORITHM",
              "TANK",
              "MISSION",
              "CODE",
              "HERO",
              "SEIZETHEDAY",
              "LIFEITSELFISAQUOTATION",
              "LIFEISVENTUREORNOTHING",
              "DONTDREAMBEIT",
              "TRYYOURBESTRATHERTHANBETHEBEST",
              "WHATWILLBEWILLBE",
              "DONTDWELLONTHEPAST",
              "PASTISJUSTPAST"
            ]
    for shift in range(26):
        decoded = caesar_decode(ciphertext, shift)
        for keyword in keywords:
            if keyword in decoded.upper():
                return decoded
    return caesar_decode(ciphertext, 3)  # 기본 fallback


def bfs_to_adjacent_supply():
    """BFS로 보급시설(F)에 인접한 가장 가까운 위치까지의 경로 찾기"""
    my_pos = get_my_position()
    if not my_pos:
        return []

    H, W = get_map_size()
    visited = set()
    queue = deque([(my_pos, [])])
    visited.add(my_pos)

    while queue:
        (r, c), path = queue.popleft()

        # F에 인접한지 확인
        for i, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and map_data[nr][nc] == 'F':
                return path  # F 인접 위치 도달

        # 4방향 탐색
        for i, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and
                (nr, nc) not in visited and
                map_data[nr][nc] in ['G', 'S']):  # 이동 가능한 지형
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [M_CMD[i]]))

    return []


def is_adjacent_to_supply():
    """현재 위치가 보급시설(F)에 인접한지 확인"""
    my_pos = get_my_position()
    if not my_pos:
        return False

    H, W = get_map_size()
    r, c = my_pos

    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and map_data[nr][nc] == 'F':
            return True

    return False


def find_enemy_tanks():
    """적 탱크들의 위치 찾기"""
    enemy_positions = []
    H, W = get_map_size()

    for r in range(H):
        for c in range(W):
            if map_data[r][c] in ['E1', 'E2', 'E3']:
                enemy_positions.append((r, c, map_data[r][c]))

    return enemy_positions


def find_enemy_turret():
    """적 포탑(X)의 위치 찾기"""
    return find_symbol(map_data, 'X')


def can_attack(my_pos, target_pos):
    """현재 위치에서 목표를 공격할 수 있는지 확인 (사거리 3, 직선/차폐 단순판정)"""
    if not my_pos or not target_pos:
        return False, -1

    mr, mc = my_pos
    tr, tc = target_pos

    # 같은 행
    if mr == tr:
        if mc < tc:  # 오른쪽
            for c in range(mc + 1, tc):
                if map_data[mr][c] not in ['G', 'S', 'W']:  # 단순 통과 허용셋
                    return False, -1
            if abs(tc - mc) <= 3:
                return True, 0  # 오른쪽
        else:  # 왼쪽
            for c in range(tc + 1, mc):
                if map_data[mr][c] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(tc - mc) <= 3:
                return True, 2  # 왼쪽

    # 같은 열
    elif mc == tc:
        if mr < tr:  # 아래
            for r in range(mr + 1, tr):
                if map_data[r][mc] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(tr - mr) <= 3:
                return True, 1  # 아래
        else:  # 위
            for r in range(tr + 1, mr):
                if map_data[r][mc] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(tr - mr) <= 3:
                return True, 3  # 위

    return False, -1


def bfs_to_attack_position(target_pos):
    """목표를 공격할 수 있는 위치까지의 최단 경로 찾기"""
    my_pos = get_my_position()
    if not my_pos or not target_pos:
        return []

    H, W = get_map_size()
    visited = set()
    queue = deque([(my_pos, [])])
    visited.add(my_pos)

    while queue:
        (r, c), path = queue.popleft()

        # 현재 위치에서 목표 공격 가능한지 확인
        can_hit, _ = can_attack((r, c), target_pos)
        if can_hit:
            return path

        # 4방향 탐색
        for i, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and
                (nr, nc) not in visited and
                map_data[nr][nc] in ['G', 'S']):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [M_CMD[i]]))

    return []


# ★ 추가: 적 탱크 HP 조회
def get_enemy_hp(enemy_code):
    try:
        if enemy_code in enemies and len(enemies[enemy_code]) >= 1:
            return int(enemies[enemy_code][0])
    except:
        pass
    return None  # HP 정보 불명


# ★ 추가: 메가 1개 이후 1순위 - 가장 가까운 적 탱크 공격/추격
def decide_vs_nearest_enemy():
    my_pos = get_my_position()
    if not my_pos:
        return "S"

    enemy_tanks = find_enemy_tanks()
    if not enemy_tanks:
        return None  # 적 탱크가 없으면 None 반환

    # 1) 지금 당장 사격 가능한 적이 있으면 우선 사격
    for (er, ec, etype) in enemy_tanks:
        can_hit, direction = can_attack(my_pos, (er, ec))
        if can_hit:
            hp = get_enemy_hp(etype)
            mega_cnt = get_mega_bomb_count()
            if hp is not None and hp <= 30:
                return S_CMD[direction]          # 30 이하면 일반탄
            elif mega_cnt > 0:
                return S_MEGA_CMD[direction]     # 그 이상이면 메가(보유 시)
            else:
                return S_CMD[direction]          # 메가 없으면 일반

    # 2) 사격 불가면, 사격 가능 위치까지 경로가 가장 짧은 적을 선택해 한 칸 전진
    best_path = None
    for (er, ec, etype) in enemy_tanks:
        path = bfs_to_attack_position((er, ec))
        if path:
            if (best_path is None) or (len(path) < len(best_path)):
                best_path = path

    if best_path:
        return best_path[0]

    return "S"  # 완전히 막혔으면 대기


###################################
# 알고리즘 함수/메서드 부분 구현 끝
###################################
parse_data(game_data)

# 반복문: 메인 프로그램 <-> 클라이언트(이 코드) 간 순차로 데이터 송수신(동기 처리)
while game_data is not None:

    ##############################
    # 알고리즘 메인 부분 구현 시작
    ##############################

    output = "S"  # 기본값: 대기

    # 현재 메가 폭탄 개수 확인
    mega_count = get_mega_bomb_count()

    # ★ 메가 1개 이상 확보 시 전역 플래그 고정
    if mega_count >= 10:
        mega_bombs_acquired = True

    # 1순위: (메가 < 1) 이고 (아직 보급 완료 아님) → 보급 인접 이동/해독
    if (mega_count < 10) and (not mega_bombs_acquired):
        if codes and is_adjacent_to_supply():
            decoded = find_valid_caesar_decode(codes[0])
            # 디버깅용 출력(원하면 주석)
            print(codes[0])
            if decoded:
                output = f"G {decoded}"
                # 실제 메가 증가 여부는 다음 턴에서 mega_count로 반영
        else:
            path = bfs_to_adjacent_supply()
            if path:
                output = path[0]

    # 2순위: 메가 1개 확보 이후 → "가장 가까운 적 탱크" 우선 공격/추격
    else:
        act = decide_vs_nearest_enemy()
        if act and act != "S":
            output = act
        else:
            # 적 탱크를 당장 처리할 수 없으면 포탑(X) 로직
            my_pos = get_my_position()
            turret_pos = find_enemy_turret()
            if turret_pos and my_pos:
                can_hit, direction = can_attack(my_pos, turret_pos)
                if can_hit:
                    if mega_count > 0:
                        output = S_MEGA_CMD[direction]
                    else:
                        output = S_CMD[direction]
                else:
                    path = bfs_to_attack_position(turret_pos)
                    if path:
                        output = path[0]
            else:
                output = "S"

    ##############################
    # 알고리즘 메인 구현 끝
    ##############################

    # 메인 프로그램에서 명령을 처리할 수 있도록 명령어를 submit()의 인자로 전달
    game_data = submit(output)

    # submit()의 리턴으로 받은 갱신된 데이터를 다시 파싱
    if game_data:
        parse_data(game_data)

# 반복문을 빠져나왔을 때 메인 프로그램과의 연결을 완전히 해제하기 위해 close() 호출
close()