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


# 파싱한 데이터를 화면에 출력(디버깅용)
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
NICKNAME = '서울18_이명재'
game_data = init(NICKNAME)

###################################
# 알고리즘 함수/메서드 부분 구현 시작
###################################

# ===== 전역 상태: S 사용 집계 & 안전가드 플래그 =====
S_USAGE_COUNT = 0                   # 모든 "S" 누적 카운트 (최대 2회만 의미있게 사용)
safety_stop_used_this_turn = False  # 이번 턴 안전가드로 정지(S) 사용 여부
stop_counted_this_turn = False      # 이번 턴에 S_USAGE_COUNT 증가 반영 여부

# 전역 변수 - 메가 폭탄 획득 여부 추적 (2개로 고정 기준)
mega_bombs_acquired = False

# 전역 변수 - 방어 루프 상태 추적
current_route = None
route_position = 0
route_direction = 1  # 1: 정방향, -1: 역방향

# ✅ 엔트 시작 턴
ENDGAME_TURN = 30

# 방향 벡터 정의
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상
M_CMD = ["R A", "D A", "L A", "U A"]
S_CMD = ["R F", "D F", "L F", "U F"]
S_MEGA_CMD = ["R F M", "D F M", "L F M", "U F M"]


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


def get_allied_turret_position():
    """아군 포탑(H)의 위치 반환"""
    return find_symbol(map_data, 'H')


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


def get_corner():
    t = get_allied_turret_position()
    if not t:
        return None
    Hn, Wn = get_map_size()
    tr, tc = t
    if tr == 0 and tc == Wn - 1:
        return "TR"
    if tr == Hn - 1 and tc == 0:
        return "BL"
    return None

def is_entry_phase():
    corner = get_corner()
    my_pos = get_my_position()
    if not corner or not my_pos:
        return False
    Hn, Wn = get_map_size()
    r, c = my_pos
    if corner == "BL":
        return (r >= Hn - 2) or (c <= 2)
    if corner == "TR":
        return (r <= 1) or (c >= Wn - 3)
    return False

def try_entry_breakT_move(apply_guard=True):
    if not is_entry_phase():
        return None
    corner = get_corner()
    my_pos = get_my_position()
    if not corner or not my_pos:
        return None
    r, c = my_pos
    d = 0 if corner == "BL" else 2  # RIGHT or LEFT
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
        return None
    if map_data[nr][nc] == 'T':
        return S_CMD[d]
    if map_data[nr][nc] == 'G':
        return apply_safety_guard(M_CMD[d]) if apply_guard else M_CMD[d]
    return None


def find_valid_caesar_decode(ciphertext):
    """유효한 카이사르 해독 찾기 (키워드 매칭)"""
    keywords = [
        "YOUWILLNEVERKNOWUNTILYOUTRY", "THEREISNOROYALROADTOLEARNING",
        "BETTERLATETHANNEVER", "THISTOOSHALLPASSAWAY", "FAITHWITHOUTDEEDSISUSELESS",
        "FORGIVENESSISBETTERTHANREVENGE", "LIFEISNOTALLBEERANDSKITTLES", "UNTILDEATHITISALLLIFE",
        "WHATEVERYOUDOMAKEITPAY", "TIMEISGOLD", "THEONLYCUREFORGRIEFISACTION",
        "GIVEMELIBERTYORGIVEMEDEATH", "APOETISTHEPAINTEROFTHESOUL", "BELIEVEINYOURSELF",
        "NOSWEATNOSWEET", "EARLYBIRDCATCHESTHEWORM", "SEEINGISBELIEVING", "ASKINGCOSTSNOTHING",
        "GOODFENCESMAKESGOODNEIGHBORS", "AROLLINGSTONEGATHERSNOMOSS", "ONESUTMOSTMOVESTHEHEAVENS",
        "LITTLEBYLITTLEDOESTHETRICK", "LIVEASIFYOUWERETODIETOMORROW", "LETBYGONESBEBYGONES",
        "THEBEGINNINGISHALFOFTHEWHOLE", "NOPAINNOGAIN", "STEPBYSTEPGOESALONGWAY",
        "THEDIFFICULTYINLIFEISTHECHOICE", "LIFEISFULLOFUPSANDDOWNS", "ROMEWASNOTBUILTINADAY",
        "IFYOUCANTBEATTHEMJOINTHEM", "NOTHINGVENTUREDNOTHINGGAINED", "KNOWLEDGEINYOUTHISWISDOMINAGE",
        "NOBEESNOHONEY", "WHERETHEREISAWILLTHEREISAWAY", "HABITISSECONDNATURE",
        "SUTMOSTMOVESTHEHEAVENS", "ONLY", "ACTION", "CUREFORGRIEFIS", "SSAFY", "BATTLE",
        "ALGORITHM", "TANK", "MISSION", "CODE", "HERO", "SEIZETHEDAY", "LIFEITSELFISAQUOTATION",
        "LIFEISVENTUREORNOTHING", "DONTDREAMBEIT", "TRYYOURBESTRATHERTHANBETHEBEST",
        "WHATWILLBEWILLBE", "DONTDWELLONTHEPAST", "PASTISJUSTPAST", "FOLLOWYOURHEART",
        "LITTLEBYLITTLEDOESTHETRICK"
    ]
    for shift in range(26):
        decoded = caesar_decode(ciphertext, shift)
        for keyword in keywords:
            if keyword in decoded.upper():
                return decoded
    return caesar_decode(ciphertext, 3)  # 기본 fallback


##############################
# 🔒 경고 구역(Forbidden Zone) 처리
##############################
def get_forbidden_cells():
    Hn, Wn = get_map_size()
    turret = get_allied_turret_position()
    forbidden = set()
    if not turret:
        return forbidden

    tr, tc = turret

    # 맨 우측상단: (0, W-1)
    if tr == 0 and tc == Wn - 1:
        candidates = [
            (0, Wn - 2), (0, Wn - 3),
            (1, Wn - 1), (1, Wn - 2), (1, Wn - 3),
            (2, Wn - 1), (2, Wn - 2), (2, Wn - 3),
        ]
        for r, c in candidates:
            if 0 <= r < Hn and 0 <= c < Wn:
                forbidden.add((r, c))

    # 맨 좌측하단: (H-1, 0)
    if tr == Hn - 1 and tc == 0:
        candidates = [
            (Hn - 1, 1), (Hn - 1, 2),
            (Hn - 2, 0), (Hn - 2, 1), (Hn - 2, 2),
            (Hn - 3, 0), (Hn - 3, 1), (Hn - 3, 2),
        ]
        for r, c in candidates:
            if 0 <= r < Hn and 0 <= c < Wn:
                forbidden.add((r, c))

    return forbidden


def is_forbidden_cell(r, c):
    return (r, c) in get_forbidden_cells()


##############################
# 이동/경로 관련 (다익스트라)
##############################
def terrain_cost(r, c):
    """
    ★ 모든 이동 로직 공통 코스트:
      - G: 1
      - T: 2 (파괴 가능 지형)
      - 그 외/금지칸: 통과 불가(None)
    """
    if is_forbidden_cell(r, c):
        return None
    cell = map_data[r][c]
    if cell == 'G':
        return 1
    if cell == 'T':
        return 2
    return None  # 이동 불가


def reconstruct_commands(prev, start, goal):
    """prev[(r,c)] = (pr,pc, dir_idx) 로부터 명령 리스트 재구성"""
    path_dirs = []
    cur = goal
    while cur != start and cur in prev:
        pr, pc, d_idx = prev[cur]
        path_dirs.append(d_idx)
        cur = (pr, pc)
    path_dirs.reverse()
    return [M_CMD[d] for d in path_dirs]


def dijkstra_to_positions(start, targets):
    """
    다익스트라로 시작점에서 targets 집합(정확히 그 좌표)까지 최소 코스트 경로.
    (G=1, T=2 허용)
    """
    Hn, Wn = get_map_size()
    INF = 10**9
    dist = [[INF]*Wn for _ in range(Hn)]
    prev = {}  # (r,c) -> (pr,pc,dir_idx)
    sr, sc = start
    dist[sr][sc] = 0
    pq = [(0, sr, sc)]

    while pq:
        cost, r, c = heapq.heappop(pq)
        if cost != dist[r][c]:
            continue
        if (r, c) in targets:
            return reconstruct_commands(prev, start, (r, c))
        for d_idx, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < Hn and 0 <= nc < Wn:
                t_cost = terrain_cost(nr, nc)
                if t_cost is None:
                    continue
                ncost = cost + t_cost
                if ncost < dist[nr][nc]:
                    dist[nr][nc] = ncost
                    prev[(nr, nc)] = (r, c, d_idx)
                    heapq.heappush(pq, (ncost, nr, nc))
    return []


def dijkstra_to_first_adjacent_of(symbol):
    """특정 심볼(예: 'F')에 인접한 임의의 칸까지 (G=1,T=2) 최소 코스트 경로."""
    start = get_my_position()
    if not start:
        return []
    Hn, Wn = get_map_size()
    adj_targets = set()
    for r in range(Hn):
        for c in range(Wn):
            if terrain_cost(r, c) is None:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < Hn and 0 <= nc < Wn and map_data[nr][nc] == symbol:
                    if not is_forbidden_cell(r, c):
                        adj_targets.add((r, c))
                    break
    return dijkstra_to_positions(start, adj_targets)


def dijkstra_to_specific(target_pos):
    """특정 좌표까지의 최소 코스트 경로(다익스트라) - (G=1, T=2 허용)"""
    start = get_my_position()
    if not start or not target_pos:
        return []
    if is_forbidden_cell(*target_pos):
        return []
    return dijkstra_to_positions(start, {target_pos})


##############################
# 전투/표적 관련
##############################
def find_enemy_tanks():
    """적 탱크들의 위치 찾기 (E1, E2, E3만)"""
    enemy_positions = []
    Hn, Wn = get_map_size()

    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] in ['E1', 'E2', 'E3']:
                enemy_positions.append((r, c, map_data[r][c]))
    return enemy_positions


def find_all_enemies():
    """모든 적(탱크+포탑) 위치"""
    res = []
    Hn, Wn = get_map_size()
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] in ['E1', 'E2', 'E3', 'X']:
                res.append((r, c, map_data[r][c]))
    return res


def can_attack(from_pos, to_pos):
    """from_pos에서 to_pos를 공격 가능? (사거리3, 직선, G/S/W만 투과)"""
    if not from_pos or not to_pos:
        return False, -1

    fr, fc = from_pos
    tr, tc = to_pos

    # 같은 행
    if fr == tr:
        if fc < tc:  # 오른쪽
            for c in range(fc + 1, tc):
                if map_data[fr][c] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(tc - fc) <= 3:
                return True, 0
        else:       # 왼쪽
            for c in range(tc + 1, fc):
                if map_data[fr][c] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(fc - tc) <= 3:
                return True, 2

    # 같은 열
    elif fc == tc:
        if fr < tr:  # 아래
            for r in range(fr + 1, tr):
                if map_data[r][fc] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(tr - fr) <= 3:
                return True, 1
        else:       # 위
            for r in range(tr + 1, fr):
                if map_data[r][fc] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(fr - tr) <= 3:
                return True, 3

    return False, -1


def get_enemy_hp(enemy_code):
    """적 탱크 HP 조회"""
    try:
        if enemy_code in enemies and len(enemies[enemy_code]) >= 1:
            return int(enemies[enemy_code][0])
    except:
        pass
    return None


def distance(pos1, pos2):
    """두 위치 간의 맨해튼 거리 계산"""
    if not pos1 or not pos2:
        return float('inf')
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def nearest_enemy_distance(pos):
    es = find_all_enemies()
    if not es:
        return float('inf')
    return min(distance(pos, (er, ec)) for er, ec, _ in es)


##############################
# 유틸 & 안전가드
##############################
def cmd_to_dir(cmd):
    for i, m in enumerate(M_CMD):
        if cmd == m:
            return i
    return None


def in_bounds(r, c):
    Hn, Wn = get_map_size()
    return 0 <= r < Hn and 0 <= c < Wn


def is_allied_tank_cell(cell_val):
    """맵 셀 값이 '다른 아군 탱크'인지 판정(M/H/지형/적 제외)"""
    if cell_val in ('', 'G', 'S', 'W', 'T', 'F', 'H', 'M', 'E1', 'E2', 'E3', 'X'):
        return False
    # allies 딕셔너리 키로도 확인 (안전성 강화)
    return cell_val in my_allies and cell_val not in ('M', 'H')


def path_has_other_ally_between(a, b):
    """a와 b가 같은 행/열일 때, 사이 경로에 '다른 아군 탱크'가 존재하면 True"""
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if ac < bc else -1
        for c in range(ac + step, bc, step):
            if is_allied_tank_cell(map_data[ar][c]):
                return True
    elif ac == bc:
        step = 1 if ar < br else -1
        for r in range(ar + step, br, step):
            if is_allied_tank_cell(map_data[r][ac]):
                return True
    return False


def enemies_that_can_shoot_pos_clear(pos):
    """pos를 '차폐 없이' 사격할 수 있는 적 좌표 목록"""
    res = []
    for er, ec, _ in find_all_enemies():
        ok, _d = can_attack((er, ec), pos)
        if ok:
            # can_attack은 이미 차폐가 없는 경우만 True
            res.append((er, ec))
    return res


def enemies_threaten(pos):
    """해당 좌표가 적의 사격경로(차폐 없음)에 있는가"""
    return len(enemies_that_can_shoot_pos_clear(pos)) > 0


def track_stop_and_return_S():
    """S 사용 1회 집계 + 이번 턴 안전정지 플래그 설정(단, 2회 제한 초과 금지)"""
    global S_USAGE_COUNT, safety_stop_used_this_turn, stop_counted_this_turn
    if S_USAGE_COUNT >= 2:
        return None
    S_USAGE_COUNT += 1
    safety_stop_used_this_turn = True
    stop_counted_this_turn = True
    return "S"


def shoot_adjacent_T_if_any():
    """인접 T 우선 사격"""
    mr, mc = get_my_position()
    if (mr, mc) is None:
        return None
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if in_bounds(nr, nc) and map_data[nr][nc] == 'T':
            return S_CMD[i]
    return None


def safe_G_move_if_any():
    """적 사격경로가 아닌 인접 G로 이동"""
    mr, mc = get_my_position()
    if (mr, mc) is None:
        return None
    best = None
    best_score = (-1, float('inf'))  # (적과의 최소거리, turret 거리) 정렬용
    trc = get_allied_turret_position()
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        if enemies_threaten((nr, nc)):
            continue
        # 스코어: 적과 최소거리 최대화, (보조) 기지와 거리 최소화
        min_enemy = float('inf')
        for er, ec, _ in find_all_enemies():
            d = distance((nr, nc), (er, ec))
            if d < min_enemy:
                min_enemy = d
        turret_d = distance((nr, nc), trc) if trc else 0
        score = (min_enemy, turret_d)
        if score > best_score:
            best_score = score
            best = i
    return M_CMD[best] if best is not None else None


def choose_shoot_T_or_safe_move():
    """1순위 인접 T 사격, 2순위 안전 G 이동"""
    shootT = shoot_adjacent_T_if_any()
    if shootT:
        return shootT
    return safe_G_move_if_any()


def get_next_pos_for_move_cmd(cmd):
    d = cmd_to_dir(cmd)
    if d is None:
        return None
    mr, mc = get_my_position()
    dr, dc = DIRS[d]
    return (mr + dr, mc + dc)

def find_allied_tanks_positions():
    H, W = get_map_size()
    out = []
    for r in range(H):
        for c in range(W):
            if is_allied_tank_cell(map_data[r][c]):
                out.append((r, c))
    return out

def move_blocks_ally_fire(next_pos):
    """
    내가 next_pos로 이동하면 '아군→적' 사격 라인을 막는가?
    (아군이 적을 공격 가능한 상태일 때, 그 사이에 내가 들어가면 True)
    """
    allies = find_allied_tanks_positions()
    enemies_all = find_all_enemies()
    for ar, ac in allies:
        for er, ec, _ in enemies_all:
            ok, _ = can_attack((ar, ac), (er, ec))
            if not ok:
                continue
            if next_pos in cells_between_on_line((ar, ac), (er, ec)):
                return True
    return False

def cells_between_on_line(a, b):
    (r1, c1), (r2, c2) = a, b
    cells = []
    if r1 == r2:
        step = 1 if c1 < c2 else -1
        for c in range(c1 + step, c2, step):
            cells.append((r1, c))
    elif c1 == c2:
        step = 1 if r1 < r2 else -1
        for r in range(r1 + step, r2, step):
            cells.append((r, c1))
    return cells

def apply_safety_guard(action):
    """
    ★ 이동명령에 '요청한 규칙' 적용 ★

    if 다음 이동칸이 적 사격경로(차폐 없음):
        if (그 경로상에 이미 다른 우리팀 탱크가 있음) or (그 '위협 적' 중 하나라도 우리 타워(H)를 때릴 수 있는 위치):
            → 이동 허용
        elif (그 경로상 아군이 나밖에 없다) or (내 이동이 '아군→적' 공격 라인을 막는다):
            → 'S'로 멈춤 (전역 카운트 관리)
               단, 이미 이번 턴 S 사용/총 2회 소진이면
               → 1순위 인접 T 사격, 2순위 '사격 경로 아닌' G로 이동
    else:
        → 이동
    """
    d = cmd_to_dir(action)
    if d is None:  # 이동 명령이 아니면 그대로
        return action

    next_pos = get_next_pos_for_move_cmd(action)
    if not next_pos or is_forbidden_cell(*next_pos) or not in_bounds(*next_pos):
        # 갈 수 없는 곳으로의 명령 → 회피 시도
        alt = choose_shoot_T_or_safe_move()
        if alt:
            return alt
        s_try = track_stop_and_return_S()
        return s_try if s_try else action  # 더 못 멈추면 원래 이동 강행

    # 내가 이동해서 '아군→적' 공격 라인을 막는다면(=아군의 사격 경로상으로 들어감) → 멈춤/회피
    if move_blocks_ally_fire(next_pos):
        s_try = track_stop_and_return_S()
        if s_try:
            return s_try
        alt = choose_shoot_T_or_safe_move()
        if alt:
            return alt
        return action  # 최후: 강행

    # 다음 칸 위협 적들
    threats = enemies_that_can_shoot_pos_clear(next_pos)

    if threats:
        # 1) 경로상에 '다른 아군' 차폐가 모두 존재하면 허용
        blocked_by_allies = all(path_has_other_ally_between(e, next_pos) for e in threats)

        # 2) 위협 적 중 하나라도 우리 타워(H)를 때릴 수 있는 자리면 허용
        hits_turret_any = False
        turret = get_allied_turret_position()
        if turret:
            for e in threats:
                if can_attack(e, turret)[0]:
                    hits_turret_any = True
                    break

        if blocked_by_allies or hits_turret_any:
            return action  # 이동 허용

        # 3) 차폐 없고 타워 사정거리도 아니면 → S 멈춤 (가능할 때만)
        s_try = track_stop_and_return_S()
        if s_try:
            return s_try

        # 4) 이미 S 소진 → 1순위 인접 T 사격, 2순위 안전 G 이동, 마지막 강행
        alt = choose_shoot_T_or_safe_move()
        if alt:
            return alt
        return action

    # 위협 없음 → 이동
    return action


##############################
# 이동/경로 유틸
##############################
def next_step_with_T_break(path):
    """
    경로가 있을 때:
      - 다음 칸이 T면 해당 방향으로 사격(S_CMD)하여 길을 연다
      - 다음 칸이 G면 그대로 전진(M_CMD, 단 안전가드 적용 지점에서 호출자가 감싼다)
    """
    if not path:
        return None
    first_cmd = path[0]
    d = cmd_to_dir(first_cmd)
    if d is None:
        return None
    mr, mc = get_my_position()
    dr, dc = DIRS[d]
    nr, nc = mr + dr, mc + dc
    if in_bounds(nr, nc):
        if map_data[nr][nc] == 'T':
            return S_CMD[d]
        if map_data[nr][nc] == 'G':
            return first_cmd
    return None


def fallback_move_away_from_enemies():
    """
    경로가 전혀 없을 때:
      - 인접 G 중 '가장 가까운 적과의 거리'가 최대가 되는 칸으로 1칸 이동
      - 후보 없으면 None
    """
    my_pos = get_my_position()
    if not my_pos:
        return None
    Hn, Wn = get_map_size()
    mr, mc = my_pos

    best_i = None
    best_score = -1
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not (0 <= nr < Hn and 0 <= nc < Wn):
            continue
        if is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        score = nearest_enemy_distance((nr, nc))
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None:
        return M_CMD[best_i]
    return None


##############################
# 정찰 경계: 벽에 붙은 수직 2점 왕복
##############################
def get_optimal_defense_positions():
    """
    정찰(패트롤)은 '벽에 붙은' 수직 2점 왕복으로만:
      - BL(H-1,0):  col=0 에서 (H-4,0) <-> (H-5,0) 우선 시도
      - TR(0,W-1):  col=W-1 에서 (3,W-1) <-> (4,W-1) 우선 시도
    """
    turret_pos = get_allied_turret_position()
    if not turret_pos:
        return []

    tr, tc = turret_pos
    Hn, Wn = get_map_size()

    def ok(r, c):
        if not (0 <= r < Hn and 0 <= c < Wn):
            return False
        if is_forbidden_cell(r, c):
            return False
        return map_data[r][c] in ('G', 'T')

    routes = []

    # BL: 왼쪽 벽(col=0)
    if tr == Hn - 1 and tc == 0:
        col = 0
        seeds = [(Hn - 4, col), (Hn - 5, col)]
        pts = [(r, c) for (r, c) in seeds if ok(r, c)]
        if len(pts) < 2:
            cand_rows = list(range(Hn - 2, max(-1, Hn - 10), -1))
            for r in cand_rows:
                if len(pts) >= 2:
                    break
                if ok(r, col) and (r, col) not in pts:
                    pts.append((r, col))
        if len(pts) >= 2:
            routes.append(("patrol_vertical_left", pts[:2]))
            return routes

    # TR: 오른쪽 벽(col=W-1)
    if tr == 0 and tc == Wn - 1:
        col = Wn - 1
        seeds = [(3, col), (4, col)]
        pts = [(r, c) for (r, c) in seeds if ok(r, c)]
        if len(pts) < 2:
            cand_rows = list(range(1, min(Hn, 10)))
            for r in cand_rows:
                if len(pts) >= 2:
                    break
                if ok(r, col) and (r, col) not in pts:
                    pts.append((r, col))
        if len(pts) >= 2:
            routes.append(("patrol_vertical_right", pts[:2]))
            return routes

    return []


def vertical_step_towards(target):
    """
    같은 열(c)이면 상/하로 한 칸만 이동/사격해서 목표 행(tr)에 다가간다.
      - 다음 칸이 T면 그 방향으로 먼저 사격(S_CMD)
      - 다음 칸이 G면 전진(M_CMD)
    """
    my_pos = get_my_position()
    if not my_pos:
        return None
    r, c = my_pos
    tr, tc = target
    if c != tc:
        return None

    if tr > r:
        d = 1  # 아래
        nr, nc = r + 1, c
    elif tr < r:
        d = 3  # 위
        nr, nc = r - 1, c
    else:
        return None

    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
        return None

    if map_data[nr][nc] == 'T':
        return S_CMD[d]
    if map_data[nr][nc] == 'G':
        return M_CMD[d]
    return None


def dijkstra_to_attack_position(target_pos):
    """
    목표를 공격할 수 있는 위치까지의 최소 코스트 경로(다익스트라).
    현재 칸에서 target_pos에 사격 가능해지는 칸 도달 시 종료. (G=1,T=2)
    """
    start = get_my_position()
    if not start or not target_pos:
        return []

    Hn, Wn = get_map_size()
    INF = 10**9
    dist = [[INF]*Wn for _ in range(Hn)]
    prev = {}
    sr, sc = start
    dist[sr][sc] = 0
    pq = [(0, sr, sc)]

    while pq:
        cost, r, c = heapq.heappop(pq)
        if cost != dist[r][c]:
            continue

        can_hit, _ = can_attack((r, c), target_pos)
        if can_hit:
            return reconstruct_commands(prev, start, (r, c))

        for d_idx, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                t_cost = terrain_cost(nr, nc)
                if t_cost is None:
                    continue
                ncost = cost + t_cost
                if ncost < dist[nr][nc]:
                    dist[nr][nc] = ncost
                    prev[(nr, nc)] = (r, c, d_idx)
                    heapq.heappush(pq, (ncost, nr, nc))
    return []


##############################
# 보급 모드 행동
##############################
def is_adjacent_to_supply():
    """현재 위치가 보급시설(F)에 인접한지 확인"""
    my_pos = get_my_position()
    if not my_pos:
        return False

    r, c = my_pos
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and map_data[nr][nc] == 'F':
            return True
    return False


def find_closest_enemy_to_turret():
    """아군 포탑(H)에 가장 가까운 적 탱크(E1/E2/E3) 하나 반환"""
    turret_pos = get_allied_turret_position()
    if not turret_pos:
        return None
    ets = find_enemy_tanks()
    if not ets:
        return None
    tr, tc = turret_pos
    return min(ets, key=lambda e: abs(e[0]-tr) + abs(e[1]-tc))


def decide_supply_phase_action():
    """
    메가탄 2개 확보 전 동작:
      1) 사거리 내 적 → (메가가 정확히 1개면 메가) 또는 (HP<=30이면 일반)
      2) F 인접 + 코드 → 해독&응답
      3) F 인접까지 경로 추종 (T면 사격, G 이동은 안전가드 적용)
      4) 실패 시 폴백(안전가드 적용)
    """
    my_pos = get_my_position()
    if not my_pos:
        return "S"

    # 1) 선공
    for er, ec, etype in find_enemy_tanks():
        ok, d = can_attack(my_pos, (er, ec))
        if ok:
            hp = get_enemy_hp(etype)
            if get_mega_bomb_count() == 1:
                return S_MEGA_CMD[d]
            if hp is not None and hp <= 30:
                return S_CMD[d]
            break

    # 2) 해독
    if codes and is_adjacent_to_supply():
        decoded = find_valid_caesar_decode(codes[0])
        print(f"암호 : {codes[0]}")
        if decoded:
            print(f"응답 : G{decoded}")
            return f"G {decoded}"

    # 3) F 인접까지
    path = dijkstra_to_first_adjacent_of('F')
    step = next_step_with_T_break(path)
    if step:
        if step in S_CMD:
            return step
        return apply_safety_guard(step)

    # 4) 폴백
    fb = fallback_move_away_from_enemies()
    if fb:
        return apply_safety_guard(fb)

    return "S"


##############################
# 방어(정찰) 모드 행동 - 수직 왕복
##############################
def decide_defense_action():
    """방어 모드: 수직 2점 왕복, 이동엔 안전가드 적용"""
    global current_route, route_position, route_direction, turn_count  # ← turn_count 추가

    # ✅ 70턴부터 엔트 시도
    if turn_count >= ENDGAME_TURN:
        pref = try_entry_breakT_move(apply_guard=True)
        if pref:
            return pref

    my_pos = get_my_position()
    if not my_pos:
        return "S"

    # 1) 사거리 내 적 사격
    for er, ec, etype in find_enemy_tanks():
        ok, d = can_attack(my_pos, (er, ec))
        if ok:
            hp = get_enemy_hp(etype)
            if hp is not None and hp <= 30:
                return S_CMD[d]
            if get_mega_bomb_count() > 0:
                return S_MEGA_CMD[d]
            return S_CMD[d]

    # 2) 정찰 루트 준비/동기화
    defense_routes = get_optimal_defense_positions()
    route_list_only = [route[1] for route in defense_routes]
    if not current_route or (current_route not in route_list_only and route_list_only):
        current_route = route_list_only[0] if route_list_only else None
        route_position = 0
        route_direction = 1

    # 3) 루트 이동 (수직 우선)
    if current_route:
        if my_pos in current_route and len(current_route) > 1:
            idx = current_route.index(my_pos)
            next_idx = idx + route_direction
            if next_idx >= len(current_route):
                next_idx = len(current_route) - 2
                route_direction = -1
            elif next_idx < 0:
                next_idx = 1
                route_direction = 1
            target = current_route[next_idx]
        else:
            target = min(current_route, key=lambda p: distance(my_pos, p))

        vs = vertical_step_towards(target)
        if vs:
            if vs in S_CMD:
                return vs
            return apply_safety_guard(vs)

        path = dijkstra_to_specific(target)
        step = next_step_with_T_break(path)
        if step:
            if step in S_CMD:
                return step
            return apply_safety_guard(step)

    # 4) 포탑에 가까운 적에게 사거리 확보 이동
    ce = find_closest_enemy_to_turret()
    if ce:
        tr, tc, _ = ce
        path = dijkstra_to_attack_position((tr, tc))
        step = next_step_with_T_break(path)
        if step:
            if step in S_CMD:
                return step
            return apply_safety_guard(step)

    # 5) 폴백
    fb = fallback_move_away_from_enemies()
    if fb:
        return apply_safety_guard(fb)

    return "S"


###################################
# 알고리즘 함수/메서드 부분 구현 끝
###################################
parse_data(game_data)
turn_count = 0

def reset_turn_flags():
    """매 턴 시작 시 안전가드/S 집계 플래그 리셋"""
    global safety_stop_used_this_turn, stop_counted_this_turn
    safety_stop_used_this_turn = False
    stop_counted_this_turn = False


def finalize_stop_count(cmd):
    """
    이번 턴 최종 명령이 'S'이고, 아직 카운트 미반영이면 카운트 증가.
    (안전가드에서 이미 반영된 경우는 중복 방지)
    """
    global S_USAGE_COUNT, stop_counted_this_turn
    if cmd == "S" and not stop_counted_this_turn and S_USAGE_COUNT < 2:
        S_USAGE_COUNT += 1
        stop_counted_this_turn = True


# 반복문: 메인 프로그램 <-> 클라이언트(이 코드) 간 순차로 데이터 송수신(동기 처리)
while game_data is not None:
    # 턴 시작: 안전가드 플래그 리셋
    reset_turn_flags()

    output = "S"

    mega_count = get_mega_bomb_count()
    if mega_count >= 2:
        mega_bombs_acquired = True

    if (mega_count < 2) and (not mega_bombs_acquired):
        output = decide_supply_phase_action()
    else:
        output = decide_defense_action()

    # 최종 명령이 S면 누락 집계 보완 (단, 2회 제한)
    finalize_stop_count(output)

    game_data = submit(output)
    if game_data:
        parse_data(game_data)
        turn_count += 1

# 종료
close()
