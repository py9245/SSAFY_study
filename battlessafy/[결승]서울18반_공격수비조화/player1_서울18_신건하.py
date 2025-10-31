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
NICKNAME = '서울18_신건하'
game_data = init(NICKNAME)

###################################
# 알고리즘 함수/메서드 부분 구현 시작
###################################

# ===== 전역 상태: S 사용 집계 & 안전가드 플래그 =====
S_USAGE_COUNT = 0                   # 모든 "S" 대기 명령 누적 카운트 (최대 2회 의미있게 사용)
safety_stop_used_this_turn = False  # 이번 턴 안전정지(S) 사용 여부

# 전역 변수 - 메가 폭탄 획득 여부 추적 (2개로 고정 기준)
mega_bombs_acquired = False

# 전역 변수 - 방어 루프 상태 추적
current_route = None
route_position = 0
route_direction = 1  # 1: 정방향, -1: 역방향

# 전역 변수 - 턴 수 관리
turn_count = 0

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


def find_valid_caesar_decode(ciphertext):
    """유효한 카이사르 해독 찾기 (키워드 매칭)"""
    keywords = [
        "YOUWILLNEVERKNOWUNTILYOUTRY", "THEREISNOROYALROADTOLEARNING",
        "BETTERLATETHANNEVER", "THISTOOSHALLPASSAWAY", "FAITHWITHOUTDEEDSISUSELESS",
        "FORGIVENESSISBETTERTHANREVENGE", "LIFEISNOTALLBEERANDSKITTLES", "UNTILDEATHITISALLLIFE",
        "WHATEVERYOUDOMAKEITPAY", "TIMEISGOLD",
        "THEONLYCUREFORGRIEFISACTION", "GIVEMELIBERTYORGIVEMEDEATH",
        "APOETISTHEPAINTEROFTHESOUL", "BELIEVEINYOURSELF",
        "NOSWEATNOSWEET", "EARLYBIRDCATCHESTHEWORM",
        "SEEINGISBELIEVING", "ASKINGCOSTSNOTHING",
        "GOODFENCESMAKESGOODNEIGHBORS", "AROLLINGSTONEGATHERSNOMOSS",
        "ONESUTMOSTMOVESTHEHEAVENS", "LITTLEBYLITTLEDOESTHETRICK",
        "LIVEASIFYOUWERETODIETOMORROW", "LETBYGONESBEBYGONES",
        "THEBEGINNINGISHALFOFTHEWHOLE", "NOPAINNOGAIN",
        "STEPBYSTEPGOESALONGWAY", "THEDIFFICULTYINLIFEISTHECHOICE",
        "LIFEISFULLOFUPSANDDOWNS", "ROMEWASNOTBUILTINADAY",
        "IFYOUCANTBEATTHEMJOINTHEM", "NOTHINGVENTUREDNOTHINGGAINED",
        "KNOWLEDGEINYOUTHISWISDOMINAGE", "NOBEESNOHONEY",
        "WHERETHEREISAWILLTHEREISAWAY", "HABITISSECONDNATURE",
        "SUTMOSTMOVESTHEHEAVENS",
        "ONLY", "ACTION", "CUREFORGRIEFIS", "SSAFY", "BATTLE",
        "ALGORITHM", "TANK", "MISSION", "CODE", "HERO", "SEIZETHEDAY",
        "LIFEITSELFISAQUOTATION", "LIFEISVENTUREORNOTHING",
        "DONTDREAMBEIT", "TRYYOURBESTRATHERTHANBETHEBEST",
        "WHATWILLBEWILLBE", "DONTDWELLONTHEPAST", "PASTISJUSTPAST",
        "FOLLOWYOURHEART", "LITTLEBYLITTLEDOESTHETRICK"
    ]
    for shift in range(26):
        decoded = caesar_decode(ciphertext, shift)
        for keyword in keywords:
            if keyword in decoded.upper():
                return decoded
    return caesar_decode(ciphertext, 3)  # 기본 fallback


##############################
# 경고 구역(Forbidden Zone) 처리
##############################
def get_forbidden_cells():
    """
    H(아군 포탑)가 '맨 우측상단(0, W-1)' 또는 '맨 좌측하단(H-1, 0)'인 경우,
    문제에서 제시한 9칸 금지 좌표를 세트로 반환.
    그 외 위치라면 빈 세트 반환.
    """
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
      - T: 2
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
    다익스트라로 시작점에서 targets 집합(정확히 그 좌표)까지 최소 코스트 경로 탐색.
    targets: set of (r,c)
    반환: 명령 리스트(한 턴 이동용이면 [0]만 사용)
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
            # 경로 복원
            cmds = reconstruct_commands(prev, start, (r, c))
            return cmds
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
    """
    특정 심볼(예: 'F')에 인접한 임의의 칸까지 도달하는 최소 코스트 경로를 다익스트라로 탐색.
    이동 가능한 칸에서, 인접 4방에 symbol이 있는 지점 도달하면 종료.
    금지칸 제외. (G=1, T=2 허용)
    """
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
    out = []
    Hn, Wn = get_map_size()
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] in ['E1', 'E2', 'E3', 'X']:
                out.append((r, c, map_data[r][c]))
    return out


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
                if map_data[mr][c] not in ['G', 'S', 'W']:
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


def get_enemy_turret_position():
    """적 포탑(X)의 위치 반환"""
    return find_symbol(map_data, 'X')


def get_our_tanks_positions():
    """아군 탱크들(M, M2, M3)의 위치 반환"""
    tanks = []
    Hn, Wn = get_map_size()
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] in ['M', 'M2', 'M3']:
                tanks.append((r, c))
    return tanks


def calculate_x_to_all():
    """X(적군 포탑)로부터 아군 탱크들(M, M2, M3)까지의 거리의 합"""
    x_pos = get_enemy_turret_position()
    if not x_pos:
        return float('inf')

    our_tanks = get_our_tanks_positions()
    total_distance = 0
    for tank_pos in our_tanks:
        total_distance += distance(x_pos, tank_pos)
    return total_distance


def calculate_h_to_enu():
    """H(아군 포탑)로부터 적군 탱크들(E1, E2, E3)까지의 거리의 합"""
    h_pos = get_allied_turret_position()
    if not h_pos:
        return float('inf')

    enemy_tanks = find_enemy_tanks()
    total_distance = 0
    for er, ec, _ in enemy_tanks:
        total_distance += distance(h_pos, (er, ec))
    return total_distance


def nearest_enemy_distance(pos):
    """해당 위치에서 가장 가까운 적까지의 맨해튼 거리"""
    es = find_all_enemies()
    if not es:
        return float('inf')
    return min(distance(pos, (er, ec)) for er, ec, _ in es)


##############################
# 유틸: 아군탱크/차폐/라인 차단 판정
##############################
def in_bounds(r, c):
    Hn, Wn = get_map_size()
    return 0 <= r < Hn and 0 <= c < Wn

def is_allied_tank_cell(cell_val):
    """맵 셀 값이 '다른 아군 탱크'인지 판정(M/H/지형/적 제외)"""
    if cell_val in ('', 'G', 'S', 'W', 'T', 'F', 'H', 'M', 'E1', 'E2', 'E3', 'X'):
        return False
    return cell_val in my_allies and cell_val not in ('M', 'H')

def find_allied_tanks_positions():
    H, W = get_map_size()
    out = []
    for r in range(H):
        for c in range(W):
            if is_allied_tank_cell(map_data[r][c]):
                out.append((r, c))
    return out

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

def enemies_threat_detail(pos):
    """
    pos가 적의 사격경로 안인가? (X 제외 - X는 포격하지 않음)
      - blocked_by_ally: 그 경로상에 '다른 아군 탱크'가 있는가
      - enemy_hits_turret: 그 적이 우리 타워(H)를 때릴 수 있는 위치인가
    """
    allies_set = set(find_allied_tanks_positions())
    turret = get_allied_turret_position()
    for er, ec, etype in find_all_enemies():
        if etype == 'X':  # X는 포격하지 않으므로 위협으로 간주하지 않음
            continue
        ok, _dir = can_attack((er, ec), pos)
        if not ok:
            continue
        blocked_by_ally = any((cr, cc) in allies_set for (cr, cc) in cells_between_on_line((er, ec), pos))
        enemy_hits_turret = False
        if turret:
            can_hit_turret, _ = can_attack((er, ec), turret)
            enemy_hits_turret = can_hit_turret
        return True, blocked_by_ally, enemy_hits_turret
    return False, False, False


##############################
# 유틸: 이동/사격/안전가드 보조
##############################
def cmd_to_dir(cmd):
    for i, m in enumerate(M_CMD):
        if cmd == m:
            return i
    return None

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
    """적 사격경로(차폐 없는) 아닌 인접 G로 이동"""
    mr, mc = get_my_position()
    if (mr, mc) is None:
        return None
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        threat, blocked_by_ally, _ = enemies_threat_detail((nr, nc))
        if not threat or blocked_by_ally:
            return M_CMD[i]
    return None

def get_next_pos_for_move_cmd(cmd):
    d = cmd_to_dir(cmd)
    if d is None:
        return None
    mr, mc = get_my_position()
    dr, dc = DIRS[d]
    return (mr + dr, mc + dc)

def enemies_that_can_shoot_pos_clear(pos):
    """적이 pos를 '차폐 없이' 사격할 수 있으면 목록 반환"""
    res = []
    for er, ec, _ in find_all_enemies():
        ok, _d = can_attack((er, ec), pos)
        if ok:
            res.append((er, ec))
    return res

def next_step_with_T_break(path):
    """
    경로(path)가 있을 때 한 칸 전진을 안전하게 변환:
      - 다음 칸이 T면: 그 방향으로 사격(S_CMD)하여 나무 제거
      - 다음 칸이 G면: 그대로 이동(M_CMD)
      - 경계/금지/이상치면: None
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
    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
        return None
    if map_data[nr][nc] == 'T':
        return S_CMD[d]
    if map_data[nr][nc] == 'G':
        return first_cmd
    return None


##############################
# 이동 안전가드 (요청 규칙 반영)
##############################
def apply_move_safety(proposed_cmd):
    """
    모든 이동에서 다음 규칙 적용:

    if 다음 이동칸이 적 사격경로:
        if (그 경로상에 이미 다른 우리팀 탱크가 있음) or (그 적이 우리 타워를 때릴 수 있는 위치):
            → 이동 허용
        elif (그 경로상 아군이 나밖에 없음) or (내 이동이 아군→적 공격 라인을 막는 경우):
            → 'S'로 멈춤 (전역 카운트 관리)
               단, 이미 이번 턴 'S' 사용했거나 전체 'S' 사용이 2회 이상이면
               → 1순위 인접 T 사격, 2순위 안전 G 이동 (사격경로 아닌)
        else:
            → 방어적으로 동일 처리
    else:
        → 이동
    """
    global safety_stop_used_this_turn, S_USAGE_COUNT

    d = cmd_to_dir(proposed_cmd)
    if d is None:
        return proposed_cmd  # 이동이 아니면 그대로

    next_pos = get_next_pos_for_move_cmd(proposed_cmd)
    if not next_pos or not in_bounds(*next_pos) or is_forbidden_cell(*next_pos):
        # 불가/금지 → S or 대안
        if S_USAGE_COUNT < 2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn = True
            S_USAGE_COUNT += 1
            return "S"
        shoot = shoot_adjacent_T_if_any()
        if shoot:
            return shoot
        safe = safe_G_move_if_any()
        if safe:
            return safe
        return "S"

    # 내가 이동해서 아군의 공격 라인을 막는가?
    if move_blocks_ally_fire(next_pos):
        if S_USAGE_COUNT < 2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn = True
            S_USAGE_COUNT += 1
            return "S"
        shoot = shoot_adjacent_T_if_any()
        if shoot:
            return shoot
        safe = safe_G_move_if_any()
        if safe:
            return safe
        return proposed_cmd  # 대안 없으면 강행

    # 다음 칸에 대한 적 위협 상세
    threat, blocked_by_ally, enemy_hits_turret = enemies_threat_detail(next_pos)

    if threat:
        # (1) 경로상에 다른 아군이 있거나 (2) 그 적이 타워를 칠 수 있는 위치라면 이동 허용
        if blocked_by_ally or enemy_hits_turret:
            return proposed_cmd

        # (3) 아군이 나밖에 없거나 (4) 위협이고 차폐도 없으며 안전치 않으면 멈춤/대안
        if S_USAGE_COUNT < 2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn = True
            S_USAGE_COUNT += 1
            return "S"

        # 이미 이번 턴 S를 썼거나 S 카운트가 2회 도달 → 대안
        shoot = shoot_adjacent_T_if_any()   # 1순위 나무(T) 부수기
        if shoot:
            return shoot
        safe = safe_G_move_if_any()         # 2순위 안전한 G 이동
        if safe:
            return safe
        return proposed_cmd                 # 마지막 대안: 이동 강행

    # 위협이 없다면 이동
    return proposed_cmd


##############################
# 좌/우 전개 유틸 (정찰 보조)
##############################
def horizontal_step_towards(target):
    """
    같은 행(r)이면 좌/우로 한 칸만 이동/사격해서 목표 열(tc)에 다가간다.
    다음 칸이 T면 그 방향으로 사격, G면 전진 (세로 이동 금지)
    """
    my_pos = get_my_position()
    if not my_pos:
        return None
    r, c = my_pos
    tr, tc = target
    if r != tr:
        return None

    if tc > c:
        d, nr, nc = 0, r, c + 1
    elif tc < c:
        d, nr, nc = 2, r, c - 1
    else:
        return None

    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
        return None
    if map_data[nr][nc] == 'T':
        return S_CMD[d]
    if map_data[nr][nc] == 'G':
        return M_CMD[d]
    return None


def horizontal_step_towards_col(target_c):
    """같은 행 여부와 무관하게 좌/우 한 칸만 시도(T면 사격, G면 전진)"""
    my_pos = get_my_position()
    if not my_pos:
        return None
    r, c = my_pos
    if target_c > c:
        d, nr, nc = 0, r, c + 1
    elif target_c < c:
        d, nr, nc = 2, r, c - 1
    else:
        return None
    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
        return None
    if map_data[nr][nc] == 'T':
        return S_CMD[d]
    if map_data[nr][nc] == 'G':
        return M_CMD[d]
    return None


def fallback_move_away_from_enemies_lr():
    """좌/우 후보 중 적과의 최소거리 최대가 되는 G로 1칸 이동"""
    my_pos = get_my_position()
    if not my_pos:
        return None
    r, c = my_pos
    best_i, best_score = None, -1
    for i in (0, 2):  # R/L
        dr, dc = DIRS[i]
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        score = nearest_enemy_distance((nr, nc))
        if score > best_score:
            best_score, best_i = score, i
    return M_CMD[best_i] if best_i is not None else None


##############################
# 방어 루트/행동 (정찰 경계: 코너별 2점 왕복)
##############################
def get_optimal_defense_positions():
    """
    정찰(패트롤) 경로 계산:
      - BL(H-1,0):  r = H-1 에서 좌↔우 스캔(기존 유지)
      - TR(0,tc):   r = 0 에서 '타워와 가로거리 3, 4' 두 점만 좌↔우 왕복  ← 수정
    스페셜 루트가 불가하면 기존 차단 라인 폴백.
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
        return map_data[r][c] in ('G', 'T')  # 이동/사격해 뚫기 허용

    # === BL: 맨 아래 행 좌↔우 패트롤 (기존 유지) ===
    if tr == Hn - 1 and tc == 0:
        r = Hn - 1
        cols = list(range(3, max(3, Wn - 1)))
        pts = [(r, c) for c in cols if ok(r, c)]
        if len(pts) >= 2:
            if len(pts) > 6:
                step = max(1, len(pts) // 6)
                pts = pts[::step]
                if len(pts) < 2:
                    pts = pts[:2]
            return [("patrol_bottom_row", pts)]

    # === TR: 우측 상단일 때 H 심볼 함수로 위치 구해서 -3, -4 적용 ===
    if tr == 0 and tc == Wn - 1:
        # 함수로 H의 실제 위치 구하기
        h_pos = get_allied_turret_position()
        if h_pos:
            hr, hc = h_pos
            candidates = []
            # H의 실제 좌표에서 열(column)에 -3, -4 적용
            for offset in (3, 4):
                nr, nc = hr, hc - offset
                if 0 <= nr < Hn and 0 <= nc < Wn and ok(nr, nc):
                    candidates.append((nr, nc))

            # 유효한 좌표가 있으면 사용
            if len(candidates) >= 1:
                return [("patrol_TR_H_offset", candidates)]

    # === 스페셜 불가 시 기존 차단 라인 폴백 ===
    defense_routes = []

    # 북
    if tr >= Hn // 2:
        nr = tr - 3
        line = []
        if 0 <= nr < Hn:
            for nc in range(max(0, tc - 1), min(Wn, tc + 4)):
                if ok(nr, nc):
                    line.append((nr, nc))
        if line:
            defense_routes.append(("north", line))

    # 동
    if tc <= Wn // 2:
        nc = tc + 3
        line = []
        if 0 <= nc < Wn:
            for nr in range(max(0, tr - 1), min(Hn, tr + 4)):
                if ok(nr, nc):
                    line.append((nr, nc))
        if line:
            defense_routes.append(("east", line))

    # 서
    if tc >= Wn // 2:
        nc = tc - 3
        line = []
        if 0 <= nc < Wn:
            for nr in range(max(0, tr - 1), min(Hn, tr + 4)):
                if ok(nr, nc):
                    line.append((nr, nc))
        if line:
            defense_routes.append(("west", line))

    # 남
    if tr <= Hn // 2:
        nr = tr + 3
        line = []
        if 0 <= nr < Hn:
            for nc in range(max(0, tc - 1), min(Wn, tc + 4)):
                if ok(nr, nc):
                    line.append((nr, nc))
        if line:
            defense_routes.append(("south", line))

    return defense_routes




def find_closest_enemy_to_turret():
    """아군 포탑(H)에 가장 가까운 적 탱크(E1/E2/E3) 반환"""
    turret_pos = get_allied_turret_position()
    if not turret_pos:
        return None
    ets = find_enemy_tanks()
    if not ets:
        return None
    tr, tc = turret_pos
    return min(ets, key=lambda e: abs(e[0]-tr) + abs(e[1]-tc))


def vertical_step_towards(target):
    """
    같은 열(c)이면 위/아래로 한 칸만 이동/사격해서 목표 행(tr)에 다가간다.
    다음 칸이 T면 그 방향으로 사격, G면 전진 (가로 이동 금지)
    """
    my_pos = get_my_position()
    if not my_pos:
        return None
    r, c = my_pos
    tr, tc = target
    if c != tc:
        return None

    if tr > r:
        d, nr, nc = 1, r + 1, c  # 아래
    elif tr < r:
        d, nr, nc = 3, r - 1, c  # 위
    else:
        return None

    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
        return None
    if map_data[nr][nc] == 'T':
        return S_CMD[d]
    if map_data[nr][nc] == 'G':
        return M_CMD[d]
    return None


def fallback_move_away_from_enemies():
    """4방향 후보 중 적과의 최소거리 최대가 되는 G로 1칸 이동"""
    my_pos = get_my_position()
    if not my_pos:
        return None
    r, c = my_pos
    best_i, best_score = None, -1
    for i in range(4):  # 4방향 모두
        dr, dc = DIRS[i]
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        score = nearest_enemy_distance((nr, nc))
        if score > best_score:
            best_score, best_i = score, i
    return M_CMD[best_i] if best_i is not None else None


def decide_defense_action():
    """방어 모드: 수직 2점 왕복, 이동엔 안전가드 적용"""
    global current_route, route_position, route_direction

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
            return apply_move_safety(vs)

        path = dijkstra_to_specific(target)
        step = next_step_with_T_break(path)
        if step:
            if step in S_CMD:
                return step
            return apply_move_safety(step)

    # 4) 포탑에 가까운 적에게 사거리 확보 이동
    ce = find_closest_enemy_to_turret()
    if ce:
        tr, tc, _ = ce
        path = dijkstra_to_attack_position((tr, tc))
        step = next_step_with_T_break(path)
        if step:
            if step in S_CMD:
                return step
            return apply_move_safety(step)

    # 5) 폴백
    fb = fallback_move_away_from_enemies()
    if fb:
        return apply_move_safety(fb)

    return "S"



def closest_supply_to_turret():
    """기지(H)에 가장 가까운 F 하나 좌표"""
    turret = get_allied_turret_position()
    if not turret:
        return None
    Hn, Wn = get_map_size()
    tr, tc = turret
    best = None
    best_d = float('inf')
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] == 'F':
                d = abs(r-tr) + abs(c-tc)
                if d < best_d:
                    best_d, best = d, (r, c)
    return best


def dijkstra_to_adjacent_of_supply_closest_to_turret():
    """기지에 가장 가까운 F의 인접칸(G/T)으로 경로 탐색"""
    start = get_my_position()
    if not start:
        return []
    f = closest_supply_to_turret()
    if not f:
        return []
    Hn, Wn = get_map_size()
    rF, cF = f
    targets = set()
    for dr, dc in DIRS:
        nr, nc = rF + dr, cF + dc
        if in_bounds(nr, nc) and not is_forbidden_cell(nr, nc) and terrain_cost(nr, nc) is not None:
            targets.add((nr, nc))
    if not targets:
        return []
    return dijkstra_to_positions(start, targets)


def dijkstra_to_attack_position(target_pos):
    """목표를 사격 가능한 칸까지 (G=1,T=2)"""
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


def decide_supply_phase_action():
    """
    mega_bombs_acquired == False 동안 행동:
      1) 사거리 내 적 → (메가=1이면 메가) 또는 (HP<=30 일반)
      2) F 인접 && 코드 → 해독/응답
      3) F 인접까지 (G=1,T=2). 첫 칸 T면 사격.
      4) 없으면 좌/우 폴백
    """
    my_pos = get_my_position()
    if not my_pos:
        return "S"

    # 1) 선공
    enemy_tanks = find_enemy_tanks()
    mega_count = get_mega_bomb_count()
    for er, ec, etype in enemy_tanks:
        can_hit, direction = can_attack(my_pos, (er, ec))
        if can_hit:
            enemy_hp = get_enemy_hp(etype)
            if mega_count == 1:
                return S_MEGA_CMD[direction]
            if enemy_hp is not None and enemy_hp <= 30:
                return S_CMD[direction]
            return S_CMD[direction]

    # 2) 해독
    if codes and is_adjacent_to_supply():
        decoded = find_valid_caesar_decode(codes[0])
        print(f"암호 : {codes[0]}")
        if decoded:
            print(f"응답 : G{decoded}")
            return f"G {decoded}"

    # 3) F 인접까지
    path = dijkstra_to_adjacent_of_supply_closest_to_turret()
    step = next_step_with_T_break(path)
    if step:
        if step in M_CMD:
            return apply_move_safety(step)
        return step

    # 4) 폴백
    fb = fallback_move_away_from_enemies_lr()
    if fb:
        if fb in M_CMD:
            return apply_move_safety(fb)
        return fb

    return "S"


##############################
# 전진 모드 관련 함수들
##############################

def get_x_attack_positions():
    """X를 일직선상 3칸 거리에서 공격할 수 있는 위치들 반환"""
    x_pos = get_enemy_turret_position()
    if not x_pos:
        return []

    Hn, Wn = get_map_size()
    xr, xc = x_pos
    positions = []

    # X를 중심으로 4방향 일직선상 3칸 거리 위치
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상

    for dr, dc in directions:
        # 1, 2, 3칸 거리의 위치들 모두 추가
        for dist in [1, 2, 3]:
            nr, nc = xr + dr * dist, xc + dc * dist
            if in_bounds(nr, nc) and not is_forbidden_cell(nr, nc):
                if terrain_cost(nr, nc) is not None:  # 이동 가능한 지형
                    positions.append((nr, nc))

    return positions


def dijkstra_find_advance_target(min_enemy_distance):
    """
    적 탱크의 사정거리(3칸)를 피하면서 X의 일직선상 3칸 거리 위치 중 가장 가까운 곳을 다익스트라로 찾기
    우선순위: 1) 적 탱크 사정거리(3칸) 밖 2) X 공격 가능 위치 3) 가장 가까운 거리
    """
    start = get_my_position()
    if not start:
        return []

    # X의 일직선상 3칸 거리 위치들을 타겟으로 설정
    x_attack_positions = get_x_attack_positions()
    if not x_attack_positions:
        return []

    enemy_tanks = find_enemy_tanks()
    if not enemy_tanks:
        # 적이 없으면 가장 가까운 X 공격 위치로 이동
        return dijkstra_to_positions(start, set(x_attack_positions))

    # 1순위: 적 탱크의 사정거리(3칸) 밖에 있는 X 공격 위치들
    safe_from_attack = []
    for pos in x_attack_positions:
        is_safe = True
        for er, ec, _ in enemy_tanks:
            if distance(pos, (er, ec)) <= 3:  # 적 탱크 사정거리 3칸
                is_safe = False
                break
        if is_safe:
            safe_from_attack.append(pos)

    if safe_from_attack:
        return dijkstra_to_positions(start, set(safe_from_attack))

    # 2순위: 적들과 min_enemy_distance 이상 떨어진 X 공격 위치들
    valid_targets = []
    for pos in x_attack_positions:
        min_dist_to_enemy = min(distance(pos, (er, ec)) for er, ec, _ in enemy_tanks)
        if min_dist_to_enemy >= min_enemy_distance:
            valid_targets.append(pos)

    if valid_targets:
        return dijkstra_to_positions(start, set(valid_targets))

    # 3순위: 모든 X 공격 위치 중 가장 가까운 곳 (최후의 선택)
    return dijkstra_to_positions(start, set(x_attack_positions))


def is_at_advance_target(min_enemy_distance):
    """현재 위치가 X의 일직선상 3칸 거리 위치이면서 적과 안전거리 확보했는지 확인"""
    my_pos = get_my_position()
    if not my_pos:
        return False

    # X의 일직선상 3칸 거리 위치들 중 하나인지 확인
    x_attack_positions = get_x_attack_positions()
    if my_pos not in x_attack_positions:
        return False

    # 모든 적과 min_enemy_distance 이상 떨어져 있는지 확인
    enemy_tanks = find_enemy_tanks()
    if not enemy_tanks:
        return True

    for er, ec, _ in enemy_tanks:
        if distance(my_pos, (er, ec)) < min_enemy_distance:
            return False

    return True


def decide_advance_action(min_enemy_distance):
    """전진 모드 행동 결정"""
    my_pos = get_my_position()
    if not my_pos:
        return "S"

    # 1) 사거리 내 적 사격 (적 탱크 우선)
    for er, ec, etype in find_enemy_tanks():
        ok, d = can_attack(my_pos, (er, ec))
        if ok:
            hp = get_enemy_hp(etype)
            if hp is not None and hp <= 30:
                return S_CMD[d]
            if get_mega_bomb_count() > 0:
                return S_MEGA_CMD[d]
            return S_CMD[d]

    # X(적 포탑) 사격 - X는 포격하지 않으므로 회피할 필요 없음
    x_pos = get_enemy_turret_position()
    if x_pos:
        can_shoot_x, direction = can_attack(my_pos, x_pos)
        if can_shoot_x:
            if get_mega_bomb_count() > 0:
                return S_MEGA_CMD[direction]
            return S_CMD[direction]

    # 2) 현재 위치가 목표 위치인지 확인
    if is_at_advance_target(min_enemy_distance):
        # 우리 타워(G) 방향으로 이동하거나 주변 나무 사격
        turret_pos = get_allied_turret_position()
        if turret_pos:
            # 타워 방향으로 한 걸음 이동
            path = dijkstra_to_specific(turret_pos)
            step = next_step_with_T_break(path)
            if step:
                if step in M_CMD:
                    return apply_move_safety(step)
                return step

        # 주변 나무 사격
        shoot_tree = shoot_adjacent_T_if_any()
        if shoot_tree:
            return shoot_tree

    # 3) 전진 목표 위치로 이동
    path = dijkstra_find_advance_target(min_enemy_distance)
    step = next_step_with_T_break(path)
    if step:
        if step in M_CMD:
            return apply_move_safety(step)
        return step

    # 4) 폴백
    fb = fallback_move_away_from_enemies_lr()
    if fb:
        if fb in M_CMD:
            return apply_move_safety(fb)
        return fb

    return "S"


###################################
# 알고리즘 함수/메서드 부분 구현 끝
###################################
parse_data(game_data)

# 반복문: 메인 프로그램 <-> 클라이언트(이 코드) 간 동기 처리
while game_data is not None:
    # 턴 수 증가
    turn_count += 1

    # 턴 시작 시 안전정지 플래그 리셋
    safety_stop_used_this_turn = False

    output = "S"  # 기본값

    # 현재 메가 폭탄 개수 확인
    mega_count = get_mega_bomb_count()

    # 메가 2개 이상 확보 시 전역 플래그 고정
    if mega_count >= 2:
        mega_bombs_acquired = True

    # 거리 계산 (턴 70 이후에만)
    x_to_all = 0
    h_to_enu = 0
    if turn_count >= 70:
        x_to_all = calculate_x_to_all()
        h_to_enu = calculate_h_to_enu()

    # (1) 보급 모드
    if (mega_count < 2) and (not mega_bombs_acquired):
        proposed = decide_supply_phase_action()
    # (2) 전진 모드 (턴 70 이상)
    elif turn_count >= 70:
        # 턴 90 이상: 거리 2
        if turn_count >= 90 and x_to_all + 1 >= h_to_enu:
            proposed = decide_advance_action(2)
        # 턴 80 이상: 거리 4
        elif turn_count >= 80 and x_to_all + 2 > h_to_enu:
            proposed = decide_advance_action(3)
        # 턴 70 이상: 거리 6
        else:
            proposed = decide_advance_action(6)
    # (3) 방어 모드
    else:
        proposed = decide_defense_action()

    # ----- 이동 안전가드 적용 -----
    final_cmd = proposed
    if proposed in M_CMD:
        final_cmd = apply_move_safety(proposed)

    # 모든 S는 전역 카운트 (안전가드에서 이미 올렸으면 여기선 증가 안 함)
    if final_cmd == "S" and not safety_stop_used_this_turn and S_USAGE_COUNT < 2:
        S_USAGE_COUNT += 1
        safety_stop_used_this_turn = True

    # 명령 전송
    game_data = submit(final_cmd)

    # 수신 데이터 파싱
    if game_data:
        parse_data(game_data)

# 종료
close()
