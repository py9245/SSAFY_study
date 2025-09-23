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
NICKNAME = '갑옷'
game_data = init(NICKNAME)

###################################
# 알고리즘 함수/메서드 부분 구현 시작
###################################

# ===== 전역 상태: S 사용 집계 & 안전가드 플래그 =====
S_USAGE_COUNT = 0                   # 모든 "S" 대기 명령 누적 카운트
safety_stop_used_this_turn = False  # 이번 턴 안전정지(S) 사용 여부

# 전역 변수 - 메가 폭탄 획득 여부 추적 (2개로 고정 기준)
mega_bombs_acquired = False

# 전역 변수 - 방어 루프 상태 추적
current_route = None
route_position = 0
route_direction = 1  # 1: 정방향, -1: 역방향

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


def nearest_enemy_distance(pos):
    """해당 위치에서 가장 가까운 적까지의 맨해튼 거리"""
    es = find_all_enemies()
    if not es:
        return float('inf')
    return min(distance(pos, (er, ec)) for er, ec, _ in es)


##############################
# 유틸: 이동/사격/안전가드
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
    """적이 pos를 '차폐 없이' 사격할 수 있으면 목록 반환"""
    res = []
    for er, ec, _ in find_all_enemies():
        ok, _d = can_attack((er, ec), pos)
        if ok:
            res.append((er, ec))
    return res


def enemies_aligned_but_blocked_by_ally(pos):
    """정렬은 됐지만 사이에 다른 아군 탱크가 있어 차폐되는 적 목록"""
    res = []
    for er, ec, _ in find_all_enemies():
        # 정렬(+사거리) 판정은 can_attack이 해주지만, 여기선 '차폐 있음'만 수집
        mr, mc = pos
        if er == mr and abs(ec - mc) <= 3 and path_has_other_ally_between((er, ec), pos):
            res.append((er, ec))
        if ec == mc and abs(er - mr) <= 3 and path_has_other_ally_between((er, ec), pos):
            res.append((er, ec))
    return res


def next_step_with_T_break(path):
    """
    경로가 있을 때:
      - 다음 칸이 T면 해당 방향으로 사격(S_CMD)하여 길을 연다
      - 다음 칸이 G면 그대로 전진(M_CMD)
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
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        # '차폐 없는' 위협이 없어야 안전
        if not enemies_that_can_shoot_pos_clear((nr, nc)):
            return M_CMD[i]
    return None


def get_next_pos_for_move_cmd(cmd):
    d = cmd_to_dir(cmd)
    if d is None:
        return None
    mr, mc = get_my_position()
    dr, dc = DIRS[d]
    return (mr + dr, mc + dc)


def apply_move_safety(proposed_cmd):
    """
    이동명령에 안전가드를 적용:
      - 다음 칸이 적의 '차폐 없는' 사격경로면:
          · 사이에 '다른 아군 탱크'가 있으면 이동 허용
          · 없으면: (S<2이고 이번 턴 안전정지 미사용) → 'S'
                   그 외 → 1) 인접 T 사격, 2) 안전한 G로 이동, 3) 원래 명령 유지
    """
    global safety_stop_used_this_turn, S_USAGE_COUNT

    d = cmd_to_dir(proposed_cmd)
    if d is None:
        return proposed_cmd  # 이동이 아니면 그대로

    next_pos = get_next_pos_for_move_cmd(proposed_cmd)
    if not next_pos or not in_bounds(*next_pos):
        return proposed_cmd

    # 적 위협 판정
    clear_threats = enemies_that_can_shoot_pos_clear(next_pos)

    if not clear_threats:
        # 위협 없음(또는 아군으로 차폐됨) → 이동
        return proposed_cmd

    # 일부/전부가 '다른 아군 탱크'로 차폐되는지 확인
    blocked_list = [e for e in clear_threats if path_has_other_ally_between(e, next_pos)]
    if blocked_list:
        # 경로에 다른 아군이 있어 사실상 안전 → 이동 허용
        return proposed_cmd

    # 여기 오면 '차폐 없는' 위협 존재 & 나만 노출
    if (not safety_stop_used_this_turn) and (S_USAGE_COUNT < 2):
        safety_stop_used_this_turn = True
        return "S"

    # S를 더 못 쓰는 경우: 1순위 T 사격, 2순위 안전 G 이동
    shootT = shoot_adjacent_T_if_any()
    if shootT:
        return shootT
    safeMove = safe_G_move_if_any()
    if safeMove:
        return safeMove

    # 그래도 대안 없으면 원래 이동 강행
    return proposed_cmd


##############################
# 방어 루트/행동 (정찰 경계: 코너별 2점 왕복)
##############################
def get_optimal_defense_positions():
    """
    정찰(패트롤)은 가로로만:
      - 기지 좌하단(H-1,0): (H-1,3) <-> (H-1,4)
      - 기지 우상단(0,W-1): (0,W-4) <-> (0,W-5)
    위 좌표가 불가하면 기존 차단 라인으로 fallback.
    """
    turret_pos = get_allied_turret_position()
    if not turret_pos:
        return []

    tr, tc = turret_pos
    Hn, Wn = get_map_size()

    def ok(r, c):
        if not in_bounds(r, c) or is_forbidden_cell(r, c):
            return False
        return terrain_cost(r, c) is not None  # G/T 허용

    # 코너별 가로 2점 경로
    special_route = []
    if tr == Hn - 1 and tc == 0:  # 좌하단
        cands = [(Hn - 1, 3), (Hn - 1, 4)]
        special_route = [(r, c) for (r, c) in cands if ok(r, c)]
    elif tr == 0 and tc == Wn - 1:  # 우상단
        cands = [(0, Wn - 4), (0, Wn - 5)]
        special_route = [(r, c) for (r, c) in cands if ok(r, c)]

    if len(special_route) >= 2:
        return [("patrol_lr", special_route)]

    # --- 스페셜 불가 시 기존 차단 라인 ---
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


def decide_defense_action():
    """방어 모드: 정찰(좌↔우) 우선, 경로 없으면 좌/우 폴백"""
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

    # 2) 정찰 루트(좌↔우) 준비
    defense_routes = get_optimal_defense_positions()
    route_list_only = [route[1] for route in defense_routes]
    if not current_route or (current_route not in route_list_only and route_list_only):
        current_route = route_list_only[0] if route_list_only else None
        route_position, route_direction = 0, 1

    # 3) 루트 이동(좌/우 우선)
    if current_route:
        if my_pos in current_route and len(current_route) > 1:
            idx = current_route.index(my_pos)
            nxt = idx + route_direction
            if nxt >= len(current_route):
                nxt = len(current_route) - 2
                route_direction = -1
            elif nxt < 0:
                nxt = 1
                route_direction = 1
            target = current_route[nxt]
        else:
            target = min(current_route, key=lambda p: distance(my_pos, p))

        # 같은 행 우선 전개
        hs = horizontal_step_towards(target)
        if hs:
            return hs

        hs2 = horizontal_step_towards_col(target[1])
        if hs2:
            return hs2

        fb = fallback_move_away_from_enemies_lr()
        if fb:
            return fb

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
            break  # 이동/해독으로

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
        return step

    # 4) 폴백
    fb = fallback_move_away_from_enemies_lr()
    if fb:
        return fb

    return "S"


###################################
# 알고리즘 함수/메서드 부분 구현 끝
###################################
parse_data(game_data)
turn_count = 0

# 반복문: 메인 프로그램 <-> 클라이언트(이 코드) 간 동기 처리
while game_data is not None:
    # 턴 시작 시 안전정지 플래그 리셋
    safety_stop_used_this_turn = False

    output = "S"  # 기본값

    # 현재 메가 폭탄 개수 확인
    mega_count = get_mega_bomb_count()

    # 메가 2개 이상 확보 시 전역 플래그 고정
    if mega_count >= 2:
        mega_bombs_acquired = True

    # (1) 보급 모드
    if (mega_count < 2) and (not mega_bombs_acquired):
        proposed = decide_supply_phase_action()
    # (2) 방어 모드
    else:
        proposed = decide_defense_action()

    # ----- 이동 안전가드 적용 -----
    final_cmd = proposed
    if proposed in M_CMD:
        final_cmd = apply_move_safety(proposed)

    # 모든 S는 전역 카운트
    if final_cmd == "S":
        S_USAGE_COUNT += 1

    # 명령 전송
    game_data = submit(final_cmd)

    # 수신 데이터 파싱
    if game_data:
        parse_data(game_data)

# 종료
close()
