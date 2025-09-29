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
    output_filename = 'game_state_output_1.txt'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f'\n----------입력 데이터----------\n{game_data}\n----------------------------\n')
        f.write(f'\n[맵 정보] ({len(map_data)} x {len(map_data[0])})\n')
        for i in range(len(map_data)):
            row_str = ''
            for j in range(len(map_data[i])):
                row_str += f'{map_data[i][j]} '
            f.write(row_str.strip() + '\n')
        f.write(f'\n[아군 정보] (아군 수: {len(my_allies)})\n')
        for k, v in my_allies.items():
            if k == 'M':
                f.write(f'M (내 탱크) - 체력: {v[0]}, 방향: {v[1]}, 보유한 일반 포탄: {v[2]}개, 보유한 메가 포탄: {v[3]}개\n')
            elif k == 'H':
                f.write(f'H (아군 포탑) - 체력: {v[0]}\n')
            else:
                f.write(f'{k} (아군 탱크) - 체력: {v[0]}\n')
        f.write(f'\n[적군 정보] (적군 수: {len(enemies)})\n')
        for k, v in enemies.items():
            if k == 'X':
                f.write(f'X (적군 포탑) - 체력: {v[0]}\n')
            else:
                f.write(f'{k} (적군 탱크) - 체력: {v[0]}\n')
        f.write(f'\n[암호문 정보] (암호문 수: {len(codes)})\n')
        for i in range(len(codes)):
            f.write(codes[i] + '\n')
    print(f"게임 상태 정보가 '{output_filename}' 파일에 저장되었습니다. 💾")


##############################
# 닉네임 설정 및 최초 연결
##############################
NICKNAME = '40턴_이명재'
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

# 방향 벡터 정의
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상
M_CMD = ["R A", "D A", "L A", "U A"]
S_CMD = ["R F", "D F", "L F", "U F"]
S_MEGA_CMD = ["R F M", "D F M", "L F M", "U F M"]

# ★ 추가: 상대 기지(X) 축방향 오프셋(열/행) 설정
BASE_OFF_C = 2  # x 좌표 +- 값(열 이동)
BASE_OFF_R = 2  # y 좌표 +- 값(행 이동)


def get_map_size():
    if map_data and map_data[0]:
        return len(map_data), len(map_data[0])
    return (0, 0)


def find_symbol(grid, symbol):
    if not grid or not grid[0]:
        return None
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] == symbol:
                return (row, col)
    return None


def get_my_position():
    return find_symbol(map_data, 'M')


def get_allied_turret_position():
    return find_symbol(map_data, 'H')


def get_enemy_turret_position():
    return find_symbol(map_data, 'X')  # ★ 추가: 적 포탑 위치


def get_mega_bomb_count():
    if 'M' in my_allies and len(my_allies['M']) >= 4:
        return int(my_allies['M'][3])
    return 0


def caesar_decode(ciphertext, shift):
    result = ""
    for char in ciphertext:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base - shift) % 26 + base)
        else:
            result += char
    return result


def find_valid_caesar_decode(ciphertext):
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
    return caesar_decode(ciphertext, 3)


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
    # G:1, T:2, 나머지/금지:불가
    if is_forbidden_cell(r, c):
        return None
    cell = map_data[r][c]
    if cell == 'G':
        return 1
    if cell == 'T':
        return 2
    return None


def reconstruct_commands(prev, start, goal):
    path_dirs = []
    cur = goal
    while cur != start and cur in prev:
        pr, pc, d_idx = prev[cur]
        path_dirs.append(d_idx)
        cur = (pr, pc)
    path_dirs.reverse()
    return [M_CMD[d] for d in path_dirs]


def dijkstra_to_positions(start, targets):
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
    start = get_my_position()
    if not start:
        return []
    Hn, Wn = get_map_size()
    targets = set()
    for r in range(Hn):
        for c in range(Wn):
            if terrain_cost(r, c) is None:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < Hn and 0 <= nc < Wn and map_data[nr][nc] == symbol:
                    if not is_forbidden_cell(r, c):
                        targets.add((r, c))
                    break
    return dijkstra_to_positions(start, targets)


def dijkstra_to_specific(target_pos):
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
    enemy_positions = []
    Hn, Wn = get_map_size()
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] in ['E1', 'E2', 'E3']:
                enemy_positions.append((r, c, map_data[r][c]))
    return enemy_positions


def find_all_enemies():
    res = []
    Hn, Wn = get_map_size()
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] in ['E1', 'E2', 'E3', 'X']:
                res.append((r, c, map_data[r][c]))
    return res


# ★ 변경: 공격 주체가 H/X(기지)이면 사거리 2, 그 외(탱크/플레이어)는 3
def can_attack(from_pos, to_pos):
    if not from_pos or not to_pos:
        return False, -1
    fr, fc = from_pos
    tr, tc = to_pos

    attacker = map_data[fr][fc] if in_bounds(fr, fc) else ''
    max_range = 2 if attacker in ('H', 'X') else 3  # ★ 핵심 수정

    if fr == tr:
        if fc < tc:
            for c in range(fc + 1, tc):
                if map_data[fr][c] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(tc - fc) <= max_range:
                return True, 0
        else:
            for c in range(tc + 1, fc):
                if map_data[fr][c] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(fc - tc) <= max_range:
                return True, 2
    elif fc == tc:
        if fr < tr:
            for r in range(fr + 1, tr):
                if map_data[r][fc] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(tr - fr) <= max_range:
                return True, 1
        else:
            for r in range(tr + 1, fr):
                if map_data[r][fc] not in ['G', 'S', 'W']:
                    return False, -1
            if abs(fr - tr) <= max_range:
                return True, 3
    return False, -1


def get_enemy_hp(enemy_code):
    try:
        if enemy_code in enemies and len(enemies[enemy_code]) >= 1:
            return int(enemies[enemy_code][0])
    except:
        pass
    return None


def distance(pos1, pos2):
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
    if cell_val in ('', 'G', 'S', 'W', 'T', 'F', 'H', 'M', 'E1', 'E2', 'E3', 'X'):
        return False
    return cell_val in my_allies and cell_val not in ('M', 'H')


def path_has_other_ally_between(a, b):
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
    res = []
    for er, ec, _ in find_all_enemies():
        ok, _d = can_attack((er, ec), pos)
        if ok:
            res.append((er, ec))
    return res


def enemies_threaten(pos):
    return len(enemies_that_can_shoot_pos_clear(pos)) > 0


def track_stop_and_return_S():
    global S_USAGE_COUNT, safety_stop_used_this_turn, stop_counted_this_turn
    if S_USAGE_COUNT >= 2:
        return None
    S_USAGE_COUNT += 1
    safety_stop_used_this_turn = True
    stop_counted_this_turn = True
    return "S"


def shoot_adjacent_T_if_any():
    mr, mc = get_my_position()
    if (mr, mc) is None:
        return None
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if in_bounds(nr, nc) and map_data[nr][nc] == 'T':
            return S_CMD[i]
    return None


def safe_G_move_if_any():
    mr, mc = get_my_position()
    if (mr, mc) is None:
        return None
    best = None
    best_score = (-1, float('inf'))
    trc = get_allied_turret_position()
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        if enemies_threaten((nr, nc)):
            continue
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


def apply_safety_guard(action):
    d = cmd_to_dir(action)
    if d is None:
        return action

    next_pos = get_next_pos_for_move_cmd(action)
    if not next_pos or is_forbidden_cell(*next_pos) or not in_bounds(*next_pos):
        alt = choose_shoot_T_or_safe_move()
        if alt:
            return alt
        s_try = track_stop_and_return_S()
        return s_try if s_try else action

    if move_blocks_ally_fire(next_pos):
        s_try = track_stop_and_return_S()
        if s_try:
            return s_try
        alt = choose_shoot_T_or_safe_move()
        if alt:
            return alt
        return action

    threats = enemies_that_can_shoot_pos_clear(next_pos)
    if threats:
        blocked_by_allies = all(path_has_other_ally_between(e, next_pos) for e in threats)

        hits_turret_any = False
        turret = get_allied_turret_position()
        if turret:
            for e in threats:
                if can_attack(e, turret)[0]:
                    hits_turret_any = True
                    break

        if blocked_by_allies or hits_turret_any:
            return action

        s_try = track_stop_and_return_S()
        if s_try:
            return s_try

        alt = choose_shoot_T_or_safe_move()
        if alt:
            return alt
        return action

    return action


##############################
# 이동/경로 유틸
##############################
def next_step_with_T_break(path):
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
    turret_pos = get_allied_turret_position()
    if not turret_pos:
        return None
    ets = find_enemy_tanks()
    if not ets:
        return None
    tr, tc = turret_pos
    return min(ets, key=lambda e: abs(e[0]-tr) + abs(e[1]-tc))


def decide_supply_phase_action():
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


##############################
# ★ 추가: X 기준 축방향(같은 행/열) 오프셋 목적지 생성
##############################
def build_enemy_base_axis_offsets(off_c, off_r):
    base = get_enemy_turret_position()
    if not base:
        return set()
    br, bc = base
    Hn, Wn = get_map_size()
    candidates = [
        (br, bc + off_c), (br, bc - off_c),   # 같은 행, 열 오프셋
        (br + off_r, bc), (br - off_r, bc),   # 같은 열, 행 오프셋
    ]
    targets = set()
    for r, c in candidates:
        if 0 <= r < Hn and 0 <= c < Wn and not is_forbidden_cell(r, c):
            if terrain_cost(r, c) is not None:  # G/T만
                targets.add((r, c))
    return targets


##############################
# ★ 40턴 이후 전진 로직 (요구 조건 반영)
##############################
def min_enemy_tank_to_my_base():
    """E1/E2/E3 중 내 기지(H)까지의 최소 거리"""
    h = get_allied_turret_position()
    if not h:
        return None
    ets = find_enemy_tanks()
    if not ets:
        return None
    return min(distance((er, ec), h) for er, ec, _ in ets)


def collect_targets_closer_to_X_by2(required_max_dist_to_X):
    """
    dist(cell, X) <= required_max_dist_to_X 를 만족하는 이동가능(G/T) 셀들의 집합
    (금지칸 제외)
    """
    X = get_enemy_turret_position()
    if not X:
        return set()
    Hn, Wn = get_map_size()
    targets = set()
    for r in range(Hn):
        for c in range(Wn):
            if terrain_cost(r, c) is None:
                continue
            if distance((r, c), X) <= required_max_dist_to_X:
                targets.add((r, c))
    return targets


def decide_midgame_action():
    """
    40턴 이후:
      - (요구3) 적 탱크 사거리 진입 시 최우선 사격
      - (요구2) X 기준 축방향 오프셋(±x, ±y) 목적지로 전진
               목적지 도착/경로 중 X 사거리면 즉시 타격
      - 그 외 기존 압박 로직 폴백
    """
    my_pos = get_my_position()
    if not my_pos:
        return "S"

    # 0) 교전 우선 (요구3)
    for er, ec, etype in find_enemy_tanks():
        ok, d = can_attack(my_pos, (er, ec))
        if ok:
            hp = get_enemy_hp(etype)
            if hp is not None and hp <= 30:
                return S_CMD[d]
            if get_mega_bomb_count() > 0:
                return S_MEGA_CMD[d]
            return S_CMD[d]

    X = get_enemy_turret_position()
    if not X:
        return decide_defense_action()

    # 현재 위치에서 X 사거리면 즉시 타격
    can_x, dir_x = can_attack(my_pos, X)
    if can_x:
        return S_MEGA_CMD[dir_x] if get_mega_bomb_count() > 0 else S_CMD[dir_x]

    # (요구2) 축방향 오프셋 목적지 시도
    offset_targets = build_enemy_base_axis_offsets(BASE_OFF_C, BASE_OFF_R)
    if offset_targets:
        if my_pos in offset_targets:
            # 목적지 도착: X 사거리면 타격
            can_x2, dir_x2 = can_attack(my_pos, X)
            if can_x2:
                return S_MEGA_CMD[dir_x2] if get_mega_bomb_count() > 0 else S_CMD[dir_x2]
        path0 = dijkstra_to_positions(my_pos, offset_targets)
        step0 = next_step_with_T_break(path0)
        if step0:
            if step0 in S_CMD:
                return step0          # T 파괴 우선
            return apply_safety_guard(step0)

    # 기존 압박(거리 조건) → X 인접칸
    baseline = min_enemy_tank_to_my_base()
    if baseline is not None:
        required = max(0, baseline - 2)
        targets = collect_targets_closer_to_X_by2(required)
        if targets:
            path = dijkstra_to_positions(get_my_position(), targets)
            step = next_step_with_T_break(path)
            if step:
                if step in S_CMD:
                    return step
                return apply_safety_guard(step)

    path2 = dijkstra_to_first_adjacent_of('X')
    step2 = next_step_with_T_break(path2)
    if step2:
        if step2 in S_CMD:
            return step2
        return apply_safety_guard(step2)

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
    global safety_stop_used_this_turn, stop_counted_this_turn
    safety_stop_used_this_turn = False
    stop_counted_this_turn = False


def finalize_stop_count(cmd):
    global S_USAGE_COUNT, stop_counted_this_turn
    if cmd == "S" and not stop_counted_this_turn and S_USAGE_COUNT < 2:
        S_USAGE_COUNT += 1
        stop_counted_this_turn = True


# 반복문
while game_data is not None:
    reset_turn_flags()
    print_data()
    output = "S"

    mega_count = get_mega_bomb_count()
    if mega_count >= 2:
        mega_bombs_acquired = True

    # 40턴 전/후 분기
    if turn_count < 40:
        if (mega_count < 2) and (not mega_bombs_acquired):
            output = decide_supply_phase_action()
        else:
            output = decide_defense_action()
    else:
        output = decide_midgame_action()

    finalize_stop_count(output)

    game_data = submit(output)
    if game_data:
        parse_data(game_data)
        turn_count += 1

# 종료
close()
