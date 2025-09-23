import sys
import socket
import heapq
from collections import defaultdict, deque

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

def submit(string_to_send):
    try:
        send_data = ARGS + string_to_send + ' '
        sock.send(send_data.encode('utf-8'))
        return receive()
    except Exception as e:
        print('[ERROR] Failed to send data. Please check if connection to the main program is valid.')
    return None

def receive():
    try:
        game_data = (sock.recv(1024)).decode()
        if game_data and game_data[0].isdigit() and int(game_data[0]) > 0:
            return game_data
        print('[STATUS] No receive data from the main program.')
        close()
    except Exception as e:
        print('[ERROR] Failed to receive data. Please check if connection to the main program is valid.')

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
map_data = [[]]
my_allies = {}
enemies = {}
codes = []

##############################
# 전역 상태 (요청 반영)
##############################
S_USAGE_COUNT = 0                    # 모든 "S" 누적 카운트 (총 2회까지만 의미 있게 사용)
safety_stop_used_this_turn = False   # 이번 턴에 안전 때문에 S를 썼는지
stop_counted_this_turn = False       # 이번 턴에 S_USAGE_COUNT를 이미 증가시켰는지
mega_bombs_acquired = False

current_route = None
route_position = 0
route_direction = 1  # 1: 정방향, -1: 역방향

# 방향/명령어
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상
M_CMD = ["R A", "D A", "L A", "U A"]
S_CMD = ["R F", "D F", "L F", "U F"]
S_MEGA_CMD = ["R F M", "D F M", "L F M", "U F M"]
TURN_COUNT = 0
ENDGAME_TURN = 60

##############################
# 파싱 / 기본 유틸
##############################
def parse_data(game_data):
    game_data_rows = game_data.split('\n')
    row_index = 0

    header = game_data_rows[row_index].split(' ')
    map_height = int(header[0]) if len(header) >= 1 else 0
    map_width = int(header[1]) if len(header) >= 2 else 0
    num_of_allies = int(header[2]) if len(header) >= 3 else 0
    num_of_enemies = int(header[3]) if len(header) >= 4 else 0
    num_of_codes = int(header[4]) if len(header) >= 5 else 0
    row_index += 1

    map_data.clear()
    map_data.extend([['' for c in range(map_width)] for r in range(map_height)])
    for i in range(0, map_height):
        col = game_data_rows[row_index + i].split(' ')
        for j in range(0, len(col)):
            map_data[i][j] = col[j]
    row_index += map_height

    my_allies.clear()
    for i in range(row_index, row_index + num_of_allies):
        ally = game_data_rows[i].split(' ')
        ally_name = ally.pop(0) if len(ally) >= 1 else '-'
        my_allies[ally_name] = ally
    row_index += num_of_allies

    enemies.clear()
    for i in range(row_index, row_index + num_of_enemies):
        enemy = game_data_rows[i].split(' ')
        enemy_name = enemy.pop(0) if len(enemy) >= 1 else '-'
        enemies[enemy_name] = enemy
    row_index += num_of_enemies

    codes.clear()
    for i in range(row_index, row_index + num_of_codes):
        codes.append(game_data_rows[i])

def get_map_size():
    if map_data and map_data[0]:
        return len(map_data), len(map_data[0])
    return (0, 0)

def find_symbol(grid, symbol):
    if not grid or not grid[0]:
        return None
    H, W = len(grid), len(grid[0])
    for r in range(H):
        for c in range(W):
            if grid[r][c] == symbol:
                return (r, c)
    return None

def find_all_symbols(symbol):
    if not map_data or not map_data[0]:
        return []
    H, W = get_map_size()
    ret = []
    for r in range(H):
        for c in range(W):
            if map_data[r][c] == symbol:
                ret.append((r, c))
    return ret

def get_my_position():
    return find_symbol(map_data, 'M')

def get_allied_turret_position():
    return find_symbol(map_data, 'H')

def get_mega_bomb_count():
    if 'M' in my_allies and len(my_allies['M']) >= 4:
        return int(my_allies['M'][3])
    return 0

def distance(a, b):
    if not a or not b:
        return float('inf')
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def in_bounds(r, c):
    H, W = get_map_size()
    return 0 <= r < H and 0 <= c < W

##############################
# 암호
##############################
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
        "YOUWILLNEVERKNOWUNTILYOUTRY","THEREISNOROYALROADTOLEARNING",
        "BETTERLATETHANNEVER","THISTOOSHALLPASSAWAY","FAITHWITHOUTDEEDSISUSELESS",
        "FORGIVENESSISBETTERTHANREVENGE","LIFEISNOTALLBEERANDSKITTLES","UNTILDEATHITISALLLIFE",
        "WHATEVERYOUDOMAKEITPAY","TIMEISGOLD","THEONLYCUREFORGRIEFISACTION","GIVEMELIBERTYORGIVEMEDEATH",
        "APOETISTHEPAINTEROFTHESOUL","BELIEVEINYOURSELF","NOSWEATNOSWEET","EARLYBIRDCATCHESTHEWORM",
        "SEEINGISBELIEVING","ASKINGCOSTSNOTHING","GOODFENCESMAKESGOODNEIGHBORS","AROLLINGSTONEGATHERSNOMOSS",
        "ONESUTMOSTMOVESTHEHEAVENS","LITTLEBYLITTLEDOESTHETRICK","LIVEASIFYOUWERETODIETOMORROW",
        "LETBYGONESBEBYGONES","THEBEGINNINGISHALFOFTHEWHOLE","NOPAINNOGAIN","STEPBYSTEPGOESALONGWAY",
        "THEDIFFICULTYINLIFEISTHECHOICE","LIFEISFULLOFUPSANDDOWNS","ROMEWASNOTBUILTINADAY",
        "IFYOUCANTBEATTHEMJOINTHEM","NOTHINGVENTUREDNOTHINGGAINED","KNOWLEDGEINYOUTHISWISDOMINAGE",
        "NOBEESNOHONEY","WHERETHEREISAWILLTHEREISAWAY","HABITISSECONDNATURE","SUTMOSTMOVESTHEHEAVENS",
        "ONLY","ACTION","CUREFORGRIEFIS","SSAFY","BATTLE","ALGORITHM","TANK","MISSION","CODE","HERO","SEIZETHEDAY",
        "LIFEITSELFISAQUOTATION","LIFEISVENTUREORNOTHING","DONTDREAMBEIT","TRYYOURBESTRATHERTHANBETHEBEST",
        "WHATWILLBEWILLBE","DONTDWELLONTHEPAST","PASTISJUSTPAST","FOLLOWYOURHEART","LITTLEBYLITTLEDOESTHETRICK"
    ]
    for shift in range(26):
        decoded = caesar_decode(ciphertext, shift)
        for kw in keywords:
            if kw in decoded.upper():
                return decoded
    return caesar_decode(ciphertext, 3)

##############################
# Forbidden Zone
##############################
def get_forbidden_cells():
    H, W = get_map_size()
    turret = get_allied_turret_position()
    forbidden = set()
    if not turret:
        return forbidden
    tr, tc = turret
    if tr == 0 and tc == W-1:
        cand = [(0, W-2),(0, W-3),(1, W-1),(1, W-2),(1, W-3),(2, W-1),(2, W-2),(2, W-3)]
        forbidden |= {(r,c) for (r,c) in cand if 0<=r<H and 0<=c<W}
    if tr == H-1 and tc == 0:
        cand = [(H-1,1),(H-1,2),(H-2,0),(H-2,1),(H-2,2),(H-3,0),(H-3,1),(H-3,2)]
        forbidden |= {(r,c) for (r,c) in cand if 0<=r<H and 0<=c<W}
    return forbidden

def is_forbidden_cell(r,c):
    return (r,c) in get_forbidden_cells()

def get_corner():
    """포탑(H) 위치로 코너 방향 판정: TR(우상) / BL(좌하)"""
    t = get_allied_turret_position()
    if not t:
        return None
    H, W = get_map_size()
    tr, tc = t
    if tr == 0 and tc == W - 1:
        return "TR"
    if tr == H - 1 and tc == 0:
        return "BL"
    return None

def is_entry_phase():
    """
    코너(기지가 BL 또는 TR) 초입 구간이면 True.
    - BL(좌하): 맵 하단 2줄이거나 좌측 3열 안쪽
    - TR(우상): 맵 상단 2줄이거나 우측 3열 안쪽
    """
    corner = get_corner()
    if not corner:
        return False
    my_pos = get_my_position()
    if not my_pos:
        return False
    H, W = get_map_size()
    r, c = my_pos
    if corner == "BL":
        return (r >= H - 2) or (c <= 2)
    if corner == "TR":
        return (r <= 1) or (c >= W - 3)
    return False

def try_entry_breakT_move(apply_guard=True):
    """
    엔트포인트일 때,
      - BL이면 오른쪽 우선(바로 오른칸이 T면 사격, G면 우측 이동)
      - TR이면 왼쪽 우선(바로 왼칸이 T면 사격, G면 좌측 이동)
    이동은 안전가드로 감쌀 수 있음(apply_guard=True)
    """
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
        return S_CMD[d]  # 먼저 부수기
    if map_data[nr][nc] == 'G':
        return apply_safety_guard(M_CMD[d]) if apply_guard else M_CMD[d]

    return None

##############################
# 이동/경로 (다익스트라)
##############################
def terrain_cost(r, c):
    if is_forbidden_cell(r, c):
        return None
    cell = map_data[r][c]
    if cell == 'G':
        return 1
    if cell == 'S':
        return 2
    return None

def terrain_cost_supply(r, c):
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

def next_step_with_T_break(path):
    """
    경로(path)가 있을 때 한 칸 전진을 안전하게 변환:
      - 다음 칸이 T면: 그 방향으로 사격(S_CMD)하여 나무 제거
      - 다음 칸이 G면: 그대로 이동(M_CMD)
      - 그 외/경계/금지: None
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
        return S_CMD[d]    # 🔫 먼저 쏴서 길 뚫기
    if map_data[nr][nc] == 'G':
        return first_cmd   # 🚶 정상 이동
    return None

def dijkstra_to_positions_generic(start, targets, cost_fn):
    H, W = get_map_size()
    INF = 10**9
    dist = [[INF]*W for _ in range(H)]
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
            if 0 <= nr < H and 0 <= nc < W:
                t_cost = cost_fn(nr, nc)
                if t_cost is None:
                    continue
                ncost = cost + t_cost
                if ncost < dist[nr][nc]:
                    dist[nr][nc] = ncost
                    prev[(nr, nc)] = (r, c, d_idx)
                    heapq.heappush(pq, (ncost, nr, nc))
    return []

def dijkstra_to_positions(start, targets):
    return dijkstra_to_positions_generic(start, targets, terrain_cost)

def dijkstra_to_specific(target_pos):
    start = get_my_position()
    if not start or not target_pos:
        return []
    if is_forbidden_cell(*target_pos):
        return []
    return dijkstra_to_positions(start, {target_pos})

def dijkstra_to_first_adjacent_of(symbol):
    start = get_my_position()
    if not start:
        return []
    H, W = get_map_size()
    targets = set()
    for r in range(H):
        for c in range(W):
            if terrain_cost(r, c) is None:
                continue
            for dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and map_data[nr][nc] == symbol:
                    if not is_forbidden_cell(r, c):
                        targets.add((r, c))
                    break
    return dijkstra_to_positions(start, targets)

def dijkstra_to_first_adjacent_of_supply(symbol):
    start = get_my_position()
    if not start:
        return []
    H, W = get_map_size()
    targets = set()
    for r in range(H):
        for c in range(W):
            if terrain_cost_supply(r, c) is None:
                continue
            for dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and map_data[nr][nc] == symbol:
                    if not is_forbidden_cell(r, c):
                        targets.add((r, c))
                    break
    return dijkstra_to_positions_generic(start, targets, terrain_cost_supply)

##############################
# 전투/표적
##############################
def find_enemy_tanks():
    H, W = get_map_size()
    out = []
    for r in range(H):
        for c in range(W):
            if map_data[r][c] in ['E1','E2','E3']:
                out.append((r,c,map_data[r][c]))
    return out

def find_all_enemies():
    H, W = get_map_size()
    out = []
    for r in range(H):
        for c in range(W):
            if map_data[r][c] in ['E1','E2','E3','X']:
                out.append((r,c,map_data[r][c]))
    return out

def can_attack(from_pos, to_pos):
    if not from_pos or not to_pos:
        return False, -1
    fr, fc = from_pos
    tr, tc = to_pos
    if fr == tr:
        if fc < tc:
            for c in range(fc+1, tc):
                if map_data[fr][c] not in ['G','S','W']:
                    return False, -1
            if abs(tc-fc) <= 3:
                return True, 0
        else:
            for c in range(tc+1, fc):
                if map_data[fr][c] not in ['G','S','W']:
                    return False, -1
            if abs(fc-tc) <= 3:
                return True, 2
    elif fc == tc:
        if fr < tr:
            for r in range(fr+1, tr):
                if map_data[r][fc] not in ['G','S','W']:
                    return False, -1
            if abs(tr-fr) <= 3:
                return True, 1
        else:
            for r in range(tr+1, fr):
                if map_data[r][fc] not in ['G','S','W']:
                    return False, -1
            if abs(fr-tr) <= 3:
                return True, 3
    return False, -1

def get_enemy_hp(enemy_code):
    try:
        if enemy_code in enemies and len(enemies[enemy_code]) >= 1:
            return int(enemies[enemy_code][0])
    except:
        pass
    return None

##############################
# 🔐 안전 지점(맨해튼≥4 & 모든 적 사정거리 밖) 탐색 유틸
##############################
ENEMY_SYMBOLS = ('X', 'E1', 'E2', 'E3')

def all_enemy_positions():
    H, W = get_map_size()
    pos = []
    for r in range(H):
        for c in range(W):
            if map_data[r][c] in ENEMY_SYMBOLS:
                pos.append((r, c))
    return pos

def min_manhattan_to_enemies(cell):
    if not cell:
        return float('inf')
    enemies = all_enemy_positions()
    if not enemies:
        return float('inf')
    r, c = cell
    return min(abs(r-er)+abs(c-ec) for er, ec in enemies)

def outside_all_firing(cell):
    """모든 적(X,E1,E2,E3)에 대해 '차폐 없이 사거리3' 밖이면 True"""
    for er, ec in all_enemy_positions():
        ok, _ = can_attack((er, ec), cell)
        if ok:
            return False
    return True

def terrain_cost_safespot(r, c):
    """안전지점 탐색 코스트: G=1, T=2 (T는 쏘며 전진), 나머지/금지 불가"""
    if is_forbidden_cell(r, c):
        return None
    cell = map_data[r][c]
    if cell == 'G':
        return 1
    if cell == 'T':
        return 2
    return None

def build_safe_targets():
    """조건 충족(G/T) 칸 전체에서 안전 지점을 수집"""
    H, W = get_map_size()
    targets = set()
    for r in range(H):
        for c in range(W):
            if terrain_cost_safespot(r, c) is None:
                continue
            if min_manhattan_to_enemies((r, c)) >= 4 and outside_all_firing((r, c)):
                targets.add((r, c))
    return targets

def path_to_nearest_safespot():
    """내 위치에서 가장 빨리 도달 가능한 안전 지점으로의 경로"""
    start = get_my_position()
    if not start:
        return []
    targets = build_safe_targets()
    if not targets:
        return []
    return dijkstra_to_positions_generic(start, targets, terrain_cost_safespot)

##############################
# 아군/차폐/라인 판정 (요청 핵심)
##############################
def is_allied_tank_cell(cell_val):
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
    """내가 next_pos로 이동하면 아군→적의 사격 라인을 가로막는가?"""
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
    pos가 적의 사격경로 안인가?
      - blocked_by_ally: 그 경로상에 '다른 아군 탱크'가 있는가
      - enemy_hits_turret: 그 적이 우리 타워(H)를 때릴 수 있는 위치인가
    """
    allies_set = set(find_allied_tanks_positions())
    turret = get_allied_turret_position()
    for er, ec, _ in find_all_enemies():
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
# 안전 가드 + S 집계/대안
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
    H, W = get_map_size()
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if 0 <= nr < H and 0 <= nc < W and map_data[nr][nc] == 'T':
            return S_CMD[i]
    return None

def safe_G_move_if_any():
    """적 사격경로(차폐 없는) 아닌 인접 G로 이동"""
    mr, mc = get_my_position()
    if (mr, mc) is None:
        return None
    H, W = get_map_size()
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if is_forbidden_cell(nr, nc):
            continue
        if map_data[nr][nc] != 'G':
            continue
        threat, blocked_by_ally, _ = enemies_threat_detail((nr, nc))
        if not threat or blocked_by_ally:
            return M_CMD[i]
    return None

def track_stop_and_return_S():
    """S 사용 시도(총 2회, 턴당 1회). 불가하면 None."""
    global S_USAGE_COUNT, safety_stop_used_this_turn, stop_counted_this_turn
    if S_USAGE_COUNT >= 2 or safety_stop_used_this_turn:
        return None
    S_USAGE_COUNT += 1
    safety_stop_used_this_turn = True
    stop_counted_this_turn = True
    return "S"

def choose_shoot_T_or_safe_move(fallback_target=None):
    """S가 더 못 쓰일 때: 1순위 T 파괴 → 2순위 안전 G 이동"""
    shoot = shoot_adjacent_T_if_any()
    if shoot:
        return shoot
    safe = safe_G_move_if_any()
    if safe:
        return safe
    return None

def apply_safety_guard(action, fallback_target=None):
    """
    이동 직전에 안전가드 적용 (요청 규칙):
    - 다음 칸이 적 사격경로이면
      - 그 경로상에 '다른 아군 탱크'가 있거나, 그 적이 우리 타워(H)를 칠 수 있는 위치라면 → 이동 허용
      - 아니면, 내가 이동해 아군→적 라인을 막는 경우 포함 → S 시도(총 2/턴1), 실패 시 T 사격→안전 G 이동, 그래도 없으면 이동 강행
    """
    d = cmd_to_dir(action)
    if d is None:
        return action  # 이동 아닌 명령

    mr, mc = get_my_position()
    dr, dc = DIRS[d]
    nr, nc = mr + dr, mc + dc

    # 불가/금지 → 대안
    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc):
        alt = choose_shoot_T_or_safe_move(fallback_target)
        if alt:
            return alt
        s_try = track_stop_and_return_S()
        return s_try if s_try else action

    # 내가 이동하면 아군의 사격 라인을 막는가?
    if move_blocks_ally_fire((nr, nc)):
        s_try = track_stop_and_return_S()
        if s_try:
            return s_try
        alt = choose_shoot_T_or_safe_move(fallback_target)
        if alt:
            return alt
        return action  # 강행

    # 다음 칸 위협 상세
    threat, blocked_by_ally, enemy_hits_turret = enemies_threat_detail((nr, nc))
    if threat:
        if blocked_by_ally or enemy_hits_turret:
            return action  # 이동 허용
        # 차폐 없음 & 타워도 못치게 되는 적 → 멈추기 시도
        s_try = track_stop_and_return_S()
        if s_try:
            return s_try
        alt = choose_shoot_T_or_safe_move(fallback_target)
        if alt:
            return alt
        return action  # 대안 없으면 강행

    # 위협 없음 → 이동
    return action

##############################
# 보급/해독
##############################
def is_adjacent_to_supply():
    my_pos = get_my_position()
    if not my_pos:
        return False
    H, W = get_map_size()
    r, c = my_pos
    for dr, dc in DIRS:
        nr, nc = r+dr, c+dc
        if 0 <= nr < H and 0 <= nc < W and map_data[nr][nc] == 'F':
            return True
    return False

def nearest_distance_to_targets(pos, targets):
    if not targets:
        return float('inf')
    pr, pc = pos
    return min(abs(pr - tr) + abs(pc - tc) for (tr, tc) in targets)

def fallback_move_adjacent_G_towards_target(target_pos):
    my_pos = get_my_position()
    if not my_pos or not target_pos:
        return None
    H, W = get_map_size()
    mr, mc = my_pos
    best = None
    best_score = float('inf')
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not in_bounds(nr, nc): continue
        if is_forbidden_cell(nr, nc): continue
        if map_data[nr][nc] != 'G': continue
        score = distance((nr, nc), target_pos)
        if score < best_score:
            best_score = score
            best = i
    if best is not None:
        return M_CMD[best]
    return None

def fallback_move_adjacent_G_towards_F():
    my_pos = get_my_position()
    if not my_pos:
        return None
    F_targets = find_all_symbols('F')
    H, W = get_map_size()
    mr, mc = my_pos
    best = None
    best_score = float('inf')
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not in_bounds(nr, nc): continue
        if is_forbidden_cell(nr, nc): continue
        if map_data[nr][nc] != 'G': continue
        score = nearest_distance_to_targets((nr, nc), F_targets)
        if score < best_score:
            best_score = score
            best = i
    if best is not None:
        return M_CMD[best]
    return None

def find_closest_supply_to_base():
    base = get_allied_turret_position()
    if not base:
        return None
    Fs = find_all_symbols('F')
    if not Fs:
        return None
    return min(Fs, key=lambda p: distance(base, p))

def dijkstra_to_adjacent_of_specific_supply(f_pos):
    start = get_my_position()
    if not start or not f_pos:
        return []
    H, W = get_map_size()
    targets = set()
    for r in range(H):
        for c in range(W):
            if terrain_cost_supply(r, c) is None:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) == f_pos and not is_forbidden_cell(r, c):
                    targets.add((r, c))
                    break
    if not targets:
        return []
    return dijkstra_to_positions_generic(start, targets, terrain_cost_supply)

##############################
# 웨이포인트 (part3 정찰)
##############################
def get_corner():
    t = get_allied_turret_position()
    if not t:
        return None
    H, W = get_map_size()
    tr, tc = t
    if tr == 0 and tc == W-1:
        return "TR"
    if tr == H-1 and tc == 0:
        return "BL"
    return None

def is_good_stand_cell(r, c):
    if not in_bounds(r, c):
        return False
    if is_forbidden_cell(r, c):
        return False
    return map_data[r][c] == 'G'

def clamp_waypoints(pts):
    return [(r,c) for (r,c) in pts if is_good_stand_cell(r,c)]

def build_part3_waypoints():
    corner = get_corner()
    if corner is None:
        return []
    H, W = get_map_size()
    if corner == "BL":
        raw = [(H-4,2),(H-4,3),(H-3,3)]
    else:  # TR
        raw = [(2, W-4),(3, W-4),(3, W-3)]
    wps = clamp_waypoints(raw)
    if len(wps) < 3:
        wps = []
        for r, c in raw:
            if in_bounds(r,c) and not is_forbidden_cell(r,c) and map_data[r][c] != 'T':
                wps.append((r,c))
    return wps[:3]

##############################
# 의사결정
##############################
def decide_supply_phase_action():
    my_pos = get_my_position()
    if not my_pos:
        return "S"

    # 1) 기회 사격
    for er, ec, et in find_enemy_tanks():
        ok, d = can_attack(my_pos, (er, ec))
        if ok:
            hp = get_enemy_hp(et)
            if get_mega_bomb_count() == 1:
                return S_MEGA_CMD[d]
            if hp is not None and hp <= 30:
                return S_CMD[d]
            return S_CMD[d]

    # 2) 해독
    if codes and is_adjacent_to_supply():
        decoded = find_valid_caesar_decode(codes[0])
        if decoded:
            return f"G {decoded}"

    # 3) 기지에서 가장 가까운 F 인접칸까지 (G=1,T=2)
    closest_f = find_closest_supply_to_base()
    if closest_f:
        path = dijkstra_to_adjacent_of_specific_supply(closest_f)
        if path:
            first_cmd = path[0]
            d = cmd_to_dir(first_cmd)
            if d is not None:
                mr, mc = get_my_position()
                dr, dc = DIRS[d]
                nr, nc = mr + dr, mc + dc
                if in_bounds(nr, nc) and map_data[nr][nc] == 'T':
                    return S_CMD[d]  # 사격은 이동 아님(안전가드 제외)
                if in_bounds(nr, nc) and map_data[nr][nc] == 'G':
                    return apply_safety_guard(first_cmd, fallback_target=closest_f)

            fb = fallback_move_adjacent_G_towards_target(closest_f)
            if fb:
                return apply_safety_guard(fb, fallback_target=closest_f)
            # ↓↓↓ 여기서부터 '안전 지점' 보급로 전진 시도
        # F 경로 실패 시에도 안전 지점 시도
    # === 안전 지점(맨해튼≥4 & 사정거리 밖)으로 보급로 전진 ===
    spath = path_to_nearest_safespot()
    sstep = next_step_with_T_break(spath)
    if sstep:
        if sstep in M_CMD:
            return apply_safety_guard(sstep)
        return sstep

    # 4) 기존 폴백
    fb = fallback_move_adjacent_G_towards_F()
    if fb:
        return apply_safety_guard(fb)
    return "S"

def decide_defense_action():
    global route_position, route_direction, current_route, TURN_COUNT

    if TURN_COUNT >= ENDGAME_TURN:
        pref = try_entry_breakT_move(apply_guard=True)
        if pref:
            return pref
    my_pos = get_my_position()
    if not my_pos:
        return "S"

    # 1) 즉시 사격
    for er, ec, et in find_enemy_tanks():
        ok, d = can_attack(my_pos, (er, ec))
        if ok:
            hp = get_enemy_hp(et)
            if hp is not None and hp <= 30:
                return S_CMD[d]
            if get_mega_bomb_count() > 0:
                return S_MEGA_CMD[d]
            return S_CMD[d]

    # 2) part3 웨이포인트 순찰
    wps = build_part3_waypoints()
    if wps:
        if my_pos in wps:
            route_position = wps.index(my_pos)
        if route_direction not in (1, -1):
            route_direction = 1
        next_idx = route_position + route_direction
        if next_idx >= len(wps):
            next_idx = len(wps)-2 if len(wps) >= 2 else 0
            route_direction = -1
        elif next_idx < 0:
            next_idx = 1 if len(wps) >= 2 else 0
            route_direction = 1
        target = wps[next_idx] if my_pos in wps else min(wps, key=lambda p: distance(my_pos, p))
        path = dijkstra_to_specific(target)
        step = next_step_with_T_break(path)
        if step:
            if step in S_CMD:  # 사격이면 안전가드 불필요
                return step
            return apply_safety_guard(step, fallback_target=target)

    # 3) 포탑 근처 한 칸(최소 방어)
    t = get_allied_turret_position()
    if t:
        mr, mc = my_pos
        for i, (dr, dc) in enumerate(DIRS):
            nr, nc = mr+dr, mc+dc
            if in_bounds(nr, nc) and is_good_stand_cell(nr, nc):
                return apply_safety_guard(M_CMD[i], fallback_target=t)

    # 4) (신규) 안전 지점으로 보급로 전진
    spath = path_to_nearest_safespot()
    sstep = next_step_with_T_break(spath)
    if sstep:
        if sstep in M_CMD:
            return apply_safety_guard(sstep)
        return sstep

    return "S"

##############################
# 턴 제어/집계 보조
##############################
def reset_turn_flags():
    global safety_stop_used_this_turn, stop_counted_this_turn
    safety_stop_used_this_turn = False
    stop_counted_this_turn = False

def finalize_stop_count(cmd):
    """이번 턴 최종 명령이 'S'이고 아직 반영 안했으면 카운트(단, 총 2회 한정)."""
    global S_USAGE_COUNT, stop_counted_this_turn
    if cmd == "S" and not stop_counted_this_turn and S_USAGE_COUNT < 2:
        S_USAGE_COUNT += 1
        stop_counted_this_turn = True

##############################
# 메인 루프
##############################
if __name__ == "__main__":
    NICKNAME = '서울18_박유신'  # 원하는 닉네임
    game_data = init(NICKNAME)
    turn_count = 0

    while game_data is not None:
        parse_data(game_data)
        reset_turn_flags()

        # 전역 턴 카운터 갱신
        TURN_COUNT = turn_count

        # 메가탄 플래그 업데이트
        mega_count = get_mega_bomb_count()
        if mega_count >= 2:
            mega_bombs_acquired = True

        # 메인 의사결정
        if (mega_count < 2) and (not mega_bombs_acquired):
            output = decide_supply_phase_action()
        else:
            output = decide_defense_action()

        # S 집계 마무리 (중복 방지)
        finalize_stop_count(output)

        # 명령 전송 및 다음 입력 수신
        game_data = submit(output)
        if game_data:
            turn_count += 1

    close()
