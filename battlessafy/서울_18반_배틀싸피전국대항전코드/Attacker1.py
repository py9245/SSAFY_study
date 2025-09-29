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


# 파싱한 데이터를 파일에 출력 (디버깅용)
def print_data():
    output_filename = 'game_state_output.txt'
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
NICKNAME = '40턴_신건하'
game_data = init(NICKNAME)

###################################
# 알고리즘 함수/메서드 부분 구현 시작
###################################

# ===== 전역 상태 =====
S_USAGE_COUNT = 0                   # 모든 "S" 대기 명령 누적 카운트 (최대 2회 의미있게 사용)
safety_stop_used_this_turn = False  # 이번 턴 안전정지(S) 사용 여부
mega_bombs_acquired = False         # 메가 2개 확보 여부
current_route = None
route_position = 0
route_direction = 1  # 1: 정방향, -1: 역방향

# 방향/명령어
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상
M_CMD = ["R A", "D A", "L A", "U A"]
S_CMD = ["R F", "D F", "L F", "U F"]
S_MEGA_CMD = ["R F M", "D F M", "L F M", "U F M"]

# ★ 변경: 기지 근처 공략용 오프셋(행/열). 필요 시 값만 조절하세요.
BASE_OFFSET_DR = 2
BASE_OFFSET_DC = 2


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


def get_my_position(): return find_symbol(map_data, 'M')
def get_allied_turret_position(): return find_symbol(map_data, 'H')
def get_enemy_turret_position(): return find_symbol(map_data, 'X')  # ★ 추가: 적 포탑 위치


def get_mega_bomb_count():
    if 'M' in my_allies and len(my_allies['M']) >= 4:
        return int(my_allies['M'][3])
    return 0


# 카이사르
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
    return caesar_decode(ciphertext, 3)


##############################
# 금지/이동 코스트
##############################
def get_forbidden_cells():
    Hn, Wn = get_map_size()
    turret = get_allied_turret_position()
    forbidden = set()
    if not turret:
        return forbidden
    tr, tc = turret
    if tr == 0 and tc == Wn - 1:
        candidates = [
            (0, Wn - 2), (0, Wn - 3),
            (1, Wn - 1), (1, Wn - 2), (1, Wn - 3),
            (2, Wn - 1), (2, Wn - 2), (2, Wn - 3),
        ]
        for r, c in candidates:
            if 0 <= r < Hn and 0 <= c < Wn:
                forbidden.add((r, c))
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


def terrain_cost(r, c):
    if is_forbidden_cell(r, c):
        return None
    cell = map_data[r][c]
    if cell == 'G':
        return 1
    if cell == 'T':
        return 2
    return None


def in_bounds(r, c):
    Hn, Wn = get_map_size()
    return 0 <= r < Hn and 0 <= c < Wn


##############################
# 경로계산
##############################
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
    Hn, Wn = get_map_size(); INF=10**9
    dist=[[INF]*Wn for _ in range(Hn)]; prev={}
    sr,sc = start; dist[sr][sc]=0
    pq=[(0,sr,sc)]
    while pq:
        cost,r,c = heapq.heappop(pq)
        if cost!=dist[r][c]: continue
        if (r,c) in targets:
            return reconstruct_commands(prev,start,(r,c))
        for d,(dr,dc) in enumerate(DIRS):
            nr,nc=r+dr,c+dc
            if 0<=nr<Hn and 0<=nc<Wn:
                t=terrain_cost(nr,nc)
                if t is None: continue
                ncost=cost+t
                if ncost<dist[nr][nc]:
                    dist[nr][nc]=ncost; prev[(nr,nc)]=(r,c,d)
                    heapq.heappush(pq,(ncost,nr,nc))
    return []


def dijkstra_to_first_adjacent_of(symbol):
    start = get_my_position()
    if not start: return []
    Hn,Wn = get_map_size()
    targets=set()
    for r in range(Hn):
        for c in range(Wn):
            if terrain_cost(r,c) is None: continue
            for dr,dc in DIRS:
                nr,nc = r+dr, c+dc
                if 0<=nr<Hn and 0<=nc<Wn and map_data[nr][nc]==symbol and not is_forbidden_cell(r,c):
                    targets.add((r,c)); break
    return dijkstra_to_positions(start, targets)


def dijkstra_to_specific(target_pos):
    start=get_my_position()
    if not start or not target_pos: return []
    if is_forbidden_cell(*target_pos): return []
    return dijkstra_to_positions(start,{target_pos})


##############################
# 적/아군 탐지 & 전투
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
    out = []
    Hn, Wn = get_map_size()
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] in ['E1', 'E2', 'E3', 'X']:
                out.append((r, c, map_data[r][c]))
    return out


# ★ 변경: 기지(H/X)는 사거리 2, 탱크는 사거리 3으로 판정
def can_attack(my_pos, target_pos):
    if not my_pos or not target_pos: return (False,-1)
    mr,mc = my_pos; tr,tc = target_pos

    attacker = map_data[mr][mc] if in_bounds(mr, mc) else ''
    max_range = 2 if attacker in ('H','X') else 3  # ★ 사거리 차등 적용

    if mr==tr:
        if mc<tc:
            for c in range(mc+1, tc):
                if map_data[mr][c] not in ['G','S','W']: return (False,-1)
            if tc-mc<=max_range: return (True,0)
        else:
            for c in range(tc+1, mc):
                if map_data[mr][c] not in ['G','S','W']: return (False,-1)
            if mc-tc<=max_range: return (True,2)
    elif mc==tc:
        if mr<tr:
            for r in range(mr+1, tr):
                if map_data[r][mc] not in ['G','S','W']: return (False,-1)
            if tr-mr<=max_range: return (True,1)
        else:
            for r in range(tr+1, mr):
                if map_data[r][mc] not in ['G','S','W']: return (False,-1)
            if mr-tr<=max_range: return (True,3)
    return (False,-1)


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
# 아군탱크/차폐/라인 차단
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
    allies = find_allied_tanks_positions()
    enemies_all = find_all_enemies()
    for ar, ac in allies:
        for er, ec, _ in enemies_all:
            ok, _ = can_attack((ar, ac), (er, ec))
            if not ok: continue
            if next_pos in cells_between_on_line((ar, ac), (er, ec)):
                return True
    return False


def enemies_threat_detail(pos):
    allies_set = set(find_allied_tanks_positions())
    turret = get_allied_turret_position()
    for er, ec, _ in find_all_enemies():
        ok, _dir = can_attack((er, ec), pos)
        if not ok: continue
        blocked_by_ally = any((cr, cc) in allies_set for (cr, cc) in cells_between_on_line((er, ec), pos))
        enemy_hits_turret = False
        if turret:
            can_hit_turret, _ = can_attack((er, ec), turret)
            enemy_hits_turret = can_hit_turret
        return True, blocked_by_ally, enemy_hits_turret
    return False, False, False


##############################
# 이동/사격 보조
##############################
def cmd_to_dir(cmd):
    for i, m in enumerate(M_CMD):
        if cmd == m: return i
    return None


def shoot_adjacent_T_if_any():
    mr, mc = get_my_position()
    if (mr, mc) is None: return None
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if in_bounds(nr, nc) and map_data[nr][nc] == 'T':
            return S_CMD[i]
    return None


def safe_G_move_if_any():
    mr, mc = get_my_position()
    if (mr, mc) is None: return None
    for i, (dr, dc) in enumerate(DIRS):
        nr, nc = mr + dr, mc + dc
        if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc): continue
        if map_data[nr][nc] != 'G': continue
        threat, blocked_by_ally, _ = enemies_threat_detail((nr, nc))
        if not threat or blocked_by_ally:
            return M_CMD[i]
    return None


def get_next_pos_for_move_cmd(cmd):
    d = cmd_to_dir(cmd)
    if d is None: return None
    mr, mc = get_my_position()
    dr, dc = DIRS[d]
    return (mr + dr, mc + dc)


def next_step_with_T_break(path):
    if not path: return None
    first_cmd = path[0]
    d = cmd_to_dir(first_cmd)
    if d is None: return None
    mr, mc = get_my_position()
    dr, dc = DIRS[d]
    nr, nc = mr + dr, mc + dc
    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc): return None
    if map_data[nr][nc] == 'T': return S_CMD[d]
    if map_data[nr][nc] == 'G': return first_cmd
    return None


##############################
# 이동 안전가드
##############################
def apply_move_safety(proposed_cmd):
    global safety_stop_used_this_turn, S_USAGE_COUNT
    d = cmd_to_dir(proposed_cmd)
    if d is None: return proposed_cmd

    next_pos = get_next_pos_for_move_cmd(proposed_cmd)
    if not next_pos or not in_bounds(*next_pos) or is_forbidden_cell(*next_pos):
        if S_USAGE_COUNT < 2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn = True; S_USAGE_COUNT += 1
            return "S"
        shoot = shoot_adjacent_T_if_any()
        if shoot: return shoot
        safe = safe_G_move_if_any()
        if safe: return safe
        return "S"

    if move_blocks_ally_fire(next_pos):
        if S_USAGE_COUNT < 2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn = True; S_USAGE_COUNT += 1
            return "S"
        shoot = shoot_adjacent_T_if_any()
        if shoot: return shoot
        safe = safe_G_move_if_any()
        if safe: return safe
        return proposed_cmd

    threat, blocked_by_ally, enemy_hits_turret = enemies_threat_detail(next_pos)
    if threat:
        if blocked_by_ally or enemy_hits_turret:
            return proposed_cmd
        if S_USAGE_COUNT < 2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn = True; S_USAGE_COUNT += 1
            return "S"
        shoot = shoot_adjacent_T_if_any()
        if shoot: return shoot
        safe = safe_G_move_if_any()
        if safe: return safe
        return proposed_cmd

    return proposed_cmd


##############################
# 좌/우 전개 유틸
##############################
def horizontal_step_towards(target):
    my_pos = get_my_position()
    if not my_pos: return None
    r, c = my_pos; tr, tc = target
    if r != tr: return None
    if tc > c: d, nr, nc = 0, r, c + 1
    elif tc < c: d, nr, nc = 2, r, c - 1
    else: return None
    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc): return None
    if map_data[nr][nc] == 'T': return S_CMD[d]
    if map_data[nr][nc] == 'G': return M_CMD[d]
    return None


def horizontal_step_towards_col(target_c):
    my_pos = get_my_position()
    if not my_pos: return None
    r, c = my_pos
    if target_c > c: d, nr, nc = 0, r, c + 1
    elif target_c < c: d, nr, nc = 2, r, c - 1
    else: return None
    if not in_bounds(nr, nc) or is_forbidden_cell(nr, nc): return None
    if map_data[nr][nc] == 'T': return S_CMD[d]
    if map_data[nr][nc] == 'G': return M_CMD[d]
    return None


def fallback_move_away_from_enemies_lr():
    my_pos = get_my_position()
    if not my_pos: return None
    r, c = my_pos; best_i=None; best_score=-1
    for i in (0,2):
        dr, dc = DIRS[i]; nr, nc = r+dr, c+dc
        if not in_bounds(nr,nc) or is_forbidden_cell(nr,nc): continue
        if map_data[nr][nc] != 'G': continue
        score = nearest_enemy_distance((nr,nc))
        if score > best_score: best_score, best_i = score, i
    return M_CMD[best_i] if best_i is not None else None


##############################
# 방어 루트 (기존 유지)
##############################
def get_optimal_defense_positions():
    turret_pos=get_allied_turret_position()
    if not turret_pos: return []
    tr,tc=turret_pos; Hn,Wn=get_map_size()
    def ok(r,c):
        if not (0<=r<Hn and 0<=c<Wn): return False
        if is_forbidden_cell(r,c): return False
        return map_data[r][c] in ('G','T')
    # BL
    if tr==Hn-1 and tc==0:
        r=Hn-1; cols=list(range(3, max(3,Wn-1)))
        pts=[(r,c) for c in cols if ok(r,c)]
        if len(pts)>=2:
            if len(pts)>6:
                step=max(1,len(pts)//6); pts=pts[::step]
                if len(pts)<2: pts=pts[:2]
            return [("patrol_bottom_row", pts)]
    # TR
    if tr==0:
        r=0; cand=[]
        for dc in (3,4):
            c=tc-dc
            if 0<=c<Wn and ok(r,c): cand.append((r,c))
        if len(cand)>=2:
            return [("patrol_top_row_3_4", cand)]
        cols=list(range(1, max(1,Wn-3)))
        pts=[(r,c) for c in cols if ok(r,c)]
        if len(pts)>=2:
            if len(pts)>6:
                step=max(1,len(pts)//6); pts=pts[::step]
                if len(pts)<2: pts=pts[:2]
            return [("patrol_top_row_fallback", pts)]
    routes=[]
    if tr>=Hn//2:
        nr=tr-3; line=[]
        if 0<=nr<Hn:
            for nc in range(max(0,tc-1), min(Wn,tc+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("north", line))
    if tc<=Wn//2:
        nc=tc+3; line=[]
        if 0<=nc<Wn:
            for nr in range(max(0,tr-1), min(Hn,tr+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("east", line))
    if tc>=Wn//2:
        nc=tc-3; line=[]
        if 0<=nc<Wn:
            for nr in range(max(0,tr-1), min(Hn,tr+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("west", line))
    if tr<=Hn//2:
        nr=tr+3; line=[]
        if 0<=nr<Hn:
            for nc in range(max(0,tc-1), min(Wn,tc+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("south", line))
    return routes


def find_closest_enemy_to_turret():
    turret_pos = get_allied_turret_position()
    if not turret_pos: return None
    ets = find_enemy_tanks()
    if not ets: return None
    tr,tc = turret_pos
    return min(ets, key=lambda e: abs(e[0]-tr)+abs(e[1]-tc))


def decide_defense_action():
    global current_route, route_position, route_direction
    my_pos = get_my_position()
    if not my_pos: return "S"

    # 1) 사거리 내 적 우선 사격
    for er,ec,etype in find_enemy_tanks():
        ok,d = can_attack(my_pos,(er,ec))
        if ok:
            hp = get_enemy_hp(etype)
            if hp is not None and hp<=30: return S_CMD[d]
            if get_mega_bomb_count()>0: return S_MEGA_CMD[d]
            return S_CMD[d]

    # 2) 정찰 루트
    defense_routes = get_optimal_defense_positions()
    route_list_only = [r[1] for r in defense_routes]
    if not current_route or (current_route not in route_list_only and route_list_only):
        current_route = route_list_only[0] if route_list_only else None
        route_position, route_direction = 0, 1

    if current_route:
        if my_pos in current_route and len(current_route) > 1:
            idx = current_route.index(my_pos)
            nxt = idx + route_direction
            if nxt >= len(current_route):
                nxt = len(current_route) - 2; route_direction = -1
            elif nxt < 0:
                nxt = 1; route_direction = 1
            target = current_route[nxt]
        else:
            target = min(current_route, key=lambda p: distance(my_pos, p))

        hs = horizontal_step_towards(target)
        if hs: return apply_move_safety(hs) if hs in M_CMD else hs
        hs2 = horizontal_step_towards_col(target[1])
        if hs2: return apply_move_safety(hs2) if hs2 in M_CMD else hs2
        path = dijkstra_to_specific(target)
        step = next_step_with_T_break(path)
        if step: return apply_move_safety(step) if step in M_CMD else step
        fb = fallback_move_away_from_enemies_lr()
        if fb: return apply_move_safety(fb) if fb in M_CMD else fb

    return "S"


##############################
# 보급(메가) 단계
##############################
def closest_supply_to_turret():
    turret = get_allied_turret_position()
    if not turret: return None
    Hn, Wn = get_map_size()
    tr, tc = turret
    best = None; best_d = float('inf')
    for r in range(Hn):
        for c in range(Wn):
            if map_data[r][c] == 'F':
                d = abs(r-tr) + abs(c-tc)
                if d < best_d: best_d, best = d, (r, c)
    return best


def dijkstra_to_adjacent_of_supply_closest_to_turret():
    start = get_my_position()
    if not start: return []
    f = closest_supply_to_turret()
    if not f: return []
    Hn, Wn = get_map_size()
    rF, cF = f
    targets = set()
    for dr, dc in DIRS:
        nr, nc = rF + dr, cF + dc
        if in_bounds(nr, nc) and not is_forbidden_cell(nr, nc) and terrain_cost(nr, nc) is not None:
            targets.add((nr, nc))
    if not targets: return []
    return dijkstra_to_positions(start, targets)


def dijkstra_to_attack_position(target_pos):
    start = get_my_position()
    if not start or not target_pos: return []
    Hn, Wn = get_map_size()
    INF=10**9
    dist=[[INF]*Wn for _ in range(Hn)]; prev={}
    sr,sc=start; dist[sr][sc]=0
    pq=[(0,sr,sc)]
    while pq:
        cost,r,c=heapq.heappop(pq)
        if cost!=dist[r][c]: continue
        can_hit,_ = can_attack((r,c), target_pos)
        if can_hit: return reconstruct_commands(prev,start,(r,c))
        for d_idx,(dr,dc) in enumerate(DIRS):
            nr,nc=r+dr,c+dc
            if in_bounds(nr,nc):
                t_cost=terrain_cost(nr,nc)
                if t_cost is None: continue
                ncost=cost+t_cost
                if ncost<dist[nr][nc]:
                    dist[nr][nc]=ncost; prev[(nr,nc)]=(r,c,d_idx)
                    heapq.heappush(pq,(ncost,nr,nc))
    return []


def is_adjacent_to_supply():
    my_pos = get_my_position()
    if not my_pos: return False
    r, c = my_pos
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and map_data[nr][nc] == 'F':
            return True
    return False


def decide_supply_phase_action():
    my_pos = get_my_position()
    if not my_pos: return "S"

    # 1) 사거리 내 적 우선 사격
    enemy_tanks = find_enemy_tanks()
    mega_count = get_mega_bomb_count()
    for er, ec, etype in enemy_tanks:
        can_hit, direction = can_attack(my_pos, (er, ec))
        if can_hit:
            enemy_hp = get_enemy_hp(etype)
            if mega_count == 1: return S_MEGA_CMD[direction]
            if enemy_hp is not None and enemy_hp <= 30: return S_CMD[direction]
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
    if step: return apply_move_safety(step) if step in M_CMD else step

    # 4) 폴백
    fb = fallback_move_away_from_enemies_lr()
    if fb: return apply_move_safety(fb) if fb in M_CMD else fb

    return "S"


##############################
# ★ 40턴 이후 전진 로직 (요구사항 반영)
##############################
def min_enemy_tank_to_my_base():
    """E1/E2/E3 중 내 기지(H)까지의 최소 거리"""
    h = get_allied_turret_position()
    if not h: return None
    ets = find_enemy_tanks()
    if not ets: return None
    return min(distance((er,ec), h) for er,ec,_ in ets)


def collect_targets_closer_to_X_by2(required_max_dist_to_X):
    X = get_enemy_turret_position()
    if not X: return set()
    Hn,Wn = get_map_size()
    targets=set()
    for r in range(Hn):
        for c in range(Wn):
            if terrain_cost(r,c) is None: continue
            if distance((r,c), X) <= required_max_dist_to_X:
                targets.add((r,c))
    return targets


# ★ 추가: 적 기지(X) 기준 오프셋 목적지들 생성 (±dr, ±dc)
def build_enemy_base_offset_targets(dr_off, dc_off):
    base = get_enemy_turret_position()
    if not base: return set()
    br, bc = base
    cand = {(br+dr_off, bc+dc_off), (br+dr_off, bc-dc_off),
            (br-dr_off, bc+dc_off), (br-dr_off, bc-dc_off)}
    Hn,Wn = get_map_size()
    targets=set()
    for r,c in cand:
        if 0<=r<Hn and 0<=c<Wn and not is_forbidden_cell(r,c) and terrain_cost(r,c) is not None:
            targets.add((r,c))
    return targets


def decide_midgame_action():
    """
    40턴 이후:
      - **요구 3**: 이동 중에도 사거리 내 적 탱크가 있으면 즉시 공격
      - **요구 2**: 적 기지(X) 기준 (±dr, ±dc) 목적지로 이동 → 도착/경로 중 사거리 되면 기지 타격
      - 보조: 기존 'X 인접/가까워지는' 전진 로직 폴백
    """
    my_pos = get_my_position()
    if not my_pos: return "S"

    # (A) 우선 교전: 적 탱크 먼저
    for er, ec, etype in find_enemy_tanks():
        ok, d = can_attack(my_pos, (er, ec))
        if ok:
            hp = get_enemy_hp(etype)
            if hp is not None and hp <= 30: return S_CMD[d]
            if get_mega_bomb_count() > 0: return S_MEGA_CMD[d]
            return S_CMD[d]

    X = get_enemy_turret_position()
    if not X:
        return decide_defense_action()

    # (B) 현재 위치에서 X 사격 가능하면 즉시 타격
    can_x, dir_x = can_attack(my_pos, X)
    if can_x:
        return S_MEGA_CMD[dir_x] if get_mega_bomb_count() > 0 else S_CMD[dir_x]

    # (C) 요구 2: 오프셋 목적지(±dr, ±dc)로 이동
    offset_targets = build_enemy_base_offset_targets(BASE_OFFSET_DR, BASE_OFFSET_DC)
    if offset_targets:
        # 이미 목적지 위에 있으면 다시 한 번 X 사격 시도
        if my_pos in offset_targets:
            can_x2, dir_x2 = can_attack(my_pos, X)
            if can_x2:
                return S_MEGA_CMD[dir_x2] if get_mega_bomb_count() > 0 else S_CMD[dir_x2]
        # 목적지 집합으로 최단 경로
        path = dijkstra_to_positions(my_pos, offset_targets)
        step = next_step_with_T_break(path)
        if step:
            return apply_move_safety(step) if step in M_CMD else step

    # (D) 기존 압박 폴백: baseline 기반 전진 or X 인접칸
    baseline = min_enemy_tank_to_my_base()
    if baseline is not None:
        required = max(0, baseline - 2)
        targets = collect_targets_closer_to_X_by2(required)
        if targets:
            path2 = dijkstra_to_positions(my_pos, targets)
            step2 = next_step_with_T_break(path2)
            if step2: return apply_move_safety(step2) if step2 in M_CMD else step2

    path3 = dijkstra_to_first_adjacent_of('X')
    step3 = next_step_with_T_break(path3)
    if step3: return apply_move_safety(step3) if step3 in M_CMD else step3

    fb = fallback_move_away_from_enemies_lr()
    if fb: return apply_move_safety(fb) if fb in M_CMD else fb
    return "S"


###################################
# 알고리즘 함수/메서드 부분 구현 끝
###################################
parse_data(game_data)
turn_count = 0  # ★ 추가: 턴 카운터

# 반복문
while game_data is not None:
    print_data()
    safety_stop_used_this_turn = False

    # 현재 메가 폭탄 개수
    mega_count = get_mega_bomb_count()
    if mega_count >= 2:
        mega_bombs_acquired = True

    # --- 의사결정 ---
    if turn_count < 40:
        # 40턴 전: 기존 로직 유지 (보급→방어)
        if (mega_count < 2) and (not mega_bombs_acquired):
            proposed = decide_supply_phase_action()
        else:
            proposed = decide_defense_action()
    else:
        # 40턴부터: ★ 요구 2 로직 반영 (오프셋 기반 기지 압박 + 사거리 내 즉시 타격)
        proposed = decide_midgame_action()

    # 이동 안전가드
    final_cmd = apply_move_safety(proposed) if proposed in M_CMD else proposed

    # S 집계
    if final_cmd == "S" and not safety_stop_used_this_turn and S_USAGE_COUNT < 2:
        S_USAGE_COUNT += 1
        safety_stop_used_this_turn = True

    # 명령 전송
    game_data = submit(final_cmd)

    # 다음 턴 준비
    if game_data:
        turn_count += 1  # ★ 매 턴 증가
        parse_data(game_data)

# 종료
close()
