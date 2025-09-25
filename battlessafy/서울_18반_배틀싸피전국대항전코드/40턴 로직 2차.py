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
map_data = [[]]  # 맵 정보
my_allies = {}   # 아군 정보 (키: 유닛명)
enemies = {}     # 적군 정보
codes = []       # 암호문


##############################
# 입력 데이터 파싱
##############################
def parse_data(game_data):
    rows = game_data.split('\n')
    i = 0
    h, w, na, ne, nc = (int(x) if x.isdigit() else 0 for x in rows[i].split(' ')[:5]); i += 1

    map_data.clear()
    map_data.extend([['' for _ in range(w)] for _ in range(h)])
    for r in range(h):
        col = rows[i + r].split(' ')
        for c in range(len(col)):
            map_data[r][c] = col[c]
    i += h

    my_allies.clear()
    for _ in range(na):
        parts = rows[i].split(' ')
        name = parts.pop(0) if parts else '-'
        my_allies[name] = parts
        i += 1

    enemies.clear()
    for _ in range(ne):
        parts = rows[i].split(' ')
        name = parts.pop(0) if parts else '-'
        enemies[name] = parts
        i += 1

    codes.clear()
    for _ in range(nc):
        codes.append(rows[i]); i += 1

# 디버깅 출력
def print_data():
    output_filename = 'game_state_output.txt'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f'\n----------입력 데이터----------\n{game_data}\n----------------------------\n')
        f.write(f'\n[맵 정보] ({len(map_data)} x {len(map_data[0])})\n')
        for row in map_data:
            f.write((' '.join(row)).strip() + '\n')
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
        for c in codes:
            f.write(c + '\n')
    print(f"게임 상태 정보가 '{output_filename}' 파일에 저장되었습니다. 💾")


##############################
# 닉네임 설정 및 최초 연결
##############################
NICKNAME = '서울18_신건하'
game_data = init(NICKNAME)

###################################
# 알고리즘 함수/메서드 부분 구현 시작
###################################

# ===== 전역 상태 =====
S_USAGE_COUNT = 0                   # 'S' 누적 카운트 (최대 2회 의미있게 사용)
safety_stop_used_this_turn = False  # 이번 턴 'S' 사용 여부
mega_bombs_acquired = False         # 메가 2개 확보 여부
current_route = None
route_position = 0
route_direction = 1  # 1: 정방향, -1: 역방향

# 방향/명령어
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 우, 하, 좌, 상
M_CMD = ["R A", "D A", "L A", "U A"]
S_CMD = ["R F", "D F", "L F", "U F"]
S_MEGA_CMD = ["R F M", "D F M", "L F M", "U F M"]

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

def get_my_position(): return find_symbol(map_data, 'M')
def get_allied_turret_position(): return find_symbol(map_data, 'H')
def get_enemy_turret_position(): return find_symbol(map_data, 'X')

def get_mega_bomb_count():
    if 'M' in my_allies and len(my_allies['M']) >= 4:
        return int(my_allies['M'][3])
    return 0

def get_my_turn_count():
    if 'M' in my_allies and len(my_allies['M']) >= 6:
        return int(my_allies['M'][5])
    return 0

# 카이사르
def caesar_decode(ciphertext, shift):
    res = ""
    for ch in ciphertext:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            res += chr((ord(ch) - base - shift) % 26 + base)
        else:
            res += ch
    return res

def find_valid_caesar_decode(ciphertext):
    keywords = [
        "YOUWILLNEVERKNOWUNTILYOUTRY","THEREISNOROYALROADTOLEARNING","BETTERLATETHANNEVER","THISTOOSHALLPASSAWAY",
        "FAITHWITHOUTDEEDSISUSELESS","FORGIVENESSISBETTERTHANREVENGE","LIFEISNOTALLBEERANDSKITTLES","UNTILDEATHITISALLLIFE",
        "WHATEVERYOUDOMAKEITPAY","TIMEISGOLD","THEONLYCUREFORGRIEFISACTION","GIVEMELIBERTYORGIVEMEDEATH",
        "APOETISTHEPAINTEROFTHESOUL","BELIEVEINYOURSELF","NOSWEATNOSWEET","EARLYBIRDCATCHESTHEWORM",
        "SEEINGISBELIEVING","ASKINGCOSTSNOTHING","GOODFENCESMAKESGOODNEIGHBORS","AROLLINGSTONEGATHERSNOMOSS",
        "ONESUTMOSTMOVESTHEHEAVENS","LITTLEBYLITTLEDOESTHETRICK","LIVEASIFYOUWERETODIETOMORROW","LETBYGONESBEBYGONES",
        "THEBEGINNINGISHALFOFTHEWHOLE","NOPAINNOGAIN","STEPBYSTEPGOESALONGWAY","THEDIFFICULTYINLIFEISTHECHOICE",
        "LIFEISFULLOFUPSANDDOWNS","ROMEWASNOTBUILTINADAY","IFYOUCANTBEATTHEMJOINTHEM","NOTHINGVENTUREDNOTHINGGAINED",
        "KNOWLEDGEINYOUTHISWISDOMINAGE","NOBEESNOHONEY","WHERETHEREISAWILLTHEREISAWAY","HABITISSECONDNATURE",
        "SUTMOSTMOVESTHEHEAVENS","ONLY","ACTION","CUREFORGRIEFIS","SSAFY","BATTLE","ALGORITHM","TANK","MISSION",
        "CODE","HERO","SEIZETHEDAY","LIFEITSELFISAQUOTATION","LIFEISVENTUREORNOTHING","DONTDREAMBEIT",
        "TRYYOURBESTRATHERTHANBETHEBEST","WHATWILLBEWILLBE","DONTDWELLONTHEPAST","PASTISJUSTPAST","FOLLOWYOURHEART",
        "LITTLEBYLITTLEDOESTHETRICK"
    ]
    for s in range(26):
        dec = caesar_decode(ciphertext, s)
        u = dec.upper()
        for kw in keywords:
            if kw in u:
                return dec
    return caesar_decode(ciphertext, 3)

##############################
# 금지/이동 코스트
##############################
def get_forbidden_cells():
    H, W = get_map_size()
    t = get_allied_turret_position()
    forb = set()
    if not t: return forb
    tr, tc = t
    if tr == 0 and tc == W-1:
        cand = [(0,W-2),(0,W-3),(1,W-1),(1,W-2),(1,W-3),(2,W-1),(2,W-2),(2,W-3)]
        forb |= {(r,c) for r,c in cand if 0<=r<H and 0<=c<W}
    if tr == H-1 and tc == 0:
        cand = [(H-1,1),(H-1,2),(H-2,0),(H-2,1),(H-2,2),(H-3,0),(H-3,1),(H-3,2)]
        forb |= {(r,c) for r,c in cand if 0<=r<H and 0<=c<W}
    return forb

def is_forbidden_cell(r,c): return (r,c) in get_forbidden_cells()

def terrain_cost(r,c):
    if is_forbidden_cell(r,c): return None
    cell = map_data[r][c]
    if cell == 'G': return 1
    if cell == 'T': return 2
    return None

def in_bounds(r,c):
    H,W = get_map_size()
    return 0 <= r < H and 0 <= c < W

##############################
# 경로계산
##############################
def reconstruct_commands(prev, start, goal):
    path_dirs = []
    cur = goal
    while cur != start and cur in prev:
        pr, pc, d = prev[cur]
        path_dirs.append(d); cur = (pr,pc)
    path_dirs.reverse()
    return [M_CMD[d] for d in path_dirs]

def dijkstra_to_positions(start, targets):
    H,W = get_map_size(); INF=10**9
    dist=[[INF]*W for _ in range(H)]; prev={}
    sr,sc = start; dist[sr][sc]=0
    pq=[(0,sr,sc)]
    while pq:
        cost,r,c = heapq.heappop(pq)
        if cost!=dist[r][c]: continue
        if (r,c) in targets:
            return reconstruct_commands(prev,start,(r,c))
        for d,(dr,dc) in enumerate(DIRS):
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W:
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
    H,W = get_map_size()
    targets=set()
    for r in range(H):
        for c in range(W):
            if terrain_cost(r,c) is None: continue
            for dr,dc in DIRS:
                nr,nc=r+dr,c+dc
                if 0<=nr<H and 0<=nc<W and map_data[nr][nc]==symbol and not is_forbidden_cell(r,c):
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
    H,W = get_map_size(); out=[]
    for r in range(H):
        for c in range(W):
            if map_data[r][c] in ['E1','E2','E3']:
                out.append((r,c,map_data[r][c]))
    return out

def find_all_enemies():
    H,W = get_map_size(); out=[]
    for r in range(H):
        for c in range(W):
            if map_data[r][c] in ['E1','E2','E3','X']:
                out.append((r,c,map_data[r][c]))
    return out

def can_attack(from_pos, to_pos):
    if not from_pos or not to_pos: return (False,-1)
    fr,fc = from_pos; tr,tc = to_pos
    if fr==tr:
        if fc<tc:
            for c in range(fc+1, tc):
                if map_data[fr][c] not in ['G','S','W']: return (False,-1)
            if tc-fc<=3: return (True,0)
        else:
            for c in range(tc+1, fc):
                if map_data[fr][c] not in ['G','S','W']: return (False,-1)
            if fc-tc<=3: return (True,2)
    elif fc==tc:
        if fr<tr:
            for r in range(fr+1, tr):
                if map_data[r][fc] not in ['G','S','W']: return (False,-1)
            if tr-fr<=3: return (True,1)
        else:
            for r in range(tr+1, fr):
                if map_data[r][fc] not in ['G','S','W']: return (False,-1)
            if fr-tr<=3: return (True,3)
    return (False,-1)

def get_enemy_hp(code):
    try:
        if code in enemies and len(enemies[code])>=1:
            return int(enemies[code][0])
    except:
        pass
    return None

def distance(a,b):
    if not a or not b: return float('inf')
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

##############################
# 아군 유닛/차폐 판정
##############################
def is_allied_tank_cell(val):
    if val in ('','G','S','W','T','F','H','M','E1','E2','E3','X'): return False
    return val in my_allies and val not in ('M','H')

def find_allied_tanks_positions():
    H,W = get_map_size(); out=[]
    for r in range(H):
        for c in range(W):
            if is_allied_tank_cell(map_data[r][c]):
                out.append((r,c))
    return out

def cells_between_on_line(a,b):
    (r1,c1),(r2,c2)=a,b; cells=[]
    if r1==r2:
        step=1 if c1<c2 else -1
        for c in range(c1+step, c2, step): cells.append((r1,c))
    elif c1==c2:
        step=1 if r1<r2 else -1
        for r in range(r1+step, r2, step): cells.append((r,c1))
    return cells

def move_blocks_ally_fire(next_pos):
    allies=find_allied_tanks_positions()
    ens=find_all_enemies()
    for ar,ac in allies:
        for er,ec,_ in ens:
            ok,_=can_attack((ar,ac),(er,ec))
            if not ok: continue
            if next_pos in cells_between_on_line((ar,ac),(er,ec)):
                return True
    return False

def enemies_threat_detail(pos):
    allies_set=set(find_allied_tanks_positions())
    turret=get_allied_turret_position()
    for er,ec,_ in find_all_enemies():
        ok,_=can_attack((er,ec),pos)
        if not ok: continue
        blocked = any(cell in allies_set for cell in cells_between_on_line((er,ec),pos))
        enemy_hits_t = False
        if turret:
            can_hit,_=can_attack((er,ec),turret)
            enemy_hits_t = can_hit
        return True, blocked, enemy_hits_t
    return False, False, False

##############################
# 이동/사격 보조
##############################
def cmd_to_dir(cmd):
    for i,m in enumerate(M_CMD):
        if cmd==m: return i
    return None

def shoot_adjacent_T_if_any():
    pos=get_my_position()
    if not pos: return None
    r,c=pos
    for i,(dr,dc) in enumerate(DIRS):
        nr,nc=r+dr,c+dc
        if in_bounds(nr,nc) and map_data[nr][nc]=='T':
            return S_CMD[i]
    return None

def safe_G_move_if_any():
    pos=get_my_position()
    if not pos: return None
    r,c=pos
    for i,(dr,dc) in enumerate(DIRS):
        nr,nc=r+dr,c+dc
        if not in_bounds(nr,nc) or is_forbidden_cell(nr,nc): continue
        if map_data[nr][nc]!='G': continue
        threat,blocked,_=enemies_threat_detail((nr,nc))
        if not threat or blocked:
            return M_CMD[i]
    return None

def get_next_pos_for_move_cmd(cmd):
    d=cmd_to_dir(cmd)
    if d is None: return None
    r,c=get_my_position()
    dr,dc=DIRS[d]
    return (r+dr,c+dc)

def next_step_with_T_break(path):
    if not path: return None
    first=path[0]; d=cmd_to_dir(first)
    if d is None: return None
    r,c=get_my_position(); dr,dc=DIRS[d]
    nr,nc=r+dr,c+dc
    if not in_bounds(nr,nc) or is_forbidden_cell(nr,nc): return None
    if map_data[nr][nc]=='T': return S_CMD[d]
    if map_data[nr][nc]=='G': return first
    return None

##############################
# 이동 안전가드
##############################
def apply_move_safety(proposed_cmd):
    global safety_stop_used_this_turn, S_USAGE_COUNT
    d=cmd_to_dir(proposed_cmd)
    if d is None: return proposed_cmd
    next_pos=get_next_pos_for_move_cmd(proposed_cmd)
    if not next_pos or not in_bounds(*next_pos) or is_forbidden_cell(*next_pos):
        if S_USAGE_COUNT<2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn=True; S_USAGE_COUNT+=1
            return "S"
        shoot=shoot_adjacent_T_if_any()
        if shoot: return shoot
        safe=safe_G_move_if_any()
        if safe: return safe
        return "S"
    if move_blocks_ally_fire(next_pos):
        if S_USAGE_COUNT<2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn=True; S_USAGE_COUNT+=1
            return "S"
        shoot=shoot_adjacent_T_if_any()
        if shoot: return shoot
        safe=safe_G_move_if_any()
        if safe: return safe
        return proposed_cmd
    threat,blocked,enemy_hits_turret = enemies_threat_detail(next_pos)
    if threat:
        if blocked or enemy_hits_turret:
            return proposed_cmd
        if S_USAGE_COUNT<2 and not safety_stop_used_this_turn:
            safety_stop_used_this_turn=True; S_USAGE_COUNT+=1
            return "S"
        shoot=shoot_adjacent_T_if_any()
        if shoot: return shoot
        safe=safe_G_move_if_any()
        if safe: return safe
        return proposed_cmd
    return proposed_cmd

##############################
# 좌/우 보조, 방어 경로
##############################
def horizontal_step_towards(target):
    my_pos=get_my_position()
    if not my_pos: return None
    r,c=my_pos; tr,tc=target
    if r!=tr: return None
    if tc>c: d,nr,nc=0,r,c+1
    elif tc<c: d,nr,nc=2,r,c-1
    else: return None
    if not in_bounds(nr,nc) or is_forbidden_cell(nr,nc): return None
    if map_data[nr][nc]=='T': return S_CMD[d]
    if map_data[nr][nc]=='G': return M_CMD[d]
    return None

def horizontal_step_towards_col(target_c):
    my_pos=get_my_position()
    if not my_pos: return None
    r,c=my_pos
    if target_c>c: d,nr,nc=0,r,c+1
    elif target_c<c: d,nr,nc=2,r,c-1
    else: return None
    if not in_bounds(nr,nc) or is_forbidden_cell(nr,nc): return None
    if map_data[nr][nc]=='T': return S_CMD[d]
    if map_data[nr][nc]=='G': return M_CMD[d]
    return None

def fallback_move_away_from_enemies_lr():
    my_pos=get_my_position()
    if not my_pos: return None
    r,c=my_pos; best_i=None; best_score=-1
    for i in (0,2):
        dr,dc=DIRS[i]; nr,nc=r+dr,c+dc
        if not in_bounds(nr,nc) or is_forbidden_cell(nr,nc): continue
        if map_data[nr][nc]!='G': continue
        es=find_all_enemies()
        if not es: continue
        score=min(distance((nr,nc),(er,ec)) for er,ec,_ in es)
        if score>best_score: best_score=score; best_i=i
    return M_CMD[best_i] if best_i is not None else None

##############################
# (수정) 기지에서 2~3칸 떨어진 '링' 패트롤 경로
##############################
def build_base_ring_patrol_points():
    """
    메가 2개 확보 후(40턴 이전) 방어 시:
    아군 포탑(H)으로부터 맨해튼 거리 2~3칸 떨어진 칸들 중
    이동/사격으로 처리 가능한 칸(G/T)에서 두 지점을 골라 왕복 패트롤.
    """
    t = get_allied_turret_position()
    if not t: return []
    tr, tc = t
    H, W = get_map_size()

    candidates = []
    for r in range(H):
        for c in range(W):
            if not in_bounds(r,c): continue
            if is_forbidden_cell(r,c): continue
            if map_data[r][c] not in ('G','T'): continue
            d = abs(r-tr) + abs(c-tc)
            if d in (2,3):
                candidates.append((r,c))

    if len(candidates) < 2:
        return []

    # 서로 가장 멀리 떨어진 2점을 선택(왕복 안정성↑)
    best_pair = None
    best_d = -1
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            dd = distance(candidates[i], candidates[j])
            if dd > best_d:
                best_d, best_pair = dd, (candidates[i], candidates[j])

    return list(best_pair) if best_pair else candidates[:2]

def get_optimal_defense_positions():
    turret_pos=get_allied_turret_position()
    if not turret_pos: return []
    tr,tc=turret_pos; H,W=get_map_size()
    def ok(r,c):
        if not in_bounds(r,c): return False
        if is_forbidden_cell(r,c): return False
        return map_data[r][c] in ('G','T')
    # BL
    if tr==H-1 and tc==0:
        r=H-1; cols=list(range(3, max(3,W-1)))
        pts=[(r,c) for c in cols if ok(r,c)]
        if len(pts)>=2:
            if len(pts)>6:
                step=max(1,len(pts)//6); pts=pts[::step]; pts=pts[:2] if len(pts)<2 else pts
            return [("patrol_bottom_row", pts)]
    # TR
    if tr==0:
        r=0; cand=[]
        for dc in (3,4):
            c=tc-dc
            if 0<=c<W and ok(r,c): cand.append((r,c))
        if len(cand)>=2:
            return [("patrol_top_row_3_4", cand)]
        cols=list(range(1, max(1,W-3)))
        pts=[(r,c) for c in cols if ok(r,c)]
        if len(pts)>=2:
            if len(pts)>6:
                step=max(1,len(pts)//6); pts=pts[::step]; pts=pts[:2] if len(pts)<2 else pts
            return [("patrol_top_row_fallback", pts)]
    routes=[]
    if tr>=H//2:
        nr=tr-3; line=[]
        if 0<=nr<H:
            for nc in range(max(0,tc-1), min(W,tc+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("north", line))
    if tc<=W//2:
        nc=tc+3; line=[]
        if 0<=nc<W:
            for nr in range(max(0,tr-1), min(H,tr+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("east", line))
    if tc>=W//2:
        nc=tc-3; line=[]
        if 0<=nc<W:
            for nr in range(max(0,tr-1), min(H,tr+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("west", line))
    if tr<=H//2:
        nr=tr+3; line=[]
        if 0<=nr<H:
            for nc in range(max(0,tc-1), min(W,tc+4)):
                if ok(nr,nc): line.append((nr,nc))
        if line: routes.append(("south", line))
    return routes

def find_closest_enemy_to_turret():
    t=get_allied_turret_position()
    if not t: return None
    ets=find_enemy_tanks()
    if not ets: return None
    tr,tc=t
    return min(ets, key=lambda e: abs(e[0]-tr)+abs(e[1]-tc))

##############################
# 방어 액션 (수정: 메가 2개 확보 & 40턴 이전엔 H에서 2~3칸 링 패트롤)
##############################
def decide_defense_action():
    global current_route, route_position, route_direction
    my_pos=get_my_position()
    if not my_pos: return "S"

    # 1) 즉시 사격(이동보다 우선)
    for er,ec,etype in find_enemy_tanks():
        ok,d=can_attack(my_pos,(er,ec))
        if ok:
            hp=get_enemy_hp(etype)
            if hp is not None and hp<=30: return S_CMD[d]
            if get_mega_bomb_count()>0: return S_MEGA_CMD[d]
            return S_CMD[d]

    # 2) (핵심 수정) 40턴 이전 & 메가포탄 '2개 이상' 확보 시 → 기지 거리 2~3 링 왕복 패트롤
    preferred_routes = []
    my_turn = get_my_turn_count()
    if my_turn < 40 and (mega_bombs_acquired or get_mega_bomb_count() >= 2):
        ring_pts = build_base_ring_patrol_points()
        if ring_pts and len(ring_pts) >= 2:
            preferred_routes = [("base_ring_patrol_2_3", ring_pts)]

    # 3) 그 외엔 기존 최적 방어 루트
    if not preferred_routes:
        preferred_routes = get_optimal_defense_positions()

    route_list=[r[1] for r in preferred_routes]
    if not current_route or (current_route not in route_list and route_list):
        current_route=route_list[0] if route_list else None
        route_position, route_direction = 0,1

    if current_route:
        my_pos=get_my_position()
        if my_pos in current_route and len(current_route)>1:
            idx=current_route.index(my_pos)
            nxt=idx+route_direction
            if nxt>=len(current_route): nxt=len(current_route)-2; route_direction=-1
            elif nxt<0: nxt=1; route_direction=1
            target=current_route[nxt]
        else:
            target=min(current_route, key=lambda p: distance(my_pos,p))

        hs=horizontal_step_towards(target)
        if hs: return apply_move_safety(hs) if hs in M_CMD else hs
        hs2=horizontal_step_towards_col(target[1])
        if hs2: return apply_move_safety(hs2) if hs2 in M_CMD else hs2
        path=dijkstra_to_specific(target)
        step=next_step_with_T_break(path)
        if step: return apply_move_safety(step) if step in M_CMD else step
        fb=fallback_move_away_from_enemies_lr()
        if fb: return apply_move_safety(fb) if fb in M_CMD else fb

    return "S"

##############################
# 보급(메가 탄) 단계
##############################
def is_adjacent_to_supply():
    pos=get_my_position()
    if not pos: return False
    r,c=pos
    for dr,dc in DIRS:
        nr,nc=r+dr,c+dc
        if in_bounds(nr,nc) and map_data[nr][nc]=='F': return True
    return False

def decide_supply_phase_action():
    """
    현 위치 기준 가장 가까운 F로 이동 → 인접 시 해독 → 메가 포탄 확보.
    """
    my_pos=get_my_position()
    if not my_pos: return "S"
    # 선제 사격
    for er,ec,etype in find_enemy_tanks():
        ok,d=can_attack(my_pos,(er,ec))
        if ok:
            hp=get_enemy_hp(etype)
            if get_mega_bomb_count()==1: return S_MEGA_CMD[d]
            if hp is not None and hp<=30: return S_CMD[d]
            return S_CMD[d]
    # 해독
    if codes and is_adjacent_to_supply():
        decoded=find_valid_caesar_decode(codes[0])
        print(f"암호 : {codes[0]}")
        if decoded:
            print(f"응답 : G{decoded}")
            return f"G {decoded}"
    # 내 위치 기준 가장 가까운 F의 인접칸으로 이동 (G=1, T=2)
    path=dijkstra_to_first_adjacent_of('F')
    step=next_step_with_T_break(path)
    if step: return apply_move_safety(step) if step in M_CMD else step
    # 폴백: 좌/우 안전 이동
    fb=fallback_move_away_from_enemies_lr()
    if fb: return apply_move_safety(fb) if fb in M_CMD else fb
    return "S"

##############################
# 40라운드 이후: '3칸 리드' 전진
##############################
def min_enemy_tank_to_H():
    t=get_allied_turret_position()
    if not t: return None
    ets=find_enemy_tanks()
    if not ets: return None
    return min(distance((er,ec), t) for er,ec,_ in ets)

def min_other_allies_to_X():
    x=get_enemy_turret_position()
    if not x: return None
    allies = [pos for pos in find_allied_tanks_positions()]
    if not allies: return None
    return min(distance(pos, x) for pos in allies)

def decide_midgame_action():
    """
    40턴 이후,
      baseline = min( min(E*→H), min(M1/M2/M3→X) )
      목표: 내 탱크의 (→X) 거리가 baseline - 3이 되도록 조절
    """
    my_pos=get_my_position()
    if not my_pos: return "S"
    # 즉시 사격
    for er,ec,etype in find_enemy_tanks():
        ok,d=can_attack(my_pos,(er,ec))
        if ok:
            hp=get_enemy_hp(etype)
            if hp is not None and hp<=30: return S_CMD[d]
            if get_mega_bomb_count()>0: return S_MEGA_CMD[d]
            return S_CMD[d]
    enemy_base=get_enemy_turret_position()
    if not enemy_base:
        return decide_defense_action()
    d_eH = min_enemy_tank_to_H()
    d_mX = min_other_allies_to_X()
    candidates=[d for d in (d_eH,d_mX) if d is not None]
    if not candidates:
        path=dijkstra_to_first_adjacent_of('X')
        step=next_step_with_T_break(path)
        if step: return apply_move_safety(step) if step in M_CMD else step
        fb=fallback_move_away_from_enemies_lr()
        if fb: return apply_move_safety(fb) if fb in M_CMD else fb
        return "S"
    baseline=min(candidates)
    my_to_X = distance(my_pos, enemy_base)
    lead = baseline - my_to_X  # 내가 더 가까울수록 값이 큼

    if lead < 3:
        path=dijkstra_to_first_adjacent_of('X')
        step=next_step_with_T_break(path)
        if step: return apply_move_safety(step) if step in M_CMD else step
        fb=fallback_move_away_from_enemies_lr()
        if fb: return apply_move_safety(fb) if fb in M_CMD else fb
        return "S"
    elif lead == 3:
        return decide_defense_action()
    else:
        return "S"

###################################
# 알고리즘 함수/메서드 부분 구현 끝
###################################
parse_data(game_data)

# 메인 루프
while game_data is not None:
    print_data()
    safety_stop_used_this_turn = False

    mega_count = get_mega_bomb_count()
    if mega_count >= 2:
        mega_bombs_acquired = True

    # 의사결정
    if (mega_count < 2) and (not mega_bombs_acquired):
        proposed = decide_supply_phase_action()
    else:
        if get_my_turn_count() >= 40:
            proposed = decide_midgame_action()
        else:
            # ★ 메가 2개 확보 & 40턴 이전이면 H에서 2~3칸 링 패트롤이 decide_defense_action()에서 우선 적용됨
            proposed = decide_defense_action()

    # 이동 안전가드 적용
    final_cmd = apply_move_safety(proposed) if proposed in M_CMD else proposed

    # S 집계
    if final_cmd == "S" and not safety_stop_used_this_turn and S_USAGE_COUNT < 2:
        S_USAGE_COUNT += 1
        safety_stop_used_this_turn = True

    # 명령 전송
    game_data = submit(final_cmd)

    # 다음 턴 준비
    if game_data:
        parse_data(game_data)

# 종료
close()
