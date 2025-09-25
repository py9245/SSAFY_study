import sys
import socket
from collections import deque
from heapq import heappop, heappush
import random

##############################
# 硫붿씤 ?꾨줈洹몃옩 ?듭떊 蹂???뺤쓽
##############################
HOST = '127.0.0.1'
PORT = 8747
ARGS = sys.argv[1] if len(sys.argv) > 1 else ''
sock = socket.socket()

##############################
# 硫붿씤 ?꾨줈洹몃옩 ?듭떊 ?⑥닔 ?뺤쓽
##############################

# 硫붿씤 ?꾨줈洹몃옩 ?곌껐 諛?珥덇린??
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

# 硫붿씤 ?꾨줈洹몃옩?쇰줈 ?곗씠??紐낅졊?? ?꾩넚
def submit(string_to_send):
    try:
        send_data = ARGS + string_to_send + ' '
        sock.send(send_data.encode('utf-8'))

        return receive()
        
    except Exception as e:
        print('[ERROR] Failed to send data. Please check if connection to the main program is valid.')

    return None

# 硫붿씤 ?꾨줈洹몃옩?쇰줈遺???곗씠???섏떊
def receive():
    try:
        game_data = (sock.recv(1024)).decode()

        if game_data and game_data[0].isdigit() and int(game_data[0]) > 0:
            return game_data

        print('[STATUS] No receive data from the main program.')    
        close()

    except Exception as e:
        print('[ERROR] Failed to receive data. Please check if connection to the main program is valid.')

# ?곌껐 ?댁젣
def close():
    try:
        if sock is not None:
            sock.close()
        print('[STATUS] Connection closed')
    
    except Exception as e:
        print('[ERROR] Network connection has been corrupted.')

##############################
# ?낅젰 ?곗씠??蹂???뺤쓽
##############################
map_data = [[]]  # 留??뺣낫. ?? map_data[0][1] - [0, 1]??吏??吏臾?
my_allies = {}  # ?꾧뎔 ?뺣낫. ?? my_allies['M'] - ?뚮젅?댁뼱 蹂몄씤???뺣낫
enemies = {}  # ?곴뎔 ?뺣낫. ?? enemies['X'] - ???ы깙???뺣낫
codes = []  # 二쇱뼱吏??뷀샇臾? ?? codes[0] - 泥?踰덉㎏ ?뷀샇臾?

##############################
# ?낅젰 ?곗씠???뚯떛
##############################

# ?낅젰 ?곗씠?곕? ?뚯떛?섏뿬 媛곴컖??由ъ뒪???뺤뀛?덈━?????
def parse_data(game_data, battle_ssafy):
    # ?낅젰 ?곗씠?곕? ?됱쑝濡??섎늻湲?
    game_data_rows = game_data.split('\n')
    row_index = 0

    # 泥?踰덉㎏ ???곗씠???쎄린
    header = game_data_rows[row_index].split(' ')
    map_height = int(header[0]) if len(header) >= 1 else 0 # 留듭쓽 ?몃줈 ?ш린
    map_width = int(header[1]) if len(header) >= 2 else 0  # 留듭쓽 媛濡??ш린
    num_of_allies = int(header[2]) if len(header) >= 3 else 0  # ?꾧뎔????
    num_of_enemies = int(header[3]) if len(header) >= 4 else 0  # ?곴뎔????
    num_of_codes = int(header[4]) if len(header) >= 5 else 0  # ?뷀샇臾몄쓽 ??
    row_index += 1

    if battle_ssafy is None: 
        battle_ssafy = Battlessafy(map_height, map_width)

    # 湲곗〈??留??뺣낫瑜?珥덇린?뷀븯怨??ㅼ떆 ?쎌뼱?ㅺ린
    map_data.clear()
    map_data.extend([[ '' for c in range(map_width)] for r in range(map_height)])
    for i in range(0, map_height):
        col = game_data_rows[row_index + i].split(' ')
        for j in range(0, len(col)):
            map_data[i][j] = col[j]
    row_index += map_height

    # 湲곗〈???꾧뎔 ?뺣낫瑜?珥덇린?뷀븯怨??ㅼ떆 ?쎌뼱?ㅺ린
    my_allies.clear()
    for i in range(row_index, row_index + num_of_allies):
        ally = game_data_rows[i].split(' ')
        ally_name = ally.pop(0) if len(ally) >= 1 else '-'
        my_allies[ally_name] = ally
    row_index += num_of_allies

    # 湲곗〈???곴뎔 ?뺣낫瑜?珥덇린?뷀븯怨??ㅼ떆 ?쎌뼱?ㅺ린
    enemies.clear()
    for i in range(row_index, row_index + num_of_enemies):
        enemy = game_data_rows[i].split(' ')
        enemy_name = enemy.pop(0) if len(enemy) >= 1 else '-'
        enemies[enemy_name] = enemy
    row_index += num_of_enemies

    # 湲곗〈???뷀샇臾??뺣낫瑜?珥덇린?뷀븯怨??ㅼ떆 ?쎌뼱?ㅺ린
    codes.clear()
    for i in range(row_index, row_index + num_of_codes):
        codes.append(game_data_rows[i])

    battle_ssafy.update_info(num_of_allies, num_of_enemies, map_data, my_allies, enemies, num_of_codes, codes)
    return battle_ssafy

# ?뚯떛???곗씠?곕? ?붾㈃??異쒕젰
def print_data():
    print(f"\n---------- Game Data ----------\n{game_data}\n----------------------------")

    if map_data and map_data[0]:
        print(f"\n[Map] ({len(map_data)} x {len(map_data[0])})")
        for row in map_data:
            print(" ".join(row))
    else:
        print("\n[Map] (empty)")

    print(f"\n[Allies] (count {len(my_allies)})")
    for name, info in my_allies.items():
        if name == "M":
            print(f"M (self) hp={info[0]} dir={info[1]} normal={info[2]} mega={info[3]}")
        elif name == "H":
            print(f"H (ally turret) hp={info[0]}")
        else:
            print(f"{name} (ally tank) hp={info[0]}")

    print(f"\n[Enemies] (count {len(enemies)})")
    for name, info in enemies.items():
        if name == "X":
            print(f"X (enemy turret) hp={info[0]}")
        else:
            print(f"{name} (enemy tank) hp={info[0]}")

    print(f"\n[Codes] (count {len(codes)})")
    for code in codes:
        print(code)
##############################
# ?됰꽕???ㅼ젙 諛?理쒖큹 ?곌껐
##############################
NICKNAME = 'attacker3'
game_data = init(NICKNAME)

###################################
# ?뚭퀬由ъ쬁 ?⑥닔/硫붿꽌??遺遺?援ы쁽 ?쒖옉
###################################

class Battlessafy:
    DIRS = [(0,1), (1,0), (0,-1), (-1,0)]
    MOVE_CMDS = {0: "R A", 1: "D A", 2: "L A", 3: "U A"}
    FIRE_CMDS = {0: "R F", 1: "D F", 2: "L F", 3: "U F"}
    SUPER_AMMO = ' M'
    MY_TANK = 'M'
    TARGET = 'X'
    BASE = 'H'
    ROCK = 'R'
    WATER = 'W'
    TREE = 'T'
    SUPPLY = 'F'
    CODE = 'G '
    ALLIES = ['M1', 'M2', 'M3']
    ENEMIES = ['E1', 'E2', 'E3']
    DISTANCE = {'G': 1, 'T': 2, 'E1': 1, 'E2': 1, 'E3': 1, 'S': 3}

    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.maps = None
        self.my_tank = None
        self.allie = None
        self.allie_num = 0
        self.enemy = None
        self.enemy_num = 0
        self.target = None
        self.dmap = None
        self.range_map = None
        self.current_target = None
        self.positions = None
        self.code_num = 0
        self.code = None
        self.tot_supper_ammo = 0

    def update_info(self, allie_num, enemy_num, map_info, allie, enemy, code_num, code): 
        self.allie_num = allie_num - 1
        self.my_tank = allie.pop(Battlessafy.MY_TANK)
        self.tot_supper_ammo = int(self.my_tank[3])
        self.allie = allie
        self.enemy_num = enemy_num - 1
        self.enemy = enemy
        self.target = enemy.pop(Battlessafy.TARGET)
        self.code_num = code_num
        self.code = code
        self.maps = map_info
        self.positions = self.find_positions()
        self.range_map = self.get_range_map()
        self.current_target = self.target_seeker('ATTACK')

    def find_positions(self):
        positions = {}
        for row in range(self.R):
            for col in range(self.C):
                if self.maps[row][col] == Battlessafy.MY_TANK:
                    positions[Battlessafy.MY_TANK] = (row, col)
                elif self.maps[row][col] == Battlessafy.TARGET:
                    positions[Battlessafy.TARGET] = (row, col)
                elif self.maps[row][col] == Battlessafy.SUPPLY: 
                    positions[Battlessafy.SUPPLY] = (row, col)
                elif self.maps[row][col] in Battlessafy.ALLIES: 
                    positions[self.maps[row][col]] = (row, col)
                elif self.maps[row][col] in Battlessafy.ENEMIES: 
                    positions[self.maps[row][col]] = (row, col)
                elif self.maps[row][col] == Battlessafy.BASE: 
                    positions[Battlessafy.BASE] = (row, col)

        return positions

    def get_range_map(self): 
        rmap = [[[] for _ in range(self.C)]  for _ in range(self.R)]
        br, bc = self.positions[Battlessafy.BASE]
        for r in range(-3, 3): 
            for c in range(-3, 3): 
                if 0 <= br+r < self.R and 0 <= bc+c < self.C: 
                    rmap[br+r][bc+c].append(Battlessafy.BASE)
        for row in range(self.R): 
            for col in range(self.C): 
                if self.maps[row][col] in Battlessafy.ENEMIES+[Battlessafy.TARGET]: 
                    for dir in range(4): 
                        r, c = Battlessafy.DIRS[dir]
                        for ofs in range(1, 4): 
                            nrow, ncol = row+(r*ofs), col+(c*ofs)
                            if (0 <= nrow < self.R and 0 <= ncol <self.C
                                and self.maps[nrow][ncol] not in [Battlessafy.ROCK, Battlessafy.TREE, Battlessafy.SUPPLY]): 
                                rmap[nrow][ncol].append(self.maps[row][col])
                            else: 
                                break
                if self.maps[row][col] == Battlessafy.SUPPLY: 
                    for dir in range(4): 
                        r, c = Battlessafy.DIRS[dir]
                        nrow, ncol = row+r, col+c
                        if (0 <= nrow < self.R and 0 <= ncol <self.C
                            and self.maps[nrow][ncol] not in [Battlessafy.ROCK, Battlessafy.TREE, Battlessafy.SUPPLY]): 
                            rmap[nrow][ncol].append(self.maps[row][col])
        return rmap
    
    def solv_code(self, code): 
        code_list = list(code)
        for i in range(len(code)): 
            code_list[i] = ord(code_list[i]) + 9
        for i in range(len(code)): 
            if code_list[i] > 90: 
                code_list[i] -= 26
            code_list[i] = chr(code_list[i])
        result = ''.join(code_list)
        return result
    
    def target_seeker(self, mode): 
        supply = self.positions.get(Battlessafy.SUPPLY)
        my_pos = self.positions.get(Battlessafy.MY_TANK)
        base = self.positions.get(Battlessafy.BASE)
        if my_pos is None or not self.range_map:
            return Battlessafy.TARGET


        if game_data:
            print(f"{NICKNAME} state:")
            print(game_data)

        if mode == 'ATTACK': 
            if int(self.target[0]) <= 30: 
                return Battlessafy.TARGET
            elif set(self.range_map[my_pos[0]][my_pos[1]]).intersection(set(Battlessafy.ENEMIES)): 
                target = self.range_map[my_pos[0]][my_pos[1]][0]
                t_idx = 1
                while target not in Battlessafy.ENEMIES: 
                    target = self.range_map[my_pos[0]][my_pos[1]][t_idx]
                    t_idx += 1
                    if t_idx == len(self.range_map[my_pos[0]][my_pos[1]]): 
                        return target
                return target
            elif supply and int(self.my_tank[3]) < 2 and self.tot_supper_ammo < 2:  
                return Battlessafy.SUPPLY
            else: 
                return Battlessafy.TARGET
        # if mode == 'DEFENCE': 
        #     for enemy, epos in self.positions.items(): 
        #         if get_manhatan_distance(my_pos, epos) < 8:
        #             return enemy 
        #     if (get_manhatan_distance(my_pos, supply) < 10 and supply 
        #           and int(self.my_tank[3]) < 3 and self.tot_supper_ammo < 3):
        #         return Battlessafy.SUPPLY
        #     elif get_manhatan_distance(my_pos, base) > 8: 
        #         print(f'me-base dis: {get_manhatan_distance(my_pos, base)}')
        #         return Battlessafy.BASE 
        #     else: 
        #         return 'Rand'
                
    
    def path_find(self):
        pqueue = []
        start = self.positions.get(Battlessafy.MY_TANK)
        if start is None or not self.range_map or not self.current_target:
            return 'S'

        visited = {start}
        command = ''
        print(f'Target: {self.current_target}')

        # if self.current_target == 'Rand': 
        #     dir = random.randint(0, 3)
        #     r, c = Battlessafy.DIRS[dir]
        #     while (self.maps[start[0]+r][self[1]+c] 
        #             in ['T', 'R', 'W', 'F', 'H']+Battlessafy.ALLIES+Battlessafy.ENEMIES): 
        #         dir = random.randint(0, 3)
        #         r, c = Battlessafy.DIRS[dir]
        #     command = Battlessafy.MOVE_CMDS[dir]
        #     return command

        if self.current_target in self.range_map[start[0]][start[1]]: 
            if self.current_target == Battlessafy.SUPPLY: 
                if self.code: 
                    codes = self.code.pop()
                    result = self.solv_code(codes)
                    self.tot_supper_ammo += 1
                    return Battlessafy.CODE+result
            for dir in range(4): 
                r, c = Battlessafy.DIRS[dir]
                for ofs in range(1, 4): 
                    nrow, ncol = start[0]+(r*ofs), start[1]+(c*ofs)
                    if 0 <= nrow < self.R and 0 <= ncol < self.C and self.maps[nrow][ncol] == self.current_target:
                        command = Battlessafy.FIRE_CMDS[dir]
                        if self.current_target == Battlessafy.TARGET: 
                            if int(self.target[0]) > 30: 
                                command = self.fire_supper(command)
                        else: 
                            if int(self.enemy.get(self.current_target)[0]) > 30: 
                                command = self.fire_supper(command)
                        return command

        for dir in range(4): 
            r, c = Battlessafy.DIRS[dir]
            nrow, ncol = start[0]+r, start[1]+c
            if (0 <= nrow < self.R and 0 <= ncol < self.C 
                and self.maps[nrow][ncol] not in 
                [Battlessafy.ROCK, Battlessafy.WATER, Battlessafy.SUPPLY, Battlessafy.TARGET]+Battlessafy.ENEMIES+Battlessafy.ALLIES): 
                dist = Battlessafy.DISTANCE[self.maps[nrow][ncol]]
                if self.range_map[nrow][ncol]: 
                    for enemy in self.range_map[nrow][ncol]: 
                        dist += Battlessafy.DISTANCE.get(enemy, 0)
                heappush(pqueue, (dist, (nrow, ncol), dir))

        while pqueue:
            dist, (row, col), init_dir = heappop(pqueue)
            visited.add((row, col))

            if self.current_target in self.range_map[row][col]: 
                r, c = Battlessafy.DIRS[init_dir]
                if self.maps[start[0]+r][start[1]+c] == Battlessafy.TREE: 
                    command = Battlessafy.FIRE_CMDS[init_dir]
                    command = self.fire(command)
                    return command
                else: 
                    command = Battlessafy.MOVE_CMDS[init_dir]
                    return command

            for dir in range(4): 
                r, c = Battlessafy.DIRS[dir]
                nrow, ncol = row+r, col+c
                if (0 <= nrow < self.R and 0 <= ncol < self.C and (nrow, ncol) not in visited
                    and self.maps[nrow][ncol] not in 
                    [Battlessafy.ROCK, Battlessafy.WATER, Battlessafy.SUPPLY, Battlessafy.TARGET]+Battlessafy.ENEMIES+Battlessafy.ALLIES): 
                    ndist = dist + Battlessafy.DISTANCE.get(self.maps[nrow][ncol], 999)
                    if self.range_map[nrow][ncol]: 
                        for enemy in self.range_map[nrow][ncol]: 
                            ndist += Battlessafy.DISTANCE.get(enemy, 0)
                    heappush(pqueue, (ndist, (nrow, ncol), init_dir))
    
    def fire(self, command): 
        if int(self.my_tank[2]) == 0: 
            return command+Battlessafy.SUPER_AMMO
        else: 
            return command

    def fire_supper(self, command): 
        if int(self.my_tank[3]) > 0: 
            return command+Battlessafy.SUPER_AMMO
        else: 
            return command


# 異쒕컻吏? 紐⑹쟻吏???꾩튂 李얘린
# def find_positions(grid, start_mark, goal_mark):
#     rows, cols = len(grid), len(grid[0])
#     start = goal = None

#     for row in range(rows):
#         for col in range(cols):
#             if grid[row][col] == start_mark:
#                 start = (row, col)

#             elif grid[row][col] == goal_mark:
#                 goal = (row, col)

#     return start, goal

# 寃쎈줈 ?먯깋 ?뚭퀬由ъ쬁
# def bfs(grid, start, target, wall):
#     rows, cols = len(grid), len(grid[0])
#     queue = deque([(start, [])])
#     visited = {start}

#     while queue:
#         (r, c), actions = queue.popleft()

#         for d, (dr, dc) in enumerate(DIRS):
#             nr, nc = r + dr, c + dc
#             if (nr, nc) == target:
#                 return actions + [FIRE_CMDS[d]]

#         for d, (dr, dc) in enumerate(DIRS):
#             nr, nc = r + dr, c + dc

#             if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != wall and (nr, nc) not in visited:
#                 visited.add((nr, nc))
#                 queue.append(((nr, nc), actions + [MOVE_CMDS[d]]))

#     return []

# 寃쎈줈 ?먯깋 蹂???뺤쓽
# DIRS = [(0,1), (1,0), (0,-1), (-1,0)]
# MOVE_CMDS = {0: "R A", 1: "D A", 2: "L A", 3: "U A"}
# FIRE_CMDS = {0: "R F", 1: "D F", 2: "L F", 3: "U F"}
# START_SYMBOL = 'M'
# TARGET_SYMBOL = 'X'
# WALL_SYMBOL = 'R'

def get_manhatan_distance(p1, p2): 
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

battle_ssafy = None
# 理쒖큹 ?곗씠???뚯떛
battle_ssafy = parse_data(game_data, battle_ssafy)

# 異쒕컻吏?? 紐⑺몴吏?먯쓽 ?꾩튂 ?뺤씤
# start, target = find_positions(map_data, START_SYMBOL, TARGET_SYMBOL)
# if not start or not target:
#     print("[ERROR] Start or target not found in map")
#     close()
#     exit()

# 理쒖큹 寃쎈줈 ?먯깋
# actions = bfs(map_data, start, target, WALL_SYMBOL)
# actions = battle_ssafy.path_find()

###################################
# ?뚭퀬由ъ쬁 ?⑥닔/硫붿꽌??遺遺?援ы쁽 ??
###################################

# 諛섎났臾? 硫붿씤 ?꾨줈洹몃옩 <-> ?대씪?댁뼵????肄붾뱶) 媛??쒖감濡??곗씠???≪닔???숆린 泥섎━)
while game_data is not None:

    ##############################
    # ?뚭퀬由ъ쬁 硫붿씤 遺遺?援ы쁽 ?쒖옉
    ##############################

    # ?뚯떛???곗씠?곕? ?붾㈃??異쒕젰?섏뿬 ?뺤씤
    # print_data()

    # ?댁쟾 寃쎈줈 ?먯깋 寃곌낵媛 議댁옱?섏? ?딆쓣 寃쎌슦 ?ㅼ떆 ?먯깋
    # if not actions:
    #     actions = battle_ssafy.path_find()

    # ?깊겕瑜??쒖뼱??紐낅졊?대? output??媛믪쑝濡?吏??type: string)
    # output = actions.pop(0) if actions else 'A'

    # 硫붿씤 ?꾨줈洹몃옩?먯꽌 紐낅졊??泥섎━?????덈룄濡?紐낅졊?대? submit()???몄옄濡??꾨떖
    actions = battle_ssafy.path_find()
    print(actions)
    game_data = submit(actions)

    # submit()??由ы꽩?쇰줈 諛쏆? 媛깆떊???곗씠?곕? ?ㅼ떆 ?뚯떛
    if game_data:
        parse_data(game_data, battle_ssafy)

    ##############################
    # ?뚭퀬由ъ쬁 硫붿씤 援ы쁽 ??
    ##############################

# 諛섎났臾몄쓣 鍮좎졇?섏솕????硫붿씤 ?꾨줈洹몃옩怨쇱쓽 ?곌껐???꾩쟾???댁젣?섍린 ?꾪빐 close() ?몄텧
close()

