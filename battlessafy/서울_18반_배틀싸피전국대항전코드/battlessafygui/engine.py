"""게임 상태 관리 및 명령 처리 엔진."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from . import maps

DIR_VECTORS = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0),
}

PASS_THROUGH_TERRAINS = {"G", "S", "W"}
BLOCKING_TERRAINS = {"R", "F"}
TREE_TERRAIN = "T"
SUPPLY_TERRAIN = "F"

NORMAL_DAMAGE = 30
MEGA_DAMAGE = 70


@dataclass
class TankState:
    id: str
    team: str
    hp: int
    direction: str
    normal_ammo: int
    mega_ammo: int
    position: Optional[Tuple[int, int]]
    alive: bool = True


@dataclass
class TurretState:
    id: str
    team: str
    hp: int
    position: Optional[Tuple[int, int]]
    alive: bool = True


class GameEngine:
    def __init__(self, parsed_map: maps.ParsedMap):
        self.parsed = parsed_map
        self.base_grid = [row[:] for row in parsed_map.base_grid]
        # 최신 지형 정보가 반영되도록 참조 갱신
        self.parsed.base_grid = self.base_grid

        self.tanks: Dict[str, TankState] = {}
        self.turrets: Dict[str, TurretState] = {}
        self.codes: List[str] = list(parsed_map.codes)
        self.turn_counter: int = 0
        self.game_over: bool = False
        self.victory_team: Optional[str] = None
        self.defeat_team: Optional[str] = None
        self.defeat_reason: Optional[str] = None

        for gid in maps.ATTACKER_TANK_IDS + maps.DEFENDER_TANK_IDS:
            pos = parsed_map.tank_positions.get(gid)
            stats = parsed_map.entity_stats[gid]
            self.tanks[gid] = TankState(
                id=gid,
                team=maps.GLOBAL_TO_TEAM[gid],
                hp=stats.hp,
                direction=stats.direction,
                normal_ammo=99,
                mega_ammo=stats.mega_ammo,
                position=pos,
            )

        for gid in (maps.ATTACKER_TURRET_ID, maps.DEFENDER_TURRET_ID):
            pos = parsed_map.turret_positions.get(gid)
            stats = parsed_map.entity_stats[gid]
            self.turrets[gid] = TurretState(
                id=gid,
                team=maps.GLOBAL_TO_TEAM[gid],
                hp=stats.hp,
                position=pos,
            )

        self.forbidden_cells = {
            "attackers": self._compute_forbidden_cells(self.turrets[maps.ATTACKER_TURRET_ID].position),
            "defenders": self._compute_forbidden_cells(self.turrets[maps.DEFENDER_TURRET_ID].position),
        }

    @classmethod
    def from_text(cls, text: str) -> "GameEngine":
        parsed = maps.parse_map_text(text)
        return cls(parsed)

    def _compute_forbidden_cells(self, turret_pos: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if turret_pos is None:
            return []
        r, c = turret_pos
        height = self.parsed.height
        width = self.parsed.width
        cells: List[Tuple[int, int]] = []
        if (r, c) == (0, width - 1):  # 우상단
            candidates = [
                (0, width - 2), (0, width - 3),
                (1, width - 1), (1, width - 2), (1, width - 3),
                (2, width - 1), (2, width - 2), (2, width - 3),
            ]
            cells.extend([(rr, cc) for rr, cc in candidates if 0 <= rr < height and 0 <= cc < width])
        if (r, c) == (height - 1, 0):  # 좌하단
            candidates = [
                (height - 1, 1), (height - 1, 2),
                (height - 2, 0), (height - 2, 1), (height - 2, 2),
                (height - 3, 0), (height - 3, 1), (height - 3, 2),
            ]
            cells.extend([(rr, cc) for rr, cc in candidates if 0 <= rr < height and 0 <= cc < width])
        return cells

    def _is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.parsed.height and 0 <= c < self.parsed.width

    def _tank_at(self, pos: Tuple[int, int]) -> Optional[TankState]:
        for tank in self.tanks.values():
            if tank.alive and tank.position == pos:
                return tank
        return None

    def _turret_at(self, pos: Tuple[int, int]) -> Optional[TurretState]:
        for turret in self.turrets.values():
            if turret.alive and turret.position == pos:
                return turret
        return None

    def _is_blocked_for_team(self, team: str, pos: Tuple[int, int]) -> bool:
        if not self._is_in_bounds(pos):
            return True
        r, c = pos
        terrain = self.base_grid[r][c]
        if terrain in {"R", "W", TREE_TERRAIN, SUPPLY_TERRAIN}:
            return True
        if pos in self.forbidden_cells.get(team, []):
            return True
        if self._tank_at(pos) is not None:
            return True
        if self._turret_at(pos) is not None:
            return True
        return False

    def process_command(self, actor_id: str, command: str) -> None:
        if self.game_over:
            return
        command = (command or "").strip()
        tank = self.tanks.get(actor_id)
        if not tank or not tank.alive:
            return

        if not command:
            return
        if command == "S":
            return

        # 보급 명령은 "G <text>"
        if command.startswith("G"):
            self._handle_supply_command(tank, command)
            return

        parts = command.split()
        if len(parts) < 2:
            return
        direction_token, action = parts[0], parts[1]
        if direction_token not in DIR_VECTORS:
            return

        if action == "A":
            self._move_tank(tank, direction_token)
        elif action == "F":
            use_mega = len(parts) >= 3 and parts[2] == "M"
            self._shoot(tank, direction_token, use_mega)
        else:
            return

    def _check_defeat(self, team: str) -> None:
        if self.game_over:
            return

        if team == "attackers":
            tank_ids = maps.ATTACKER_TANK_IDS
            turret_id = maps.ATTACKER_TURRET_ID
        else:
            tank_ids = maps.DEFENDER_TANK_IDS
            turret_id = maps.DEFENDER_TURRET_ID

        dead_tanks = sum(1 for tid in tank_ids if not self.tanks[tid].alive)
        turret_dead = not self.turrets[turret_id].alive

        if dead_tanks >= 2 or turret_dead:
            self.game_over = True
            self.defeat_team = team
            self.defeat_reason = "turret" if turret_dead else "tanks"
            if team == "attackers":
                self.victory_team = "defenders"
            elif team == "defenders":
                self.victory_team = "attackers"
            else:
                self.victory_team = None


    def _line_of_fire_exists(self, attacker_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> bool:
        if attacker_pos is None or target_pos is None:
            return False
        ar, ac = attacker_pos
        tr, tc = target_pos
        dr = tr - ar
        dc = tc - ac
        if dr != 0 and dc != 0:
            return False
        distance = abs(dr) + abs(dc)
        if distance == 0 or distance > 3:
            return False
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
        for step in range(1, distance + 1):
            r = ar + step_r * step
            c = ac + step_c * step
            if not self._is_in_bounds((r, c)):
                return False
            if step < distance:
                if self._tank_at((r, c)) is not None:
                    return False
                if self._turret_at((r, c)) is not None:
                    return False
                terrain = self.base_grid[r][c]
                if terrain == TREE_TERRAIN:
                    return False
                if terrain in BLOCKING_TERRAINS:
                    return False
                if terrain in PASS_THROUGH_TERRAINS:
                    continue
                return False
            else:
                terrain = self.base_grid[r][c]
                if terrain == TREE_TERRAIN:
                    return False
                if terrain in BLOCKING_TERRAINS:
                    return False
        return True

    def _would_enter_enemy_fire(self, tank: TankState, new_pos: Tuple[int, int]) -> bool:
        enemy_team = 'attackers' if tank.team == 'defenders' else 'defenders'
        for other in self.living_tanks():
            if other.team != enemy_team or other.position is None:
                continue
            if self._line_of_fire_exists(other.position, new_pos):
                return True
        for turret in self.living_turrets():
            if turret.team != enemy_team or turret.position is None:
                continue
            if self._line_of_fire_exists(turret.position, new_pos):
                return True
        return False

    def _move_tank(self, tank: TankState, direction_token: str) -> None:
        delta = DIR_VECTORS[direction_token]
        if not tank.position:
            return
        new_pos = (tank.position[0] + delta[0], tank.position[1] + delta[1])
        if self._is_blocked_for_team(tank.team, new_pos):
            tank.direction = direction_token
            return
        if self._would_enter_enemy_fire(tank, new_pos):
            tank.direction = direction_token
            return
        # 이동
        tank.position = new_pos
        tank.direction = direction_token

    def _shoot(self, tank: TankState, direction_token: str, use_mega: bool) -> None:
        if use_mega:
            if tank.mega_ammo <= 0:
                return
            damage = MEGA_DAMAGE
            tank.mega_ammo -= 1
        else:
            if tank.normal_ammo <= 0:
                return
            damage = NORMAL_DAMAGE
            tank.normal_ammo -= 1

        tank.direction = direction_token
        delta = DIR_VECTORS[direction_token]
        if not tank.position:
            return
        cur_row, cur_col = tank.position
        for step in range(1, 4):
            target = (cur_row + delta[0] * step, cur_col + delta[1] * step)
            if not self._is_in_bounds(target):
                break

            other_tank = self._tank_at(target)
            if other_tank:
                self._apply_damage_to_tank(other_tank, damage)
                break

            turret = self._turret_at(target)
            if turret:
                self._apply_damage_to_turret(turret, damage)
                break

            terrain = self.base_grid[target[0]][target[1]]
            if terrain == TREE_TERRAIN:
                self.base_grid[target[0]][target[1]] = "G"
                break
            if terrain in BLOCKING_TERRAINS:
                break
            # 통과 가능한 지형이면 계속 진행
            if terrain in PASS_THROUGH_TERRAINS:
                continue
            # 기타 지형은 그대로 중단
            break
    def _handle_supply_command(self, tank: TankState, command: str) -> None:
        if tank.position is None:
            return
        parts = command.split(" ", 1)
        if len(parts) != 2 or not parts[1]:
            return
        if not self._is_adjacent_to_supply(tank.position):
            return

        tank.mega_ammo += 1

        if self.codes:
            self.codes.pop(0)
        self.codes.clear()
        self.codes.append("NMKX")

    def _is_adjacent_to_supply(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if self.base_grid[r][c] == SUPPLY_TERRAIN:
            return True
        for dr, dc in DIR_VECTORS.values():
            nr, nc = r + dr, c + dc
            if self._is_in_bounds((nr, nc)) and self.base_grid[nr][nc] == SUPPLY_TERRAIN:
                return True
        return False

    def _apply_damage_to_tank(self, target: TankState, damage: int) -> None:
        target.hp -= damage
        if target.hp <= 0:
            target.alive = False
            target.position = None
            self._check_defeat(target.team)

    def _apply_damage_to_turret(self, turret: TurretState, damage: int) -> None:
        turret.hp -= damage
        if turret.hp <= 0:
            turret.alive = False
            if turret.position:
                r, c = turret.position
                self.base_grid[r][c] = "G"
            turret.position = None
            self._check_defeat(turret.team)

    def build_game_data(self, tank_id: str) -> str:
        view = maps.build_perspective_for(tank_id)

        tank_stats: Dict[str, maps.EntityStats] = {}
        tank_positions: Dict[str, Optional[Tuple[int, int]]] = {}
        for gid, tank in self.tanks.items():
            tank_positions[gid] = tank.position
            tank_stats[gid] = maps.EntityStats(
                hp=max(tank.hp, 0),
                direction=tank.direction,
                normal_ammo=max(tank.normal_ammo, 0),
                mega_ammo=max(tank.mega_ammo, 0),
            )

        turret_stats: Dict[str, maps.EntityStats] = {}
        turret_positions: Dict[str, Optional[Tuple[int, int]]] = {}
        for gid, turret in self.turrets.items():
            turret_positions[gid] = turret.position
            turret_stats[gid] = maps.EntityStats(hp=max(turret.hp, 0))

        codes_for_view = list(self.codes)
        tank = self.tanks.get(tank_id)
        if tank and tank.alive and tank.position and self._is_adjacent_to_supply(tank.position):
            if codes_for_view[-1:] != ["NMKX"]:
                codes_for_view.append("NMKX")

        if self.game_over:
            summary = f"GAMEOVER {self.victory_team or 'none'}"
            if self.defeat_team:
                summary += f" LOSE:{self.defeat_team} CAUSE:{self.defeat_reason or 'unknown'}"
            if summary not in codes_for_view:
                codes_for_view.append(summary)

        return maps.state_to_text(
            self.parsed,
            tank_stats,
            tank_positions,
            turret_stats,
            turret_positions,
            codes_for_view,
            view,
        )

    def advance_turn(self) -> None:
        self.turn_counter += 1

    def living_tanks(self) -> Iterable[TankState]:
        return (tank for tank in self.tanks.values() if tank.alive)

    def living_turrets(self) -> Iterable[TurretState]:
        return (turret for turret in self.turrets.values() if turret.alive)

    def get_snapshot(self) -> Dict[str, object]:
        return {
            "turn": self.turn_counter,
            "base_grid": [row[:] for row in self.base_grid],
            "tanks": {
                gid: {
                    "hp": tank.hp,
                    "direction": tank.direction,
                    "normal_ammo": tank.normal_ammo,
                    "mega_ammo": tank.mega_ammo,
                    "position": tank.position,
                    "alive": tank.alive,
                }
                for gid, tank in self.tanks.items()
            },
            "turrets": {
                gid: {
                    "hp": turret.hp,
                    "position": turret.position,
                    "alive": turret.alive,
                }
                for gid, turret in self.turrets.items()
            },
            "codes": list(self.codes),
            "game_over": self.game_over,
            "victory_team": self.victory_team,
            "defeat_team": self.defeat_team,
            "defeat_reason": self.defeat_reason,
        }

