"""맵 파싱과 시리얼라이저 유틸리티.

현재 게임 로직은 두 팀(공격/수비)의 탱크와 포탑을 전제로 동작한다.
본 모듈은 맵 텍스트를 내부 상태로 변환하고, 특정 봇 시각에서
동일한 포맷의 game_data 문자열을 생성하는 기능을 제공한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

BASE_TERRAINS = {"G", "S", "W", "T", "F", "R"}

# 전역 ID 정의
ATTACKER_TANK_IDS = ["A1", "A2", "A3"]
DEFENDER_TANK_IDS = ["B1", "B2", "B3"]
ATTACKER_TURRET_ID = "AH"
DEFENDER_TURRET_ID = "BH"

TOKEN_TO_GLOBAL = {
    "M": "A1",
    "M1": "A1",
    "M2": "A2",
    "M3": "A3",
    "E1": "B1",
    "E2": "B2",
    "E3": "B3",
    "H": ATTACKER_TURRET_ID,
    "X": DEFENDER_TURRET_ID,
}

GLOBAL_TO_TEAM = {
    "A1": "attackers",
    "A2": "attackers",
    "A3": "attackers",
    ATTACKER_TURRET_ID: "attackers",
    "B1": "defenders",
    "B2": "defenders",
    "B3": "defenders",
    DEFENDER_TURRET_ID: "defenders",
}

DIRECTION_FALLBACK = "U"


@dataclass
class EntityStats:
    hp: int
    direction: str = DIRECTION_FALLBACK
    normal_ammo: int = 0
    mega_ammo: int = 0


@dataclass
class ParsedMap:
    height: int
    width: int
    base_grid: List[List[str]]
    tank_positions: Dict[str, Tuple[int, int]]
    turret_positions: Dict[str, Tuple[int, int]]
    entity_stats: Dict[str, EntityStats]
    codes: List[str]


class MapFormatError(ValueError):
    """입력 맵이 기대한 형식을 따르지 않을 때 발생."""


def _token_to_global_id(token: str) -> Optional[str]:
    return TOKEN_TO_GLOBAL.get(token)


def parse_map_text(text: str) -> ParsedMap:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise MapFormatError("맵 텍스트가 비어 있습니다.")

    header_tokens = lines[0].split()
    if len(header_tokens) < 4:
        raise MapFormatError("헤더 정보가 부족합니다. (height width allies enemies [codes])")

    height = int(header_tokens[0])
    width = int(header_tokens[1])
    num_allies = int(header_tokens[2])
    num_enemies = int(header_tokens[3])
    num_codes = int(header_tokens[4]) if len(header_tokens) >= 5 else 0

    if len(lines) < 1 + height:
        raise MapFormatError("맵 행 수가 헤더에 명시된 값보다 적습니다.")

    base_grid: List[List[str]] = [["G" for _ in range(width)] for _ in range(height)]
    tank_positions: Dict[str, Tuple[int, int]] = {}
    turret_positions: Dict[str, Tuple[int, int]] = {}

    row_idx = 1
    for r in range(height):
        row_tokens = lines[row_idx + r].split()
        if len(row_tokens) != width:
            raise MapFormatError(f"{r}번째 맵 행의 토큰 수가 {width}와 다릅니다.")
        for c, token in enumerate(row_tokens):
            global_id = _token_to_global_id(token)
            if global_id:
                if global_id in tank_positions or global_id in turret_positions:
                    raise MapFormatError(f"중복 배치된 엔티티 토큰: {token} @ ({r}, {c})")
                if global_id in (ATTACKER_TURRET_ID, DEFENDER_TURRET_ID):
                    turret_positions[global_id] = (r, c)
                else:
                    tank_positions[global_id] = (r, c)
                base_grid[r][c] = "G"
            elif token in BASE_TERRAINS:
                base_grid[r][c] = token
            else:
                raise MapFormatError(f"알 수 없는 셀 토큰 '{token}' @ ({r}, {c})")
    row_idx += height

    entity_stats: Dict[str, EntityStats] = {}

    def _parse_entity_line(line: str) -> Tuple[str, EntityStats]:
        parts = line.split()
        if not parts:
            raise MapFormatError("엔티티 정보가 비어 있습니다.")
        token = parts[0]
        global_id = _token_to_global_id(token)
        if not global_id:
            raise MapFormatError(f"알 수 없는 엔티티 토큰 '{token}'")
        hp = int(parts[1]) if len(parts) >= 2 else 0
        direction = parts[2] if len(parts) >= 3 else DIRECTION_FALLBACK
        normal = int(parts[3]) if len(parts) >= 4 else 0
        mega = int(parts[4]) if len(parts) >= 5 else 0
        return global_id, EntityStats(hp=hp, direction=direction, normal_ammo=normal, mega_ammo=mega)

    for _ in range(num_allies):
        if row_idx >= len(lines):
            raise MapFormatError("헤더에 명시된 아군 정보 수와 입력이 일치하지 않습니다.")
        gid, stats = _parse_entity_line(lines[row_idx])
        entity_stats[gid] = stats
        row_idx += 1

    enemy_lines: List[str] = []
    for _ in range(num_enemies):
        if row_idx >= len(lines):
            raise MapFormatError("헤더에 명시된 적군 정보 수와 입력이 일치하지 않습니다.")
        enemy_lines.append(lines[row_idx])
        row_idx += 1

    # 추가로 남은 라인이 있고 첫 토큰이 'X'라면 포탑 정보를 확장 처리한다.
    if row_idx < len(lines):
        next_token = lines[row_idx].split()[0]
        if next_token == "X":
            enemy_lines.append(lines[row_idx])
            row_idx += 1

    for line in enemy_lines:
        gid, stats = _parse_entity_line(line)
        entity_stats[gid] = stats

    codes = lines[row_idx:]
    if len(codes) != num_codes:
        # 실제 남은 코드 수로 갱신 (맵 데이터 불일치에 관대한 처리)
        num_codes = len(codes)

    for gid in ATTACKER_TANK_IDS + DEFENDER_TANK_IDS:
        if gid not in tank_positions:
            raise MapFormatError(f"탱크 {gid}의 위치 정보가 누락되었습니다.")
        if gid not in entity_stats:
            # 입력에 탄약/방향 정보가 없다면 기본값 사용
            entity_stats[gid] = EntityStats(hp=100)

    for gid in (ATTACKER_TURRET_ID, DEFENDER_TURRET_ID):
        if gid not in turret_positions:
            raise MapFormatError(f"포탑 {gid}의 위치 정보가 누락되었습니다.")
        if gid not in entity_stats:
            entity_stats[gid] = EntityStats(hp=100)

    return ParsedMap(
        height=height,
        width=width,
        base_grid=base_grid,
        tank_positions=tank_positions,
        turret_positions=turret_positions,
        entity_stats=entity_stats,
        codes=codes,
    )


@dataclass
class ViewPerspective:
    primary: str
    allies: List[str]
    turret: str
    enemies: List[str]
    enemy_turret: str


def build_perspective_for(global_tank_id: str) -> ViewPerspective:
    team = GLOBAL_TO_TEAM[global_tank_id]
    if team == "attackers":
        allies = [gid for gid in ATTACKER_TANK_IDS if gid != global_tank_id]
        enemies = DEFENDER_TANK_IDS
        turret = ATTACKER_TURRET_ID
        enemy_turret = DEFENDER_TURRET_ID
    else:
        allies = [gid for gid in DEFENDER_TANK_IDS if gid != global_tank_id]
        enemies = ATTACKER_TANK_IDS
        turret = DEFENDER_TURRET_ID
        enemy_turret = ATTACKER_TURRET_ID
    return ViewPerspective(
        primary=global_tank_id,
        allies=allies,
        turret=turret,
        enemies=enemies,
        enemy_turret=enemy_turret,
    )


def perspective_token_mapping(view: ViewPerspective) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    mapping[view.primary] = "M"
    ally_labels = ["M2", "M3"]
    for gid, label in zip(view.allies, ally_labels):
        mapping[gid] = label
    mapping[view.turret] = "H"
    enemy_labels = ["E1", "E2", "E3"]
    for gid, label in zip(view.enemies, enemy_labels):
        mapping[gid] = label
    mapping[view.enemy_turret] = "X"
    return mapping


def format_entity_line(token: str, stats: EntityStats) -> str:
    if token in {"H", "X"}:
        return f"{token} {stats.hp}"
    return f"{token} {stats.hp} {stats.direction} {stats.normal_ammo} {stats.mega_ammo}"


def state_to_text(
    parsed: ParsedMap,
    tank_stats: Dict[str, EntityStats],
    tank_positions: Dict[str, Tuple[int, int]],
    turret_stats: Dict[str, EntityStats],
    turret_positions: Dict[str, Tuple[int, int]],
    codes: Iterable[str],
    perspective: ViewPerspective,
) -> str:
    mapping = perspective_token_mapping(perspective)

    height = parsed.height
    width = parsed.width

    grid = [[cell for cell in row] for row in parsed.base_grid]

    # 보급소(F)는 base_grid에 이미 포함되어 있음. 탱크/포탑만 오버레이한다.
    for gid, pos in tank_positions.items():
        if pos is None:
            continue
        token = mapping.get(gid)
        if token:
            r, c = pos
            grid[r][c] = token
    for gid, pos in turret_positions.items():
        if pos is None:
            continue
        token = mapping.get(gid)
        if token:
            r, c = pos
            grid[r][c] = token

    map_lines = [" ".join(row) for row in grid]

    def _collect_stats(ids: List[str]) -> List[Tuple[str, EntityStats]]:
        out: List[Tuple[str, EntityStats]] = []
        for gid in ids:
            token = mapping.get(gid)
            if token and gid in tank_stats:
                out.append((token, tank_stats[gid]))
        return out

    allies = [("M", tank_stats[perspective.primary])]
    allies.extend(_collect_stats(perspective.allies))
    allies.append(("H", turret_stats[perspective.turret]))

    enemies = _collect_stats(perspective.enemies)
    enemies.append(("X", turret_stats[perspective.enemy_turret]))

    code_list = list(codes)
    header = f"{height} {width} {len(allies)} {len(enemies)} {len(code_list)}"

    body_lines: List[str] = []
    body_lines.extend(map_lines)
    body_lines.extend(format_entity_line(token, stats) for token, stats in allies)
    body_lines.extend(format_entity_line(token, stats) for token, stats in enemies)
    body_lines.extend(code_list)

    return "\n".join([header, *body_lines])

