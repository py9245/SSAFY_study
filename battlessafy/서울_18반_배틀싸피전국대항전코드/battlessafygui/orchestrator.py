"""봇 프로세스 관리 및 턴 오케스트레이션."""
from __future__ import annotations

import os
import runpy
import sys
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Deque, Dict, List, Optional

from .compat_socket import install_fake_socket
from .engine import GameEngine
from . import maps

DEFAULT_COMMAND_TIMEOUT = 1.0


def _bot_worker(conn: Connection, script_path: str) -> None:
    restore = install_fake_socket(conn)
    abs_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(abs_path)
    try:
        if script_dir and script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        os.chdir(script_dir or os.getcwd())
        sys.argv = [abs_path]
        runpy.run_path(abs_path, run_name="__main__")
    finally:
        restore()


@dataclass
class BotSpec:
    name: str
    path: str
    tank_id: str
    team: str


class BotProcess:
    def __init__(self, spec: BotSpec):
        self.spec = spec
        self._conn = None
        self._process = None
        self._queue: Deque[str] = deque()
        self._alive = False
        self._waiting_for_state = False
        self._connected = False
        self._initialized = False

    def create_process(self) -> None:
        """새로운 프로세스를 생성합니다."""
        if self._process and self._process.is_alive():
            self.terminate()
        parent_conn, child_conn = Pipe()
        self._conn = parent_conn
        self._process = Process(target=_bot_worker, args=(child_conn, self.spec.path), daemon=True)
        self._queue.clear()
        self._alive = False
        self._waiting_for_state = False
        self._connected = False
        self._initialized = False

    def start(self) -> None:
        if not self._process:
            self.create_process()
        self._process.start()
        self._alive = True

    def terminate(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
        if self._process:
            self._process.join(timeout=1.0)
        self._alive = False

    def _pump_messages(self) -> None:
        if not self._conn:
            return
        while self._conn.poll(0):
            msg_type, payload = self._conn.recv()
            if msg_type == "connect":
                self._connected = True
            elif msg_type == "send":
                command = payload.decode("utf-8", errors="ignore").strip()
                if command:
                    self._queue.append(command)
            elif msg_type == "close":
                self._alive = False
                break
            elif msg_type == "data":
                # 봇이 먼저 데이터를 push 할 일은 없지만, 프로토콜 호환성을 위해 무시.
                continue

    def get_next_command(self, timeout: float) -> Optional[str]:
        deadline = time.time() + timeout
        while True:
            self._pump_messages()
            if self._queue:
                command = self._queue.popleft()
                # 명령을 받았으므로 봇이 응답을 기다리고 있음을 표시
                self._waiting_for_state = True
                return command
            if not self._process or not self._process.is_alive():
                self._alive = False
                return None
            if timeout is not None and time.time() >= deadline:
                return None
            time.sleep(0.005)  # 더 빠른 폴링

    def send_state(self, game_data: str, *, reset_queue: bool = False) -> None:
        if not self._conn:
            return
        payload = game_data.encode("utf-8")
        if reset_queue:
            # 이전 상태를 기준으로 만든 명령이 남아있다면 폐기한다.
            self._queue.clear()
        # 즉시 상태 전송
        try:
            self._conn.send(("data", payload))
            self._waiting_for_state = False
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._alive = False

    @property
    def waiting_for_state(self) -> bool:
        return self._waiting_for_state

    @property
    def is_alive(self) -> bool:
        return self._process and self._process.is_alive()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def mark_initialized(self) -> None:
        self._initialized = True


class TurnOrchestrator:
    def __init__(
        self,
        engine: GameEngine,
        bot_specs: List[BotSpec],
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
    ):
        self.engine = engine
        self.bot_specs = bot_specs
        self.command_timeout = command_timeout
        self.bot_processes: Dict[str, BotProcess] = {spec.name: BotProcess(spec) for spec in bot_specs}
        self.turn_order = bot_specs[:]  # 이미 원하는 순서로 전달된다고 가정
        self.turn_index = 0
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        # 봇 프로세스들은 필요할 때마다 생성함
        self.started = True

    def stop(self) -> None:
        for bot in self.bot_processes.values():
            bot.terminate()
        self.started = False

    def _normalize_command(self, raw: str) -> str:
        raw = raw.strip()
        if not raw:
            return "S"
        init_idx = raw.find("INIT ")
        if init_idx != -1:
            return raw[init_idx:]
        return raw

    def step(self) -> Dict[str, object]:
        if not self.started:
            raise RuntimeError("오케스트레이터가 시작되지 않았습니다.")

        spec = self.turn_order[self.turn_index]
        bot = self.bot_processes[spec.name]

        print(f"[DEBUG] Starting {spec.name}({spec.tank_id})")

        # 새로운 프로세스 생성 및 시작
        bot.create_process()
        bot.start()

        # 초기화 (INIT 핸드셰이크)
        init_cmd = bot.get_next_command(timeout=self.command_timeout)
        if not init_cmd or "INIT" not in init_cmd:
            raise RuntimeError(f"봇 {spec.name}이 INIT 핸드셰이크를 보내지 않았습니다.")

        # 게임 데이터 전송
        game_data = self.engine.build_game_data(spec.tank_id)
        bot.send_state(game_data)
        print(f"[DEBUG] Game data sent to {spec.name}({spec.tank_id})")

        # 명령 받기
        command = bot.get_next_command(timeout=self.command_timeout)
        if command is None:
            normalized = "S"
            print(f"[DEBUG] {spec.name}({spec.tank_id}): No command received -> 'S'")
        else:
            normalized = self._normalize_command(command)
            print(f"[DEBUG] {spec.name}({spec.tank_id}): Raw='{command}' -> Normalized='{normalized}'")

        # 명령 처리
        if normalized and not normalized.startswith("INIT"):
            print(f"[DEBUG] {spec.name}({spec.tank_id}): Processing command '{normalized}'")
            self.engine.process_command(spec.tank_id, normalized)
        else:
            print(f"[DEBUG] {spec.name}({spec.tank_id}): Skipping command processing")

        # 봇 프로세스 종료
        bot.terminate()
        print(f"[DEBUG] {spec.name}({spec.tank_id}) terminated")

        # 턴 완료 후 카운터 증가
        self.engine.advance_turn(spec.tank_id)
        self.turn_index = (self.turn_index + 1) % len(self.turn_order)
        return self.engine.get_snapshot()

    def get_snapshot(self) -> Dict[str, object]:
        return self.engine.get_snapshot()

