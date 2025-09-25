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
        parent_conn, child_conn = Pipe()
        self._conn = parent_conn
        self._process = Process(target=_bot_worker, args=(child_conn, spec.path), daemon=True)
        self._queue: Deque[str] = deque()
        self._alive = False
        self._waiting_for_state = False
        self._connected = False

    def start(self) -> None:
        self._process.start()
        self._alive = True

    def terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=1.0)
        self._alive = False

    def _pump_messages(self) -> None:
        while self._conn.poll(0):
            msg_type, payload = self._conn.recv()
            if msg_type == "connect":
                self._connected = True
            elif msg_type == "send":
                command = payload.decode("utf-8", errors="ignore").strip()
                if command:
                    self._queue.append(command)
                    self._waiting_for_state = True
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
                return self._queue.popleft()
            if not self._process.is_alive():
                self._alive = False
                return None
            if timeout is not None and time.time() >= deadline:
                return None
            time.sleep(0.01)

    def send_state(self, game_data: str) -> None:
        payload = game_data.encode("utf-8")
        self._conn.send(("data", payload))
        self._waiting_for_state = False

    @property
    def waiting_for_state(self) -> bool:
        return self._waiting_for_state

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()


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
        for process in self.bot_processes.values():
            process.start()
        # 핸드셰이크 처리
        for spec in self.turn_order:
            bot = self.bot_processes[spec.name]
            init_cmd = bot.get_next_command(timeout=self.command_timeout)
            if not init_cmd or "INIT" not in init_cmd:
                raise RuntimeError(f"봇 {spec.name}이 INIT 핸드셰이크를 보내지 않았습니다.")
            game_data = self.engine.build_game_data(spec.tank_id)
            bot.send_state(game_data)
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

        command = bot.get_next_command(timeout=self.command_timeout)
        if command is None:
            normalized = "S"
        else:
            normalized = self._normalize_command(command)

        if normalized.startswith("INIT"):
            # 예외 상황: 봇이 재시작되었을 때. 최신 상태 재전송 후 다음 명령 대기.
            bot.send_state(self.engine.build_game_data(spec.tank_id))
            command = bot.get_next_command(timeout=self.command_timeout)
            normalized = self._normalize_command(command or "")

        if normalized and not normalized.startswith("INIT"):
            self.engine.process_command(spec.tank_id, normalized)
        self.engine.advance_turn()

        # 명령 송신자에게 최신 상태 전달
        bot.send_state(self.engine.build_game_data(spec.tank_id))

        # 다른 봇이 응답을 기다리고 있다면 최신 상태 브로드캐스트
        for other_spec in self.turn_order:
            other_bot = self.bot_processes[other_spec.name]
            if other_bot is bot:
                continue
            if other_bot.waiting_for_state:
                other_bot.send_state(self.engine.build_game_data(other_spec.tank_id))

        self.turn_index = (self.turn_index + 1) % len(self.turn_order)
        return self.engine.get_snapshot()

    def get_snapshot(self) -> Dict[str, object]:
        return self.engine.get_snapshot()

