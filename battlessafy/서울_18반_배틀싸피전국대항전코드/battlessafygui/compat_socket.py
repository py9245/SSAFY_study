"""봇 프로세스 안에서 TCP 소켓을 흉내 내는 가짜 소켓 구현."""
from __future__ import annotations

import socket
from multiprocessing.connection import Connection
from typing import Callable, Optional, Tuple


class FakeSocket:
    def __init__(self, conn: Connection):
        self._conn = conn
        self._buffer = bytearray()
        self._closed = False

    def connect(self, address: Tuple[str, int]) -> None:
        self._conn.send(("connect", address))

    def send(self, data: bytes) -> int:
        if self._closed:
            return 0
        payload = bytes(data)
        self._conn.send(("send", payload))
        return len(payload)

    def sendall(self, data: bytes) -> None:
        self.send(data)

    def recv(self, bufsize: int) -> bytes:
        if self._closed:
            return b""
        while len(self._buffer) < bufsize:
            msg_type, payload = self._conn.recv()
            if msg_type == "data":
                self._buffer.extend(payload)
            elif msg_type == "close":
                self._closed = True
                break
            else:
                # 미지정 이벤트는 무시
                continue
            if self._closed or self._buffer:
                break
        if not self._buffer:
            return b""
        chunk = self._buffer[:bufsize]
        del self._buffer[:bufsize]
        return bytes(chunk)

    def close(self) -> None:
        if not self._closed:
            self._conn.send(("close", None))
            self._closed = True
        self._conn.close()

    def settimeout(self, _timeout: Optional[float]) -> None:  # pragma: no cover
        # 해당 봇 로직은 타임아웃을 사용하지 않지만, 호환성을 위해 존재한다.
        return

    def gettimeout(self) -> Optional[float]:  # pragma: no cover
        return None


def install_fake_socket(conn: Connection) -> Callable[[], None]:
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    sock_instance: Optional[FakeSocket] = None

    def socket_factory(*_args, **_kwargs):
        nonlocal sock_instance
        if sock_instance is None:
            sock_instance = FakeSocket(conn)
        return sock_instance

    def fake_create_connection(address, timeout=None, source_address=None):
        sock = socket_factory()
        sock.connect(address)
        return sock

    socket.socket = socket_factory  # type: ignore[assignment]
    socket.create_connection = fake_create_connection  # type: ignore[assignment]

    def restore() -> None:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]

    return restore

