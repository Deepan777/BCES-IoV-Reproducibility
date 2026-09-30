"""Process-local guard that makes accidental network use fail closed."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch


class NetworkAccessProhibited(RuntimeError):
    pass


@dataclass
class NetworkGuardState:
    attempts: int = 0


@contextmanager
def deny_network_access() -> Iterator[NetworkGuardState]:
    state = NetworkGuardState()

    def blocked(*_args: object, **_kwargs: object) -> None:
        state.attempts += 1
        raise NetworkAccessProhibited("network access is forbidden in Phase 2")

    with (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket, "getaddrinfo", blocked),
        patch.object(socket.socket, "connect", blocked),
        patch.object(socket.socket, "connect_ex", blocked),
    ):
        yield state
