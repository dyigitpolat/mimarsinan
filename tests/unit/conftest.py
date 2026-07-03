"""Unit-suite guard: no network access beyond loopback (device guards live in
the root conftest, session-wide)."""

import socket

import pytest

_LOOPBACK_HOSTS = {"localhost", "::1", ""}


def _is_loopback(address) -> bool:
    if not isinstance(address, tuple):
        return True
    host = address[0]
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise RuntimeError(
                f"unit-test network access is disabled (attempted connect to {address!r}); "
                "use local fixtures or move the test to an integration tier"
            )
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
