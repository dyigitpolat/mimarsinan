"""The unit suite must never touch the network (loopback excepted)."""

import socket

import pytest


class TestNoNetworkGuard:
    def test_outbound_connect_is_blocked(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with pytest.raises(RuntimeError, match="network access is disabled"):
                sock.connect(("93.184.216.34", 80))

    def test_outbound_connect_by_hostname_is_blocked(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with pytest.raises(RuntimeError, match="network access is disabled"):
                sock.connect(("example.com", 443))

    def test_loopback_connect_is_allowed(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect(("127.0.0.1", port))
