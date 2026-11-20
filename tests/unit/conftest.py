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


@pytest.fixture
def window_firing_declared(monkeypatch):
    """[U-LIF2] Set the lif_window_firing entry's data (its declared set and default) for one test: the generic code
    at a set a later decision may declare (both values reach integrate_then_emit)."""
    import importlib

    def set_(values=("fire_while_integrating", "integrate_then_emit"), default="fire_while_integrating"):
        axis = importlib.import_module("mimarsinan.chip_simulation.window_firing_axis")
        monkeypatch.setattr(axis, "DECLARED_WINDOW_FIRINGS", tuple(values))
        monkeypatch.setattr(axis, "DEFAULT_WINDOW_FIRING", default)
    return set_


@pytest.fixture
def policy_builds(monkeypatch):
    """[U-LIF2 r2] The point every mode-policy build receives, in order: the builder is replaced in its own module and in
    every loaded module that imported it by name, so a site's build is seen wherever the site reads the builder."""
    import sys

    import mimarsinan.chip_simulation.spiking_mode_policy as source

    real = source.policy_for_spiking_mode
    seen = []

    def spy(point, *args, **kwargs):
        seen.append(point)
        return real(point, *args, **kwargs)

    spy.policy_builds_real = real  # a module first imported while a spy was installed holds that spy: replaced too
    for name, module in list(sys.modules.items()):
        held = getattr(module, "policy_for_spiking_mode", None) if name.startswith("mimarsinan") else None
        if held is real or getattr(held, "policy_builds_real", None) is real:
            monkeypatch.setattr(module, "policy_for_spiking_mode", spy)
    return seen
