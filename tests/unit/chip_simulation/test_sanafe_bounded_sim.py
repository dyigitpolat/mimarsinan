"""simulate_chip_bounded: SANA-FE chip.sim under a wall cap with one fresh-chip retry."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.execution_bounds import SimulationTimeoutError
from mimarsinan.chip_simulation.sanafe.runner.bounded_sim import simulate_chip_bounded


class _RecordingChip:
    created: list["_RecordingChip"] = []
    sim_behavior = staticmethod(lambda: {"spikes": 3})

    def __init__(self, arch):
        self.arch = arch
        self.loaded = None
        self.sim_calls: list[dict] = []
        _RecordingChip.created.append(self)

    def load(self, net):
        self.loaded = net

    def sim(self, timesteps, *, spike_trace=None, potential_trace=None,
            message_trace=None):
        self.sim_calls.append({
            "timesteps": timesteps,
            "spike_trace": spike_trace,
            "potential_trace": potential_trace,
            "message_trace": message_trace,
        })
        return _RecordingChip.sim_behavior()


@pytest.fixture(autouse=True)
def _reset_chip_class():
    _RecordingChip.created = []
    _RecordingChip.sim_behavior = staticmethod(lambda: {"spikes": 3})
    yield


def _fake_module():
    return SimpleNamespace(SpikingChip=_RecordingChip)


def test_normal_completion_returns_chip_and_results_unchanged():
    arch, net = object(), object()
    payload = {"spikes": 42, "energy": {"total": 1.0}}
    _RecordingChip.sim_behavior = staticmethod(lambda: payload)

    chip, results = simulate_chip_bounded(
        _fake_module(), arch, net, 8,
        spike_trace=True, potential_trace=False, message_trace=True,
        timeout_s=30.0,
    )

    assert results is payload
    assert len(_RecordingChip.created) == 1
    assert chip is _RecordingChip.created[0]
    assert chip.arch is arch
    assert chip.loaded is net
    assert chip.sim_calls == [{
        "timesteps": 8,
        "spike_trace": True,
        "potential_trace": False,
        "message_trace": True,
    }]


def test_hung_sim_retries_on_a_fresh_chip_then_fails_loud():
    _RecordingChip.sim_behavior = staticmethod(lambda: time.sleep(30.0))

    t0 = time.monotonic()
    with pytest.raises(SimulationTimeoutError, match="twice"):
        simulate_chip_bounded(
            _fake_module(), object(), object(), 8,
            spike_trace=True, potential_trace=False, message_trace=False,
            timeout_s=0.2,
        )
    assert time.monotonic() - t0 < 10.0
    assert len(_RecordingChip.created) == 2, "retry must build a fresh chip"


def test_sim_exception_propagates_without_retry():
    def _boom():
        raise ValueError("sim bug")

    _RecordingChip.sim_behavior = staticmethod(_boom)

    with pytest.raises(ValueError, match="sim bug"):
        simulate_chip_bounded(
            _fake_module(), object(), object(), 8,
            spike_trace=True, potential_trace=False, message_trace=False,
            timeout_s=30.0,
        )
    assert len(_RecordingChip.created) == 1
