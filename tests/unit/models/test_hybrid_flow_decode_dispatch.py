"""Lock SpikingHybridCoreFlow.forward dispatch on policy.decode_mode (Vector V2).

The forward used to select ``_forward_rate`` / ``_forward_ttfs`` via a
``is_cascaded_ttfs`` + ``requires_ttfs_firing`` predicate cascade. V2 routes it
through ``policy.decode_mode()`` (``count`` → rate, ``timing`` → ttfs). These
tests pin that the policy-driven routing is byte-identical to the legacy cascade
for every (mode × schedule), without building a real mapping.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_semantics import (
    is_cascaded_ttfs,
    requires_ttfs_firing,
)
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

CASES = [
    ("lif", "cascaded"),
    ("rate", "cascaded"),
    ("ttfs", "cascaded"),
    ("ttfs", "synchronized"),
    ("ttfs_quantized", "cascaded"),
    ("ttfs_quantized", "synchronized"),
    ("ttfs_cycle_based", "cascaded"),
    ("ttfs_cycle_based", "synchronized"),
]


def _legacy_route(spiking_mode: str, schedule: str) -> str:
    """The pre-V2 predicate cascade, returning the chosen forward's name."""
    if is_cascaded_ttfs(spiking_mode, schedule):
        return "_forward_rate"
    if requires_ttfs_firing(spiking_mode):
        return "_forward_ttfs"
    return "_forward_rate"


class _StubFlow(SpikingHybridCoreFlow):
    """Bypass __init__ to test forward()'s routing in isolation."""

    def __init__(self, spiking_mode: str, schedule: str):
        torch.nn.Module.__init__(self)
        self.spiking_mode = spiking_mode
        self.ttfs_cycle_schedule = schedule
        self.preprocessor = torch.nn.Identity()
        self.simulation_length = 4
        self.routed_to: str | None = None

    def _forward_rate(self, x):
        self.routed_to = "_forward_rate"
        return x

    def _forward_ttfs(self, x):
        self.routed_to = "_forward_ttfs"
        return x

    def _evict_segment_cache(self):
        pass


@pytest.mark.parametrize("mode,schedule", CASES)
def test_forward_routes_by_decode_mode(mode, schedule):
    flow = _StubFlow(mode, schedule)
    flow.forward(torch.zeros(1, 3))
    assert flow.routed_to == _legacy_route(mode, schedule)


@pytest.mark.parametrize("mode,schedule", CASES)
def test_decode_mode_maps_to_legacy_route(mode, schedule):
    """``timing`` ⇔ ttfs forward, ``count`` ⇔ rate forward, per the old cascade."""
    decode = policy_for_spiking_mode(mode, schedule).decode_mode()
    expected = "_forward_ttfs" if decode == "timing" else "_forward_rate"
    assert expected == _legacy_route(mode, schedule)
