"""[ODIN C4] A host-mediated pass schedule must be CAUSAL, not index-ordered.

A pass is host-mediated: the host reads one core's counts back and words them
as the next core's stimulus. So a core may only run after every core it reads,
and the shipped executor refuses a bundle whose ``pass_order`` says otherwise
(``require_pass_order_is_causal``). Every bundle before the three-hop MLP had
core-index order == topological order by luck, which is exactly why the freezer
emitting ``range(len(cores))`` went unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.odin_hacc.fabric_builds import (
    PassOrderError,
    causal_core_order,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping


def _core(sources) -> HardCore:
    core = HardCore(axons_per_core=len(sources), neurons_per_core=2,
                    has_bias_capability=False)
    core.core_matrix = np.zeros((len(sources), 2), dtype=np.float64)
    core.axon_sources = list(sources)
    core.threshold = 1.0
    core.available_axons = 0
    core.available_neurons = 0
    return core


def _mapping(cores) -> HardCoreMapping:
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = list(cores)
    mapping.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    return mapping


def _entry(count: int):
    return [SpikeSource(-2, index, is_input=True) for index in range(count)]


class TestTheOrderIsTopologicalAndNotIndexOrder:
    def test_the_three_hop_cascade_runs_producer_first(self):
        """The C3 MLP's own shape: core 1 -> core 2 -> readout core 0."""
        mapping = _mapping([
            _core([SpikeSource(2, 0)]),   # 0: the readout, fed by core 2
            _core(_entry(2)),             # 1: fed by the entry raster
            _core([SpikeSource(1, 0)]),   # 2: fed by core 1
        ])
        assert causal_core_order(mapping) == (1, 2, 0)

    def test_an_already_causal_mapping_keeps_its_index_order(self):
        """Byte-identical for every bundle frozen before this: a two-hop
        mapping whose index order is already topological must not move."""
        mapping = _mapping([_core(_entry(2)), _core([SpikeSource(0, 0)])])
        assert causal_core_order(mapping) == (0, 1)

    def test_independent_cores_break_ties_on_the_lowest_index(self):
        """Determinism: a bundle's pass order is part of its self-hash."""
        mapping = _mapping([_core(_entry(1)), _core(_entry(1)), _core(_entry(1))])
        assert causal_core_order(mapping) == (0, 1, 2)

    def test_a_deep_chain_is_fully_reversed_when_it_has_to_be(self):
        mapping = _mapping([
            _core([SpikeSource(1, 0)]),
            _core([SpikeSource(2, 0)]),
            _core([SpikeSource(3, 0)]),
            _core(_entry(1)),
        ])
        assert causal_core_order(mapping) == (3, 2, 1, 0)


class TestAnUnorderableMappingRefuses:
    def test_a_cycle_has_no_causal_order_and_says_so(self):
        mapping = _mapping([
            _core([SpikeSource(1, 0)]),
            _core([SpikeSource(0, 0)]),
        ])
        with pytest.raises(PassOrderError, match="cycle"):
            causal_core_order(mapping)

    def test_a_route_to_a_core_that_does_not_exist_refuses(self):
        mapping = _mapping([_core([SpikeSource(7, 0)])])
        with pytest.raises(PassOrderError, match="core 7"):
            causal_core_order(mapping)


class TestTheFreezerEmitsThatOrder:
    def test_pass_builds_come_back_in_causal_order(self):
        """The ONE place that chooses builds is the one that orders them."""
        from mimarsinan.chip_simulation.odin_hacc.fabric_builds import pass_builds

        mapping = _mapping([
            _core([SpikeSource(2, 0)]),
            _core(_entry(2)),
            _core([SpikeSource(1, 0)]),
        ])
        assert causal_core_order(mapping) == (1, 2, 0)
        assert [b.index for b in _stub_builds(pass_builds, mapping)] == [1, 2, 0]


def _stub_builds(pass_builds, mapping):
    """``pass_builds`` with its per-core construction stubbed to an index tag.

    The ORDER is what this file is about; what a build IS has its own gates.
    """
    import mimarsinan.chip_simulation.odin_hacc.fabric_builds as module

    class _Tag:
        def __init__(self, index):
            self.index = index

    original = module.PassBuild
    module.PassBuild = lambda _m, index, *a, **k: _Tag(index)
    try:
        return pass_builds(
            mapping, [object()], weight_bits=4, effective_max_axons=127,
            weight_sign_granularity="per_axon", soma_law=_Law(), membrane_init=0)
    finally:
        module.PassBuild = original


class _Law:
    membrane_bits = 8
