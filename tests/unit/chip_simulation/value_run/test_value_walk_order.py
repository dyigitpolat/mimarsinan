"""R6: the value walk orders cores by data dependency, ties broken by stored latency (U1 pass projection)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import run_neural_segment_values
from mimarsinan.chip_simulation.value_run.value_execution import (
    ValueWalkOrderError,
    dependency_order,
    ensure_core_latencies,
    segment_walk_order,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan
from mimarsinan.torch_mapping.converter import convert_torch_model
from unit.chip_simulation.value_run.integer_programs import build_integer_program, lenet5, mlp


def _affine_chain_segment():
    """One neural segment of two convolutions: second-layer cores read first-layer cores."""
    torch.manual_seed(2)
    model = nn.Sequential(nn.Conv1d(2, 3, 3), nn.Conv1d(3, 2, 3)).eval()
    flow = convert_torch_model(model, (2, 8), 2, device="cpu", packaging=MVM_PACKAGING).eval()
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64).map(repr_)
    stage = next(s for s in build_identity_hybrid_mapping(ir_graph=ir).stages if s.kind == "neural")
    return stage.hard_core_mapping, (2 * 8)


def _latency_order(hcm):
    return sorted(range(len(hcm.cores)), key=lambda i: int(hcm.cores[i].latency or 0))


class TestDependencyOrder:
    def test_stored_latencies_that_contradict_the_dependencies_cannot_reorder_the_walk(self):
        hcm, width = _affine_chain_segment()
        x = torch.randn(5, width, dtype=torch.float64)
        reference = run_neural_segment_values(hcm, x)
        consumers = {i for i, c in enumerate(hcm.cores) if any(s.kind == "core" for s in c.get_axon_source_spans())}
        assert consumers and len(consumers) < len(hcm.cores)
        for index, core in enumerate(hcm.cores):
            core.latency = 0 if index in consumers else 5
        latency_first = _latency_order(hcm)[0]
        assert latency_first in consumers, "the stored latencies now put a consumer before its producers"
        order = segment_walk_order(hcm)
        position = {core: rank for rank, core in enumerate(order)}
        for consumer in consumers:
            for span in hcm.cores[consumer].get_axon_source_spans():
                if span.kind == "core":
                    assert position[int(span.src_core)] < position[consumer]
        assert torch.equal(run_neural_segment_values(hcm, x), reference)

    @pytest.mark.parametrize("build", [
        lambda: build_integer_program(mlp(), (8,), num_classes=4),
        lambda: build_integer_program(lenet5(), (1, 28, 28), num_classes=10),
    ])
    def test_the_order_equals_the_latency_order_wherever_that_is_topological(self, build):
        program = build()
        mappings = [program.identity().hybrid_mapping, program.packed_mapping(axons=1024, neurons=64)]
        checked = 0
        for hybrid in mappings:
            for stage in hybrid.stages:
                if stage.kind != "neural":
                    continue
                hcm = stage.hard_core_mapping
                ensure_core_latencies(hcm)
                assert segment_walk_order(hcm) == _latency_order(hcm)
                checked += len(hcm.cores)
        assert checked > 0

    def test_a_cycle_refuses_by_name(self):
        def core(source, latency):
            span = SpikeSourceSpan(kind="core", src_core=source, src_start=0, length=1, dst_start=0)
            return types.SimpleNamespace(latency=latency, get_axon_source_spans=lambda: [span])

        hcm = types.SimpleNamespace(cores=[core(1, 0), core(0, 1)])
        with pytest.raises(ValueWalkOrderError, match="cycle"):
            segment_walk_order(hcm)


class TestGenericOrder:
    """One walk-order contract for every walk over a segment's cores: topological, ties by (latency, index)."""

    def test_every_core_follows_the_cores_it_reads_and_ties_go_by_latency_then_index(self):
        sources = [set(), {0}, set(), {1, 2}, set()]
        assert dependency_order(sources, [3, 0, 1, 0, 1]) == [2, 4, 0, 1, 3]
        assert dependency_order(sources, [0, 0, 0, 0, 0]) == [0, 1, 2, 3, 4]

    def test_a_latency_order_that_is_already_topological_is_returned_unchanged(self):
        sources = [set(), set(), {0}, {1, 2}]
        latencies = [0, 1, 1, 2]
        assert dependency_order(sources, latencies) == sorted(range(4), key=lambda i: (latencies[i], i))

    def test_a_cycle_refuses_by_name(self):
        with pytest.raises(ValueWalkOrderError, match="2 cores"):
            dependency_order([{1}, {0}, set()], [0, 0, 0])
