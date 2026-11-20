"""[thru:w1] Layer copies are cost-only (PL/thru/design.md (c) 'Schedule', (e) 'Schedule'): a program built with
``layer_replication`` equals the identity program at atol 0 on the value executor and its int64 golden (the lane-fused
residual block at 8, 3 and 1 cores) and on the cycle-accurate streamed-LIF executor (outputs and per-node spike
counts), the rate and synchronized LIF executors and every TTFS variant; each group's first pass loads every copy
(cores ordered by key, then copy) and every later pass runs on them, each core at its head ordinal, verified pass by
pass; W15's static check establishes Theorem 1 on each of these programs (the TTFS forms it reads as cycle executors:
synchronized and cascaded; analytical and quantized TTFS are outside Theorem 1 by name). Copies of a key may share a
core: a core holding an instance of the pass runs wholly, and the pass declares the columns of its other copies idle
(loaded, run on off axons, read by nothing), which the check exempts from its unplaced-column rule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts
from mimarsinan.certification.program_check import CheckDeclaration
from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.mapping.support.wavefront_stages import (
    SCHEDULE_CORE_ORDINALS, SCHEDULE_DARK_CORES, SCHEDULE_IDLE_COLUMNS, idle_columns, wavefront_head)

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV
from unit.chip_simulation.value_run.integer_programs import assert_exact, integer_inputs
from unit.mapping.position_multiplexing import vehicles as W02

from . import vehicles as V

COPIES = {"layer_replication": True}
#: The LIF chains' chip: 8 of the position-multiplexing LIF vehicles' cores, so a copy serves several positions.
LIF_COPY_CORES = [dict(V.LIF_CORES[0], count=8)]


@pytest.fixture(scope="module")
def block():
    return V.lane_program("block")


@pytest.fixture(scope="module")
def built(block):
    return {count: {name: V.build(block.ir, V.cores(count), True, **kw) for name, kw in (("one", {}), ("copies", COPIES))}
            for count in (8, 3, 1)}


def _layout_cores(hybrid) -> list:
    return [wavefront_head(s).layout_cores for s in V.heads(hybrid)]


class TestValueSemantics:
    @pytest.mark.parametrize("count", [8, 3, 1])
    def test_copies_equal_the_identity_and_the_golden_at_atol_0(self, block, built, count):
        x = integer_inputs(block, 8)
        copies = built[count]["copies"]
        want = V.value(block.identity().hybrid_mapping, x)
        assert_exact(V.value(copies, x), want)
        codes, steps = run_golden(copies, x)
        assert_exact(torch.as_tensor(codes * steps), want)
        assert len(V.neural(copies)) <= len(V.neural(built[count]["one"]))

    def test_copies_take_the_free_cores_and_cut_the_passes(self, built):
        one, copies = built[8]["one"], built[8]["copies"]
        assert len(V.neural(copies)) < len(V.neural(one))
        assert sum(_layout_cores(copies)) > sum(_layout_cores(one))
        assert max(len(s.hard_core_mapping.cores) for s in V.neural(copies)) <= 8

    def test_a_group_limit_is_exact_and_established_too(self, block, built):
        """At 8 cores under a limit of 2 cores per copy of a group, the groups close earlier and copies fill the rest."""
        x = integer_inputs(block, 8)
        limited = V.build(block.ir, V.cores(8), True, wavefront_group_cores=2, **COPIES)
        want = V.value(block.identity().hybrid_mapping, x)
        assert_exact(V.value(limited, x), want)
        codes, steps = run_golden(limited, x)
        assert_exact(torch.as_tensor(codes * steps), want)
        assert len(V.heads(limited)) > len(V.heads(built[8]["copies"]))
        report = P.run(limited, block.ir, UV.value_decl(V.cores(8)))
        assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())

    def test_the_lane_free_ir_is_exact_too(self):
        program = W02.integer_resnet()
        x = integer_inputs(program, 8)
        hybrid = V.build(program.ir, [{"max_axons": 160, "max_neurons": 32, "count": 64, "has_bias": True}], True,
                         **COPIES)
        assert_exact(V.value(hybrid, x), V.value(program.identity().hybrid_mapping, x))


class TestTheLoadLayout:
    @pytest.mark.parametrize("count", [8, 3])
    def test_each_group_s_first_pass_loads_every_copy_and_every_later_pass_runs_resident(self, built, count):
        remaining, seen = 0, 0
        for stage in V.neural(built[count]["copies"]):
            head = wavefront_head(stage)
            if head is not None:
                assert not stage.schedule_weights_resident
                assert head.layout_cores == len(stage.hard_core_mapping.cores)
                remaining, seen = head.passes - 1, seen + head.passes
                continue
            if remaining:
                assert stage.schedule_weights_resident and getattr(stage, SCHEDULE_CORE_ORDINALS, None) is not None
                remaining -= 1
        assert seen > 1 and not remaining

    @pytest.mark.parametrize("count", [8, 3])
    def test_every_resident_core_runs_its_heads_copy_core_by_core(self, built, count):
        checked = 0
        for stage, core, loaded in V.resident_pairs(built[count]["copies"]):
            assert np.array_equal(np.asarray(core.get_core_matrix()), np.asarray(loaded.get_core_matrix()))
            bias, ref = getattr(core, "hardware_bias", None), getattr(loaded, "hardware_bias", None)
            assert (bias is None) == (ref is None) and (bias is None or np.array_equal(bias, ref))
            assert float(core.threshold) == float(loaded.threshold)
            assert (core.axons_per_core, core.neurons_per_core) == (loaded.axons_per_core, loaded.neurons_per_core)
            checked += 1
        assert checked > 20

    def test_a_dark_core_holds_no_instance_and_a_later_pass_no_dark_core(self, built):
        for stage in V.neural(built[8]["copies"]):
            placements = stage.hard_core_mapping.soft_core_placements_per_hard_core
            dark = getattr(stage, SCHEDULE_DARK_CORES, frozenset())
            assert all(bool(ps) != (i in dark) for i, ps in enumerate(placements))
            if stage.schedule_weights_resident:
                assert not dark

    def test_every_wavefront_pass_runs_at_its_own_timing(self, built):
        stages = [s for s in V.neural(built[8]["copies"])
                  if wavefront_head(s) is not None or getattr(s, SCHEDULE_CORE_ORDINALS, None) is not None]
        assert stages and all(int(core.latency) == 0 for s in stages for core in s.hard_core_mapping.cores)


class TestTheStaticCheck:
    @pytest.mark.parametrize("count", [8, 3, 1])
    def test_a_value_program_of_copies_is_established(self, block, built, count):
        report = P.run(built[count]["copies"], block.ir, UV.value_decl(V.cores(count)))
        assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
        h10 = report.results["H10"]
        assert h10.status == ESTABLISHED and h10.checked > 0 and "wavefront group" in h10.note, h10.note

    @pytest.mark.parametrize("semantics", ["streamed_lif", "synchronized_ttfs", "cascaded_ttfs"])
    def test_a_lif_chain_of_copies_is_established(self, semantics):
        """The LIF chain at integral weights (so H8's integrality binds), as the wavefront's own check reads it."""
        rep = W02.lif_conv_chain(seed=0, widths=(6, 6))
        for perceptron in rep.get_perceptrons():
            with torch.no_grad():
                perceptron.layer.weight.copy_(torch.round(perceptron.layer.weight * 5))
                if perceptron.layer.bias is not None:
                    perceptron.layer.bias.fill_(1.0)
                perceptron.base_activation.activation_scale.fill_(6.0)
        ir = W02.lif_ir(rep, 64, 16)
        program = V.build(ir, LIF_COPY_CORES, True, allow_coalescing=False, **COPIES)
        assert any(getattr(s, SCHEDULE_CORE_ORDINALS, None) for s in V.neural(program))
        decl = CheckDeclaration(semantics=semantics, window=W02.T,
                                pass_transfer=VERBATIM if semantics == "streamed_lif" else COLLAPSE,
                                core_types=tuple(dict(c) for c in LIF_COPY_CORES))
        report = P.run(program, ir, decl)
        assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
        assert report.results["H10"].status == ESTABLISHED and report.results["H10"].checked > 0


class TestIdleCopies:
    """At one core every copy of a group shares it: a pass that runs one copy's instance runs the core wholly."""

    def test_an_idle_copy_s_columns_are_declared_and_hold_no_instance(self, built):
        declared = 0
        for stage in V.neural(built[1]["copies"]):
            dark = getattr(stage, SCHEDULE_DARK_CORES, frozenset())
            hcm = stage.hard_core_mapping
            for h, (core, records) in enumerate(zip(hcm.cores, hcm.soft_core_placements_per_hard_core)):
                placed = {int(r["neuron_offset"]) + k for r in records for k in range(int(r["neurons"]))}
                used = set(range(int(core.neurons_per_core) - int(core.available_neurons)))
                idle = idle_columns(stage).get(h, frozenset())
                assert not idle & placed
                if h not in dark:
                    assert used - placed == idle
                declared += len(idle)
        assert declared > 0

    def test_an_undeclared_idle_copy_is_an_unplaced_column(self, block, built):
        import copy

        bad = copy.deepcopy(built[1]["copies"])
        stage = next(s for s in V.neural(bad) if getattr(s, SCHEDULE_IDLE_COLUMNS, None))
        delattr(stage, SCHEDULE_IDLE_COLUMNS)
        tiles = P.run(bad, block.ir, UV.value_decl(V.cores(1))).results["tiles"]
        assert tiles.status == FAILED and {v.rule for v in tiles.violations} == {"unplaced_column"}

    def test_no_idle_column_without_copies(self, built):
        for count in (8, 3, 1):
            assert not any(getattr(s, SCHEDULE_IDLE_COLUMNS, None) for s in V.neural(built[count]["one"]))


def _lif_chain(widths):
    ir = W02.lif_ir(W02.lif_conv_chain(widths=widths), 64, 16)
    return ir, V.build(ir, LIF_COPY_CORES, True, allow_coalescing=False, **COPIES)


@pytest.fixture(scope="module", params=[(6, 6), (6, 20)], ids=["chain", "split_neurons"])
def chain(request):
    return _lif_chain(request.param)


class TestLifSemantics:
    def test_copies_equal_identity_outputs_and_counts(self, chain):
        ir, copies = chain
        identity = W02.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
        one = V.build(ir, LIF_COPY_CORES, True, allow_coalescing=False)
        assert len(V.neural(copies)) < len(V.neural(one))
        for batch in range(2):
            x = torch.rand(4, *W02.LIF_SHAPE, generator=torch.Generator().manual_seed(batch))
            with torch.no_grad():
                want = identity(x)
                assert torch.equal(W02.lif_flow(copies)(x), want)
            assert 0 < float(want.mean()) < W02.T and float(want.std()) > 0
            cert, report, _ = certify_twin_flow_counts(ir, identity, W02.lif_flow(copies), x, backend="hcm",
                                                       discipline="streaming", reference_discipline="streaming")
            assert cert.passed and cert.neuron_windows_compared > 0, (cert.summary(), report)

    @pytest.mark.parametrize("synchronized", [False, True], ids=["rate", "synchronized_counts"])
    def test_lif_without_the_cycle_accurate_forward(self, chain, synchronized):
        from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

        ir, copies = chain

        def flow(hybrid):
            return SpikingHybridCoreFlow(W02.LIF_SHAPE, hybrid, simulation_length=W02.T, spiking_mode="lif",
                                         thresholding_mode="<=", lif_execution_synchronized=synchronized)

        x = torch.rand(4, *W02.LIF_SHAPE, generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            want = flow(build_identity_hybrid_mapping(ir_graph=ir))(x)
            assert torch.equal(flow(copies)(x), want) and float(want.std()) > 0

    @pytest.mark.parametrize("variant", ["analytical", "quantized", "synchronized", "cascaded"])
    def test_every_ttfs_variant(self, chain, variant):
        from unit.mapping.position_multiplexing.test_programs import _ttfs_flow

        ir, copies = chain
        x = torch.rand(5, *W02.LIF_SHAPE, generator=torch.Generator().manual_seed(11))
        with torch.no_grad():
            want = _ttfs_flow(build_identity_hybrid_mapping(ir_graph=ir), variant)(x)
            assert torch.equal(_ttfs_flow(copies, variant)(x), want) and float(want.std()) > 0
