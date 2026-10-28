"""[U-F4] Pipelined position multiplexing is cost-only: the pipelined program equals the per-pass program (W02) and
the identity program at atol 0, on the value executor and its int64 golden (a lane-fused residual network at three
core budgets) and on the cycle-accurate streamed-LIF executor (outputs and per-node spike counts), with fewer passes,
each group's copies loaded once by its first pass and run on, core by core, by every later pass."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.wavefront_stages import (
    SCHEDULE_CORE_ORDINALS, SCHEDULE_DARK_CORES, wavefront_head)
from mimarsinan.mapping.weight_programming import weight_programming_report

from unit.chip_simulation.value_run.integer_programs import assert_exact, integer_inputs
from unit.mapping.position_multiplexing import vehicles as W02

from . import vehicles as V


@pytest.fixture(scope="module")
def block():
    return V.lane_program("block")


@pytest.fixture(scope="module")
def built(block):
    return {count: {p: V.build(block.ir, V.cores(count), p) for p in (None, True)} for count in (8, 3, 1)}


class TestValueSemantics:
    @pytest.mark.parametrize("count", [8, 3, 1])
    def test_pipelined_equals_per_pass_identity_and_golden_at_atol_0(self, block, built, count):
        x = integer_inputs(block, 8)
        per_pass, pipelined = built[count][None], built[count][True]
        want = V.value(block.identity().hybrid_mapping, x)
        assert_exact(V.value(per_pass, x), want)
        got = V.value(pipelined, x)
        assert_exact(got, want)
        codes, steps = run_golden(pipelined, x)
        assert_exact(torch.as_tensor(codes * steps), want)
        assert len(V.neural(pipelined)) <= len(V.neural(per_pass))

    def test_levels_share_passes_where_the_copies_fit_together(self, block, built):
        pipelined, per_pass = built[8][True], built[8][None]
        assert len(V.neural(pipelined)) < len(V.neural(per_pass))
        assert max(len(levels) for levels in V.levels_per_pass(pipelined, block.ir)) >= 3
        assert max(len(levels) for levels in V.levels_per_pass(per_pass, block.ir)) == 1

    def test_a_budget_that_holds_no_two_levels_together_runs_one_level_per_pass(self, block, built):
        assert max(len(levels) for levels in V.levels_per_pass(built[1][True], block.ir)) == 1

    def test_the_lane_free_ir_is_exact_too(self):
        program = W02.integer_resnet()
        x = integer_inputs(program, 8)
        hybrid = V.build(program.ir, [{"max_axons": 160, "max_neurons": 32, "count": 64, "has_bias": True}], True)
        assert_exact(V.value(hybrid, x), V.value(program.identity().hybrid_mapping, x))


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("count", [96, 24], ids=["96_cores", "24_cores"])
def test_the_quarter_width_resnet18_is_exact_with_fo_star_s_pass_shape(count):
    """The quarter-width lane-fused ResNet-18 at 32 x 32: at 96 cores every level shares one wavefront (FO*'s pass
    shape: 1,266 passes for W02's 6,465), at 24 cores several groups; both equal W02's per-pass program and the
    identity program at atol 0."""
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from unit.mapping.lanes.test_lane_shape_walk import _fixture

    platform, program = _fixture("resnet18_quarter", ("L", "E"), multiplexing=True, axons=256, neurons=64,
                                 count=count, passes=1 << 16, coalescing=True, hardware_bias=True)
    x = integer_inputs(program, 4)
    want = V.value(program.identity().hybrid_mapping, x)
    per_pass = build_hybrid_mapping_for_pipeline(program.ir, platform)
    pipelined = build_hybrid_mapping_for_pipeline(program.ir, {**platform, "pipelined_multiplexing": True})
    assert_exact(V.value(per_pass, x), want)
    assert_exact(V.value(pipelined, x), want)
    assert len(V.neural(pipelined)) < len(V.neural(per_pass))
    if count == 96:
        assert (len(V.neural(per_pass)), len(V.neural(pipelined)), len(V.heads(pipelined))) == (6465, 1266, 1)
    else:
        assert len(V.heads(pipelined)) > 1


class TestTheLoadLayout:
    def test_each_group_s_first_pass_loads_and_every_later_pass_runs_resident(self, built):
        hybrid = built[8][True]
        remaining, seen = 0, 0
        for stage in V.neural(hybrid):
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

    def test_every_resident_core_runs_its_heads_copy_core_by_core(self, built):
        checked = 0
        for stage, core, loaded in V.resident_pairs(built[8][True]):
            assert np.array_equal(np.asarray(core.get_core_matrix()), np.asarray(loaded.get_core_matrix()))
            bias, ref = getattr(core, "hardware_bias", None), getattr(loaded, "hardware_bias", None)
            assert (bias is None) == (ref is None) and (bias is None or np.array_equal(bias, ref))
            assert float(core.threshold) == float(loaded.threshold)
            assert (core.axons_per_core, core.neurons_per_core) == (loaded.axons_per_core, loaded.neurons_per_core)
            checked += 1
        assert checked > 20

    def test_a_later_pass_holds_only_the_cores_its_instances_use(self, built):
        for stage in V.neural(built[8][True]):
            placements = stage.hard_core_mapping.soft_core_placements_per_hard_core
            dark = getattr(stage, SCHEDULE_DARK_CORES, frozenset())
            assert all(bool(ps) != (i in dark) for i, ps in enumerate(placements))
            if stage.schedule_weights_resident:
                assert not dark

    def test_every_bank_is_programmed_once(self, built):
        report = weight_programming_report(built[8][True])
        assert report.params_programmed == report.params_unique

    def test_every_wavefront_pass_runs_at_its_own_timing(self, built):
        """Every core of a wavefront pass reads carried inputs only, so every core starts at latency 0."""
        stages = [s for s in V.neural(built[8][True])
                  if wavefront_head(s) is not None or getattr(s, SCHEDULE_CORE_ORDINALS, None) is not None]
        assert stages and all(int(core.latency) == 0 for s in stages for core in s.hard_core_mapping.cores)


class TestLifSemantics:
    @pytest.mark.parametrize("widths", [(6, 6), (6, 20)], ids=["chain", "split_neurons"])
    def test_pipelined_equals_identity_outputs_and_counts(self, widths):
        ir = W02.lif_ir(W02.lif_conv_chain(widths=widths), 64, 16)
        identity = W02.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
        pipelined = V.build(ir, V.LIF_CORES, True, allow_coalescing=False)
        per_pass = V.build(ir, V.LIF_CORES, None, allow_coalescing=False)
        assert len(V.neural(pipelined)) < len(V.neural(per_pass))
        for batch in range(2):
            x = torch.rand(4, *W02.LIF_SHAPE, generator=torch.Generator().manual_seed(batch))
            with torch.no_grad():
                want = identity(x)
                assert torch.equal(W02.lif_flow(pipelined)(x), want)
            assert 0 < float(want.mean()) < W02.T and float(want.std()) > 0
            cert, report, _ = certify_twin_flow_counts(ir, identity, W02.lif_flow(pipelined), x, backend="hcm",
                                                       discipline="streaming", reference_discipline="streaming")
            assert cert.passed and cert.neuron_windows_compared > 0, (cert.summary(), report)


@pytest.fixture(scope="module", params=[(6, 6), (6, 20)], ids=["chain", "split_neurons"])
def chain(request):
    ir = W02.lif_ir(W02.lif_conv_chain(widths=request.param), 64, 16)
    return ir, V.build(ir, V.LIF_CORES, True, allow_coalescing=False)


class TestTheOtherExecutorsCompose:
    """Rate-coded LIF, the synchronized LIF count twin and every TTFS variant as a run executes it run each pass on its
    own cores, so the pipelined program equals the identity program at atol 0 there too."""

    @pytest.mark.parametrize("synchronized", [False, True], ids=["rate", "synchronized_counts"])
    def test_lif_without_the_cycle_accurate_forward(self, chain, synchronized):
        from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

        ir, pipelined = chain

        def flow(hybrid):
            return SpikingHybridCoreFlow(W02.LIF_SHAPE, hybrid, simulation_length=W02.T, spiking_mode="lif",
                                         thresholding_mode="<=", lif_execution_synchronized=synchronized)

        x = torch.rand(4, *W02.LIF_SHAPE, generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            want = flow(build_identity_hybrid_mapping(ir_graph=ir))(x)
            assert torch.equal(flow(pipelined)(x), want) and float(want.std()) > 0

    @pytest.mark.parametrize("variant", ["analytical", "quantized", "synchronized", "cascaded"])
    def test_every_ttfs_variant(self, chain, variant):
        from unit.mapping.position_multiplexing.test_programs import _ttfs_flow

        ir, pipelined = chain
        x = torch.rand(5, *W02.LIF_SHAPE, generator=torch.Generator().manual_seed(11))
        with torch.no_grad():
            want = _ttfs_flow(build_identity_hybrid_mapping(ir_graph=ir), variant)(x)
            assert torch.equal(_ttfs_flow(pipelined, variant)(x), want) and float(want.std()) > 0


class TestRefusalsByName:
    def test_per_hop_retiming_is_refused(self):
        ir = W02.lif_ir(W02.lif_conv_chain(), 64, 16)
        with pytest.raises(PositionMultiplexingError, match="per-hop re-timing"):
            V.build(ir, V.LIF_CORES, True, retimed_level_stages=True)
        with pytest.raises(PositionMultiplexingError, match="per-hop re-timing"):
            V.build(ir, V.LIF_CORES, True, per_hop=True)

    @pytest.mark.parametrize("backend", ["nevresim", "sanafe", "lava"])
    def test_the_backends_refuse_a_wavefront_program(self, built, backend):
        """nevresim, SANA-FE and Lava run every core they are handed: a wavefront head's dark cores would run and enter
        their event census, so each refuses the program by name (the HCM, value and TTFS executors run it)."""
        from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
        from mimarsinan.models.spiking.hybrid.carry import require_backend_replay

        pipelined, per_pass = built[8][True], built[8][None]
        behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<=",
                                        spike_generation_mode="Uniform")

        def enter(hybrid):
            if backend == "nevresim":
                require_backend_replay(hybrid, "nevresim", transfer="verbatim", spike_mode="Uniform", spiking_mode="lif")
            elif backend == "sanafe":
                from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

                SanafeRunner(mapping=hybrid, simulation_length=4, behavior=behavior)
            else:
                from mimarsinan.chip_simulation.lava_loihi.runner import LavaLoihiRunner

                LavaLoihiRunner(hybrid, 4, behavior)

        with pytest.raises(PositionMultiplexingError, match=f"the {backend} backend is not composed"):
            enter(pipelined)
        enter(per_pass)
        require_backend_replay(pipelined, "hcm", transfer="verbatim", spike_mode="Uniform", spiking_mode="lif")

    def test_a_pass_budget_below_the_wavefront_is_refused(self, block):
        with pytest.raises(PositionMultiplexingError, match="pipelined.*above max_schedule_passes 4"):
            V.build(block.ir, V.cores(8), True, max_schedule_passes=4)


class TestOffByDefault:
    @pytest.mark.parametrize("value", [None, False])
    def test_absent_or_false_builds_the_per_pass_program(self, block, value):
        a = V.build(block.ir, V.cores(3), value)
        b = W02.build(block.ir, V.cores(3), True, allow_coalescing=True, allow_neuron_splitting=True)
        assert [s.name for s in a.stages] == [s.name for s in b.stages]
        assert W02.built_banks(a, block.ir) == W02.built_banks(b, block.ir)
        assert not V.heads(a)
