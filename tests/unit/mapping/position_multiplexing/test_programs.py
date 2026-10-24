"""[W02] Position multiplexing is cost-only: the multiplexed program's outputs equal the replicated program's and the
identity program's at atol 0, on value semantics (G2's integer law, with the int64 golden) and on LIF semantics
(outputs and per-node spike counts), every multiplexed program multi-pass; the two planes plan the same passes and
the same residency, and the decision's cost is read through the existing pass, latency and programming terms."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts, flow_node_counts
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import candidate_programming_census
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.weight_programming import weight_programming_report

from unit.chip_simulation.value_run.integer_programs import assert_exact, integer_inputs
from unit.mapping.packing.test_neuron_split_before_coalescing import SHAPE as WIDE_SHAPE, _wide_cnn
from unit.mapping.residency.executed import vehicles as RV

from . import vehicles as V

VALUE_CORES = [{"max_axons": 160, "max_neurons": 32, "count": 64, "has_bias": True}]
LIF_CORES = [{"max_axons": 64, "max_neurons": 16, "count": 64, "has_bias": True}]


def _value(hybrid, x):
    with torch.no_grad():
        return ValueHybridCoreFlow(hybrid, dtype=torch.float64)(x.double())


@pytest.fixture(scope="module")
def resnet():
    return V.integer_resnet()


class TestValueSemantics:
    def test_multiplexed_equals_replicated_identity_and_golden_at_atol_0(self, resnet):
        x = integer_inputs(resnet, 16)
        replicated, multiplexed = (V.build(resnet.ir, VALUE_CORES, m) for m in (False, True))
        assert len(V.neural(multiplexed)) > len(V.neural(replicated)) >= 1
        want = _value(resnet.identity().hybrid_mapping, x)
        assert_exact(_value(replicated, x), want)
        got = _value(multiplexed, x)
        assert_exact(got, want)
        codes, steps = run_golden(multiplexed, x)
        assert_exact(torch.as_tensor(codes * steps), want)

    def test_every_bank_is_programmed_once(self, resnet):
        replicated, multiplexed = (weight_programming_report(V.build(resnet.ir, VALUE_CORES, m)) for m in (False, True))
        assert multiplexed.params_programmed == multiplexed.params_unique == replicated.params_unique
        assert replicated.params_programmed > 10 * replicated.params_unique

    def test_wide_many_neuron_banks_multiplex_with_the_neuron_split(self):
        from unit.chip_simulation.value_run.integer_programs import build_integer_program

        program = build_integer_program(_wide_cnn(), WIDE_SHAPE, num_classes=5, max_axons=16, max_neurons=8,
                                        allow_coalescing=True)
        cores = [{"max_axons": 16, "max_neurons": 8, "count": 64, "has_bias": True}]
        hybrid = V.build(program.ir, cores, True, allow_coalescing=True, allow_neuron_splitting=True)
        assert sum(s.schedule_weights_resident for s in V.neural(hybrid)) > 0
        x = integer_inputs(program, 16)
        assert_exact(_value(hybrid, x), _value(program.identity().hybrid_mapping, x))


def _lif_vehicle():
    rep = V.lif_conv_chain()
    return V.lif_ir(rep, 64, 16)


class TestLifSemantics:
    @pytest.mark.parametrize("widths", [(6, 6), (6, 20)], ids=["chain", "split_neurons"])
    def test_multiplexed_equals_replicated_and_identity_outputs_and_counts(self, widths):
        ir = V.lif_ir(V.lif_conv_chain(widths=widths), 64, 16)
        identity = V.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
        programs = {m: V.build(ir, LIF_CORES, m, allow_neuron_splitting=True) for m in (False, True)}
        multiplexed = programs[True]
        assert len(V.neural(multiplexed)) >= 2 * 25 and sum(s.schedule_weights_resident for s in V.neural(multiplexed))
        for batch in range(3):
            x = torch.rand(4, *V.LIF_SHAPE, generator=torch.Generator().manual_seed(batch))
            with torch.no_grad():
                want = identity(x)
                for hybrid in programs.values():
                    assert torch.equal(V.lif_flow(hybrid)(x), want)
            assert 0 < float(want.mean()) < V.T and float(want.std()) > 0     # active and discriminating
            for hybrid in programs.values():
                cert, report, _ = certify_twin_flow_counts(ir, identity, V.lif_flow(hybrid), x, backend="hcm",
                                                           discipline="streaming", reference_discipline="streaming")
                assert cert.passed and cert.neuron_windows_compared > 0, (cert.summary(), report)
        counts = flow_node_counts(V.lif_flow(multiplexed), x, discipline="streaming")
        assert sum(float(c.sum()) for c in counts.values()) > 0

    def test_a_token_bank_feeding_a_deep_chain_multiplexes(self):
        """The residency review's multi-instance family: a token-wise LIF layer (one bank, several tokens) whose
        token feeds a deep chain, so the bank's instances sit at one level of a multi-level segment."""
        rep = RV.token_repr(deep=(8, 8), tokens=3, deep_tokens=(0,))
        ir = RV.ir_of(rep)
        identity = RV.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
        multiplexed = V.build(ir, RV.cores(16), True)
        assert sum(s.schedule_weights_resident for s in V.neural(multiplexed)) >= 2
        x = torch.rand(6, RV.IN, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            assert torch.equal(RV.lif_flow(multiplexed)(x), identity(x))


class TestPlanes:
    @pytest.mark.parametrize("which", ["value", "lif"])
    def test_both_planes_plan_the_same_passes_and_residency(self, resnet, which):
        ir, cores = (resnet.ir, VALUE_CORES) if which == "value" else (_lif_vehicle(), LIF_CORES)
        hybrid = V.build(ir, cores, True)
        frag = V.fragments(ir, cores, True)
        assert V.built_banks(hybrid, ir) == V.planned_banks(frag, list(ir.layout_softcores))
        assert [s.schedule_weights_resident for s in V.neural(hybrid)] == [p.resident for p in frag.pass_programs]

    def test_the_programmed_bytes_are_one_copy_on_both_planes(self, resnet):
        bits = 8
        frag = V.fragments(resnet.ir, VALUE_CORES, True)
        planned = candidate_programming_census(frag, weight_bits=bits)
        hybrid = V.build(resnet.ir, VALUE_CORES, True)
        executed = sum(params_bytes(CoreOccupancy.from_hard_core(core).cells_used, bits)
                       for stage in V.neural(hybrid) if not stage.schedule_weights_resident
                       for core in stage.hard_core_mapping.cores)
        assert planned.reprogrammed_bytes == executed
        replicated = candidate_programming_census(V.fragments(resnet.ir, VALUE_CORES, False), weight_bits=bits)
        assert planned.reprogrammed_bytes < replicated.reprogrammed_bytes


class TestTimeCost:
    def test_passes_and_latency_grow_while_programming_shrinks(self, resnet):
        from mimarsinan.search.problems.joint.candidate_fragments import (
            StageSemantics, candidate_program_facts, with_stage_placements)

        specs = list(resnet.ir.layout_softcores)
        facts = {}
        for m in (False, True):
            frag = with_stage_placements(specs, V.fragments(resnet.ir, VALUE_CORES, m), StageSemantics())
            facts[m] = (len(frag.pass_placements), candidate_program_facts(specs, frag, StageSemantics(), timesteps=1,
                                                                            weight_bits=8))
        (p_rep, f_rep), (p_mux, f_mux) = facts[False], facts[True]
        assert p_mux > p_rep and f_mux.latency_steps > f_rep.latency_steps
        assert f_mux.programming.reprogrammed_bytes < f_rep.programming.reprogrammed_bytes


class TestRefusals:
    def test_a_pass_budget_below_the_positions_is_refused_by_name(self, resnet):
        from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
        from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy

        strategy = MappingStrategy.resolve(ChipCapabilities(allow_scheduling=True, max_schedule_passes=16,
                                                            position_multiplexing=True))
        with pytest.raises(PositionMultiplexingError, match="above max_schedule_passes 16"):
            build_hybrid_hard_core_mapping(ir_graph=resnet.ir, cores_config=VALUE_CORES, strategy=strategy,
                                           max_schedule_passes=16)

    def test_an_ir_without_a_layout_record_is_refused_by_name(self):
        """The dependency levels come from the layout walk; a hand-built IR has none to order passes by."""
        from unit.mapping.bank_clustered_vehicles import token_graph

        with pytest.raises(PositionMultiplexingError, match="no layout record"):
            V.build(token_graph(4), [{"max_axons": 32, "max_neurons": 32, "count": 2}], True)


#: The four TTFS variants as a run declares them; their one legal input encoding and firing are TTFS.
TTFS_VARIANTS = ("analytical", "quantized", "synchronized", "cascaded")


def _ttfs_flow(hybrid, variant: str, **overrides):
    from mimarsinan.chip_simulation.activation_semantics import resolve_activation_semantics
    from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

    config = {"spiking_family": "ttfs", "spiking_variant": variant}
    semantics = resolve_activation_semantics(config)
    kwargs = dict(spiking_mode=semantics.legacy_spiking_mode, ttfs_cycle_schedule=semantics.legacy_ttfs_cycle_schedule
                  if semantics.variant != "cascaded" else "cascaded", firing_mode="TTFS", spike_mode="TTFS",
                  thresholding_mode="<=", pass_transfer=run_pass_transfer(config))
    kwargs.update(overrides)
    return SpikingHybridCoreFlow(V.LIF_SHAPE, hybrid, simulation_length=V.T, **kwargs)


class TestTtfsSemantics:
    """[W02 fix round, F1] Every TTFS variant as a run executes it (spike_generation_mode and firing_mode TTFS, the
    only legal values of the family; the run's pass transfer, ``run_pass_transfer``): the multiplexed program and a
    replicated program cut into passes equal the identity program at atol 0. A carried count's single-spike
    re-encode is the time code's lossless codec, so the collapse transfer is exact under the TTFS encoding."""

    @pytest.mark.parametrize("variant", TTFS_VARIANTS)
    @pytest.mark.parametrize("widths", [(6, 6), (6, 20)], ids=["chain", "split_neurons"])
    def test_multiplexed_and_cut_equal_the_identity(self, variant, widths):
        ir = V.lif_ir(V.lif_conv_chain(widths=widths), 64, 16)
        identity = _ttfs_flow(build_identity_hybrid_mapping(ir_graph=ir), variant)
        cut_cores = [dict(LIF_CORES[0], count=3)]
        programs = {"multiplexed": V.build(ir, LIF_CORES, True, allow_neuron_splitting=True),
                    "cut": V.build(ir, cut_cores, False, allow_neuron_splitting=True)}
        assert all(len(V.neural(h)) > 2 for h in programs.values())
        for batch in range(2):
            x = torch.rand(5, *V.LIF_SHAPE, generator=torch.Generator().manual_seed(100 + batch))
            with torch.no_grad():
                want = identity(x)
                assert float(want.std()) > 0
                for label, hybrid in programs.items():
                    assert torch.equal(_ttfs_flow(hybrid, variant)(x), want), (variant, label)

    def test_a_rate_encoding_under_cascaded_ttfs_carrying_counts_is_refused_by_name(self):
        """Outside the family's legal encodings a carried count's re-encode is an even train, not the time code; the
        flow refuses such a program by name where it carries across a pass, and runs a program that carries none."""
        from mimarsinan.models.spiking.hybrid.carry import PassReplayRefusal

        ir = _lif_vehicle()
        with pytest.raises(PassReplayRefusal, match="spike_generation_mode"):
            _ttfs_flow(V.build(ir, LIF_CORES, True, allow_neuron_splitting=True), "cascaded", spike_mode="Uniform")
        _ttfs_flow(build_identity_hybrid_mapping(ir_graph=ir), "cascaded", spike_mode="Uniform")


def _residency_mismatches(hybrid) -> tuple:
    """Per resident pass and hard-core ordinal, the core against its run head's: matrix, bias, threshold, latency and
    readout law, bit for bit; returns (cores checked, mismatches)."""
    import numpy as np

    head, checked, bad = None, 0, []
    for stage in V.neural(hybrid):
        cores = stage.hard_core_mapping.cores
        if not stage.schedule_weights_resident:
            head = cores
            continue
        assert head is not None and len(cores) == len(head)
        for i, (core, ref) in enumerate(zip(cores, head)):
            bias, ref_bias = getattr(core, "hardware_bias", None), getattr(ref, "hardware_bias", None)
            same = (np.array_equal(np.asarray(core.get_core_matrix()), np.asarray(ref.get_core_matrix()))
                    and (bias is None) == (ref_bias is None) and (bias is None or np.array_equal(bias, ref_bias))
                    and float(core.threshold) == float(ref.threshold) and core.latency == ref.latency
                    and repr(getattr(core, "readout_law", None)) == repr(getattr(ref, "readout_law", None)))
            checked += 1
            if not same:
                bad.append((stage.schedule_pass_index, i))
    return checked, bad


class TestTheResidentPassesRunTheHeadsCopies:
    """[W02 fix round, F5] A resident pass executes on its run head's copies: the value executor aliases them, and the
    LIF executors run each pass on its own matrices, so the one-copy claim needs every resident core equal to its head's
    (the build-time region check compares regions only). Checked core by core on value and LIF vehicles."""

    def test_value_vehicles(self, resnet):
        checked, bad = _residency_mismatches(V.build(resnet.ir, VALUE_CORES, True))
        assert checked > 100 and not bad

    @pytest.mark.parametrize("widths", [(6, 6), (6, 20)], ids=["chain", "split_neurons"])
    def test_lif_vehicles(self, widths):
        ir = V.lif_ir(V.lif_conv_chain(widths=widths), 64, 16)
        checked, bad = _residency_mismatches(V.build(ir, LIF_CORES, True, allow_neuron_splitting=True))
        assert checked > 20 and not bad

    def test_a_token_bank_feeding_a_deep_chain(self):
        ir = RV.ir_of(RV.token_repr(deep=(8, 8), tokens=3, deep_tokens=(0,)))
        checked, bad = _residency_mismatches(V.build(ir, RV.cores(16), True))
        assert checked >= 2 and not bad
