"""CL-SG exactness: a per-axon deployment (mixed-sign rows split into W+ then W- pairs) gives every neuron the
identity program's output at atol 0, under the value, LIF and TTFS semantics, single- and multi-pass. Under TTFS at a
window S that is not a power of two, an offload-placed raw input included, this rests on the executor's per-neuron
membrane lattice (``chip_simulation/ttfs/membrane_lattice.py``; ``test_ttfs_lattice_exactness.py``); analytical
TTFS, outside the integer scope by name, is equal up to float rounding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from conftest import make_tiny_ir_graph

import mimarsinan.certification.randomized.cost_draws as cost_draws
import mimarsinan.certification.randomized.executors as ex
from mimarsinan.certification.randomized import runner
from mimarsinan.certification.randomized.arms import ARMS, ON_SIGN_SPLIT, ON_SIGN_SPLIT_SCHEDULED
from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
from mimarsinan.mapping.export.sign_rows import SignRowsError, assert_sign_rows
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.platform_constraints import PER_AXON
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from mimarsinan.mapping.latency.depth_balancing import UnbalancedFanInError

from unit.certification.x3a_window import map_through_the_window

def _per_axon(max_axons, max_neurons, count, *, has_bias=False):
    return {"max_axons": max_axons, "max_neurons": max_neurons, "count": count, "has_bias": has_bias,
            "weight_sign_granularity": PER_AXON}


def _hard_cores(mapping):
    for stage in mapping.stages:
        if stage.kind == "neural":
            yield from stage.hard_core_mapping.cores


def _split_rows(mapping):
    """Physical rows beyond the logical ones: the second row of every split pair the deployment placed."""
    extra = 0
    for core in _hard_cores(mapping):
        for p in core.matrix_placements:
            rows = getattr(p, "keep_rows", None)
            if getattr(p, "row_signs", None) is not None and rows is not None:
                extra += int(np.sum(np.diff(rows) == 0))
    return extra


def _x3a_programs(arm_id, seed):
    """An X3a instance of a sign-split stratum: the deployment IR at its per-axon cores, its identity mapping and
    its per-axon deployment (one IR, so node ids agree), the run of its semantics and its input."""
    arm = ARMS[arm_id]
    instance = runner.instance_for(seed, arm)
    constraints = cost_draws.platform_constraints(runner.targeted_declaration(seed, arm),
                                                  hardware_bias=instance.hardware_bias)
    ir, deployed = runner.deployment(instance, constraints, seed, arm)
    _ir, identity = runner.deployment(instance, constraints, seed, arm, identity=True)
    return instance, identity, deployed


def _node_counts(mapping, instance, *, firing, synchronized):
    """Per node, the window counts of a LIF execution (the pass transfer the toolchain's rule gives it)."""
    run = ex.RUNS["synchronized_lif" if synchronized else "streamed_lif"]
    flow = SpikingHybridCoreFlow((instance.network.input_width,), mapping, simulation_length=instance.timesteps,
                                 firing_mode=firing, spike_mode="Uniform", thresholding_mode="<=",
                                 spiking_mode="lif", cycle_accurate_lif_forward=True,
                                 lif_execution_synchronized=synchronized, pass_transfer=run.transfer)
    captured = {}

    def record(stage, counts):
        for sl in stage.output_map:
            captured[int(sl.node_id)] = counts[:, sl.offset:sl.offset + sl.size].detach().clone()

    flow.stage_count_recorder = record
    with torch.no_grad():
        captured[-1] = flow(ex.input_tensor(instance)).detach().clone()
    return captured


#: LIF instances of the sign-split strata (streamed and synchronized semantics; 13 and 43 run multi-pass).
LIF_CASES = [(ON_SIGN_SPLIT, 3), (ON_SIGN_SPLIT, 26), (ON_SIGN_SPLIT_SCHEDULED, 10),
             (ON_SIGN_SPLIT_SCHEDULED, 13), (ON_SIGN_SPLIT_SCHEDULED, 43)]


class TestLifFlows:
    """The per-cycle law under both firing modes (subtractive Default, Novena) and both LIF executions."""

    @pytest.mark.parametrize("firing", ["Default", "Novena"])
    @pytest.mark.parametrize("synchronized", [False, True])
    @pytest.mark.parametrize("arm,seed", LIF_CASES)
    def test_every_node_of_a_per_axon_deployment_equals_the_identity_program(self, arm, seed, firing,
                                                                              synchronized, monkeypatch):
        """[U-TGAUGE5] The streamed run reads the window: its draw is mapped through the mapping's window pass (the
        relays the LIF recipe arms); the relay-free streamed draw is refused by name (``TestTheWindowRule``)."""
        if not synchronized:
            map_through_the_window(monkeypatch)
        instance, identity, deployed = _x3a_programs(arm, seed)
        assert instance.semantics in ("streamed_lif", "synchronized_lif")
        assert _split_rows(deployed) > 0
        if synchronized and firing == "Novena":
            # [U-LIF2, p-11(b)] the count executor's closed form is the integrate-then-emit count of a subtractive
            # reset (a zero-reset emission window emits one spike at most), so it refuses a zero reset by name.
            from mimarsinan.models.spiking.serial.refusals import ZeroResetWindowCountError

            for mapping in (identity, deployed):
                with pytest.raises(ZeroResetWindowCountError, match="firing_mode='Novena'"):
                    _node_counts(mapping, instance, firing=firing, synchronized=synchronized)
            return
        want = _node_counts(identity, instance, firing=firing, synchronized=synchronized)
        got = _node_counts(deployed, instance, firing=firing, synchronized=synchronized)
        assert want.keys() == got.keys()
        for node in want:
            assert torch.equal(want[node], got[node]), node
        assert any(bool(t.abs().sum() > 0) for t in want.values())

    @pytest.mark.parametrize("seed", [13, 43])
    def test_the_multi_pass_cases_run_several_passes(self, seed):
        _instance, _identity, deployed = _x3a_programs(ON_SIGN_SPLIT_SCHEDULED, seed)
        segments = [s.schedule_segment_index for s in deployed.stages if s.kind == "neural"]
        assert len(segments) > len(set(segments))


def _integer_tiny_graph():
    """The tiny two-core graph on the 4-bit symmetric grid (|w| <= 7), threshold 4."""
    np.random.seed(11)
    ir = make_tiny_ir_graph()
    for node in ir.nodes:
        if isinstance(node, NeuralCore):
            node.core_matrix = np.clip(np.round(node.core_matrix * 40.0), -7, 7).astype(np.float32)
            node.threshold = 4.0
    return ir


class TestTtfsContracts:
    @pytest.mark.parametrize("spiking_mode,schedule", [("ttfs_quantized", None),
                                                       ("ttfs_cycle_based", "synchronized"),
                                                       ("ttfs_cycle_based", "cascaded")])
    def test_tiny_graph_segments_are_identical(self, spiking_mode, schedule):
        ir = _integer_tiny_graph()
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[_per_axon(32, 32, 4)])
        assert _split_rows(deployed) > 0
        for x in np.random.default_rng(2).random((3, 1, 8)):
            runs = [run_ttfs_hybrid_contract(m, x, simulation_length=4, spiking_mode=spiking_mode,
                                             ttfs_cycle_schedule=schedule) for m in (identity, deployed)]
            want, got = runs[0].record.segments, runs[1].record.segments
            assert want.keys() == got.keys()
            for stage in want:
                np.testing.assert_array_equal(want[stage].seg_output, got[stage].seg_output)


def _integer_mixer_graph():
    """The mini mixer of the identity-mapping tests (a bank-backed convolution, permutes, a host mean, a classifier)
    on the 4-bit grid."""
    path = Path(__file__).resolve().parents[1] / "test_identity_hybrid_mapping.py"
    spec = importlib.util.spec_from_file_location("_identity_hybrid_mapping_vehicles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ir, input_shape = module._make_mini_mixer_ir_graph()
    for bank in ir.weight_banks.values():
        bank.core_matrix = np.clip(np.round(np.asarray(bank.core_matrix) * 20), -7, 7)
    for node in ir.get_neural_cores():
        if node.core_matrix is not None:
            node.core_matrix = np.clip(np.round(node.core_matrix * 20), -7, 7)
        node.threshold = 3.0
    return ir, input_shape


class TestBankBackedPrograms:
    """A bank-backed soft core's split drops its bank bookkeeping, so the executors' bank fast paths (which slice the
    logical bank by the placement's row count) never read it; the deployment equals the identity program."""

    @pytest.mark.parametrize("spiking_mode,schedule", [("ttfs_quantized", None),
                                                       ("ttfs_cycle_based", "synchronized")])
    def test_the_mixer_segments_are_identical(self, spiking_mode, schedule):
        ir, input_shape = _integer_mixer_graph()
        assert ir.weight_banks
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[_per_axon(64, 64, 20)])
        assert _split_rows(deployed) > 0
        placements = [p for s in deployed.stages if s.kind == "neural"
                      for row in s.hard_core_mapping.soft_core_placements_per_hard_core for p in row]
        assert all("weight_bank_id" not in p for p in placements)
        x = torch.rand(1, *input_shape, generator=torch.Generator().manual_seed(3)).reshape(1, -1).double().numpy()
        runs = [run_ttfs_hybrid_contract(m, x, simulation_length=4, spiking_mode=spiking_mode,
                                         ttfs_cycle_schedule=schedule) for m in (identity, deployed)]
        want, got = runs[0].record.segments, runs[1].record.segments
        assert want.keys() == got.keys()
        for stage in want:
            np.testing.assert_array_equal(want[stage].seg_output, got[stage].seg_output)
        assert any(np.abs(want[stage].seg_output).sum() > 0 for stage in want)


class TestSignRowGates:
    def test_every_row_of_a_per_axon_deployment_carries_one_sign(self):
        deployed = build_hybrid_hard_core_mapping(ir_graph=_integer_tiny_graph(), cores_config=[_per_axon(32, 32, 4)])
        assert_sign_rows(deployed, weight_bits=4)

    def test_the_q_min_code_is_refused_by_name_never_saturated(self):
        """At 4 bits the per-synapse grid is [-8, 7]; a per-axon row stores a 3-bit magnitude, so -8 is refused."""
        ir = _integer_tiny_graph()
        ir.nodes[0].core_matrix[0, 0] = -8.0
        deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[_per_axon(32, 32, 4)])
        assert_sign_rows(deployed, weight_bits=8)
        with pytest.raises(SignRowsError, match="magnitude 8"):
            assert_sign_rows(deployed, weight_bits=4)
        constraints = {"cores": [_per_axon(32, 32, 4)], "weight_bits": 4}
        with pytest.raises(SignRowsError):
            build_hybrid_mapping_for_pipeline(ir, constraints, pipeline_config={})
        per_synapse = {"cores": [{"max_axons": 32, "max_neurons": 32, "count": 4, "has_bias": False}],
                       "weight_bits": 4}
        build_hybrid_mapping_for_pipeline(ir, per_synapse, pipeline_config={})

    @pytest.mark.parametrize("order", ["per_axon_first", "per_synapse_first"])
    def test_a_fused_core_with_a_per_axon_component_is_gated_in_either_declaration_order(self, order):
        """[X3] Exactness review reproducer H: a fan-in-200 core holding the 4-bit q_min code fuses two 256-row
        cores of one geometry, one per-synapse and one per-axon. The fused core is per-axon when any component is
        (the fold of ``pool_sign_granularity``), so the q_min gate refuses by name whichever type is declared first."""
        from mimarsinan.mapping.ir import IRGraph, IRSource
        from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy

        rng = np.random.default_rng(5)
        w = rng.integers(-7, 8, size=(201, 16)).astype(np.float64)
        w[::3, :] = -8.0
        w[1::3, 0] = 5.0
        src = np.array([IRSource(node_id=-2, index=j) for j in range(200)] + [IRSource(node_id=-3, index=0)],
                       dtype=object)
        core = NeuralCore(id=0, name="wide", input_sources=src, core_matrix=w, threshold=16.0, latency=0)
        core.coalescing_group_id, core.coalescing_role = 0, "master"
        ir = IRGraph(nodes=[core], output_sources=np.array([IRSource(node_id=0, index=j) for j in range(16)],
                                                            dtype=object))
        per_synapse = {"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": False}
        pool = [_per_axon(256, 256, 1), per_synapse]
        if order == "per_synapse_first":
            pool.reverse()
        constraints = {"cores": pool, "allow_coalescing": True, "allow_neuron_splitting": False, "weight_bits": 4}
        strategy = MappingStrategy.resolve(ChipCapabilities.from_platform_constraints(constraints))
        mapping = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=pool, strategy=strategy)
        fused = [c for c in _hard_cores(mapping) if getattr(c, "fused_component_axons", None)]
        assert fused and all(getattr(c, "weight_sign_granularity", None) == PER_AXON for c in fused)
        with pytest.raises(SignRowsError, match="magnitude 8"):
            assert_sign_rows(mapping, weight_bits=4)

    def test_a_row_holding_both_signs_on_a_per_axon_core_is_refused_by_name(self):
        deployed = build_hybrid_hard_core_mapping(ir_graph=_integer_tiny_graph(), cores_config=[_per_axon(32, 32, 4)])
        core = next(_hard_cores(deployed))
        core.core_matrix = np.array(core.get_core_matrix(), copy=True)
        core.core_matrix[0, :2] = (3.0, -2.0)
        with pytest.raises(SignRowsError, match="both signs"):
            assert_sign_rows(deployed, weight_bits=8)


#: X3a strata of the per-axon split (``on_sign_split`` and its scheduled twin): every core type declares
#: per_axon, the rest of the draw is the declared one. Blocks of seeds, each compared per neuron at atol 0; a
#: counterexample is admitted only where the paired control (the same IR and cores, the split switched off) is
#: one too with the same mismatches, so the split adds no wrong window (the H0 counterexamples ``retiming_dropped``
#: of the step 3 screen ran equal after the CL1 fix round's leveling).
BLOCKS = [(arm, start) for arm in (ON_SIGN_SPLIT, ON_SIGN_SPLIT_SCHEDULED) for start in range(0, 64, 8)]
RUNS = (runner.EQUAL, runner.VACUOUS, runner.COUNTEREXAMPLE, runner.REFUSED, runner.REFUSED_UNNAMED,
        runner.REFERENCE_ERROR)


def split_off(seed, arm):
    """The paired control: the same instance, IR and cores with the split and its row gate switched off."""
    with mock.patch("mimarsinan.mapping.packing.hybrid_segment.apply_sign_split", lambda cores, granularity: None), \
            mock.patch("mimarsinan.pipelining.core.simulation_factory.assert_sign_rows",
                       lambda mapping, weight_bits=None: None):
        return runner.run_instance(seed, arm)


@pytest.mark.parametrize("arm,start", BLOCKS)
def test_the_sign_split_strata_add_no_wrong_window(arm, start):
    verdicts = [runner.run_instance(seed, arm) for seed in range(start, start + 8)]
    for v in verdicts:
        assert v.outcome in RUNS, (v.seed, v.outcome, v.detail)
        assert {core.get("weight_sign_granularity") for core in v.declaration["cores"]} == {PER_AXON}
        if v.outcome == runner.COUNTEREXAMPLE:
            control = split_off(v.seed, arm)
            assert control.outcome == runner.COUNTEREXAMPLE and not control.sign_split, (v.seed, control.outcome)
            # [MN5] The same wrong windows both ways: the split adds none to a base counterexample.
            assert (control.mismatches, control.windows) == (v.mismatches, v.windows), v.seed
    assert all(v.sign_split for v in verdicts if v.outcome == runner.EQUAL)


#: One seed per in-scope semantics of the scheduled sign-split stratum, and its multi-pass seeds.
COVERAGE = {"streamed_lif": 10, "cascaded_ttfs": 12, "synchronized_lif": 13, "windowed_lif": 16, "value": 20,
            "synchronized_ttfs": 41}
MULTI_PASS = (13, 20, 36, 38, 43)


def test_the_sign_split_strata_reach_every_semantics_and_multi_pass_programs():
    assert ARMS[ON_SIGN_SPLIT].semantics == ARMS["on"].semantics
    assert set(COVERAGE) == set(ARMS[ON_SIGN_SPLIT].semantics)
    for semantics, seed in COVERAGE.items():
        v = runner.run_instance(seed, ON_SIGN_SPLIT_SCHEDULED)
        assert (v.semantics, v.outcome, v.sign_split) == (semantics, runner.EQUAL, True), seed
    for seed in MULTI_PASS:
        v = runner.run_instance(seed, ON_SIGN_SPLIT_SCHEDULED)
        assert (v.outcome, v.multipass, v.sign_split) == (runner.EQUAL, True, True), seed


#: [MN2] CL-SG design section 6 (nevresim parity on a split mapping): cascaded TTFS seeds of the sign-split strata
#: whose identity program equals its HCM run on nevresim (step 3's ``clsg/nevresim`` rows), single- and multi-pass.
NEVRESIM_CASES = [(ON_SIGN_SPLIT, 80), (ON_SIGN_SPLIT_SCHEDULED, 90)]


@pytest.mark.slow
@pytest.mark.parametrize("arm_id,seed", NEVRESIM_CASES)
def test_a_split_mapping_runs_on_nevresim_as_the_identity_program(arm_id, seed, tmp_path):
    """The per-axon deployment (rows split) against the identity program, both compiled and run on nevresim, at
    atol 0 on every neuron, and the identity program on nevresim against its HCM run."""
    arm = ARMS[arm_id]
    verdict = runner.run_instance(seed, arm_id)
    assert verdict.outcome == runner.EQUAL and verdict.sign_split, verdict.to_dict()
    instance = runner.instance_for(seed, arm)
    assert instance.semantics == "cascaded_ttfs"
    constraints = cost_draws.platform_constraints(verdict.declaration, hardware_bias=instance.hardware_bias)
    ref_ir, ref_map, ref_nodes = runner.reference_nodes(instance, arm)
    ir, mapping = runner.deployment(instance, constraints, seed, arm)
    assert _split_rows(mapping) > 0
    nev_ref, _ = ex.run_nevresim_nodes(ref_map, instance, str(tmp_path / "identity"))
    nev_dep, _ = ex.run_nevresim_nodes(mapping, instance, str(tmp_path / "deployed"))
    windows, mismatches, first = runner.compare(ref_ir, nev_ref, ir, nev_dep)
    assert windows > 0 and mismatches == 0, first
    assert runner.compare(ref_ir, ref_nodes, ref_ir, nev_ref)[1] == 0


class TestTheWindowRule:
    """[U-TGAUGE5] This file's refusal test: the relay-free streamed draw of seed 3 holds a live edge at a latency gap
    other than one, so the streamed LIF executor refuses its identity program by name."""

    def test_the_relay_free_streamed_draw_is_refused_by_name(self):
        instance, identity, _deployed = _x3a_programs(ON_SIGN_SPLIT, 3)
        with pytest.raises(UnbalancedFanInError, match="no mapping armed"):
            _node_counts(identity, instance, firing="Default", synchronized=False)
