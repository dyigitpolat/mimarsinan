"""[U-LIF2 option A] The widening by data and the rest of the unit: both firing forms of a synchronized LIF hop are
declared values and integrate then emit is the default (the owner's decision D-1, option A).

W10: the walk's count branch reads ``lif_window_count`` of each hop's window charge (never the IF kernel), and the
tuners read the run's resolved point (the common part's surviving mutant M38). W11: a stage of dependency depth D runs
(D + 1) T + D cycles under integrate then emit, D read from the stage's wiring on every plane; the per-cycle stage is
priced from the latency its executor loops on (the review's p-1). W12: the neuron cycles per window. W13: the event
census armed where no external backend executes the point. W14: the certificate cells at the run's value. W16: the
toolchain's own windowed documents declare firing during the window.

Red on the executor side's tree (``ulif2_o1a``): the declared set holds one value there, and none of the names below
exists. The unit's new symbols are imported inside each test, so each test reds by itself."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mimarsinan.config_schema.runtime import derive_run_config

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
KEY = "lif_window_firing"
ROOT = Path(__file__).resolve().parents[3]
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}
T = 15


def _new(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _run(**dp):
    return derive_run_config({"model_type": "simple_mlp", "spiking_family": "lif", "spiking_variant": "synchronized",
                              **dp}, dict(_CORES))


# -- the widening by data -------------------------------------------------------------------------------------------

def test_option_a_declares_both_values_and_integrate_then_emit_is_the_default():
    axis = importlib.import_module("mimarsinan.chip_simulation.window_firing_axis")
    assert axis.DECLARED_WINDOW_FIRINGS == (FWI, ITE) and axis.DEFAULT_WINDOW_FIRING == ITE
    silent, declaring = _run(), _run(lif_window_firing=FWI)
    assert silent[KEY] == ITE and declaring[KEY] == FWI
    for enable in ("enable_nevresim_simulation", "enable_sanafe_simulation", "enable_loihi_simulation"):
        assert silent[enable] is False and declaring[enable] is True, enable


@pytest.mark.parametrize("backend,executes_ite", [
    ("hcm", True), ("unified", True), ("hybrid", True), ("scm", True), ("training", True),
    ("nevresim", False), ("sanafe", False), ("lava", False), ("loihi", False), ("odin_rtl", False),
    ("odin_fpga", False), ("odin_hacc", False)])
def test_the_toolchain_executors_declare_both_values_and_every_external_backend_fires_while_integrating(
        backend, executes_ite):
    from mimarsinan.chip_simulation.spiking_semantics import backend_capabilities

    assert backend_capabilities(backend).window_firings == frozenset({FWI, ITE} if executes_ite else {FWI})


# -- W10: the walk's count branch and the tuners' reads ---------------------------------------------------------------

@pytest.mark.parametrize("mode", ["<=", "<"])
@pytest.mark.parametrize("v0", [0.0, -0.25])
def test_w10_the_count_branch_is_the_window_count_of_the_charge_with_no_if_kernel(monkeypatch, mode, v0):
    from mimarsinan.models.nn.activations.lif import LIFActivation
    from mimarsinan.models.spiking.wire_semantics import lif_window_count

    window_counts = _new("mimarsinan.models.nn.lif_kernels", "window_counts")
    lif = LIFActivation(T=T, activation_scale=0.7, thresholding_mode=mode, membrane_init=v0)
    monkeypatch.setattr(LIFActivation, "_spikes_and_scale", lambda *_a, **_k: pytest.fail("the IF kernel ran"))
    x = torch.linspace(-1.0, 1.5, 401, dtype=torch.float64)
    theta = float(lif.activation_scale)  # the activation's own (float32) threshold
    with window_counts(lif):
        got = lif(x) * T / theta
    want = lif_window_count(x / theta * T + v0, 1.0, T, compare_mode=mode)
    assert torch.equal(got.round(), want) and float((got - want).abs().max()) < 1e-9
    assert not lif._window_count_mode  # the mode is the walk's scope only


def test_w10_the_count_branch_decides_ties_on_the_chip_lattice_in_the_measurement_plane():
    from mimarsinan.models.nn.activations.lif import LIFActivation
    from mimarsinan.models.nn.lif_kernels import measurement_plane

    window_counts = _new("mimarsinan.models.nn.lif_kernels", "window_counts")
    theta = torch.tensor(0.3, dtype=torch.float32)
    lif = LIFActivation(T=7, activation_scale=theta.clone(), thresholding_mode="<=")
    lif.set_membrane_lattice(1.0)
    x = torch.arange(8, dtype=torch.float32) * theta / 7  # each charge k lands an ulp below k in float32 (k = 1, 2, 4, 5)
    assert (torch.floor(x / theta * 7) != torch.arange(8, dtype=torch.float32)).any()
    with measurement_plane(), window_counts(lif):
        got = (lif(x) * 7 / theta).round()
    assert got.tolist() == [float(k) for k in range(8)]


@pytest.mark.parametrize("mode,count", [("<=", 8.0), ("<", 7.0)])
def test_w10_the_count_branch_reads_the_comparator_at_an_integer_charge(mode, count):
    from mimarsinan.models.nn.activations.lif import LIFActivation

    window_counts = _new("mimarsinan.models.nn.lif_kernels", "window_counts")
    lif = LIFActivation(T=16, activation_scale=1.0, thresholding_mode=mode)
    with window_counts(lif):
        got = lif(torch.tensor([0.5])) * 16  # a charge of exactly 8 thresholds
    assert got.tolist() == [count]


def test_w10_the_count_branch_trains_through_the_clamped_charge():
    from mimarsinan.models.nn.activations.lif import LIFActivation

    window_counts = _new("mimarsinan.models.nn.lif_kernels", "window_counts")
    lif = LIFActivation(T=15, activation_scale=2.0)
    x = torch.tensor([-1.0, 0.5, 1.0, 1.9, 3.0], requires_grad=True)
    with window_counts(lif):
        lif(x).sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]  # the straight-through clamp of the charge to the window


def test_w10_the_walk_runs_each_hop_by_its_window_count_and_agrees_with_the_kernel_off_ties(monkeypatch):
    from unit.spiking.host_gather_forms import lif_net

    from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
    from mimarsinan.models.nn.activations.lif import LIFActivation
    from mimarsinan.spiking import segment_policies
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    _new("mimarsinan.models.nn.lif_kernels", "window_counts")
    shape, flow, _perceptrons, _thetas = lif_net("encoding_skip", 3, 0.9)
    x = torch.rand((16,) + tuple(shape), generator=torch.Generator().manual_seed(3))
    real = LIFActivation._spikes_and_scale
    calls = []
    monkeypatch.setattr(LIFActivation, "_spikes_and_scale", lambda *a, **k: calls.append(1) or real(*a, **k))
    with torch.no_grad():
        counted = chip_aligned_segment_forward(flow, x, T, synchronized=True, soma_law=DEFAULT_SOMA_LAW)
        in_counted = len(calls)  # the host-side encoder's rate only: no synchronized hop reaches the IF kernel
        monkeypatch.setattr(segment_policies, "window_counts", lambda _lif: _Nothing())
        kernel = chip_aligned_segment_forward(flow, x, T, synchronized=True, soma_law=DEFAULT_SOMA_LAW)
    assert len(calls) - in_counted > in_counted and torch.equal(counted, kernel)


class _Nothing:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False


@pytest.mark.parametrize("declared,synchronized", [(None, True), (FWI, False), (ITE, True)])
def test_w10_the_lif_tuner_reads_the_run_point(tmp_path, declared, synchronized):
    """The common part's surviving mutant M38: the tuner's two-window flag is the run's resolved value."""
    from conftest import MockPipeline, default_config, make_tiny_supermodel

    from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg.update({"spiking_mode": "lif", "firing_mode": "Default", "thresholding_mode": "<", "simulation_steps": 4})
    if declared is not None:
        cfg[KEY] = declared
    cfg = derive_run_config(cfg, dict(_CORES))
    assert cfg[KEY] == (declared or ITE)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    tuner = LIFAdaptationTuner(pipeline, model=make_tiny_supermodel(), target_accuracy=0.5, lr=0.001,
                               adaptation_manager=AdaptationManager())
    assert tuner._synchronized is synchronized


# -- W11: the latency rule, on every plane ----------------------------------------------------------------------------

def test_w11_a_stage_of_depth_d_runs_d_plus_one_windows_and_d_delivery_cycles():
    rule = _new("mimarsinan.chip_simulation.stage_timesteps", "integrate_then_emit_cycles")
    from mimarsinan.chip_simulation.stage_timesteps import executed_stage_timesteps, program_latency_steps
    from mimarsinan.models.spiking.hybrid.executors.gated_window import gated_window_cycles

    assert [rule(d, T) for d in (0, 1, 3, 5)] == [0, 31, 63, 95]
    assert all(gated_window_cycles(d, T) == rule(d, T) for d in range(8))
    kw = dict(timesteps=T, max_latency=0, is_cycle=False, is_cascade=False)
    assert executed_stage_timesteps(**kw, integrates_then_emits=True, depth=3) == 63
    assert executed_stage_timesteps(**kw) == 16  # firing while integrating: T + L + 1
    with pytest.raises(ValueError, match="dependency depth"):
        executed_stage_timesteps(**kw, integrates_then_emits=True)
    with pytest.raises(ValueError, match="TTFS"):
        executed_stage_timesteps(**dict(kw, is_cycle=True), latency_group_count=1, integrates_then_emits=True, depth=1)
    assert program_latency_steps(stage_max_latencies=[0] * 8, timesteps=T, is_cycle=False, is_cascade=False,
                                 integrates_then_emits=True, stage_depths=[1] * 8) == 248
    with pytest.raises(ValueError, match="stage depths"):
        program_latency_steps(stage_max_latencies=[0, 0], timesteps=T, is_cycle=False, is_cascade=False,
                              integrates_then_emits=True, stage_depths=[1])


def _candidate(spec, tags):
    """Softcores of one fused stage (one pass, one segment) and their wire census from ``spec`` (name, producers)."""
    from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
    from mimarsinan.mapping.noc.wire_census import LayoutWireCensus

    index = {name: i for i, (name, _producers) in enumerate(spec)}
    softcores = [LayoutSoftCoreSpec(input_count=8, output_count=6, latency_tag=tag, segment_id=0)
                 for tag in tags]
    pairs = {(index[p], index[name]): 6 for name, producers in spec for p in producers}
    census = LayoutWireCensus(pair_wires=pairs, input_wires=tuple(0 if any(True for _ in p) else 8 for _n, p in spec),
                              on_wires=(0,) * len(spec))
    noc = SimpleNamespace(pass_placements=[[(i, i) for i in range(len(spec))]], census=census, runs=None,
                          pass_programs=[SimpleNamespace(wavefront=False)])
    return softcores, noc


_CHAIN3 = ((("a", ()), ("b", ("a",)), ("c", ("b",))), (0, 1, 2))
_SKIP = ((("a", ()), ("b", ("a",)), ("c", ("a", "b"))), (1, 1, 2))  # the tags a relay-free skip stage holds


@pytest.mark.parametrize("program,fwi,ite", [(_CHAIN3, 18, 63), (_SKIP, 18, 63)])
def test_w11_the_candidate_prices_the_stage_from_its_wiring_at_both_values(program, fwi, ite):
    from mimarsinan.search.problems.joint.candidate_fragments import candidate_latency_steps, stage_semantics_of

    depths = _new("mimarsinan.mapping.noc.stages", "execution_stage_depths")
    softcores, noc = _candidate(*program)
    assert depths(softcores, noc.pass_placements, retimed=False, census=noc.census) == (3,)
    assert depths(softcores, noc.pass_placements, retimed=False, census=None) is None
    priced = {value: candidate_latency_steps(softcores, noc, stage_semantics_of("lif", None, retimed=False,
                                                                                window_firing=value), T)
              for value in (FWI, ITE)}
    assert priced == {FWI: fwi, ITE: ite}  # the skip stage's tags would price 17 and 47
    blind = SimpleNamespace(**{**vars(noc), "census": None})
    assert candidate_latency_steps(softcores, blind, stage_semantics_of("lif", None, retimed=False,
                                                                        window_firing=ITE), T) is None


def test_w11_the_search_and_its_twin_take_the_run_window_firing():
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import firing_semantics_kwargs
    from mimarsinan.search.problems.joint.candidate_fragments import stage_semantics_of

    for value in (FWI, ITE):
        cfg = _run(lif_window_firing=value)
        kwargs = firing_semantics_kwargs(DeploymentPlan.resolve(cfg), cfg)
        assert kwargs["window_firing"] == value
        semantics = stage_semantics_of(kwargs["spiking_mode"], kwargs["ttfs_cycle_schedule"],
                                       retimed=kwargs["per_hop_retiming"], window_firing=kwargs["window_firing"])
        assert semantics.integrates_then_emits is (value == ITE)
    streamed = _run(spiking_variant="streamed")
    assert firing_semantics_kwargs(DeploymentPlan.resolve(streamed), streamed)["window_firing"] is None


def test_w11_a_record_no_simulator_timed_states_its_stage_cycles():
    steps = _new("mimarsinan.chip_simulation.stage_timesteps", "sealed_stage_steps")
    program = _new("mimarsinan.chip_simulation.stage_timesteps", "sealed_program_steps")

    assert steps(timesteps=T, window_firing=ITE, depth=3, delays=[0, 1, 2]) == 63
    assert steps(timesteps=T, window_firing=FWI, depth=3, delays=[1, 1, 2]) == 18  # p-1: the loop, not the span
    assert steps(timesteps=T, window_firing=ITE, depth=None, delays=[0]) is None  # a format-1 record
    assert steps(timesteps=T, window_firing=None, depth=1, delays=[0]) is None  # away from the synchronized variant
    assert steps(timesteps=T, window_firing=FWI, depth=1, delays=[0, None]) is None
    segments = [SimpleNamespace(dependency_depth=1, cores=[SimpleNamespace(static_delay_levels=0)])] * 8
    assert (program(segments, T, ITE), program(segments, T, FWI)) == (248, 128)  # C5's 8 stages: 248 and 128


@pytest.mark.parametrize("retimed", [False, True])
def test_w11_the_sealed_stage_cycles_are_the_executor_loop_on_a_relay_free_skip_program(retimed):
    """The review's p-1 at both values: the record's rule equals the loop each executor runs."""
    from unit.models.gated_window_programs import SKIP, identity_program, ite_flow, layered_instance, neural_stages

    from mimarsinan.deployment_record.build.program_census import segment_record
    from mimarsinan.mapping.latency.chip import ChipLatency
    from mimarsinan.models.spiking.hybrid.executors.gated_window import gated_window_plan

    steps = _new("mimarsinan.chip_simulation.stage_timesteps", "sealed_stage_steps")
    instance = layered_instance(SKIP, timesteps=T, seed=11)
    mapping = identity_program(instance, retimed=retimed)
    flow = ite_flow(instance, mapping)
    for index, stage in enumerate(neural_stages(mapping)):
        record = segment_record(index, index, 0, stage, weight_bits=8)
        delays = [c.static_delay_levels for c in record.cores]
        per_cycle = int(ChipLatency(stage.hard_core_mapping).calculate()) + T
        gated = gated_window_plan(flow._get_segment_tensors(stage, torch.device("cpu")), T).cycles
        assert steps(timesteps=T, window_firing=FWI, depth=record.dependency_depth, delays=delays) == per_cycle
        assert steps(timesteps=T, window_firing=ITE, depth=record.dependency_depth, delays=delays) == gated


def test_w11_a_record_no_simulator_timed_seals_its_program_cycles(tmp_path):
    import dataclasses

    from unit.deployment_record.record_fixtures import make_schedule
    from unit.pipelining.pipeline_steps.test_deployment_record_step import (
        DEPLOYMENT_RECORD_FILENAME, _build, _hcm_entry,
    )

    from mimarsinan.deployment_record.schema import SegmentRecord, load_deployment_record

    program = _new("mimarsinan.chip_simulation.stage_timesteps", "sealed_program_steps")
    schedule = make_schedule()  # its cores state their latencies, each pass its depth: a program this unit seals
    stages = tuple(dataclasses.replace(st, dependency_depth=2, cores=tuple(
        dataclasses.replace(c, static_delay_levels=i % 2) for i, c in enumerate(st.cores)))
        if isinstance(st, SegmentRecord) else st for st in schedule.stages)
    entry = dict(_hcm_entry(), schedule=dataclasses.replace(schedule, stages=stages).to_dict())
    pipeline, step = _build(tmp_path, sanafe=False, hcm_entry=entry)
    step.run()
    record = load_deployment_record(str(Path(pipeline.working_directory) / DEPLOYMENT_RECORD_FILENAME))
    expected = program(record.schedule.segments(), record.timing.s_global, record.identity.window_firing)
    assert record.timing.per_segment == () and expected
    assert record.timing.latency.compute_steps == expected
    from mimarsinan.deployment_record.build.program_census import record_program_passes

    rows = record_program_passes(record).rows  # the record plane reads the same sealed stage cycles
    assert sum(int(row.compute_steps) for row in rows) == expected


def test_w11_the_layout_hook_and_the_candidate_facts_carry_the_run_window_firing():
    from mimarsinan.search.problems.joint.candidate_fragments import (
        candidate_fragments, candidate_program_facts, stage_semantics_of,
    )
    from mimarsinan.search.problems.joint.layout_hook import JointLayoutMixin

    class Host(JointLayoutMixin):
        spiking_mode, ttfs_cycle_schedule, per_hop_retiming, window_firing = "lif", "cascaded", True, ITE

    assert Host().stage_semantics.window_firing == ITE
    facts = candidate_program_facts([], None, stage_semantics_of("lif", None, retimed=True, window_firing=ITE),
                                    timesteps=T, weight_bits=8)
    assert facts.window_firing == ITE
    _physics, context = candidate_fragments({"simulation_steps": T, "cores": _CORES["cores"]}, program=facts)
    assert context.neuron_cycles_per_window == 2 * T


def test_w11_the_timing_fragment_takes_the_program_cycles_only_without_simulator_timing():
    from mimarsinan.deployment_record.schema.physics import SegmentTimingRecord
    from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import timing_record

    untimed = timing_record(s_global=T, depth=8, per_segment=(), host_ops_s_raw_total=None, unsimulated_steps=248)
    timed = timing_record(s_global=T, depth=1, per_segment=(SegmentTimingRecord(0, 16, 1e-6),), host_ops_s_raw_total=None,
                          unsimulated_steps=248)
    assert (untimed.latency.compute_steps, timed.latency.compute_steps) == (248, 16)
    assert untimed.latency.compute_sim_time_s is None


# -- W12: the cost rule -----------------------------------------------------------------------------------------------

def test_w12_the_neuron_terms_multiply_the_neuron_cycles_per_window():
    from mimarsinan.deployment_record.cost.absolute.formulas import ENERGY_COMPONENTS
    from mimarsinan.deployment_record.platform_physics.vocabulary import ROWS
    from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS

    cycles = _new("mimarsinan.chip_simulation.stage_timesteps", "neuron_cycles_per_window")
    assert (cycles(T, ITE), cycles(T, FWI), cycles(T, None)) == (30, 15, 15)
    factors = {name: factors for name, factors, _label in ENERGY_COMPONENTS}
    for name in ("e_neuron_update", "e_leak_per_neuron_step"):
        assert factors[name] == ("neurons_used", "neuron_cycles_per_window"), name
        assert {row[0]: row[4] for row in ROWS}[name] == "neurons_used x neuron_cycles_per_window", name
    assert "neuron_cycles_per_window" in QUANTITY_SPECS


@pytest.mark.parametrize("firing,expected", [(ITE, 2 * T), (FWI, T), (None, T)])
def test_w12_the_record_and_the_candidate_state_the_neuron_cycles(firing, expected):
    import dataclasses

    from unit.deployment_record.record_fixtures import make_full_record

    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
    from mimarsinan.deployment_record.quantities.from_candidate import CandidateQuantityContext
    from mimarsinan.deployment_record.quantities.from_record import from_record

    record = make_full_record()
    record = dataclasses.replace(record, identity=dataclasses.replace(record.identity, window_firing=firing),
                                 timing=dataclasses.replace(record.timing, s_global=T))
    assert from_record(record).get("neuron_cycles_per_window").value == expected
    context = candidate_context_from_platform({"simulation_steps": T, "cores": _CORES["cores"]}, window_firing=firing)
    assert context.neuron_cycles_per_window == expected
    assert CandidateQuantityContext(timesteps=T).neuron_cycles_per_window is None  # absent reads S


# -- W13's arming: the event census from the executor of record ------------------------------------------------------

def test_w13_a_run_that_integrates_then_emits_takes_the_executor_census():
    from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import get_pipeline_step_specs

    census = _new("mimarsinan.chip_simulation.window_firing", "derived_census_samples")
    silent, per_cycle = _run(), _run(lif_window_firing=FWI)
    assert (census(silent), census(per_cycle)) == (1, 0)
    assert census(_run(sanafe_sample_count=3)) == 3
    assert DeploymentPlan.resolve(silent).event_census_samples == 1
    assert DeploymentPlan.resolve(_run(event_census_samples=0)).event_census_samples == 0  # a declaration stands
    assert DeploymentPlan.resolve(per_cycle).event_census_samples == 0
    names = [name for name, _cls in get_pipeline_step_specs(silent)]
    assert "Event Census" in names and BACKEND_REGISTRY.selected_step_specs(DeploymentPlan.resolve(silent)) == []


# -- W14: the certificates on the toolchain's own simulator ----------------------------------------------------------

def test_w14_each_certificate_discipline_runs_its_executor():
    from mimarsinan.certification import count_alignment

    flow = SimpleNamespace(lif_execution_synchronized=False, window_count_closed_form=False)
    for discipline, state in ((FWI, (False, False)), ("streaming", (False, False)), (ITE, (True, False)),
                              ("synchronized", (True, True))):
        prev = count_alignment._set_discipline(flow, discipline)
        assert (flow.lif_execution_synchronized, flow.window_count_closed_form) == state, discipline
        flow.lif_execution_synchronized, flow.window_count_closed_form = prev
    with pytest.raises(ValueError, match="discipline"):
        count_alignment.flow_node_counts(flow, torch.zeros(1, 1), discipline="per_event")


@pytest.mark.parametrize("dp,expected", [({}, ITE), ({KEY: FWI}, FWI), ({"spiking_variant": "streamed"}, FWI)])
def test_w14_k7_runs_both_sides_at_the_run_value(monkeypatch, dp, expected):
    from mimarsinan.pipelining.core import spike_count_gate as gate
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    plan = DeploymentPlan.resolve(_run(**dp))
    assert _new("mimarsinan.pipelining.core.spike_count_gate", "_run_discipline")(plan) == expected
    seen = {}

    class _Stop(Exception):
        pass

    def certify(*_a, discipline, reference_discipline, **_k):
        seen.update(discipline=discipline, reference=reference_discipline)
        raise _Stop

    monkeypatch.setattr(gate, "certify_twin_flow_counts", certify)
    monkeypatch.setattr(gate, "_certificate_samples", lambda *_a: torch.zeros(1, 1))
    monkeypatch.setattr(gate, "build_identity_mapping_for_pipeline", lambda *_a, **_k: object())
    monkeypatch.setattr(gate, "build_spiking_hybrid_flow", lambda *_a, **_k: object())
    with pytest.raises(_Stop):
        gate._certify_streaming_twin(SimpleNamespace(config={}), None, None, None, 1, plan)
    assert seen == {"discipline": expected, "reference": expected}


@pytest.mark.parametrize("semantics,firing", [("windowed_lif", FWI), ("synchronized_lif", ITE)])
def test_w14_each_randomized_synchronized_semantics_declares_the_value_its_flow_runs(monkeypatch, semantics, firing):
    from mimarsinan.certification.randomized import executors as ex
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point

    run = ex.RUNS[semantics]
    assert resolve_spiking_point(run.config).window_firing == firing
    built = {}

    class _Stop(Exception):
        pass

    def flow(*_a, **kwargs):
        built.update(kwargs)
        raise _Stop

    monkeypatch.setattr(ex, "SpikingHybridCoreFlow", flow)
    instance = SimpleNamespace(semantics=semantics, timesteps=4, network=SimpleNamespace(input_width=2))
    with pytest.raises(_Stop):
        ex.run_nodes(object(), instance)
    assert built["spiking_point"].window_firing == firing
    assert built["lif_execution_synchronized"] is (firing == ITE)


# -- W16: the toolchain's own windowed documents ---------------------------------------------------------------------

def _windowed_documents():
    from mimarsinan.chip_simulation.window_firing import window_firing_applies

    paths = sorted((ROOT / "templates").glob("tier_*/t*.json")) + sorted(
        (ROOT / "tests" / "fixtures" / "deployment_configs").glob("*.json"))
    return [p for p in paths if window_firing_applies(json.loads(p.read_text()).get("deployment_parameters") or {})]


def test_w16_the_toolchain_windowed_documents_declare_firing_during_the_window():
    documents = _windowed_documents()
    assert len(documents) == 23
    assert {p.stem: json.loads(p.read_text())["deployment_parameters"].get(KEY) for p in documents} == {
        p.stem: FWI for p in documents}


def test_w16_the_template_generator_declares_firing_during_the_window_for_windowed_cells():
    import importlib.util

    spec = importlib.util.spec_from_file_location("template_generate", ROOT / "templates" / "generate.py")
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    assert generate.MODES["lifsync"][KEY] == FWI
    assert all(KEY not in mode for name, mode in generate.MODES.items() if name != "lifsync")
