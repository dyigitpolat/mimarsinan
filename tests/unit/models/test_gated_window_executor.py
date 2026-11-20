"""[U-LIF2 W9] The gated cycle-level executor of integrate then emit, load-bearing, with its closed form beside it.

A synchronized LIF hop that integrates then emits runs on the per-cycle LIF physics with the fire gate: a core of
dependency level l (read from its execution stage's own wiring, never from the latency tags) integrates every arrival
from the stage's start until cycle l (T + 1) + T, takes its stage input, its always-on axons and its bias on its own T
cycles, then fires and subtracts on T input-free cycles. A stage of depth D runs (D + 1) T + D cycles. The packed
executor and the reference loop (the recording path) both run the gate; the count executor is its closed form, equal to
it on every neuron window and reached only by its name (the owner's law of 2026-07-22: never load-bearing).

The review's p-2 (the membrane guard precharged as the closed form adds it; the phase dither legal and inert), p-3 (the
depth read per execution stage, pass and segment, with a carried input a stage input) and p-11(b) (the gated executor
refuses a zero reset by name) are tested here. Every flow states integrate then emit explicitly (its two-window flag)."""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest
import torch

from mimarsinan.certification.randomized.generator import GeneratorSpace, generate
from mimarsinan.certification.randomized.instance import SYNCHRONIZED_LIF
from mimarsinan.models.nn.lif_kernels import MembraneRailTouchedError
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE
from mimarsinan.models.spiking.serial.refusals import (
    CycleAtomicRefusalError,
    SaturatingMembraneRefusalError,
    ZeroResetWindowCountError,
)

from unit.mapping.residency.executed import runs as R
from unit.mapping.residency.executed import vehicles as V
from unit.models import gated_window_programs as G

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
FUSED_PROGRAMS = G.chains((1, 3, 5)) + [("skip", G.SKIP), ("input_at_depth", G.INPUT_AT_DEPTH)]


def _equal_on_every_window(flow, runs) -> int:
    """Each stage: the packed executor, the reference loop and the closed form give one count per neuron window; the
    two gated executors give one segment output and one final membrane. Returns the windows compared."""
    compared = 0
    for stage, train in runs:
        packed, packed_out = G.window_counts(flow, stage, train, "packed")
        reference, reference_out = G.window_counts(flow, stage, train, "reference")
        closed, closed_out = G.window_counts(flow, stage, train, "closed_form")
        windows, differ = G.differing_windows(packed, closed)
        assert differ == 0, f"{stage.name}: {differ} of {windows} windows differ from the closed form"
        assert G.differing_windows(reference, packed) == (windows, 0)
        assert torch.equal(packed_out, closed_out) and torch.equal(reference_out, packed_out)
        compared += windows
    return compared


# -- the gate's structure: levels from the wiring, the depth rule ----------------------------------------------------

def test_w9_a_stage_depth_is_read_from_its_wiring_and_runs_its_depth_rule():
    """The skip shape a -> b -> c with a -> c is one fused stage of three levels: depth 3, (D + 1) T + D cycles."""
    instance = G.layered_instance(G.SKIP, timesteps=7, seed=3)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=False))
    (stage, train), = G.stage_runs(flow, G.inputs(instance))
    seg = flow._get_segment_tensors(stage, train.device)
    levels = G.new(G.GATED, "stage_dependency_levels")(seg)
    reads = [sorted({int(sp.src_core) for sp in spans if sp.kind == "core"}) for spans in seg["axon_spans"]]
    for core, read in enumerate(reads):
        assert levels[core] == (0 if not read else 1 + max(levels[r] for r in read))
    assert sorted(levels) == [0, 1, 2]
    plan = G.new(G.GATED, "gated_window_plan")(seg, 7)
    assert (plan.depth, plan.cycles) == (3, 4 * 7 + 3)
    assert plan.starts == tuple(level * (7 + 1) for level in levels)
    cycles = G.new(G.GATED, "gated_window_cycles")
    assert [cycles(d, 15) for d in (0, 1, 3, 5)] == [0, 31, 63, 95]


def test_w9_a_re_timed_level_stage_is_depth_one():
    instance = G.layered_instance(G.chain(3), timesteps=5, seed=4)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=True))
    runs = G.stage_runs(flow, G.inputs(instance))
    assert len(runs) == 3
    assert [G.depth_of(flow._get_segment_tensors(stage, train.device)) for stage, train in runs] == [1, 1, 1]


def test_w9_a_cycle_of_in_stage_reads_is_refused_by_name():
    from types import SimpleNamespace

    span = lambda src: SimpleNamespace(kind="core", src_core=src)  # noqa: E731
    seg = {"cores": [SimpleNamespace(latency=0), SimpleNamespace(latency=0)],
           "axon_spans": [[span(1)], [span(0)]]}
    from mimarsinan.chip_simulation.value_run.value_execution import ValueWalkOrderError

    with pytest.raises(ValueWalkOrderError, match="cycle"):   # the one R6 walk order's refusal
        G.new(G.GATED, "stage_dependency_levels")(seg)


# -- equality with the closed form on every neuron window ------------------------------------------------------------

@pytest.mark.parametrize("timesteps", [1, 4, 15])
@pytest.mark.parametrize("name,spec", FUSED_PROGRAMS, ids=[n for n, _s in FUSED_PROGRAMS])
def test_w9_the_gated_executors_equal_the_closed_form_on_every_window_of_a_fused_stage(name, spec, timesteps):
    instance = G.layered_instance(spec, timesteps=timesteps, seed=11 + timesteps, batch=16)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=False))
    runs = G.stage_runs(flow, G.inputs(instance))
    assert len(runs) == 1
    assert _equal_on_every_window(flow, runs) == 16 * sum(w for _n, _s, w in spec)


@pytest.mark.parametrize("extension", ["none", "phase_dither", "membrane_guard"])
@pytest.mark.parametrize("retimed", [False, True], ids=["fused", "retimed"])
@pytest.mark.parametrize("seed", range(6))
def test_w9_the_gated_executors_equal_the_closed_form_on_generated_programs(seed, retimed, extension):
    """X3a's generator (signed integer weights, skips, concats, host ops, residual adds), re-timed and fused, with
    each execution extension: every neuron window of every execution stage."""
    instance = generate(seed, GeneratorSpace(timesteps=(1, 12), batch=6), semantics=SYNCHRONIZED_LIF)
    kwargs = {"phase_dither": extension == "phase_dither",
              "lif_membrane_init": -0.5 if extension == "membrane_guard" else 0.0}
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=retimed), **kwargs)
    runs = G.stage_runs(flow, G.inputs(instance))
    assert runs and _equal_on_every_window(flow, runs) > 0


def test_w9_a_fused_stage_cut_into_passes_reads_its_depth_per_pass_and_runs_the_whole_programs_function():
    """The review's p-3: a fused stage cut into passes runs each pass at the depth of its own wiring, a carried slice
    read as a stage input; every pass equals its closed form, and the cut program's run equals the whole program's."""
    instance = G.layered_instance(G.chain(4, width=8), timesteps=6, seed=21, input_width=8, batch=12)
    ir = G.integer_ir(instance, max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS)
    cut = V.build(ir, 2, None)
    assert R.multi_pass(cut, ir)
    flow = G.ite_flow(instance, cut)
    runs = G.stage_runs(flow, G.inputs(instance))
    depths = [G.depth_of(flow._get_segment_tensors(stage, train.device)) for stage, train in runs]
    assert len(runs) >= 2 and max(depths) >= 2
    produced = [{int(sl.node_id) for sl in stage.output_map} for stage, _t in runs]
    assert any(int(sl.node_id) in set().union(*produced[:k]) for k, (stage, _t) in enumerate(runs) if k
               for sl in stage.input_map)                       # a later pass reads a slice an earlier pass carried
    _equal_on_every_window(flow, runs)
    whole = G.ite_flow(instance, V.build(ir, 64, None))
    assert not R.multi_pass(whole.hybrid_mapping, ir)
    assert torch.equal(G.run_point(instance, cut, executor_flow=flow), G.run_point(instance, None, executor_flow=whole))


@pytest.mark.parametrize("vehicle,count", [("skew3", 2), ("late_skip", 3)])
def test_w9_a_cut_with_delays_and_pins_runs_equal_to_its_identity_program(vehicle, count):
    """The residency vehicles whose cuts carry delayed slices and pin a producer at its consumer's latency: under
    integrate then emit every pass equals its closed form (a delivery delay never applies to a count)."""
    ir = V.ir_of(V.branchy_repr(vehicle))
    cut = V.build(ir, count, None)
    flow = R.flow(cut, **R.SYNCHRONIZED_LIF)
    runs = G.stage_runs(flow, torch.rand(6, V.IN, generator=torch.Generator().manual_seed(0)))
    _equal_on_every_window(flow, runs)


# -- the flow's executor of record ------------------------------------------------------------------------------------

def test_w9_an_integrate_then_emit_flow_runs_the_gated_executor_never_the_closed_form(monkeypatch):
    dispatch = G.new(G.GATED_REFERENCE, "run_integrate_then_emit").__module__
    import importlib

    module = importlib.import_module(dispatch)
    calls = {"gated": 0}
    real_gated = module.run_neural_segment_gated

    def gated(*args, **kwargs):
        calls["gated"] += 1
        return real_gated(*args, **kwargs)

    def closed_form(*_args, **_kwargs):
        raise AssertionError("the closed form ran as the executor of record")

    monkeypatch.setattr(module, "run_neural_segment_gated", gated)
    monkeypatch.setattr(module, "run_neural_segment_counts", closed_form)
    instance = G.layered_instance(G.chain(3), timesteps=6, seed=5)
    out = G.run_point(instance, G.identity_program(instance, retimed=False))
    assert calls["gated"] == 1 and out.shape[0] == len(instance.inputs)


def test_w9_the_closed_form_runs_only_when_asked_by_its_name_and_equals_the_gated_run():
    instance = G.layered_instance(G.SKIP, timesteps=9, seed=6, batch=12)
    mapping = G.identity_program(instance, retimed=False)
    flow = G.ite_flow(instance, mapping)
    assert flow.window_count_closed_form is False
    with torch.no_grad():
        gated = flow(G.inputs(instance))
        flow.window_count_closed_form = True
        closed = flow(G.inputs(instance))
    assert torch.equal(gated, closed)


def test_w9_the_recording_path_records_the_gated_executor():
    """A recorded run (the event census's, the parity gates') executes the gate: each core's recorded outputs are its
    gated window counts, and its recorded arrivals per axon are the closed form's window inputs (input counts, T on an
    always-on axon, a producer's count)."""
    instance = G.layered_instance(G.INPUT_AT_DEPTH, timesteps=8, seed=7, batch=1)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=False))
    (stage, train), = G.stage_runs(flow, G.inputs(instance))
    with torch.no_grad():
        out, record = flow.forward_with_recording(G.inputs(instance))
    assert torch.equal(out, G.run_point(instance, None, executor_flow=flow))
    gated, _ = G.window_counts(flow, stage, train, "packed")
    seg = flow._get_segment_tensors(stage, train.device)
    (segment,) = record.segments.values()
    for core in segment.cores:
        i = core.core_index
        assert core.output_spike_count.tolist() == gated[i][0].to(torch.int64).tolist()
        window = torch.zeros(1, core.n_in_used, dtype=COMPUTE_DTYPE)
        buffers = [gated.get(k, torch.zeros(1, 1, dtype=COMPUTE_DTYPE)) for k in range(len(seg["cores"]))]
        seg["axon_fill_plans"][i].apply(window, input_spikes=train.sum(0).to(COMPUTE_DTYPE), buffers=buffers,
                                        on_value=float(instance.timesteps))
        assert core.input_spike_count.tolist() == window[0].to(torch.int64).tolist()


def test_w9_a_signed_register_runs_inside_its_rails_and_a_rail_reached_is_refused_by_its_assertion():
    """The closed form refuses every bounded register; the gated executor runs a signed register, which holds the
    unbounded accumulator's number while no rail is reached, and its own assertion refuses a rail."""
    instance = G.layered_instance(G.chain(3), timesteps=6, seed=8, batch=10)
    mapping = G.identity_program(instance, retimed=False)
    wide = G.run_point(instance, mapping, soma_law=G.soma(membrane_bits=24, membrane_signed=True))
    assert torch.equal(wide, G.run_point(instance, mapping))
    with pytest.raises(MembraneRailTouchedError):
        G.run_point(instance, mapping, soma_law=G.soma(membrane_bits=3, membrane_signed=True))


def test_w9_the_reference_loop_runs_when_the_packed_executor_is_switched_off(monkeypatch):
    import importlib

    module = importlib.import_module(G.new(G.GATED_REFERENCE, "run_integrate_then_emit").__module__)
    calls = {"reference": 0}
    real = module.run_neural_segment_gated_reference

    def reference(*args, **kwargs):
        calls["reference"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "run_neural_segment_gated_reference", reference)
    instance = G.layered_instance(G.SKIP, timesteps=5, seed=9, batch=6)
    mapping = G.identity_program(instance, retimed=False)
    flow = G.ite_flow(instance, mapping)
    flow.use_packed_cycle_executor = False
    assert torch.equal(G.run_point(instance, None, executor_flow=flow), G.run_point(instance, mapping))
    assert calls["reference"] == 1


# -- the review's p-2 and p-11(b) -------------------------------------------------------------------------------------

@pytest.mark.parametrize("guard", [-0.5, 0.25])
def test_p2_the_membrane_guard_is_precharged_as_the_closed_form_adds_it(guard):
    instance = G.layered_instance(G.INPUT_AT_DEPTH, timesteps=6, seed=10, batch=12)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=False), lif_membrane_init=guard)
    _equal_on_every_window(flow, G.stage_runs(flow, G.inputs(instance)))
    plain = G.run_point(instance, G.identity_program(instance, retimed=False))
    assert not torch.equal(G.run_point(instance, None, executor_flow=flow), plain)   # the guard is not vacuous here


@pytest.mark.parametrize("retimed", [False, True], ids=["fused", "retimed"])
def test_p2_the_phase_dither_is_legal_and_inert(retimed):
    instance = G.layered_instance(G.SKIP, timesteps=8, seed=12, batch=12)
    mapping = G.identity_program(instance, retimed=retimed)
    flow = G.ite_flow(instance, mapping, phase_dither=True)
    assert torch.equal(G.run_point(instance, None, executor_flow=flow), G.run_point(instance, mapping))
    _equal_on_every_window(flow, G.stage_runs(flow, G.inputs(instance)))


@pytest.mark.parametrize("case,error", [
    ("zero_reset", ZeroResetWindowCountError),
    ("per_event", CycleAtomicRefusalError),
    ("clamping_register", SaturatingMembraneRefusalError),
])
def test_p11b_the_gated_executor_refuses_by_name_what_its_emission_window_cannot_run(case, error):
    instance = G.layered_instance(G.chain(2), timesteps=4, seed=13)
    mapping = G.identity_program(instance, retimed=False)
    kwargs = {"zero_reset": dict(firing_mode="Novena"),
              "per_event": dict(firing_mode="Novena", soma_law=G.soma(firing_granularity="per_event",
                                                                     firing_mode="Novena")),
              "clamping_register": dict(soma_law=G.soma(membrane_bits=8))}[case]
    with pytest.raises(error, match="the gated integrate-then-emit executor"):
        G.run_point(instance, mapping, **kwargs)


# -- the capability column and the closed form's one caller ----------------------------------------------------------

def test_w9_the_hybrid_flows_rows_execute_both_window_firings_and_no_external_row_does():
    from mimarsinan.chip_simulation.spiking_semantics import backend_capabilities

    both = frozenset({"fire_while_integrating", "integrate_then_emit"})
    flows = ("hcm", "unified", "hybrid", "scm", "training")  # [U-LIF2 W10] the training walk's count branch too
    for backend in flows + ("nevresim", "sanafe", "lava", "loihi", "odin_rtl", "odin_fpga", "odin_hacc"):
        expected = both if backend in flows else frozenset({"fire_while_integrating"})
        assert backend_capabilities(backend).window_firings == expected, backend


def _calls(name: str):
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                if called == name:
                    yield path.relative_to(SRC).as_posix()


def _writes(attribute: str):
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for t in ast.walk(target):
                        if isinstance(t, ast.Attribute) and t.attr == attribute:
                            yield path.relative_to(SRC).as_posix()


def test_w9_the_closed_form_has_one_caller_and_is_asked_for_only_by_the_gauge():
    """The count executor runs in one place, behind its name (``window_count_closed_form``), which only a flow's own
    construction (off) and the certification's analytic gauge (count_alignment's ``synchronized`` discipline) write."""
    assert sorted(set(_calls("run_neural_segment_counts"))) == [
        "models/spiking/hybrid/executors/gated_reference.py"]
    assert sorted(set(_writes("window_count_closed_form"))) == [
        "certification/count_alignment.py", "models/spiking/hybrid/flow.py"]


# -- the recording path on a cut, the layout's window, the byte count, the gauge's restore, an owed raster ------------

def test_w9_a_recorded_run_of_a_cut_with_delayed_slices_equals_its_unrecorded_run():
    """t07 on a cut: a recorded run (batch 1 per sample, the reference loop) reads each slice's window count as the
    unrecorded run (the packed executor) does, so the two give one output and one count per stage, and no delivery
    delay applies to a count (the vehicle's cut carries a slice delayed by one cycle)."""
    from mimarsinan.mapping.packing.pass_timing import replay_delays

    ir = V.ir_of(V.branchy_repr("late_skip"))
    cut = V.build(ir, 2, None)
    assert any(replay_delays(stage) for stage in V.neural_passes(cut))
    flow = R.flow(cut, **R.SYNCHRONIZED_LIF)
    x = torch.rand(3, V.IN, generator=torch.Generator().manual_seed(1))
    unrecorded: list = []
    flow.stage_count_recorder = lambda stage, counts: unrecorded.append(counts.detach().clone())
    with torch.no_grad():
        plain = flow(x)
    flow.stage_count_recorder = None
    with torch.no_grad():
        recorded, records = flow.forward_with_recordings(x)
    assert torch.equal(recorded, plain)
    stages = len(unrecorded)
    for sample, record in enumerate(records):
        counts = [record.segments[k].seg_output_spike_count for k in sorted(record.segments)]
        assert len(counts) == stages
        for stage in range(stages):
            assert counts[stage].tolist() == unrecorded[stage][sample].to(torch.int64).tolist()


def test_w9_the_gated_layout_follows_the_window_it_runs():
    """The packed layout's starts are the window's: a stage run at another window rebuilds it (equal to a fresh run)."""
    instance = G.layered_instance(G.SKIP, timesteps=5, seed=14, batch=4)
    mapping = G.identity_program(instance, retimed=False)
    short, long_ = G.ite_flow(instance, mapping), G.ite_flow(instance, mapping)
    long_.simulation_length = 9
    (stage, train), = G.stage_runs(short, G.inputs(instance))
    seg = short._get_segment_tensors(stage, train.device)
    run = G.new(G.GATED, "run_neural_segment_gated")
    policy = G._policy(short)
    common = dict(stage=stage, batch_size=int(train.shape[1]), device=train.device, policy=policy)
    long_train = torch.cat([train, train[:4]], dim=0)
    with torch.no_grad():
        run(short, train, seg=seg, T=5, **common)
        reused = run(long_, long_train, seg=seg, T=9, **common)
        fresh = run(long_, long_train, seg=long_._get_segment_tensors(stage, train.device), T=9, **common)
    assert torch.equal(reused, fresh)


def test_w9_the_segment_budget_counts_the_gated_layout():
    from mimarsinan.models.spiking.hybrid.segment_cache import segment_entry_nbytes

    instance = G.layered_instance(G.chain(3), timesteps=4, seed=15)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=False))
    (stage, train), = G.stage_runs(flow, G.inputs(instance))
    seg = flow._get_segment_tensors(stage, train.device)
    assert "packed_gated" in seg
    without = {k: v for k, v in seg.items() if k != "packed_gated"}
    weights = sum(int(b.weights.numel()) * b.weights.element_size() for b in seg["packed_gated"].buckets)
    assert segment_entry_nbytes(seg) >= segment_entry_nbytes(without) + weights > segment_entry_nbytes(without)


def test_w9_the_analytic_gauge_asks_for_the_closed_form_and_restores_both_flags(monkeypatch):
    """count_alignment's synchronized discipline (the analytic gauge) runs the closed form by its name, and hands the
    flow back as it found it: its two-window flag and its closed-form name."""
    from mimarsinan.certification.count_alignment import flow_perceptron_counts

    module = importlib.import_module(G.new(G.GATED_REFERENCE, "run_integrate_then_emit").__module__)
    calls = {"closed": 0}
    real = module.run_neural_segment_counts

    def closed_form(*args, **kwargs):
        calls["closed"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "run_neural_segment_counts", closed_form)
    instance = G.layered_instance(G.chain(2), timesteps=4, seed=16)
    ir = G.integer_ir(instance)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=False))
    flow_perceptron_counts(ir, flow, G.inputs(instance), discipline="synchronized")
    assert calls["closed"] == 1
    assert (flow.lif_execution_synchronized, flow.window_count_closed_form) == (True, False)
    flow_perceptron_counts(ir, flow, G.inputs(instance), discipline="streaming")
    assert (flow.lif_execution_synchronized, flow.window_count_closed_form, calls["closed"]) == (True, False, 1)


def test_w9_a_stage_that_owes_a_raster_is_refused_on_integrate_then_emit():
    """Integrate then emit carries counts, never a raster: a stage asked for its raster (a later pass's carry, the
    per-event raster seam) is refused by name, judged on the path that executes."""
    instance = G.layered_instance(G.chain(2), timesteps=4, seed=17)
    flow = G.ite_flow(instance, G.identity_program(instance, retimed=False))
    flow.stage_raster_recorder = lambda stage, train: None
    with pytest.raises(NotImplementedError, match="must carry a spike raster"):
        G.run_point(instance, None, executor_flow=flow)


def test_w9_the_integration_cycle_projects_onto_the_chip_lattice_as_the_per_cycle_cycle_does():
    """The fire gate changes when a neuron fires, never its arithmetic: an integration cycle holds the register and the
    chip lattice exactly as a per-cycle cycle does before it fires (float noise never decides a threshold tie)."""
    from mimarsinan.models.spiking.cycle_policy import LIFCyclePolicy

    policy = LIFCyclePolicy("Default", integer_lattice=True)
    noise = torch.tensor([1e-7, -1e-7, 2e-7, -2e-7] * 3, dtype=COMPUTE_DTYPE)
    contribution = (torch.arange(12, dtype=COMPUTE_DTYPE) / 2 + noise).reshape(1, -1)
    theta = torch.tensor(2.5, dtype=COMPUTE_DTYPE)
    gated, per_cycle = policy.make_state(1, 12, "cpu", COMPUTE_DTYPE), policy.make_state(1, 12, "cpu", COMPUTE_DTYPE)
    policy.integrate(gated, contribution)
    fired = policy.advance(per_cycle, contribution, theta, thresholding_mode="<=", output_dtype=COMPUTE_DTYPE)
    assert torch.equal(gated["memb"], per_cycle["memb"] + fired * theta)
    assert torch.equal(gated["memb"], torch.arange(12, dtype=COMPUTE_DTYPE).reshape(1, -1) / 2)


# -- [U-LIF2 r6, m-2] the cost rule prices what the gate runs ---------------------------------------------------------

#: The verify's program P2: a skip wire without relays (d reads a, three levels below it) in one fused stage.
P2 = (("a", (G.INPUT,), 4), ("b", ("a",), 4), ("c", ("b",), 4), ("d", ("a", "c"), 4))


class _CountingPolicy:
    """The flow's policy with the neuron-cycles it runs counted: each integrate and each emit of a bucket's slice."""

    def __init__(self, policy):
        self.policy, self.neuron_cycles = policy, 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def integrate(self, state, charge):
        self.neuron_cycles += int(charge.shape[1])
        return self.policy.integrate(state, charge)

    def emit(self, state, theta, **kwargs):
        self.neuron_cycles += int(theta.numel())
        return self.policy.emit(state, theta, **kwargs)


@pytest.mark.parametrize("spec,retimed,arrivals", [(P2, False, 4), (P2, True, 0), (G.chain(3, 4), False, 0),
                                                   (G.SKIP, False, 6), (G.INPUT_AT_DEPTH, False, 5)],
                         ids=["the verify's P2", "P2 re-timed", "a chain of three", "a skip over one level",
                              "a skip with an input at depth"])
def test_r6_m2_the_cost_rule_prices_each_core_at_twice_its_window_plus_its_arrival_cycles(spec, retimed, arrivals):
    """[U-LIF2 r6, m-2] Under integrate then emit the record's census seals the producer windows a stage's neurons
    integrate before their own windows (``SegmentRecord.window_arrivals``, from the gate's own reading of the wiring)
    and the cost rule prices the driven neurons at T (2 + arrivals / neurons): equal to the neuron-cycles the gated
    packed executor runs (its integrate and emit slices counted), on P2 (144 for 16 neurons at T = 4) as on programs
    whose every in-stage wire spans one level (2T) and on re-timed stages (depth 1). Red on the final tree of r5: the
    rule priced 2T, 128 against the executor's 144 on P2."""
    from mimarsinan.chip_simulation.stage_timesteps import neuron_cycles_per_window
    from mimarsinan.chip_simulation.window_firing_axis import INTEGRATE_THEN_EMIT
    from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping

    T = 4
    instance = G.layered_instance(spec, timesteps=T, seed=2)
    mapping = G.identity_program(instance, retimed=retimed)
    flow = G.ite_flow(instance, mapping)
    executed = 0
    for stage, train in G.stage_runs(flow, G.inputs(instance)):
        seg = flow._get_segment_tensors(stage, train.device)
        policy = _CountingPolicy(G._policy(flow))
        with torch.no_grad():
            G.new(G.GATED, "run_neural_segment_gated")(flow, train, seg=seg, stage=stage, T=T,
                                                       batch_size=int(train.shape[1]), device=train.device,
                                                       policy=policy)
        executed += policy.neuron_cycles
    segments = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0, timesteps=T,
                                            pass_transfer="collapse").segments()
    neurons = sum(core.neurons_used for segment in segments for core in segment.cores)
    taken = sum(int(getattr(segment, "window_arrivals", None) or 0) for segment in segments)
    price = neurons * neuron_cycles_per_window(T, INTEGRATE_THEN_EMIT, neurons=neurons, window_arrivals=taken)
    assert price == executed, (price, executed, neurons, taken)
    assert taken == arrivals
    if spec is P2 and not retimed:
        assert (neurons, executed) == (16, 144)


@pytest.mark.parametrize("firing", ["integrate_then_emit", "fire_while_integrating"])
def test_r6_m2_the_record_prices_the_arrivals_its_census_seals(firing):
    """[U-LIF2 r6, m-2] A sealed record prices the arrivals its passes state: under integrate then emit the neuron
    cycles per window are T (2 + arrivals / neurons_used), so the neuron terms price 2T a driven neuron plus T for each
    producer window taken early; firing while integrating prices T whatever the passes state."""
    import dataclasses

    from unit.deployment_record.record_fixtures import make_full_record

    from mimarsinan.deployment_record.quantities.from_record import from_record
    from mimarsinan.deployment_record.schema import SegmentRecord

    T, early = 5, 12
    record = make_full_record()
    stages = tuple(dataclasses.replace(s, window_arrivals=early) if isinstance(s, SegmentRecord) else s
                   for s in record.schedule.stages)
    record = dataclasses.replace(record, identity=dataclasses.replace(record.identity, window_firing=firing),
                                 timing=dataclasses.replace(record.timing, s_global=T),
                                 schedule=dataclasses.replace(record.schedule, stages=stages))
    values = from_record(record)
    neurons, taken = values.get("neurons_used").value, early * sum(isinstance(s, SegmentRecord) for s in stages)
    priced = neurons * values.get("neuron_cycles_per_window").value
    assert taken > 0 and priced == pytest.approx(2 * T * neurons + T * taken if firing == "integrate_then_emit"
                                                 else T * neurons)
