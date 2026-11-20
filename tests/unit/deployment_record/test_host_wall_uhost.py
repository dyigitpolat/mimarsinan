"""[U-HOST] (b) One source of truth for the per-inference host wall, and no measured wall in a priced term.

The timer records the wall, the calls and the samples each call carried; the runner that drives the traversal states
the count (never a tensor shape); ``stage_timing.host_wall`` returns the per-inference wall under its one definition
(amortized: each op's wall over the inferences its calls carried) and every consumer reads it (the record step's
sealed fields, the record's quantities, the objective, the measured plane, the ODIN FPGA step); the raw total is a
labelled diagnostic; a wall without its count is refused by name. No pricing site reads the raw field, and scaling the
development machine's wall by 10 changes no priced term (tests T-B1 to T-B6: the check's M-1 tests (a) and (b), t-4).
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
from spiking_points import point_of_mode

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.hybrid_run.stage_timing import StageTimer
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost import DeploymentCostModel
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.objectives import OBJECTIVES, DeploymentRecordView
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block, host_descriptions
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema import ComputeOpRecord
from mimarsinan.pipelining.pipeline_steps.verification import deployment_record_walls as walls
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import fold_compute_walls

from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.record_fixtures import make_full_record

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
#: The synthetic clock's parts: a per-call part C and a per-sample part S (seconds), and the traversal size N.
C, S, N = 0.010, 0.002, 25


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Stage:
    def __init__(self, kind, name):
        self.kind, self.name = kind, name


def _op_names(record):
    return [s.name for s in record.schedule.stages if isinstance(s, ComputeOpRecord)]


def _timed_rows(record, traversals, samples):
    """``traversals`` traversals of ``samples`` samples each through the stage loop, one timer, the synthetic clock
    charging C per call and S per sample: the timer's rows."""
    clock = _Clock()
    timer = StageTimer(clock=clock)
    stages = [_Stage("compute", name) for name in _op_names(record)]

    def on_compute(ctx):
        clock.now += C + S * samples

    for _ in range(traversals):
        run_hybrid_stages(type("M", (), {"stages": stages})(), {}, on_neural=lambda ctx: None, on_compute=on_compute,
                          stage_timer=timer, timed_samples=samples)
    return timer.compute_stage_walls()


def _record(traversals, samples):
    record = make_full_record()
    return replace(record, schedule=fold_compute_walls(record.schedule, _timed_rows(record, traversals, samples)))


#: (traversals, samples per traversal, the definition's per-inference wall per op)
CASES = [(1, N, C / N + S), (N, 1, C + S)]


class TestTheSyntheticClock:
    @pytest.mark.parametrize("traversals,samples,expected", CASES)
    def test_every_consumer_reads_the_definitions_value(self, traversals, samples, expected):
        """T-B1 (the check's (a)): one N-sample traversal and N one-sample traversals, a per-call and a per-sample
        part: each consumer of the host wall reads the one definition's value (red at R5: no count, raw totals)."""
        from mimarsinan.chip_simulation.hybrid_run.stage_timing import PER_INFERENCE_DEFINITION, host_wall

        record = _record(traversals, samples)
        ops = len(_op_names(record))
        assert host_wall(record.schedule.stages).per_inference_s == pytest.approx(ops * expected)
        assert "amortized" in PER_INFERENCE_DEFINITION and host_wall(()).definition == PER_INFERENCE_DEFINITION
        assert walls.host_ops_wall_s_per_pass(record.schedule) == pytest.approx(ops * expected)  # the sealed field
        assert walls.host_ops_wall_s(record.schedule) == pytest.approx(ops * traversals * (C + S * samples))
        assert from_record(record).get("host_ops_s").value == pytest.approx(ops * expected)
        assert OBJECTIVES.get("host_op_wall_s").value(DeploymentRecordView(record=record)) == pytest.approx(
            ops * expected)
        terms = {t.name: t for t in DeploymentCostModel().latency(record)}
        assert terms["host_wall_diagnostic_s"].value == pytest.approx(ops * expected)

    def test_the_odin_fpga_step_reads_the_definitions_value(self):
        """T-B1 at the ODIN FPGA step: N deployed runs of one sample each (a fresh timer per run), the raw sum a
        labelled diagnostic (red at R5: the raw sum over all runs was the step's host_ops_s)."""
        from types import SimpleNamespace

        from mimarsinan.pipelining.pipeline_steps.verification.odin.odin_fpga_simulation_step import _timing_fragment

        record = make_full_record()
        runs = [SimpleNamespace(timings=(), compute_stage_walls=_timed_rows(record, 1, 1)) for _ in range(N)]
        fragment = _timing_fragment(runs, {"programming_s": 0.0, "execution_s": 0.0})
        ops = len(_op_names(record))
        assert fragment["host_ops_s"] == pytest.approx(ops * (C + S))
        assert fragment["host_ops_s_raw_total"] == pytest.approx(ops * N * (C + S))
        assert "never priced" in fragment["host_ops_s_raw_total_label"]


class TestTheRunnersStateTheirCounts:
    def test_the_simulation_runner_states_its_traversals_count(self, monkeypatch):
        """T-B6: the nevresim simulation runner runs each host op once over all its samples, and states that count
        (its test data's length, never a tensor shape) for each timed call."""
        import numpy as np

        import mimarsinan.chip_simulation.simulation_runner.hybrid as hybrid
        from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner
        from types import SimpleNamespace

        samples = 7
        for name, stub in (("prepare_all_segments", lambda runner, mapping: {}),
                           ("boundary_normalization_scales", lambda mapping: {}),
                           ("resolve_stage_compute_scales", lambda *a, **k: (1.0, 1.0)),
                           ("host_op_input_state", lambda op, buf, *a: buf),
                           ("host_op_for_evaluation", lambda op, *a, **k: op),
                           ("execute_compute_op_numpy", lambda *a, **k: np.zeros((samples, 1))),
                           ("gather_final_output_numpy", lambda *a, **k: np.zeros((samples, 2))),
                           ("program_output_gauge", lambda *a, **k: np.ones(2))):  # [composed head, rule 19]
            monkeypatch.setattr(hybrid, name, stub)
        runner = SimulationRunner.__new__(SimulationRunner)
        runner.test_data = [(np.zeros(4), np.zeros(1)) for _ in range(samples)]
        runner.spiking_mode, runner.simulation_length, runner.host_compute_device = "lif", 4, "cpu"
        runner.stage_timer = StageTimer()
        runner._evaluate_chip_output = lambda predictions: 0.0
        op = SimpleNamespace(id=1, input_sources=np.array([], dtype=object))
        mapping = SimpleNamespace(stages=[SimpleNamespace(kind="compute", name="op_x", compute_op=op)],
                                  output_sources=np.array([], dtype=object), node_output_shifts=None)
        runner._run_hybrid(mapping)
        (row,) = runner.stage_timer.compute_stage_walls()
        assert (row["name"], row["invocations"], row["samples_per_call"]) == ("op_x", 1, samples)


    def test_the_odin_fpga_runner_states_its_runs_count(self, monkeypatch):
        """T-B6: the ODIN FPGA runner's run() is one sample (its contract), and it states that count."""
        import contextlib

        import numpy as np

        import mimarsinan.chip_simulation.odin_fpga.runner as odin
        from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
        from types import SimpleNamespace

        monkeypatch.setattr(odin, "execute_compute_stage", lambda *a, **k: np.zeros((1, 1)))
        monkeypatch.setattr(odin, "boundary_normalization_scales", lambda mapping: {})
        monkeypatch.setattr(odin, "DeviceSession", lambda transport: contextlib.nullcontext())
        op = SimpleNamespace(id=1, input_sources=np.array([], dtype=object))
        mapping = SimpleNamespace(stages=[SimpleNamespace(kind="compute", name="op_x", compute_op=op)])
        mapping._carried_output_cache = {}
        contract = SimpleNamespace(soma_law=lambda: DEFAULT_SOMA_LAW, spiking_mode="lif",
                                   spiking_point=lambda: point_of_mode("lif"))  # [composed head, rule 19]
        runner = odin.OdinFpgaRunner(mapping, 4, contract=contract, transport=SimpleNamespace(name="fake"),
                                     weight_bits=4, effective_max_axons=127, max_neurons=256, pass_transfer="verbatim")
        (row,) = runner.run(np.zeros((1, 3)), sample_index=0).compute_stage_walls
        assert (row["name"], row["invocations"], row["samples_per_call"]) == ("op_x", 1, 1)


class TestAWallWithoutItsCount:
    def test_the_timer_refuses_a_call_without_its_count(self):
        """T-B2: the timer records the samples each call carried; a call without them refuses by name."""
        timer = StageTimer(clock=_Clock())
        with pytest.raises(ToolchainRefusal, match="op_x"):
            with timer.time_compute_stage(0, "op_x", None):
                pass

    def test_a_timed_stage_loop_without_the_traversals_count_refuses(self):
        """T-B2: the runner states its traversal's count; a timed loop without it refuses by name."""
        stages = [_Stage("compute", "op_x")]
        with pytest.raises(ToolchainRefusal, match="op_x"):
            run_hybrid_stages(type("M", (), {"stages": stages})(), {}, on_neural=lambda ctx: None,
                              on_compute=lambda ctx: None, stage_timer=StageTimer(clock=_Clock()))

    def test_the_fold_refuses_a_row_without_its_count(self):
        """T-B2: a fragment's wall row without its samples per call (a pre-rule cache) refuses to seal by name."""
        record = make_full_record()
        name = _op_names(record)[0]
        row = {"stage_index": 0, "name": name, "wall_s_total": 0.5, "invocations": 1}
        with pytest.raises(ToolchainRefusal, match=name):
            fold_compute_walls(record.schedule, [row])

    def test_a_sealed_wall_without_its_count_is_named_at_every_consumer(self):
        """T-B2: a record sealed before the rule (a wall, its calls, no samples per call): never put per
        inference; named where it is read (red at R5: the raw total read as one sample's)."""
        record = make_full_record()
        stages = tuple(replace(s, wall_s_total=0.5, invocations=1) if isinstance(s, ComputeOpRecord) else s
                       for s in record.schedule.stages)
        record = replace(record, schedule=replace(record.schedule, stages=stages))
        name = _op_names(record)[0]
        with pytest.raises(ToolchainRefusal, match=name):
            walls.host_ops_wall_s_per_pass(record.schedule)
        q = from_record(record)
        assert not q.has("host_ops_s") and name in (q.withheld_reason("host_ops_s") or "")
        assert "host_op_wall_s" not in {s.key for s in OBJECTIVES.available_for(DeploymentRecordView(record=record))}
        assert any("withheld" in n and name in n for n in DeploymentCostModel().evaluate(record).notes)


def _attribute_reads(attr):
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.ctx, ast.Load):
                hits.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return hits


class TestTheRatchet:
    @pytest.mark.parametrize("attr", ["host_ops_s", "host_ops_s_per_pass"])
    def test_no_module_reads_the_sealed_wall_fields(self, attr):
        """T-B3 (the check's (b)): no site reads timing.latency.host_ops_s (the raw total) or its sealed per-pass
        copy; every consumer reads the one function on the schedule (red at R5: combine, extractors, from_record)."""
        assert _attribute_reads(attr) == []

    @pytest.mark.parametrize("token", ["host_ops_s", "host_compute_rate"])
    def test_no_pricing_module_reads_a_measured_wall_or_its_relation(self, token):
        """T-B3: the pricers (deployment_record/cost) name neither the measured wall quantity nor the row relating
        another machine to the host (red at R5: absolute/context.py priced host_ops_s / host_compute_rate)."""
        hits = []
        for path in sorted((SRC / "deployment_record" / "cost").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and token in node.value:
                    hits.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert hits == []

    def test_every_timed_stage_loop_states_its_traversals_count(self):
        """T-B3: every call of the stage loop that passes a timer passes the count its runner states (red at R5: the
        simulation runner, SANA-FE and the ODIN FPGA runner passed none)."""
        missing = []
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == (
                        "run_hybrid_stages"):
                    words = {k.arg: k.value for k in node.keywords}
                    timer = words.get("stage_timer")
                    if timer is not None and not (isinstance(timer, ast.Constant) and timer.value is None) \
                            and "timed_samples" not in words:
                        missing.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert missing == []


def _host_variants(name):
    yield "no host", get_platform_physics(name)
    yield "identified host", resolve_platform_physics(name, merge_override_blocks(
        {"host": host_description_block(host_descriptions()[0])}))


def _priced(pricing):
    return (sorted((t.name, t.band.low, t.band.nominal, t.band.high) for t in pricing.terms),
            sorted((r.name, r.reason) for r in pricing.refusals))


class TestNoPricedTermReadsTheWall:
    @pytest.mark.parametrize("name", available_profiles())
    def test_t4_scaling_the_wall_by_ten_changes_no_priced_term(self, name):
        """T-B4 (t-4): for every registry profile, without a host and with the identified host, the priced terms and
        refusals are equal at the wall and at ten times it (red at R5 on the seven profiles with host_compute_rate)."""
        base = dict(synaptic_events=1e6, host_macs=1e6, compute_op_count=3, cores_physical=8, latency_steps=32,
                    timesteps=4, neurons_used=100, segment_cores=1)
        for label, physics in _host_variants(name):
            at = _priced(price_absolute(quantities(**base, host_ops_s=0.05), physics))
            ten = _priced(price_absolute(quantities(**base, host_ops_s=0.5), physics))
            assert at == ten, (name, label)

    def test_t4_the_measured_plane_keeps_its_total_when_the_wall_moves(self):
        """T-B5: the measured cost plane's total is the chip side; the wall is a diagnostic beside it."""
        record = _record(1, 4)
        at = DeploymentCostModel().evaluate(record)
        stages = tuple(replace(s, wall_s_total=10 * s.wall_s_total) if isinstance(s, ComputeOpRecord) else s
                       for s in record.schedule.stages)
        ten = DeploymentCostModel().evaluate(replace(record, schedule=replace(record.schedule, stages=stages)))

        def kept(report):
            return sorted((t.name, t.value) for group in (report.latency, report.throughput, report.energy)
                          for t in group if t.name != "host_wall_diagnostic_s")

        assert kept(at) == kept(ten)
        assert {t.name: t.value for t in ten.latency}["host_wall_diagnostic_s"] == pytest.approx(
            10 * {t.name: t.value for t in at.latency}["host_wall_diagnostic_s"])
