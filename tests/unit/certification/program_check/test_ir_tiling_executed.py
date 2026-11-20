"""[E1 fix round] The ``ir-tiling`` condition reads what each IR's identity program executes, not only the IR's
attributes: the latency and stage it runs each neuron at under timed semantics (``executed_latency``, the core graph's
ALAP latencies; ``timing_derivation`` holds that derivation to the check's identity program), each node through the
compaction the identity builder applies (kept rows and columns only), the reference given as the IR at unbounded
geometry (``reference_geometry``), and under a dithered encode the channel table each identity program carries
(``phase_channel``; R4 holds the deployment's table to the IR neurons' channels directly).

Every fault below is a cost-only decision or a builder defect the step-1 predicate established while the deployment
departs from the identity program at unbounded geometry (``RA/11_wave2/e1/review_soundness``,
``review_fidelity``); each case runs that sampled comparison beside the static verdict."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing import phase_channels as PCH

from unit.certification.program_check import tiling_vehicles as V

TIMED = ("streamed_lif", "cascaded_ttfs")
TIMING_FREE = ("windowed_lif", "synchronized_lif")


def _rules(case) -> set:
    return {v.rule for v in case.report.results["ir-tiling"].violations}


def _failed(case, rule: str, condition: str = "ir-tiling") -> None:
    result = case.report.results[condition]
    assert result.status == FAILED, (condition, result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    assert case.report.theorem1.startswith("failed") and condition in case.report.theorem1, case.report.theorem1


def _established(case) -> None:
    result = case.report.results["ir-tiling"]
    assert result.status == ESTABLISHED, [v.text() for v in result.violations]
    assert case.report.theorem1.startswith("established on the program against the IR at unbounded geometry")


# ── executed latency (soundness B1) ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("semantics", TIMED)
def test_a_tiling_that_splits_a_producers_consumers_moves_its_executed_latency(semantics):
    """Witness A2 at ``max_neurons`` 2: every IR neuron keeps its fan-in and IR latency attribute, but the tile read
    only by E runs at 0 where the unbounded P runs at 1, so E reads it a cycle earlier: the outputs differed.
    [U-TGAUGE5] The unbounded P is raised for D, so E reads it at a latency gap of 0: the per-cycle executor refuses
    the identity program by name (this file's refusal test), and the check still names the departure."""
    case = V.case(V.consumer_split(semantics), max_neurons=2)
    assert case.refused is not None and "no mapping armed" in case.refused, case.refused
    _failed(case, "executed_latency")


@pytest.mark.parametrize("semantics", TIMING_FREE)
def test_the_split_consumers_run_equal_where_timing_is_not_observable(semantics):
    case = V.case(V.consumer_split(semantics), max_neurons=2)
    assert case.windows and case.mismatches == 0
    _established(case)


def test_per_tile_dead_row_masks_that_move_an_executed_latency():
    """M1: the dead-row rule on both IRs (N-identity holds): the tile holding c0 no longer lists b0, so ChipLatency
    runs it one cycle earlier than the unbounded C."""
    case = V.case(V.two_depth("streamed_lif"), max_neurons=1, mutate=lambda ir, u: V.dead_row_masks(ir, u))
    assert case.refused is not None  # [U-TGAUGE5] a0 read at two depths: the window rule refuses the identity run
    assert case.report.results["N-identity"].status == ESTABLISHED
    _failed(case, "executed_latency")


def test_an_identity_builder_whose_timing_departs_from_the_core_graph(monkeypatch):
    """The predicate derives each IR's identity timing from its core graph and holds the derivation to the identity
    program the check builds: an identity builder whose ChipLatency skips the shift alignment departs from it.
    [U-LIVE] At two neurons a core no tile holds a row dead in its own columns; at one neuron a core C's tiles do, and
    the one liveness predicate runs each where its live reads put it (the departure the dead-row masks test names)."""
    from mimarsinan.mapping.latency.chip import ChipLatency

    assert V.case(V.two_depth("streamed_lif"), max_neurons=2).report.results["ir-tiling"].status == ESTABLISHED
    monkeypatch.setattr(ChipLatency, "_align_shiftable_cores", lambda self, current: current)
    _failed(V.case(V.two_depth("streamed_lif"), max_neurons=2), "timing_derivation")


# ── the compaction the identity builder applies (soundness MJ1) ─────────────────────────────────────────────────

@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif", "synchronized_lif"))
def test_a_tile_whose_compaction_drops_a_live_row(semantics):
    """M2: one tile of C drops its live a0 row at compaction, the unbounded C keeps it. [U-TGAUGE5] Under the
    streamed run (a per-cycle cascade) a0 is read at two depths, which the window rule refuses by name."""
    vehicle = V.two_depth(semantics)
    c = vehicle[1]().perceptron_of["C"]
    case = V.case(vehicle, max_neurons=1, mutate=lambda ir, u: V.drop_live_row(ir, c))
    if semantics in TIMED:
        assert case.refused is not None, case.refused
    else:
        assert case.mismatches > 0
    _failed(case, "fan_in")


def test_a_column_one_ir_compacts_and_the_other_keeps():
    vehicle = V.host_compacted("windowed_lif", dead=1)
    a = vehicle[1]().perceptron_of["A"]
    case = V.case(vehicle, max_neurons=2, mutate=lambda ir, u: V.drop_dead_channel(ir, a, 1))
    _failed(case, "missing_neuron")


# ── the reference (soundness MJ2) ───────────────────────────────────────────────────────────────────────────────

def _split(max_neurons: int = 1):
    return V.consumer_split("synchronized_lif"), max_neurons


def test_the_ir_at_unbounded_geometry_is_a_reference():
    vehicle, n = _split()
    _established(V.case(vehicle, max_neurons=n))


def test_the_deployments_own_tiled_ir_given_as_the_reference():
    vehicle, n = _split()
    _failed(V.case(vehicle, max_neurons=n, reference=lambda ir, u: ir), "reference_geometry")


def test_an_ir_at_another_declared_geometry_given_as_the_reference():
    vehicle, n = _split()
    instance, mapped = vehicle
    other = V.integer_ir(mapped(), instance.semantics, 2)
    _failed(V.case(vehicle, max_neurons=n, reference=lambda ir, u: other), "reference_geometry")


def _host_reference(edit):
    def reference(ir, u):
        out = copy.deepcopy(u)
        edit(out)
        return out
    return V.case(V.host_compacted("synchronized_lif", dead=1), max_neurons=2, reference=reference)


def test_a_reference_holding_one_host_op_name_twice():
    def edit(u):
        op = next(n for n in u.nodes if isinstance(n, ComputeOp) and n.name == "host_H")
        twin = copy.copy(op)
        twin.id = max(int(n.id) for n in u.nodes) + 1
        u.nodes.append(twin)
    _failed(_host_reference(edit), "reference_geometry")


def test_a_reference_holding_an_unkeyed_neural_node():
    def edit(u):
        next(n for n in u.get_neural_cores() if n.perceptron_index is not None).perceptron_index = None
    _failed(_host_reference(edit), "reference_geometry")


def test_a_reference_holding_a_coalescing_group():
    def edit(u):
        u.get_neural_cores()[-1].coalescing_group_id = 0
    _failed(_host_reference(edit), "reference_geometry")


# ── channel tables on the reindexed branch (fidelity major 2) ───────────────────────────────────────────────────

def _forward_map(remap):
    """A packer reindex read forward: per packed output its own index, the column the compaction moved it from lost."""
    if not remap or all(int(o) == int(n) for o, n in remap.items()):
        return None
    return np.arange(len(remap), dtype=np.int64)


def _compacted(semantics: str, dead: int):
    vehicle = V.host_compacted(semantics, dead=dead)
    a = vehicle[1]().perceptron_of["A"]

    def mutate(ir, u):
        V.drop_dead_channel(ir, a, dead)
        V.drop_dead_channel(u, a, dead)
    return vehicle, mutate


@pytest.mark.parametrize("semantics", ("windowed_lif", "streamed_lif"))
@pytest.mark.parametrize("dead", (0, 1))
def test_a_packer_compacted_dithered_deployment_equals_the_unbounded_identity(semantics, dead):
    vehicle, mutate = _compacted(semantics, dead)
    case = V.case(vehicle, max_neurons=2, dithered=True, mutate=mutate)
    assert case.windows and case.mismatches == 0
    tables = (case.program if dead == 0 else case.theorem).node_phase_channels
    assert any(np.ndim(v) for v in tables.values())     # the reindexed branch: one channel per packed output
    _established(case)


def test_a_reindex_fault_confined_to_the_unbounded_identity_program(monkeypatch):
    """A's middle channel dead: the tiled IR's tiles keep identity reindex maps, the unbounded A a moved one, so a
    forward-read reindex moves only the unbounded identity program's key. R4 holds (the deployment rotates every neuron
    by its channel); the deployment departs from Theorem 1's identity program; ir-tiling names the table."""
    monkeypatch.setattr(PCH, "_packed_sources", _forward_map)
    vehicle, mutate = _compacted("windowed_lif", 1)
    case = V.case(vehicle, max_neurons=2, dithered=True, mutate=mutate)
    assert case.mismatches > 0
    assert case.report.results["R4"].status != FAILED
    _failed(case, "phase_channel")


def test_a_reindex_fault_in_the_deployments_channel_table(monkeypatch):
    """A's first channel dead: the deployment's tile reindexes too, so its table keys a1 on channel 0. Before the fix
    R4 compared it with the identity program's table, built by the same function; it now reads the IR neuron."""
    monkeypatch.setattr(PCH, "_packed_sources", _forward_map)
    vehicle, mutate = _compacted("windowed_lif", 0)
    case = V.case(vehicle, max_neurons=2, dithered=True, mutate=mutate)
    _failed(case, "phase_channel", condition="R4")


def test_a_bank_backed_node_whose_mask_is_not_its_columns_has_no_channel():
    """``_columns`` holds a bank-backed node's pre-pruning survivors to its width as it does an owned matrix's."""
    bank = WeightBank(id=0, core_matrix=np.ones((3, 4), dtype=np.float64))
    sources = np.array([IRSource(-2, 0), IRSource(-2, 1), IRSource(-3, 0)], dtype=object)
    node = NeuralCore(id=0, name="banked", input_sources=sources, core_matrix=None, weight_bank_id=0,
                      weight_row_slice=(0, 2), perceptron_index=0, perceptron_output_slice=(0, 3), threshold=1.0)
    node.pre_pruning_col_mask = [False, False, False]
    ir = IRGraph(nodes=[node], output_sources=np.array([IRSource(0, 0)], dtype=object), weight_banks={0: bank})
    assert 0 not in PCH.node_phase_channels(ir)
    node.pre_pruning_col_mask = [False, True, False]
    assert np.array_equal(PCH.node_phase_channels(ir)[0], [0, 2])


# ── the declaration keys the timing rules (E1 verify N1, integration2/e1) ───────────────────────────────────────

def _tiling_alone(case, semantics: str, facts):
    """``ir-tiling`` on the case's two IRs through ``check_ir_tiling`` with the given identity facts."""
    from mimarsinan.certification.program_check import CheckDeclaration
    from mimarsinan.certification.program_check.report import ConditionResult
    from mimarsinan.certification.program_check.unbounded import check_ir_tiling

    decl = CheckDeclaration(semantics=semantics, window=V.T, pass_transfer=V.TRANSFER[semantics],
                            dithered_encode=False, core_types=tuple(dict(c) for c in V.constraints(2)["cores"]))
    result = ConditionResult("ir-tiling", ESTABLISHED)
    check_ir_tiling(case.ir, case.unbounded, declaration=decl, result=result, identity_facts=facts)
    return result


def test_facts_without_executed_latencies_under_a_timed_declaration_fail_by_name():
    """A caller that hands ``check_ir_tiling`` identity facts without latencies under a timed declaration: the timing
    rules compare nothing, so the condition fails ``timing_derivation`` by name, never holds (A2, whose executed
    latency differs, would otherwise read established)."""
    from mimarsinan.certification.program_check.unbounded.latency import IdentityFacts

    case = V.case(V.consumer_split("streamed_lif"), max_neurons=2)
    facts = case.report.identity_facts
    assert _tiling_alone(case, "streamed_lif", facts).violation_count
    stripped = IdentityFacts(latency=None, stage=facts.stage, channels=facts.channels)
    result = _tiling_alone(case, "streamed_lif", stripped)
    assert "timing_derivation" in {v.rule for v in result.violations}, result.note


@pytest.mark.parametrize("semantics", TIMING_FREE)
def test_the_note_names_only_what_was_compared(semantics):
    """Timing-free and undithered: the condition holds and its note claims neither executed latency nor channels."""
    case = V.case(V.consumer_split(semantics), max_neurons=2)
    result = _tiling_alone(case, semantics, case.report.identity_facts)
    assert result.violation_count == 0
    assert "executed latency" not in result.note and "channel" not in result.note, result.note
