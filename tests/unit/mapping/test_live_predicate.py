"""[U-LIVE] One liveness predicate for every scheduling rule: one test per site.

The class (the exactness diagnosis's class A with its check's m-2 and m-3): a wired axon whose weight row is all zero,
from any source (Weight Quantization rounding a weight to code zero, an exact zero of the trained module, a pruned
column), counted as a source by one scheduling rule and not by another. ChipLatency's core sweep, its shift alignment
and its consumer map counted every wired axon, while its own per-neuron delays and the relay planner counted only
axons with a nonzero weight: a dead axon raised its reader, the alignment raised an entry core to serve it, and a
shallower live reader of that core read it a cycle late. The one rule: an axon is live for scheduling exactly when it
is not off, not a segment input, not always-on, and its weight row holds a nonzero weight
(``mapping/latency/upstream.py``), and every site reads it. Each test below is a site's behaviour on the smallest form
that shows it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fake_cores import FakeCore

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.latency.core_graph import alap_latencies
from mimarsinan.mapping.latency.cycle_windows import core_window_gaps, ir_window_gaps
from mimarsinan.mapping.latency.depth_balancing import find_intra_segment_depth_gaps
from mimarsinan.mapping.latency.ir import IRLatency


# ---------------------------------------------------------------------------------------------------------------------
# hard cores: E reads the stage input, A reads E, C reads E and reads A through all-zero rows only
# ---------------------------------------------------------------------------------------------------------------------
def _inp(i):
    return SpikeSource(-2, i, is_input=True)


def _core_src(c, n):
    return SpikeSource(c, n)


def _hcore(sources, matrix):
    return FakeCore(axon_sources=list(sources), core_matrix=np.asarray(matrix, dtype=np.float64), latency=None)


LIVE2 = [[0.5, 0.2], [0.1, 0.6]]


def _three_cores():
    e = _hcore([_inp(0), _inp(1)], LIVE2)
    a = _hcore([_core_src(0, 0), _core_src(0, 1)], LIVE2)
    c = _hcore([_core_src(0, 0), _core_src(0, 1), _core_src(1, 0), _core_src(1, 1)],
               [[0.4], [0.3], [0.0], [0.0]])
    return SimpleNamespace(cores=[e, a, c], output_sources=[_core_src(1, 0), _core_src(1, 1), _core_src(2, 0)])


def _live_gaps(mapping):
    lat = [int(c.latency) for c in mapping.cores]
    gaps = []
    for k, core in enumerate(mapping.cores):
        m = core.get_core_matrix()
        for a, s in enumerate(core.axon_sources):
            if s.core_ >= 0 and not s.is_input_ and np.any(m[a] != 0):
                gaps.append(lat[k] - lat[s.core_])
    return gaps


def test_chip_latency_counts_a_dead_axon_for_no_rule():
    """The diagnosis's first test: every live edge at gap 1, latencies 0, 1, 1 (the window unit's tree: 1, 1, 2, A
    reading E at gap 0)."""
    mapping = _three_cores()
    ChipLatency(mapping).calculate()
    assert [c.latency for c in mapping.cores] == [0, 1, 1]
    assert set(_live_gaps(mapping)) == {1}
    assert core_window_gaps(mapping) == []


def test_the_sweep_raises_a_reader_by_its_live_sources_only():
    """C reads E live and the end of a three-core chain through zero rows: 1 plus its live sources' latency."""
    e = _hcore([_inp(0), _inp(1)], LIVE2)
    p1 = _hcore([_core_src(0, 0), _core_src(0, 1)], LIVE2)
    p2 = _hcore([_core_src(1, 0), _core_src(1, 1)], LIVE2)
    p3 = _hcore([_core_src(2, 0), _core_src(2, 1)], LIVE2)
    c = _hcore([_core_src(0, 0), _core_src(0, 1), _core_src(3, 0), _core_src(3, 1)], [[0.4], [0.3], [0.0], [0.0]])
    mapping = SimpleNamespace(cores=[e, p1, p2, p3, c], output_sources=[_core_src(3, 0), _core_src(4, 0)])
    ChipLatency(mapping).calculate()
    assert mapping.cores[4].latency == mapping.cores[0].latency + 1
    assert set(_live_gaps(mapping)) == {1}


def test_the_alignment_raises_an_entry_core_for_its_live_readers_only():
    """E is read live by A and through a zero row by D, a deep core: the alignment serves A, never D's dead row."""
    e = _hcore([_inp(0), _inp(1)], LIVE2)
    a = _hcore([_core_src(0, 0), _core_src(0, 1)], LIVE2)
    b = _hcore([_core_src(1, 0), _core_src(1, 1)], LIVE2)
    d = _hcore([_core_src(2, 0), _core_src(2, 1), _core_src(0, 0), _core_src(0, 1)], [[0.4], [0.3], [0.0], [0.0]])
    mapping = SimpleNamespace(cores=[e, a, b, d], output_sources=[_core_src(1, 0), _core_src(3, 0)])
    ChipLatency(mapping).calculate()
    assert [c.latency for c in mapping.cores] == [0, 1, 2, 3]
    assert core_window_gaps(mapping) == []


def _random_mapping(rng, n_cores=7):
    cores = []
    for k in range(n_cores):
        n_ax, n_out = int(rng.integers(2, 6)), int(rng.integers(1, 4))
        sources, rows = [], []
        for a in range(n_ax):
            pick = rng.random()
            if k == 0 or pick < 0.2:
                sources.append(_inp(a))
            elif pick < 0.27:
                sources.append(SpikeSource(-1, 0, is_off=True))
            else:
                p = int(rng.integers(0, k))
                sources.append(_core_src(p, int(rng.integers(0, cores[p].core_matrix.shape[1]))))
            dead = rng.random() < 0.35
            rows.append(np.zeros(n_out) if dead else rng.uniform(0.1, 1.0, size=n_out) * (rng.random(n_out) < 0.8))
        cores.append(_hcore(sources, np.vstack(rows)))
    outputs = [_core_src(n_cores - 1, 0)] + [_core_src(int(k), 0) for k in rng.integers(0, n_cores, size=2)]
    return SimpleNamespace(cores=cores, output_sources=outputs)


@pytest.mark.parametrize("seed", range(40))
def test_the_core_graph_reads_one_source_set_with_chip_latency(seed):
    """Property (the diagnosis's fourth test): on random cores with all-zero rows, ChipLatency's latencies equal
    ``core_graph.alap_latencies`` over each core's live sources (a nonzero row reading a stage core), the source set
    the relay planner's reading gives: one schedule whatever rule reads it."""
    mapping = _random_mapping(np.random.default_rng(seed))
    ChipLatency(mapping).calculate()
    live = []
    for core in mapping.cores:
        m = core.get_core_matrix()
        live.append(sorted({int(s.core_) for a, s in enumerate(core.axon_sources)
                            if s.core_ >= 0 and not (s.is_off_ or s.is_input_) and np.any(m[a] != 0)}))
    expected = alap_latencies(live)
    for k, core in enumerate(mapping.cores):
        if live[k] or any(k in srcs for srcs in live):
            assert core.latency == expected[k], (k, [c.latency for c in mapping.cores], expected)


# ---------------------------------------------------------------------------------------------------------------------
# IR: the same form as IR nodes (E over the input, A over E, C over E and over A through zero rows)
# ---------------------------------------------------------------------------------------------------------------------
def _node(node_id, name, sources, matrix):
    return NeuralCore(id=node_id, name=name, input_sources=np.asarray(sources, dtype=object),
                      core_matrix=np.asarray(matrix, dtype=np.float64), threshold=1.0, latency=None,
                      parameter_scale=torch.tensor(0.0))


def _reads(node_id, n=2):
    return [IRSource(node_id, i) for i in range(n)]


def _ir_form(dead_rows=None, chain=0):
    dead = np.zeros((2, 1)) if dead_rows is None else np.asarray(dead_rows, dtype=np.float64)
    nodes = [_node(0, "E", [IRSource(-2, 0), IRSource(-2, 1)], LIVE2), _node(1, "A", _reads(0), LIVE2)]
    last = 1
    for k in range(chain):
        nodes.append(_node(2 + k, f"P{k}", _reads(last), LIVE2))
        last = 2 + k
    nodes.append(_node(last + 1, "C", _reads(0) + _reads(last), np.vstack([[[0.4], [0.3]], dead])))
    outs = [IRSource(1, 0), IRSource(1, 1), IRSource(last + 1, 0)]
    return IRGraph(nodes=nodes, output_sources=np.asarray(outs, dtype=object))


def test_ir_latency_takes_no_tier_from_a_dead_read():
    """IRLatency (the IR's tiers and the record's ``ir_max_latency``) reads the predicate: C over E live and over the
    end of a chain through zero rows sits one tier above E (the window unit's tree: one above the chain)."""
    graph = _ir_form(chain=2)
    depth = IRLatency(graph).calculate()
    lat = {n.name: n.latency for n in graph.nodes}
    assert lat["C"] == lat["E"] + 1 == 1
    assert depth == 4  # the chain E, A, P0, P1 is the deepest path


def test_the_window_assertion_reads_the_latencies_of_the_live_reads():
    """``ir_window_gaps`` on a program whose dead read was never turned off (built outside the mapping): no live edge
    reads a gap other than one (the window unit's tree reported E -> A at gap 0, a false refusal)."""
    assert ir_window_gaps(_ir_form()) == []


def test_the_relay_planner_counts_a_placeholder_row_as_the_latency_engine_does():
    """The relay planner reads the one predicate: a bound operand bank's placeholder row (NaN until its payload is
    written) delivers charge to ChipLatency's per-neuron delays, so the planner sees the gap through it (the window
    unit's tree read it dead, ``abs(NaN) > 0`` being false)."""
    graph = _ir_form(dead_rows=[[np.nan], [np.nan]], chain=2)
    gaps = find_intra_segment_depth_gaps(graph)
    assert [(g.producer_id, g.consumer_id, g.gap) for g in gaps] == [(0, 4, 4)]


# ---------------------------------------------------------------------------------------------------------------------
# the pass projection: two passes of one fused segment, the second reading a carried slice through zero rows
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class _Slice:
    node_id: int
    offset: int
    size: int


def _placement(node, axons, neurons):
    return {"ir_node_id": node, "axon_offset": 0, "axons": axons, "neuron_offset": 0, "neurons": neurons}


def _pass(name, cores, placements, input_map, output_map):
    hcm = SimpleNamespace(cores=cores, soft_core_placements_per_hard_core=placements,
                          output_sources=[_core_src(len(cores) - 1, 0)])
    return SimpleNamespace(name=name, kind="neural", hard_core_mapping=hcm, schedule_segment_index=0,
                           input_map=input_map, output_map=[_Slice(n, 0, 1) for n in output_map])


def _two_passes():
    e = _hcore([_inp(0), _inp(1)], LIVE2)
    a = _hcore([_core_src(0, 0), _core_src(0, 1)], LIVE2)
    first = _pass("p1", [e, a], [[_placement(10, 2, 2)], [_placement(11, 2, 2)]], [_Slice(99, 0, 2)], [10, 11])
    c = _hcore([_inp(0), _inp(1), _inp(2), _inp(3)], [[0.4], [0.3], [0.0], [0.0]])
    second = _pass("p2", [c], [[_placement(12, 4, 1)]], [_Slice(10, 0, 2), _Slice(11, 2, 2)], [12])
    return first, second


def test_the_pass_projection_pins_no_pass_for_a_dead_carried_read():
    """``fused_latencies`` over the live reads: the second pass reads the first pass's A only through zero rows, so the
    fused segment runs E, A, C at 0, 1, 1 and the first pass keeps its own timing (the window unit's tree pinned it at
    0, 0: A reading E at gap 0)."""
    from mimarsinan.mapping.latency.chip import PINNED_LATENCIES
    from mimarsinan.mapping.packing.pass_timing import _segment_graph, fused_latencies, project_passes

    first, second = _two_passes()
    reads, _placed, _produced = _segment_graph([first, second])
    assert fused_latencies(reads) == {10: 0, 11: 1, 12: 1}
    project_passes([first, second])
    assert getattr(first.hard_core_mapping, PINNED_LATENCIES, None) is None
    assert getattr(second, "replay_delays", None) is None


def test_a_stage_input_read_only_through_zero_rows_is_no_late_reader():
    """``late_stage_input_readers`` (the stage-input timing rule) reads the predicate: a core above latency 0 whose
    stage-input row is all zero reads no stage input (the window unit's tree named it a late reader)."""
    from mimarsinan.mapping.packing.pass_timing import late_stage_input_readers

    e = _hcore([_inp(0), _inp(1)], LIVE2)
    a = _hcore([_core_src(0, 0), _core_src(0, 1), _inp(0)], [[0.5, 0.2], [0.1, 0.6], [0.0, 0.0]])
    stage = _pass("s", [e, a], [[_placement(10, 2, 2)], [_placement(11, 3, 2)]], [_Slice(99, 0, 2)], [11])
    assert late_stage_input_readers(stage) == []


# ---------------------------------------------------------------------------------------------------------------------
# the static check's model of the schedule
# ---------------------------------------------------------------------------------------------------------------------
def test_the_static_checks_identity_timing_reads_the_live_rows():
    """``identity_timing`` (the static check's model of ChipLatency on the identity program) reads the predicate: E,
    A, C at 0, 1, 1 (the window unit's tree: 1, 1, 2)."""
    from mimarsinan.certification.program_check.unbounded.latency import identity_timing

    latency, _stage = identity_timing(_ir_form())
    assert latency == {0: 0, 1: 1, 2: 1}


# ---------------------------------------------------------------------------------------------------------------------
# the predicate itself
# ---------------------------------------------------------------------------------------------------------------------
def test_the_predicate_reads_off_input_always_on_and_the_weight_row():
    from mimarsinan.mapping.latency.upstream import charged_rows, live_axons

    m = np.array([[0.5, 0.0], [0.0, 0.0], [np.nan, 0.0], [0.2, 0.1], [0.3, 0.0]])
    off = np.array([False, False, False, True, False])
    outside = np.array([False, False, False, False, True])
    assert live_axons(m, off, outside).tolist() == [True, False, True, False, False]
    padded = np.vstack([m, np.zeros((3, 2))])
    assert charged_rows(padded, 5).tolist() == [True, False, True, True, True]
    assert charged_rows(np.vstack([m, [[0.0, 1.0]]]), 5).tolist() == [True] * 5  # a nonzero row past the axons
    assert charged_rows(None, 3).tolist() == [True] * 3
    assert charged_rows(m, 6).tolist() == [True, False, True, True, True, False]  # an axon past the rows
    assert charged_rows(m, 5, columns=np.array([1])).tolist() == [False, False, False, True, False]


def test_an_ir_nodes_compaction_masks_are_read_by_the_predicate():
    """A row whose nonzero weights all sit in columns its compaction drops is dead (the deployed core holds no such
    weight); a pruned row is dead."""
    from mimarsinan.mapping.latency.upstream import ir_live_reads

    node = _node(5, "N", _reads(0) + [IRSource(-3, 0)], [[0.4, 0.0], [0.0, 0.2], [0.1, 0.1]])
    assert ir_live_reads(node).tolist() == [True, True, False]
    node.pruned_row_mask = [False, False, False]
    node.pruned_col_mask = [False, True]
    assert ir_live_reads(node).tolist() == [True, False, False]
    node.pruned_row_mask = [True, False, False]
    node.pruned_col_mask = [False, False]
    assert ir_live_reads(node).tolist() == [False, True, False]


# ---------------------------------------------------------------------------------------------------------------------
# the remaining readers: a delayed carried slice, an in-pass read of the fused segment, the removal's reach, a unit table
# ---------------------------------------------------------------------------------------------------------------------
def test_a_delayed_carried_slice_read_only_through_zero_rows_is_no_window_gap():
    """``carried_window_gaps`` reads the predicate: a slice delivered a cycle late to rows that are all zero is no read
    of it at another gap (a live row of the same slice is)."""
    core = _hcore([_inp(0), _inp(1), _inp(2), _inp(3)], [[0.4], [0.3], [0.0], [0.0]])
    mapping = SimpleNamespace(cores=[core])
    slices = [_Slice(10, 0, 2), _Slice(11, 2, 2)]
    from mimarsinan.mapping.latency.cycle_windows import carried_window_gaps

    assert carried_window_gaps(mapping, slices, {1: -1}) == []
    assert [(g.producer, g.consumer, g.gap) for g in carried_window_gaps(mapping, slices, {0: -1})] == [(10, 0, 0)]


def test_an_in_pass_read_through_zero_rows_moves_no_fused_latency():
    """``fused_latencies`` over the live reads of one pass: C reads E live and A only through zero rows; the pass's own
    timing reads the same reads, so it is not pinned."""
    from mimarsinan.mapping.packing.pass_timing import _segment_graph, fused_latencies

    e = _hcore([_inp(0), _inp(1)], LIVE2)
    a = _hcore([_core_src(0, 0), _core_src(0, 1)], LIVE2)
    c = _hcore([_core_src(0, 0), _core_src(0, 1), _core_src(1, 0), _core_src(1, 1)], [[0.4], [0.3], [0.0], [0.0]])
    stage = _pass("p", [e, a, c], [[_placement(10, 2, 2)], [_placement(11, 2, 2)], [_placement(12, 4, 1)]],
                  [_Slice(99, 0, 2)], [11, 12])
    reads, _placed, _produced = _segment_graph([stage])
    assert fused_latencies(reads) == {10: 0, 11: 1, 12: 1}
    from mimarsinan.mapping.latency.chip import PINNED_LATENCIES
    from mimarsinan.mapping.packing.pass_timing import project_passes

    project_passes([stage])  # the pass's own timing reads the same live reads: nothing to pin
    assert getattr(stage.hard_core_mapping, PINNED_LATENCIES, None) is None


def test_a_core_the_removal_leaves_partly_unread_has_its_unread_neurons_marked():
    """The removal's reach: Q is read only through C's zero rows and goes; P, whose neuron 1 only Q read, keeps its
    neuron 0 (read by R) and has neuron 1 marked compacted."""
    from mimarsinan.mapping.latency.cycle_windows import silence_dead_segment_reads

    p = _node(0, "P", [IRSource(-2, 0), IRSource(-2, 1)], LIVE2)
    r = _node(1, "R", [IRSource(0, 0)], [[0.5]])
    q = _node(2, "Q", [IRSource(0, 1)], [[0.5]])
    c = _node(3, "C", [IRSource(1, 0), IRSource(2, 0)], [[0.4], [0.0]])
    graph = IRGraph(nodes=[p, r, q, c], output_sources=np.asarray([IRSource(1, 0), IRSource(3, 0)], dtype=object))
    dead = silence_dead_segment_reads(graph)
    assert (tuple(dead), dead.neurons) == ((1, 1), 1)
    assert [int(n.id) for n in graph.nodes] == [0, 1, 3]
    assert list(p.pruned_col_mask) == [False, True]


def test_a_unit_table_reads_a_node_outside_the_unit_as_no_member():
    """``ir_live_reads`` with the unit as a table by node id: a read of a node the table does not hold is outside."""
    from mimarsinan.mapping.latency.upstream import ir_live_reads

    node = _node(5, "N", [IRSource(0, 0), IRSource(1, 0)], [[0.4], [0.2]])
    table = np.full(6, -1, dtype=np.int64)
    table[[0, 5]] = [0, 1]
    assert ir_live_reads(node, members=table).tolist() == [True, False]
    assert ir_live_reads(node, members={0, 5}).tolist() == [True, False]


def test_an_off_axon_charges_no_delay_and_an_off_output_reads_none():
    """An off axon delivers nothing even where its row holds a weight (the IR may mark an axon off without zeroing its
    row), and an output source marked off is a direct signal whatever core index it carries."""
    e = _hcore([_inp(0), _inp(1)], LIVE2)
    b = _hcore([SpikeSource(0, 0, is_off=True)], [[0.7]])
    mapping = SimpleNamespace(cores=[e, b], output_sources=[_core_src(1, 0)])
    assert ChipLatency(mapping).calculate() == 0
    assert ChipLatency(mapping).get_delay_for(SpikeSource(0, 0, is_off=True)) == 0


def test_a_cycle_through_dead_rows_only_is_no_dependency():
    """X reads Y only through a zero row and Y reads X: the live core graph is acyclic, so ChipLatency orders and
    schedules it (X at 0, Y at 1) where a wired reading looped."""
    x = _hcore([_inp(0), _core_src(1, 0)], [[0.6], [0.0]])
    y = _hcore([_core_src(0, 0)], [[0.5]])
    mapping = SimpleNamespace(cores=[x, y], output_sources=[_core_src(1, 0)])
    ChipLatency(mapping).calculate()
    assert [c.latency for c in mapping.cores] == [0, 1]


def test_a_neuron_marked_unread_leaves_its_core_a_dead_row_and_the_pass_turns_it_off():
    """The fixed point between the reads turned off and the neurons marked: C's dead read leaves Q unread (removed),
    so X's neuron 1 is unread and marked; X's row on Y's neuron 0 then holds weight only in that marked column, so it
    is dead in turn and turned off, leaving Y's neuron 0 unread and marked."""
    from mimarsinan.mapping.latency.cycle_windows import silence_dead_segment_reads

    y = _node(0, "Y", [IRSource(-2, 0), IRSource(-2, 1)], LIVE2)
    x = _node(1, "X", [IRSource(0, 0), IRSource(0, 1)], [[0.0, 0.7], [0.5, 0.0]])
    r = _node(2, "R", [IRSource(1, 0), IRSource(0, 1)], [[0.4], [0.3]])
    q = _node(3, "Q", [IRSource(1, 1)], [[0.5]])
    c = _node(4, "C", [IRSource(2, 0), IRSource(3, 0)], [[0.4], [0.0]])
    graph = IRGraph(nodes=[y, x, r, q, c], output_sources=np.asarray([IRSource(4, 0)], dtype=object))
    dead = silence_dead_segment_reads(graph)
    assert [int(n.id) for n in graph.nodes] == [0, 1, 2, 4]
    assert list(x.pruned_col_mask) == [False, True] and list(y.pruned_col_mask) == [True, False]
    assert x.input_sources.flatten()[0].is_off() and (tuple(dead), dead.neurons) == ((2, 1), 2)


# ---------------------------------------------------------------------------------------------------------------------
# [r2] m-1: one function for an output source's direct signal (ChipLatency's two sites read it)
# ---------------------------------------------------------------------------------------------------------------------
def test_a_direct_signal_is_off_a_stage_input_or_the_always_on_line_and_nothing_else():
    from mimarsinan.mapping.latency.upstream import direct_signals
    sources = [SpikeSource(0, 0, is_off=True), SpikeSource(-2, 3, is_input=True),
               SpikeSource(-3, 0, is_always_on=True), SpikeSource(1, 2), SpikeSource(-1, 0)]
    assert direct_signals(sources).tolist() == [True, True, True, False, False]
    assert direct_signals([]).tolist() == []


def test_chip_latency_reads_each_output_sources_direct_signal_through_the_one_function(monkeypatch):
    """Both of ChipLatency's direct-signal sites (an output source's delay and the output sources at once) read
    ``direct_signals``: an output source the function calls direct reads no delay, whatever its producer's, and the
    whole program's outputs are read in one call."""
    import mimarsinan.mapping.latency.chip as chip

    mapping = _three_cores()
    assert ChipLatency(mapping).get_delay_for(_core_src(2, 0)) == 2
    monkeypatch.setattr(chip, "direct_signals", lambda sources: np.ones(len(list(sources)), dtype=bool))
    assert ChipLatency(mapping).get_delay_for(_core_src(2, 0)) == 0
    seen = []
    real = __import__("mimarsinan.mapping.latency.upstream", fromlist=["direct_signals"]).direct_signals
    monkeypatch.setattr(chip, "direct_signals", lambda sources: seen.append(list(sources)) or real(sources))
    mapping = _three_cores()
    assert ChipLatency(mapping).calculate() == 2
    assert seen == [list(mapping.output_sources)]


def test_the_static_checks_identity_timing_reads_no_source_through_a_column_its_producer_does_not_keep():
    """C's one charged row reads A's second neuron, which A's compaction does not keep: the identity program turns
    that read off, so C has no source and runs at 0 (counting A would put C at 2)."""
    from mimarsinan.certification.program_check.unbounded.latency import identity_timing

    nodes = [_node(0, "E", [IRSource(-2, 0), IRSource(-2, 1)], LIVE2), _node(1, "A", _reads(0), LIVE2),
             _node(2, "C", [IRSource(1, 1), IRSource(-2, 0)], [[0.7], [0.4]])]
    nodes[1].pruned_row_mask, nodes[1].pruned_col_mask = [False, False], [False, True]
    graph = IRGraph(nodes=nodes, output_sources=np.array([IRSource(1, 0), IRSource(2, 0)], dtype=object))
    latency, _stage = identity_timing(graph)
    assert latency[2] == 0 and latency[1] == 1
