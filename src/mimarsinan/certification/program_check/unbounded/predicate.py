"""[E1 F1] The IR at the declared geometry against the IR at unbounded geometry, per IR neuron (condition
``ir-tiling``): the static form of Theorem 1's clause that core geometry, the IR's tiling, is cost-only.

The reference must be an IR at unbounded geometry (``reference_geometry``, ``reference``). Both IRs are keyed in one
key space and read through the compaction the identity builder applies (``keys``). The predicate holds when every
neuron of the IR at unbounded geometry is held once by the tiled IR (``missing_neuron``, ``extra_neuron``,
``duplicate_neuron``), each with its whole fan-in in one node (``partial_sum_group``), equal to the unbounded neuron's
summed per source identity (``fan_in``); when each neuron keeps the unbounded neuron's hardware bias, threshold,
scales, the boundary maps the encode and decode read, boundary grid, readout law (as the function it declares,
``laws``) and activation kinds (``bias``, ``threshold``, ``neuron_scale``, ``boundary_grid``, ``law``, ``activation``);
under timed semantics, when each IR's identity program runs it at the same latency in the same stage
(``executed_latency``, from each IR's core graph, ``latency``; the derivation held to the check's identity program,
``timing_derivation``); when every host op is the same function of the same neurons (``host_op``) and the network
reads the same neurons (``program_outputs``); and, under a phase-dithered encode, when the channel table each IR's
identity program carries keys every neuron on its channel at unbounded geometry (``phase_channel``). A neural node of
the tiled IR with no perceptron identity fails by name (``unkeyed_neuron``). The step from two IRs equal in all of this
to two identity programs computing the same outputs is the ``neuron-keyed identity builder`` clause the verdict names."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.certification.program_check.report import ASSUMED_CONDITION, ConditionResult, Violation
from mimarsinan.certification.program_check.tiles import _same_value, _values_equal
from mimarsinan.certification.program_check.unbounded.keys import (
    WHOLE_FAN_IN_ROLES, KeyedIR, KeySpace, kept, keyed, node_width,
)
from mimarsinan.certification.program_check.unbounded.latency import IdentityFacts, facts_of, identity_timing
from mimarsinan.certification.program_check.unbounded.node_values import NeuronPair
from mimarsinan.mapping.support.value_layouts import neuron_entries
from mimarsinan.certification.program_check.unbounded.laws import column_bounds, output_levels, same_function
from mimarsinan.certification.program_check.unbounded.reference import check_reference
from mimarsinan.certification.program_check.unit_forms.lanes import laws_equal as _laws_equal
from mimarsinan.certification.program_check.view import Canonical, first_difference, same
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.phase_channels import node_phase_channels
from mimarsinan.mapping.support.activation_scales import (
    compute_node_buffer_scales, compute_node_input_scales, compute_node_output_scales,
)

#: Per-neuron values a core's executor reads, as node attributes (scales) and as the boundary maps every program built
#: from the IR carries (``node_activation_scales`` and its kin, computed by these functions), each read by the layout
#: its node declares (``NeuralCore.VALUE_LAYOUTS``).
NEURON_SCALES = ("activation_scale", "parameter_scale", "input_activation_scale", "bias_scale")
NODE_MAPS = (("node_activation_scales", compute_node_output_scales),
             ("node_input_activation_scales", compute_node_input_scales),
             ("node_buffer_scales", compute_node_buffer_scales))
#: Per-node kinds the identity builder and the executors read (the IR ``latency`` attribute is none of them: the
#: executors run a core at its executed latency, ``executed_latency``).
LABELS = {"activation_type": "activation", "normalization_type": "activation"}


def _take(can: Canonical, columns: np.ndarray) -> Canonical:
    weights = can.weights[:, columns]
    live = np.any(weights != 0, axis=1)
    return Canonical(can.codes[live], weights[live])


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _values_equal(a, b)
    return _same_value(a, b)


def _identity_remaps(graph) -> Dict[int, Dict[int, int]]:
    """The reindex map soft-core compaction gives every neural node (``compact_soft_core_mapping``: its kept
    columns renumbered in order), as the identity builder passes them to ``node_phase_channels``."""
    return {int(n.id): {int(old): new for new, old in enumerate(kept(n)[1].tolist())} for n in graph.get_neural_cores()}


def _entry(table, nid: int, size: int) -> Optional[np.ndarray]:
    entry = table.get(int(nid))
    if entry is None:
        return None
    if np.ndim(entry) == 0:
        return int(entry) + np.arange(size, dtype=np.int64)
    return np.asarray(entry, dtype=np.int64)[:size] if len(entry) >= size else None


class _Compare:
    def __init__(self, t: KeyedIR, u: KeyedIR, result: ConditionResult) -> None:
        self.t, self.u, self.result = t, u, result
        self.maps = {name: (fn(t.graph), fn(u.graph)) for name, fn in NODE_MAPS}
        self.levels: Optional[Tuple[Dict[int, float], Dict[int, float]]] = None
        self.paired: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {}

    def fail(self, rule: str, where: str, detail: str) -> None:
        self.result.add(Violation("ir-tiling", rule, where, detail))

    def neurons(self) -> None:
        t, u = self.t, self.u
        for nid in t.unkeyed:
            self.fail("unkeyed_neuron", f"node {nid}", "a neural node with no perceptron identity (or a pre-pruning "
                      "mask that is not its columns): no neuron of the IR at unbounded geometry is its")
        for code, held, other in t.duplicated:
            self.fail("duplicate_neuron", f"nodes {held[0]} and {other[0]}",
                      f"both hold {t.space.text(code) if code >= 0 else 'one host-op name'}")
        for nid, node in t.by_id.items():
            role = (getattr(node, "coalescing_role", None), getattr(node, "psum_role", None))
            if isinstance(node, NeuralCore) and (role[0] not in WHOLE_FAN_IN_ROLES or role[1] not in (None,)):
                self.fail("partial_sum_group", f"node {nid}", f"roles {role}: a share of a neuron's fan-in the "
                          "predicate does not reassemble (no builder of this tree emits one)")
        missing, extra = sorted(set(u.neurons) - set(t.neurons)), sorted(set(t.neurons) - set(u.neurons))
        for rule, codes in (("missing_neuron", missing), ("extra_neuron", extra)):
            if codes:
                self.fail(rule, "IR", f"{len(codes)} neuron(s) one IR holds and the other does not (or compacts "
                          f"away), first {t.space.text(codes[0])}")

    def pairs(self) -> Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]:
        """Per unbounded node, the tiled nodes holding its kept neurons with their columns in each (before
        compaction)."""
        out: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
        for nid, columns in self.t.cols.items():
            groups: Dict[int, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
            codes = self.t.codes[nid]
            for column in columns.tolist():
                home = self.u.neurons.get(int(codes[column]))
                if home is not None:
                    groups[home[0]][0].append(column)
                    groups[home[0]][1].append(home[1])
            for unode, (mine, theirs) in groups.items():
                out[unode].append((nid, np.asarray(mine, dtype=np.int64), np.asarray(theirs, dtype=np.int64)))
        self.paired = out
        return out

    def tiles(self) -> None:
        for unode, tiles in self.pairs().items():
            theirs = self.u.by_id[unode]
            can_u = self.u.block(theirs)
            width_u = node_width(theirs, self.u.graph)
            for nid, mine_cols, their_cols in tiles:
                mine = self.t.by_id[nid]
                self.result.checked += len(mine_cols)
                a, b = _take(self.t.block(mine), mine_cols), _take(can_u, their_cols)
                if not same(a, b):
                    diff = first_difference(a, b)
                    detail = "" if diff is None else (f"source {self.t.space.text(diff[0])} column {diff[1]}: "
                                                      f"{diff[2]} against {diff[3]}")
                    self.fail("fan_in", self.where(nid, mine_cols, unode),
                              f"the tile's fan-in is not the unbounded neuron's: {detail}")
                self.values(mine, theirs, mine_cols, their_cols, node_width(mine, self.t.graph), width_u,
                            self.where(nid, mine_cols, unode))

    @staticmethod
    def where(nid: int, columns: np.ndarray, unode: int) -> str:
        return f"node {nid} columns {int(columns[0])}-{int(columns[-1])} (unbounded node {unode})"

    def values(self, mine, theirs, mine_cols, their_cols, width_t: int, width_u: int, where: str) -> None:
        pair = NeuronPair((self.t, mine, mine_cols, width_t), (self.u, theirs, their_cols, width_u))

        def check(rule: str, name: str, a: Any, b: Any, detail: str) -> None:  # [U-PCH3] by the node's declaration
            refused = pair.differs(name, a, b)
            if refused is not None:
                self.fail(rule, where, refused or detail)

        check("bias", "hardware_bias", getattr(mine, "hardware_bias", None), getattr(theirs, "hardware_bias", None),
              "the hardware bias differs from the unbounded neuron's")
        check("threshold", "threshold", mine.threshold, theirs.threshold,
              f"threshold {mine.threshold} against {theirs.threshold}")
        for name in NEURON_SCALES:
            check("neuron_scale", name, getattr(mine, name, None), getattr(theirs, name, None),
                  f"{name} differs from the unbounded neuron's")
        for name, (table_t, table_u) in self.maps.items():
            check("neuron_scale", name, table_t.get(int(mine.id)), table_u.get(int(theirs.id)),
                  f"the programs' {name} differ from the unbounded neuron's")
        if not _same_value(getattr(mine, "boundary_grid", None), getattr(theirs, "boundary_grid", None)):
            self.fail("boundary_grid", where, "the boundary grid differs from the unbounded neuron's")
        if not self.same_law(mine, theirs, mine_cols, their_cols):
            self.fail("law", where, "the readout law declares another function than the unbounded neuron's")
        for name, rule in LABELS.items():
            if getattr(mine, name, None) != getattr(theirs, name, None):
                self.fail(rule, where, f"{name} {getattr(mine, name, None)} against {getattr(theirs, name, None)}")

    def same_law(self, mine, theirs, mine_cols, their_cols) -> bool:
        laws = (getattr(mine, "readout_law", None), getattr(theirs, "readout_law", None))
        if laws[0] is None or laws[1] is None:
            return laws[0] is laws[1]
        a, b = (neuron_entries(node, "readout_law", cols, width=len(cols), value=law)  # [U-PCH3] its declaration
                for node, cols, law in ((mine, mine_cols, laws[0]), (theirs, their_cols, laws[1])))
        if _laws_equal(a, b):
            return True
        if self.levels is None:
            self.levels = (output_levels(self.t), output_levels(self.u))
        return same_function(a, b, column_bounds(self.t, mine, mine_cols, self.levels[0]),
                             column_bounds(self.u, theirs, their_cols, self.levels[1]))

    def timing(self, facts: IdentityFacts, *, timed: bool) -> None:
        """Under timed semantics (the declaration's, never the facts' content): each paired tile runs where its
        unbounded node runs (``executed_latency``), both read off the IRs' core graphs, the tiled IR's derivation held
        to the check's identity program first; facts with no latency under a timed declaration fail by name."""
        if not timed:
            return
        if facts.latency is None:
            self.fail("timing_derivation", "IR", "a timed declaration, but the identity facts carry no executed "
                      "latency: the timing rules would compare nothing")
            return
        lat_t, stage_t = identity_timing(self.t.graph)
        lat_u, stage_u = identity_timing(self.u.graph)
        for nid in sorted(set(lat_t) | set(facts.latency)):
            got, want = (lat_t.get(nid), stage_t.get(nid)), (facts.latency.get(nid), facts.stage.get(nid))
            if got != want:
                self.fail("timing_derivation", f"node {nid}", f"the core graph runs it at (latency, stage) {got}, "
                          f"the identity program at {want}: the derivation the unbounded side rests on departs")
        for unode, tiles in self.paired.items():
            for nid, columns, _theirs in tiles:
                self.result.checked += 1
                mine, theirs = (lat_t.get(nid), stage_t.get(nid)), (lat_u.get(unode), stage_u.get(unode))
                if mine != theirs:
                    self.fail("executed_latency", self.where(nid, columns, unode),
                              f"the identity program runs the tile at (latency, stage) {mine}, the identity program at "
                              f"unbounded geometry runs its neurons at {theirs}")

    def hosts(self) -> None:
        t, u = self.t, self.u
        for name in sorted(set(t.hosts) | set(u.hosts)):
            mine, theirs = t.hosts.get(name), u.hosts.get(name)
            self.result.checked += 1
            where = f"host op {name}"
            if mine is None or theirs is None:
                self.fail("host_op", where, "held by one IR only")
                continue
            src_t, src_u = t.source_codes(t.node_sources(mine)), u.source_codes(u.node_sources(theirs))
            params = set(mine.params) == set(theirs.params) and all(_same_value(mine.params[k], theirs.params[k])
                                                                    for k in mine.params)
            shapes = (mine.input_shape, mine.output_shape) == (theirs.input_shape, theirs.output_shape)
            shift = _same(getattr(mine, "_negative_shift", None), getattr(theirs, "_negative_shift", None))
            if not (np.array_equal(src_t, src_u) and params and shapes and shift and mine.op_type == theirs.op_type):
                self.fail("host_op", where, "its input neurons, module, parameters, shapes or boundary shift differ "
                          "from the unbounded IR's host op")

    def outputs(self) -> None:
        self.result.checked += 1
        flat = [np.asarray(k.graph.output_sources, dtype=object).flatten().tolist() for k in (self.t, self.u)]
        if not np.array_equal(self.t.source_codes(flat[0]), self.u.source_codes(flat[1])):
            self.fail("program_outputs", "IR", "the network reads other neurons than the IR at unbounded geometry")

    def phases(self, facts: IdentityFacts) -> None:
        """The channel table each identity program carries keys every kept neuron (program output ``k``) on its
        channel: the tiled IR's as the check's identity program carries it, the unbounded IR's as the identity builder
        builds it (``node_phase_channels`` with compaction's reindex maps), that derivation held to the former."""
        derived_t = node_phase_channels(self.t.graph, _identity_remaps(self.t.graph))
        for k, table in ((self.t, facts.channels), (self.u, node_phase_channels(self.u.graph,
                                                                                 _identity_remaps(self.u.graph)))):
            for nid, columns in k.cols.items():
                if nid in k.unit_keyed:   # [E6] a lane core or a relay holds no perceptron channel to key
                    continue
                want = k.channels[nid][columns]
                got = _entry(table, nid, len(want))
                if got is None or not np.array_equal(got, want):
                    self.fail("phase_channel", f"node {nid} of the {k.label} IR", f"its identity program rotates "
                              f"outputs 0..{len(want) - 1} by channels {None if got is None else got[:6].tolist()}, "
                              f"its neurons' channels are {want[:6].tolist()}")
        for nid in self.t.cols:
            if nid in self.t.unit_keyed:
                continue
            a, b = _entry(derived_t, nid, len(self.t.cols[nid])), _entry(facts.channels, nid, len(self.t.cols[nid]))
            if a is None or b is None or not np.array_equal(a, b):
                if a is None and b is None:
                    continue
                self.fail("phase_channel", f"node {nid} of the deployed IR", "the table the identity builder would "
                          "carry departs from the one the check's identity program carries")


def check_ir_tiling(ir_graph: Any, unbounded_ir: Optional[Any], *, declaration: Any, result: ConditionResult,
                    identity_facts: Optional[IdentityFacts] = None) -> None:
    """Fill ``result`` (``ir-tiling``): ``assumed`` without an IR at unbounded geometry, else every rule above.
    ``identity_facts``: what the check's identity program of ``ir_graph`` executes (built here when not given)."""
    if unbounded_ir is None:
        result.status = ASSUMED_CONDITION
        result.note = "no IR at unbounded geometry given: the IR tiling stays a reference assumption"
        return
    if identity_facts is None:
        identity = build_identity_hybrid_mapping(ir_graph=ir_graph, retimed_level_stages=declaration.windowed)
        identity_facts = facts_of(identity, timed=declaration.timed)
    space = KeySpace()
    compare = _Compare(keyed(ir_graph, space, "deployed"), keyed(unbounded_ir, space, "unbounded"), result)
    result.checked += check_reference(compare.u, compare.fail)
    compare.neurons()
    compare.tiles()
    compare.timing(identity_facts, timed=declaration.timed)
    compare.hosts()
    compare.outputs()
    if declaration.dithered_encode:
        compare.phases(identity_facts)
    if not result.violation_count:
        kinds = ", ".join(["neuron"] + ["timing"] * bool(declaration.timed) + ["host-op"])
        held = ", ".join(["fan-in", "function"] + ["executed latency"] * bool(declaration.timed)
                         + ["channel"] * bool(declaration.dithered_encode))
        result.note = (f"{result.checked} {kinds} checks: the tiled IR holds every neuron of the IR at unbounded "
                       f"geometry once, with its {held}")
