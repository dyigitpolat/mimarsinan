"""One hand-built witness per hypothesis: a difference with it off, equality (or a named refusal) with it on.

Each witness is a minimal program, its inputs and one explicit packing (fragments, passes, pins, slots
and axon orders chosen by hand, no RNG). ``run_witness(label)`` returns the divergence of the off arm
(None would be a witness without teeth) and the on arm's outcome: ``equal`` (the same artifact runs
exactly with the switch on) or ``refused:<label>`` (the artifact check of the hypothesis refuses it).
Relation ``network`` witnesses (H2, H3, H4) compare the reference with the integer network N instead.
Values were derived by hand from the kernels; the unit tests pin each one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import kernels as K
from .hypotheses import BY_LABEL, admit_network, validate
from .machine import run_packed
from .packer import Fragment, HardCore, _place, core_scalar, core_timing, finish
from .program import CoreType, Declaration, Node, Program
from .reference import run_network, run_reference
from .schedule import Packed, identity_pack
from .screen import divergence


def _prog(decl: Declaration, spec: Sequence[tuple]) -> Program:
    nodes: List[Node] = []
    for row in spec:
        if row[0] == "input":
            nodes.append(Node(len(nodes), "input", row[1]))
        elif row[0] == "host":
            nodes.append(Node(len(nodes), "host", nodes[row[2][0]].size, tuple(row[2]), row[1]))
        else:
            _, srcs, w, b, theta = row[:5]
            w = np.array(w, np.int64)
            nodes.append(Node(len(nodes), "neural", w.shape[1], tuple(srcs), weights=w, bias=np.array(b, np.int64),
                              theta=theta, shift=row[5] if len(row) > 5 else 0))
    return Program(decl, nodes)


def _core(prog: Program, hid: int, frags: Sequence[Fragment], axons: Optional[List[int]] = None) -> HardCore:
    lead = frags[0]
    core = HardCore(hid, 0, prog.eta()[lead.layer], core_timing(prog, lead.layer), core_scalar(prog, lead.layer),
                    not lead.primary)
    for f in frags:
        _place(core, prog, f)
    if axons is not None:
        core.axons = list(axons)
    return core


def _full(prog: Program, layer: int) -> Fragment:
    fan = tuple(int(w) for w in prog.fanin(layer))
    return Fragment(layer, tuple(range(prog.nodes[layer].size)), fan, -1, True, 0)


def _packed(prog, cores, passes, pinned=None, slot_of=None, K=1) -> Packed:
    finish(prog, cores, None)
    if pinned is None:
        pinned = {c.hid: c.hid for c in cores}
    return Packed(cores, [(prog.eta()[cores[h[0]].slots[0][0]], tuple(h)) for h in passes], pinned, slot_of or {},
                  K, 8)


def _decl(sem: str, **kw) -> Declaration:
    return Declaration(sem, core_types=(CoreType(8, 8, 8),), **kw)


def _cases() -> Dict[str, Tuple[Program, Dict[int, np.ndarray], Any]]:
    out: Dict[str, Tuple[Program, Dict[int, np.ndarray], Any]] = {}
    # H1-theta: theta 1 and 2 on one core that keeps theta 1: 1 spike against 0.
    p = _prog(_decl("sync_lif", T=1), [("input", 1), ("neural", [0], [[1]], [0], 1), ("neural", [0], [[1]], [0], 2)])
    out["H1-theta"] = (p, {0: np.array([[1]])}, lambda p: _packed(p, [_core(p, 0, [_full(p, 1), _full(p, 2)])], [[0]]))
    # H1-latency: B (latency 1) on C's core (latency 0) reads A one cycle early: 1 spike against 2.
    p = _prog(_decl("streamed_lif", T=2), [("input", 1), ("neural", [0], [[1]], [0], 1),
                                           ("neural", [1], [[1]], [0], 1), ("neural", [0], [[1]], [0], 1)])
    out["H1-latency"] = (p, {0: np.array([[2]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 1)]), _core(p, 1, [_full(p, 3), _full(p, 2)])], [[0, 1]]))
    # H5-collapse: P fires late ([0, 0, 1, 0]); its count re-encoded arrives at cycle 0: Q 2 against 4.
    p = _prog(_decl("streamed_lif", T=4), [("input", 1), ("neural", [0], [[1]], [0], 2),
                                           ("neural", [1], [[4]], [0], 1)])
    out["H5-collapse"] = (p, {0: np.array([[2]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 1)]), _core(p, 1, [_full(p, 2)])], [[0], [1]]))
    # H5-skew: P is raised (ALAP) to latency 1; Q (latency 1, delay -1) moves to its own pass, which
    # times Q at 0 and reads P's raster without the delay: Q 2 against 1.
    p = _prog(_decl("streamed_lif", T=2), [("input", 1), ("neural", [0], [[1]], [0], 1),
                                           ("neural", [1], [[2]], [0], 1), ("neural", [1], [[1]], [0], 1),
                                           ("neural", [1, 3], [[1], [1]], [0], 1)])
    out["H5-skew"] = (p, {0: np.array([[1]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 1)]), _core(p, 1, [_full(p, 3)]), _core(p, 2, [_full(p, 4)]),
            _core(p, 3, [_full(p, 2)])], [[0, 1, 2], [3]]))
    # R2 (pins kept, delay dropped): the H5-skew vehicle; Q at its fused latency reads P's raster one cycle
    # early: 2 against 1. R1 (own timing, delay kept): P's deepest consumer S sits in the later pass, so
    # pass 1 does not raise P; Q then reads both of P's spikes: 2 against 1.
    out["R2"] = (out["H5-skew"][0], out["H5-skew"][1], out["H5-skew"][2])
    out["R1"] = (out["H5-skew"][0], {0: np.array([[2]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 1)]), _core(p, 1, [_full(p, 2)]), _core(p, 2, [_full(p, 3)]),
            _core(p, 3, [_full(p, 4)])], [[0, 1], [2, 3]]))
    # H6: acc 3 + 3 = 6 fused in two parts, each clipped to 2 bits ([-2, 1]): 6 against 2.
    p = _prog(_decl("value", act_bits=4, partial_sum_bits=2), [("input", 2), ("neural", [0], [[1], [1]], [0], 1)])
    out["H6"] = (p, {0: np.array([[3, 3]])}, lambda p: _packed(
        p, [_core(p, 0, [Fragment(1, (0,), (0,), 0, True, 0)]), _core(p, 1, [Fragment(1, (0,), (1,), 0, False, 0)])],
        [[0, 1]]))
    # H7: theta 5, w = [+3, -3], cycle 1 events a and b; rank order fires once (6 >= 5), slot order b
    # first never fires: 1 spike adjacent, 0 interleaved (D17's ODIN witness on a per-event zero-reset core).
    p = _prog(_decl("streamed_lif", T=2, granularity="per_event", reset="zero"),
              [("input", 1), ("input", 1), ("neural", [1], [[1]], [0], 2), ("neural", [0, 2], [[3], [-3]], [0], 5)])
    out["H7"] = (p, {0: np.array([[2]]), 1: np.array([[2]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 2)]), _core(p, 1, [_full(p, 3)], axons=[2, 0])], [[0, 1]]))
    # H8: (2**24 + 1) + 1 in float32 stays 2**24: exact 2**24 + 2 against 2**24.
    p = _prog(_decl("value", act_bits=26, accumulator="fp32"), [("input", 3), ("neural", [0], [[1], [1], [1]], [0], 1)])
    out["H8"] = (p, {0: np.array([[1 << 24, 1, 1]])}, lambda p: _packed(p, [_core(p, 0, [_full(p, 1)])], [[0]]))
    # H9: B's pass runs before A's, so B reads A's carry before it exists.
    p = _prog(_decl("sync_lif", T=2), [("input", 1), ("neural", [0], [[1]], [0], 1), ("neural", [1], [[1]], [0], 1)])
    out["H9"] = (p, {0: np.array([[2]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 1)]), _core(p, 1, [_full(p, 2)])], [[1], [0]]))
    # H10: B is written into A's pinned slot, so the second batch runs B in A's pass and A never again.
    out["H10"] = (p, {0: np.array([[2], [2]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 1)]), _core(p, 1, [_full(p, 2)])], [[0], [1]], pinned={0: 0}, slot_of={1: 0}))
    # R3: N (segment 1) reads P (segment 0) directly; the unscoped store hands it P's late raster
    # instead of the even re-encode of P's count: 2 against 4.
    p = _prog(_decl("streamed_lif", T=4), [("input", 1), ("neural", [0], [[1]], [0], 2), ("host", "relay", [1]),
                                           ("neural", [1, 2], [[4], [0]], [0], 1)])
    out["R3"] = (p, {0: np.array([[2]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 1)]), _core(p, 1, [_full(p, 3)])], [[0], [1]]))
    # R6: P is raised (ALAP) to latency 1 beside its shallow consumer Q; ordered by that latency with
    # Q's core first, Q reads P before P has run: 0 against 1.
    p = _prog(_decl("sync_lif", T=1), [("input", 1), ("neural", [0], [[1]], [0], 1),
                                       ("neural", [1], [[1]], [0], 1), ("neural", [1], [[1]], [0], 1),
                                       ("neural", [1, 3], [[1], [1]], [0], 1)])
    out["R6"] = (p, {0: np.array([[1]])}, lambda p: _packed(
        p, [_core(p, 0, [_full(p, 2)]), _core(p, 1, [_full(p, 1)]), _core(p, 2, [_full(p, 3)]),
            _core(p, 3, [_full(p, 4)])], [[0, 1, 2, 3]]))
    # Relation network: H2 (P raised to Q's latency: Q receives 1 of P's 2 spikes), H3 (an early fire on
    # +3 - 1 that the terminal charge 1 does not pay for), H4 (a burst of 4 at Q's last cycle).
    sl = dict(T=2)
    out["H2"] = (_prog(_decl("streamed_lif", **sl), [("input", 1), ("neural", [0], [[1]], [0], 1),
                                                       ("neural", [1], [[1]], [0], 1),
                                                       ("neural", [1, 2], [[1], [1]], [0], 2)]),
                 {0: np.array([[2]])}, None)
    out["H3"] = (_prog(_decl("streamed_lif", T=3), [("input", 1), ("input", 1), ("neural", [0, 1], [[3], [-1]], [0], 2)]),
                 {0: np.array([[1]]), 1: np.array([[2]])}, None)
    out["H4"] = (_prog(_decl("streamed_lif", **sl), [("input", 1), ("neural", [0], [[1]], [0], 2),
                                                       ("neural", [1], [[4]], [0], 1)]),
                 {0: np.array([[2]])}, None)
    return out


def run_witness(label: str) -> Dict[str, Any]:
    prog, inputs, make = _cases()[label]
    B = next(iter(inputs.values())).shape[0]
    ident, _ = run_packed(prog, identity_pack(prog, B), inputs, exact=True)
    ref, diag = run_reference(prog, inputs)
    assert divergence(prog, ref, ident) is None
    if make is None:
        on_ok, on_refused = admit_network(prog, diag, frozenset())
        off_ok, _ = admit_network(prog, diag, frozenset({label}))
        net = run_network(prog, inputs)
        assert net is not None
        return {"label": label, "relation": "network", "off": divergence(prog, net, ref) if off_ok else "not admitted",
                "on": "refused:" + "+".join(on_refused) if not on_ok else "equal"}
    packed = make(prog)
    refused = validate(prog, packed, frozenset())
    out_off, _ = run_packed(prog, packed, inputs, frozenset({label}))
    on = "refused:" + "+".join(refused) if refused else (
        "equal" if divergence(prog, ident, run_packed(prog, packed, inputs)[0]) is None else "DIFFERS")
    return {"label": label, "relation": BY_LABEL[label].relation, "off": divergence(prog, ident, out_off), "on": on}


def all_witnesses() -> List[Dict[str, Any]]:
    return [run_witness(label) for label in BY_LABEL]


def foreign_event_witness() -> Dict[str, int]:
    """Finding: a core-wide per-event loop (fire checks at foreign axons' events) breaks coalescing.

    Neuron N1 (theta 2, zero reset) fires on +3 at cycle 0 and keeps a refractory +3, so it enters cycle 1
    above threshold; its own cycle-1 event is -2. Alone (the identity core) its first cycle-1 check comes
    after the -2 (m = 1, no spike): 1 spike. Coalesced with a foreign axon of lower rank that spikes at
    cycle 1, a core-wide loop checks N1 at that foreign event first (m = 3, a spike) and then applies the
    -2: 2 spikes, in canonical rank order. Checking only at the neuron's own events gives 1 in both.
    """
    from .program import Declaration as _D

    d = _D("streamed_lif", T=2, granularity="per_event", reset="zero")
    own = np.array([[3], [3], [-2]], np.int64)  # axons a1, a2, a3 of N1
    with_foreign = np.array([[3], [3], [0], [-2]], np.int64)  # a foreign axon f ranked before a3
    x0 = np.array([[1, 1, 0]]), np.array([[1, 1, 0, 0]])
    x1 = np.array([[0, 0, 1]]), np.array([[0, 0, 1, 1]])
    counts = {}
    for name, w, xs, own_only in (("identity", own, (x0[0], x1[0]), True), ("coalesced_own", with_foreign,
                                  (x0[1], x1[1]), True), ("coalesced_core_wide", with_foreign, (x0[1], x1[1]), False)):
        m, total = np.zeros((1, 1), np.int64), 0
        for x in xs:
            m, f = K.event_fold(m, x, w, np.zeros(1, np.int64), 2, d, range(w.shape[0]), own_only)
            total += int(f.sum())
        counts[name] = total
    return counts
