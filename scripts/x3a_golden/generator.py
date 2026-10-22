"""The seeded X3a model-level generator: declarations, integer DAGs with motifs, inputs and packing knobs.

Ranges (plan section 10.2): DAG depth at most 8 and width at most 32 with chain, skip, fork, concat,
host-op, residual and cross-segment motifs; integer weights in [-8, 8], theta in [1, 16], bias in [-4, 4]
(value: bias in [-4, 4] times 2**shift); regime, comparator, reset, granularity, membrane, T in 1-8, 1-3
core types with axons and neurons in [4, 64], up to 32 cores, pass budgets in {1..8, none}.

An off arm draws from the domain where its hypothesis can bite (``DOMAINS``: the regimes of its
hypothesis and a targeted motif or knob), recorded with every instance; the all-on arm draws from the
whole domain. Every draw comes from one ``random.Random`` seeded by the instance's (arm, seed) string,
so an instance is a pure function of its seed.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from .packer import DEFAULT_KNOBS
from .program import SEMANTICS, STREAMED_LIF, VALUE, CoreType, Declaration, Node, Program

_STREAMED_CHAIN = {"semantics": (STREAMED_LIF,), "granularity": "per_cycle", "reset": "subtract", "small": True}
#: Charges that never fall (no overshoot) and never pass two thresholds in a cycle (no burst debt).
_SAFE = {"weights": (0, 2), "bias": (0, 1), "theta": "cover"}
DOMAINS: Dict[str, Dict[str, Any]] = {
    "all_on": {},
    "H1-theta": {"knobs": {"coalesce": 0.95, "prefer_bad_theta": 0.9}, "parallel": 0.9, "small": True, "shared": 0.0},
    "H1-latency": {"knobs": {"coalesce": 0.95, "prefer_bad_latency": 0.9}, "small": True, "shared": 1.0},
    "H2": dict(_STREAMED_CHAIN, skip=0.8, fork=0.5, **_SAFE),
    "H3": dict(_STREAMED_CHAIN, skip=0.0, cross=0.0, weights=(-2, 2), bias=(-1, 1), theta="cover"),
    "H4": dict(_STREAMED_CHAIN, skip=0.0, cross=0.0, weights=(0, 8), bias=(0, 2), theta=(1, 3), min_depth=3),
    "H5-collapse": {"semantics": (STREAMED_LIF, "sync_ttfs"), "knobs": {"cut": 0.9}, "granularity": "per_cycle"},
    "H5-skew": {"semantics": (STREAMED_LIF,), "knobs": {"cut": 0.9}, "fork": 0.9, "skip": 0.8, "host": 0.0,
                "parallel": 0.0, "min_depth": 3},
    "H6": {"knobs": {"fuse": 1.0}, "granularity": "per_cycle", "weights": (1, 8), "partial_bits": 2},
    "H7": {"semantics": (STREAMED_LIF,), "granularity": "per_event", "signed": True, "small": True},
    "H8": dict(skip=0.0, cross=0.0, range=True, semantics=(VALUE, STREAMED_LIF, "sync_lif", "sync_ttfs"), **_SAFE),
    "H9": {"knobs": {"cut": 0.9}, "min_depth": 2},
    "H10": {"knobs": {"pin": 0.8, "cut": 0.9, "prefer_pinned_slot": 0.9}, "samples": 6},
    "R1": {"semantics": (STREAMED_LIF,), "knobs": {"cut": 0.9}, "fork": 0.9, "skip": 0.8, "host": 0.0,
           "parallel": 0.0, "min_depth": 3},
    "R2": {"semantics": (STREAMED_LIF,), "knobs": {"cut": 0.9}, "fork": 0.9, "skip": 0.8, "host": 0.0,
           "parallel": 0.0, "min_depth": 3},
    "R3": {"semantics": (STREAMED_LIF,), "granularity": "per_cycle", "knobs": {"cut": 0.9}, "cross": 0.95},
    "R6": {"semantics": ("sync_lif", "sync_ttfs", VALUE), "skip": 0.8, "fork": 0.9, "host": 0.0, "parallel": 0.0,
           "knobs": {"cut": 0.1}, "min_depth": 3},
}
#: Negative controls ("LABEL@variant": LABEL off, drawn where the box predicts it is not needed).
CONTROLS: Dict[str, Dict[str, Any]] = {
    "H5-collapse@windowed": {"semantics": ("sync_lif", VALUE), "knobs": {"cut": 0.9}},
    "H5-skew@windowed": {"semantics": ("sync_lif", "sync_ttfs", VALUE), "knobs": {"cut": 0.9}, "fork": 0.9,
                         "skip": 0.8, "host": 0.0, "parallel": 0.0, "min_depth": 3},
    "R3@windowed": {"semantics": ("sync_lif", "sync_ttfs", VALUE), "knobs": {"cut": 0.9}, "cross": 0.95},
}
DOMAINS.update(CONTROLS)


def _declaration(rng: random.Random, dom: Dict[str, Any]) -> Declaration:
    sem = rng.choice(dom.get("semantics", SEMANTICS))
    gran = dom.get("granularity") or ("per_event" if sem == STREAMED_LIF and rng.random() < 0.3 else "per_cycle")
    reset = dom.get("reset") or ("subtract" if rng.random() < 0.75 else "zero")
    bits = rng.choice([None, None, None, 16, 8, 4])
    act_bits, acc = rng.randint(2, 8), "int64"
    if sem == VALUE:
        acc = rng.choice(["int64", "fp32"])
    if dom.get("range"):
        if sem == VALUE:
            act_bits, acc = 25, "fp32"
        else:
            bits, gran, reset = rng.choice([3, 4, 5]), "per_cycle", "subtract"
    n_types = rng.randint(1, 3)
    budget = rng.randint(n_types, 32)
    counts = [1] * n_types
    for _ in range(rng.randint(0, budget - n_types)):
        counts[rng.randrange(n_types)] += 1
    small = rng.random() < 0.6
    types = tuple(CoreType(rng.randint(4, 16 if small else 64), rng.randint(4, 16 if small else 64), c)
                  for c in counts)
    return Declaration(sem, T=rng.randint(1, 8), comparator=rng.choice(["<=", "<"]), reset=reset, granularity=gran,
                       membrane_bits=bits, act_bits=act_bits, core_types=types,
                       pass_budget=rng.choice([None, None, rng.randint(1, 8)]), accumulator=acc,
                       partial_sum_bits=dom.get("partial_bits") or rng.randint(3, 6))


def _longest(nodes: List[Node]) -> List[int]:
    """Neural layers on the longest path ending at each node."""
    out: List[int] = []
    for node in nodes:
        out.append(max((out[s] for s in node.sources), default=0) + (node.kind == "neural"))
    return out


def prog_fanins(nodes: List[Node]) -> List[int]:
    return [sum(nodes[s].size for s in n.sources) for n in nodes if n.kind == "neural"]


def _fit(decl: Declaration, fanins: List[int]) -> Declaration:
    """Raise one core type's count (or, for per-event cores, axons) inside the declared ranges so the widest
    fan-in fits one pass; a declaration the ranges cannot fit is kept and refused by name at packing."""
    widest = max(fanins, default=1)
    types = list(decl.core_types)
    spare = 32 - sum(ct.count for ct in types)
    if decl.granularity == "per_event":
        k = max(range(len(types)), key=lambda i: types[i].axons)
        types[k] = CoreType(min(64, max(types[k].axons, widest)), types[k].neurons, types[k].count)
    elif all(-(-widest // ct.axons) > ct.count for ct in types):
        k = max(range(len(types)), key=lambda i: types[i].axons)
        need = -(-widest // types[k].axons)
        types[k] = CoreType(types[k].axons, types[k].neurons, min(types[k].count + spare, max(need, types[k].count)))
    return dataclasses.replace(decl, core_types=tuple(types))


def _width(rng: random.Random, dom: Dict[str, Any]) -> int:
    if dom.get("small"):
        return rng.randint(1, 4)
    return 1 + int(rng.random() ** 2 * 31)


def _layer(rng: random.Random, nid: int, sources: List[int], fan_in: int, size: int, decl: Declaration,
           dom: Dict[str, Any]) -> Node:
    density = rng.choice([0.3, 0.6, 1.0])
    lo, hi = dom.get("weights") or (-8 if dom.get("signed") or rng.random() < 0.7 else -2, 8)
    w = np.array([[rng.randint(lo, hi) if rng.random() < density else 0 for _ in range(size)]
                  for _ in range(fan_in)], np.int64).reshape(fan_in, size)
    shift = rng.randint(0, 4) if decl.semantics == VALUE else 0
    blo, bhi = dom.get("bias", (-4, 4))
    bias = np.array([rng.randint(blo, bhi) << shift for _ in range(size)], np.int64)
    rule = dom.get("theta")
    if rule == "cover":
        need = (np.clip(w, 0, None).sum(axis=0) + np.clip(bias, 0, None)).max(initial=1)
        theta = int(min(16, max(1, need)))
    elif isinstance(rule, tuple):
        theta = rng.randint(*rule)
    else:
        theta = rng.choice([1, 1, 2]) if dom.get("parallel") and rng.random() < 0.5 else rng.randint(1, 16)
    if dom.get("_shared") is not None and rule is None:
        theta, shift = dom["_shared"]
        bias = np.array([rng.randint(blo, bhi) << shift for _ in range(size)], np.int64)
    return Node(nid, "neural", size, tuple(sorted(sources)), weights=w, bias=bias, theta=theta, shift=shift)


def generate(rng: random.Random, arm: str) -> Tuple[Program, Dict[int, np.ndarray], Dict[str, float], Dict[str, int]]:
    dom = dict(DOMAINS[arm])
    decl = _declaration(rng, dom)
    if rng.random() < dom.get("shared", 0.6):  # one threshold (shift) for every layer, as a fixed-threshold chip
        dom["_shared"] = (rng.randint(1, 16), rng.randint(0, 4) if decl.semantics == VALUE else 0)
    nodes: List[Node] = []
    motifs = {k: 0 for k in ("chain", "skip", "fork", "concat", "host", "residual", "cross", "parallel", "shared")}
    motifs["shared"] = int(dom.get("_shared") is not None)
    for _ in range(rng.randint(1, 2)):
        nodes.append(Node(len(nodes), "input", rng.randint(1, 16 if not dom.get("small") else 4)))
    depth = rng.randint(dom.get("min_depth", 1), 8)
    head, pre_host = nodes[-1].nid, []
    for level in range(depth):
        srcs = {head}
        motifs["chain"] += 1
        # A source keeps the new layer at depth level + 1 (so no path exceeds 8 layers).
        longest = _longest(nodes)
        shallow = [n.nid for n in nodes if longest[n.nid] <= level]
        if len(nodes) > 2 and rng.random() < dom.get("skip", 0.3):
            srcs.add(rng.choice(shallow))
            motifs["skip"] += 1
        cross = [n for n in pre_host if n in shallow]
        if cross and rng.random() < dom.get("cross", 0.15):
            srcs.add(rng.choice(cross))
            motifs["cross"] += 1
        if len(nodes) > 1 and rng.random() < 0.2:
            srcs.add(rng.choice(shallow))
            motifs["concat"] += len(srcs) > 1
        srcs_l = sorted(srcs)
        fan = sum(nodes[s].size for s in srcs_l)
        layer = _layer(rng, len(nodes), srcs_l, fan, _width(rng, dom), decl, dom)
        nodes.append(layer)
        head = layer.nid
        if rng.random() < dom.get("parallel", 0.25):
            sib = _layer(rng, len(nodes), srcs_l, fan, layer.size if rng.random() < 0.7 else _width(rng, dom),
                         decl, dom)
            nodes.append(sib)
            op, args = ("max", (layer.nid, sib.nid)) if sib.size == layer.size else ("relay", (layer.nid,))
            nodes.append(Node(len(nodes), "host", layer.size, args, op))
            motifs["parallel"] += 1
            motifs["host"] += 1
            pre_host.extend([layer.nid, sib.nid])
            head = nodes[-1].nid
        if level < depth - 1 and rng.random() < dom.get("fork", 0.2):
            nodes.append(_layer(rng, len(nodes), [layer.nid], layer.size, _width(rng, dom), decl, dom))
            motifs["fork"] += 1
        r = rng.random()
        longest = _longest(nodes)
        same = [n.nid for n in nodes if n.size == nodes[head].size and n.nid != head and longest[n.nid] <= level + 1]
        r = r / dom.get("host", 1.0) if dom.get("host", 1.0) > 0 else 1.0
        if r < 0.25 and same:
            if nodes[head].kind == "neural":
                pre_host.append(head)
            nodes.append(Node(len(nodes), "host", nodes[head].size, tuple(sorted((rng.choice(same), head))), "add"))
            motifs["residual"] += 1
            motifs["host"] += 1
            head = nodes[-1].nid
        elif r < 0.4:
            if nodes[head].kind == "neural":
                pre_host.append(head)
            nodes.append(Node(len(nodes), "host", nodes[head].size, (head,), rng.choice(["relay", "max"])))
            motifs["host"] += 1
            head = nodes[-1].nid
    prog = Program(_fit(decl, prog_fanins(nodes)), nodes)
    samples = dom.get("samples", 4)
    hi = decl.hi
    inputs = {n.nid: np.array([[rng.randint(0, hi) for _ in range(n.size)] for _ in range(samples)], np.int64)
              for n in nodes if n.kind == "input"}
    knobs = dict(DEFAULT_KNOBS, **dom.get("knobs", {}))
    return prog, inputs, knobs, motifs
