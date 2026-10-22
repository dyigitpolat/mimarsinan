"""The independent random packer: every X3a cost-only decision drawn at the model level.

Decisions, each drawn from the instance's packing RNG (``knobs`` sets their probabilities):

- **neuron split**: a layer's neurons are partitioned into groups of at most the core type's neurons;
- **exact fan-in fusion**: a layer's fan-in is partitioned into parts of at most the type's axons; the
  parts of one neuron group form a fusion group whose primary part holds the membranes and the partial
  parts deliver their integer partial charges to it every cycle (a fused wide core);
- **coalescing**: compatible fragments of one segment share a hard core (union of their axons);
- **placement permutation** and **axon-slot assignment**: every core's slot, axon order and neuron order
  are random permutations (``schedule.py`` draws passes, residency and slots).

Hypothesis guards the packer honours unless the arm switches them off (``off``): H1-theta (fragments
of one core share the core-scalar parameter, theta or shift), H1-latency (they share the fused latency
or group). Per-event cores take no fan-in fusion (refused by name when the fan-in exceeds every type).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .program import STREAMED_LIF, VALUE, Program


class GoldenRefusal(Exception):
    """A declaration the model-level packer cannot honour, refused by name (never a silent fallback)."""

    def __init__(self, name: str, detail: str = ""):
        super().__init__(f"{name}: {detail}")
        self.name = name


DEFAULT_KNOBS = {"fuse": 0.25, "split": 0.35, "coalesce": 0.8, "pin": 0.35, "cut": 0.35,
                 "prefer_bad_theta": 0.0, "prefer_bad_latency": 0.0, "prefer_pinned_slot": 0.0}


@dataclass
class Fragment:
    layer: int
    neurons: Tuple[int, ...]
    wires: Tuple[int, ...]
    group: int
    primary: bool
    ctype: int


@dataclass
class HardCore:
    hid: int
    ctype: int
    segment: int
    latency: int
    scalar: int
    partial: bool
    slots: List[Tuple[int, int, int]] = field(default_factory=list)
    slot_wires: List[FrozenSet[int]] = field(default_factory=list)
    axons: List[int] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    frag_scalars: List[int] = field(default_factory=list)
    frag_latencies: List[int] = field(default_factory=list)
    groups: List[int] = field(default_factory=list)


def core_timing(prog: Program, layer: int) -> int:
    """The fused latency (streamed LIF) or the ASAP group (synchronized, value) of a layer."""
    return prog.latency()[layer] if prog.decl.semantics == STREAMED_LIF else prog.depth()[layer]


def core_scalar(prog: Program, layer: int) -> int:
    node = prog.nodes[layer]
    return node.shift if prog.decl.semantics == VALUE else node.theta


def _chunks(rng: random.Random, items: List[int], cap: int, minimum: int, most: int = 1 << 30) -> List[List[int]]:
    """A random partition of ``items`` into ``minimum`` to ``most`` contiguous chunks of 1 to ``cap`` items."""
    n = len(items)
    low = max(minimum, -(-n // cap))
    k = rng.randint(low, max(low, min(n, most, low + 2)))
    sizes = [1] * k
    for _ in range(n - k):
        sizes[rng.choice([j for j in range(k) if sizes[j] < cap])] += 1
    out, at = [], 0
    for size in sizes:
        out.append(items[at:at + size])
        at += size
    return out


def fragments(prog: Program, rng: random.Random, knobs: Dict[str, float]) -> List[Fragment]:
    d, out, group = prog.decl, [], 0
    for node in prog.nodes:
        if node.kind != "neural":
            continue
        fan = [int(w) for w in prog.fanin(node.nid)]
        types = list(range(len(d.core_types)))
        if d.granularity == "per_event":
            types = [k for k in types if d.core_types[k].axons >= len(fan)]
            if not types:
                raise GoldenRefusal("EventSerialFusionRefused", f"layer {node.nid} fan-in {len(fan)}")
        fits = [k for k in types if -(-len(fan) // d.core_types[k].axons) <= d.core_types[k].count]
        if not fits:
            raise GoldenRefusal("CapacityRefused", f"layer {node.nid} fan-in {len(fan)} needs more fused cores")
        k = rng.choice(fits)
        ct = d.core_types[k]
        must_fuse = len(fan) > ct.axons
        fuse = must_fuse or (d.granularity != "per_event" and len(fan) >= 2 and ct.count >= 2
                             and rng.random() < knobs["fuse"])
        wires = list(fan)
        if fuse:
            rng.shuffle(wires)
            parts = [sorted(p) for p in _chunks(rng, wires, ct.axons, 2, ct.count)]
        else:
            parts = [wires]
        neurons = list(range(node.size))
        split = node.size > ct.neurons or (node.size >= 2 and rng.random() < knobs["split"])
        if split:
            rng.shuffle(neurons)
            ngroups = [sorted(g) for g in _chunks(rng, neurons, ct.neurons, 2 if node.size >= 2 else 1)]
        else:
            ngroups = [neurons]
        for ng in ngroups:
            gid = -1
            if len(parts) > 1:
                gid, group = group, group + 1
            lead = rng.randrange(len(parts))
            for j, part in enumerate(parts):
                out.append(Fragment(node.nid, tuple(ng), tuple(part), gid, j == lead, k))
    return out


def _fits(prog: Program, core: HardCore, frag: Fragment, off: FrozenSet[str]) -> bool:
    ct = prog.decl.core_types[core.ctype]
    if core.segment != prog.eta()[frag.layer] or core.ctype != frag.ctype or core.partial == frag.primary:
        return False
    if frag.group >= 0 and frag.group in core.groups:
        return False
    if any(lay == frag.layer and n in frag.neurons for lay, n, _ in core.slots):
        return False
    if "H1-latency" not in off and core.latency != core_timing(prog, frag.layer):
        return False
    if "H1-theta" not in off and core.scalar != core_scalar(prog, frag.layer):
        return False
    return len(set(core.axons) | set(frag.wires)) <= ct.axons and len(core.slots) + len(frag.neurons) <= ct.neurons


def _place(core: HardCore, prog: Program, frag: Fragment) -> None:
    core.axons = sorted(set(core.axons) | set(frag.wires))
    for n in frag.neurons:
        core.slots.append((frag.layer, n, frag.group))
        core.slot_wires.append(frozenset(frag.wires))
    core.frag_scalars.append(core_scalar(prog, frag.layer))
    core.frag_latencies.append(core_timing(prog, frag.layer))
    if frag.group >= 0:
        core.groups.append(frag.group)


def coalesce(prog: Program, frags: Sequence[Fragment], rng: random.Random, knobs: Dict[str, float],
             off: FrozenSet[str]) -> List[HardCore]:
    cores: List[HardCore] = []
    order = list(frags)
    rng.shuffle(order)
    for frag in order:
        fit = [c for c in cores if _fits(prog, c, frag, off)]
        bad = [c for c in fit if c.scalar != core_scalar(prog, frag.layer) and rng.random() < knobs["prefer_bad_theta"]]
        bad += [c for c in fit if c.latency != core_timing(prog, frag.layer)
                and rng.random() < knobs["prefer_bad_latency"]]
        if bad:
            core = rng.choice(bad)
        elif fit and rng.random() < knobs["coalesce"]:
            core = rng.choice(fit)
        else:
            core = HardCore(len(cores), frag.ctype, prog.eta()[frag.layer], core_timing(prog, frag.layer),
                            core_scalar(prog, frag.layer), not frag.primary)
            cores.append(core)
        _place(core, prog, frag)
    return cores


def finish(prog: Program, cores: List[HardCore], rng: Optional[random.Random]) -> None:
    """Axon-slot and neuron-slot permutations, then each core's weight block and bias."""
    index = {}
    for node in prog.nodes:
        if node.kind == "neural":
            index[node.nid] = {int(w): i for i, w in enumerate(prog.fanin(node.nid))}
    for core in cores:
        if rng is not None:
            rng.shuffle(core.axons)
            perm = list(range(len(core.slots)))
            rng.shuffle(perm)
            core.slots = [core.slots[i] for i in perm]
            core.slot_wires = [core.slot_wires[i] for i in perm]
        core.weights = np.zeros((len(core.axons), len(core.slots)), np.int64)
        core.bias = np.zeros(len(core.slots), np.int64)
        for j, ((layer, n, _), wires) in enumerate(zip(core.slots, core.slot_wires)):
            node = prog.nodes[layer]
            for i, w in enumerate(core.axons):
                if w in wires:
                    core.weights[i, j] = node.weights[index[layer][w], n]
            if not core.partial:
                core.bias[j] = node.bias[n]
