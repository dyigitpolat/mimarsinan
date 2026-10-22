"""Pass partition and order, residency and placement of a packed program; the identity packing.

- **Units.** Hard cores linked by a data wire into a strongly connected set, or by a fusion group (a
  partial core and its primary must run together), form one unit; a unit never straddles passes.
- **Pass partition and order** (H9, a DAG-monotone cut): per segment, a random linear extension of the
  unit DAG is cut into passes at random points and wherever the non-pinned cores of a pass would exceed
  the slots of their type. With H9 off, units are dealt to passes and the passes ordered at random.
- **Residency** (H10): a random set of cores is pinned to dedicated slots for the whole run (every pass,
  every batch of K inputs); the other cores are rewritten into free slots before their pass. With H10
  off the allocator ignores the pins, so a non-pinned core may overwrite a pinned slot.
- **Pass budget**: more passes than ``pass_budget`` refuses by name, as does a unit larger than the chip.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Tuple

from .packer import GoldenRefusal, HardCore, core_scalar, core_timing, finish
from .program import Program


@dataclass
class Packed:
    """A packed program: cores (weights rebuilt from slots, slot wires and axon order), passes in execution
    order, pinned slots, the slot of every other core, batch K. ``to_json`` is the exchange form a code-level
    exporter writes so the golden can run a toolchain mapping ("golden on packed artifacts")."""

    cores: List[HardCore]
    passes: List[Tuple[int, Tuple[int, ...]]]
    pinned: Dict[int, int]
    slot_of: Dict[int, int]
    batch_K: int
    n_slots: int
    flags: Dict[str, int] = field(default_factory=dict)

    _CORE_FIELDS = ("hid", "ctype", "segment", "latency", "scalar", "partial", "slots", "axons",
                    "frag_scalars", "frag_latencies", "groups")

    def to_json(self) -> Dict[str, object]:
        cores = []
        for c in self.cores:
            row: Dict[str, object] = {k: getattr(c, k) for k in self._CORE_FIELDS}
            row["slots"] = [list(t) for t in c.slots]
            row["slot_wires"] = [sorted(w) for w in c.slot_wires]
            cores.append(row)
        return {"schema": "x3a-golden-packed/1", "cores": cores, "passes": [[s, list(h)] for s, h in self.passes],
                "pinned": {str(k): v for k, v in self.pinned.items()},
                "slot_of": {str(k): v for k, v in self.slot_of.items()}, "batch_K": self.batch_K,
                "n_slots": self.n_slots}

    @staticmethod
    def from_json(prog: Program, doc: Dict[str, Any]) -> "Packed":
        cores = []
        for row in doc["cores"]:
            core = HardCore(row["hid"], row["ctype"], row["segment"], row["latency"], row["scalar"], row["partial"])
            core.slots = [tuple(t) for t in row["slots"]]
            core.slot_wires = [frozenset(w) for w in row["slot_wires"]]
            core.axons, core.groups = list(row["axons"]), list(row["groups"])
            core.frag_scalars, core.frag_latencies = list(row["frag_scalars"]), list(row["frag_latencies"])
            cores.append(core)
        finish(prog, cores, None)
        return Packed(cores, [(s, tuple(h)) for s, h in doc["passes"]], {int(k): v for k, v in doc["pinned"].items()},
                      {int(k): v for k, v in doc["slot_of"].items()}, doc["batch_K"], doc["n_slots"])


def producers(prog: Program, cores: List[HardCore]) -> Dict[Tuple[int, int], int]:
    """(layer, neuron) to the hid of the core that holds its membrane."""
    return {(lay, n): c.hid for c in cores if not c.partial for lay, n, _ in c.slots}


def _units(prog: Program, cores: List[HardCore], seg: int) -> Tuple[List[List[int]], Dict[int, set]]:
    owner = prog.owner()
    prod = producers(prog, cores)
    mine = [c for c in cores if c.segment == seg]
    adj: Dict[int, set] = {c.hid: set() for c in mine}
    for c in mine:
        for w in c.axons:
            nid = int(owner[w])
            src = prod.get((nid, w - prog.base(nid)))
            if src is not None and src in adj and src != c.hid:
                adj[src].add(c.hid)
        if c.partial:
            for lay, n, _ in c.slots:
                adj[c.hid].add(prod[(lay, n)])
                adj[prod[(lay, n)]].add(c.hid)
    index, low, stack, on, sccs, counter = {}, {}, [], set(), [], [0]

    def strong(v: int) -> None:  # Tarjan; programs are small, recursion depth stays below 64 cores
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for u in sorted(adj[v]):
            if u not in index:
                strong(u)
                low[v] = min(low[v], low[u])
            elif u in on:
                low[v] = min(low[v], index[u])
        if low[v] == index[v]:
            comp = []
            while True:
                u = stack.pop()
                on.discard(u)
                comp.append(u)
                if u == v:
                    break
            sccs.append(sorted(comp))

    for v in sorted(adj):
        if v not in index:
            strong(v)
    return sccs, adj


def _linear(rng: random.Random, units: List[List[int]], adj: Dict[int, set]) -> List[List[int]]:
    of = {h: i for i, u in enumerate(units) for h in u}
    succ = {i: {of[v] for h in u for v in adj[h]} - {i} for i, u in enumerate(units)}
    indeg = {i: 0 for i in succ}
    for i in succ:
        for j in succ[i]:
            indeg[j] += 1
    ready, out = [i for i in succ if indeg[i] == 0], []
    while ready:
        i = ready.pop(rng.randrange(len(ready)))
        out.append(units[i])
        for j in sorted(succ[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    assert len(out) == len(units), "unit graph is acyclic by construction"
    return out


def _cut(rng, order, cores, pinned, free, knobs) -> List[List[int]]:
    passes: List[List[int]] = [[]]
    used: Dict[int, int] = {}
    for unit in order:
        need: Dict[int, int] = {}
        for h in unit:
            if h not in pinned:
                need[cores[h].ctype] = need.get(cores[h].ctype, 0) + 1
        if any(n > free[k] for k, n in need.items()):
            raise GoldenRefusal("CapacityRefused", f"unit of {len(unit)} cores exceeds the free slots")
        over = any(used.get(k, 0) + n > free[k] for k, n in need.items())
        if passes[-1] and (over or rng.random() < knobs["cut"]):
            passes.append([])
            used = {}
        passes[-1].extend(unit)
        for k, n in need.items():
            used[k] = used.get(k, 0) + n
    return passes


def _passes(prog, cores, seg_units, rng, knobs, off, pinned, free):
    passes: List[Tuple[int, Tuple[int, ...]]] = []
    for seg, (units, adj) in seg_units.items():
        if "H9" in off:
            units = list(units)
            rng.shuffle(units)
            chunks = _cut(rng, units, cores, pinned, free, knobs)
            rng.shuffle(chunks)
        else:
            chunks = _cut(rng, _linear(rng, units, adj), cores, pinned, free, knobs)
        passes.extend((seg, tuple(ch)) for ch in chunks)
    return passes


def schedule(prog: Program, cores: List[HardCore], rng: random.Random, knobs, off: FrozenSet[str],
             batch: int) -> Packed:
    d = prog.decl
    starts, at = [], 0
    for ct in d.core_types:
        starts.append(at)
        at += ct.count
    seg_units = {seg: _units(prog, cores, seg) for seg in sorted({c.segment for c in cores})}
    pool = {k: list(range(starts[k], starts[k] + ct.count)) for k, ct in enumerate(d.core_types)}
    pinned: Dict[int, int] = {}
    for c in cores:
        if rng.random() < knobs["pin"] and len(pool[c.ctype]) > 1:
            pinned[c.hid] = pool[c.ctype].pop(rng.randrange(len(pool[c.ctype])))
    for units, _ in seg_units.values():
        for unit in units:  # unpin until every unit's rewritten cores fit the free slots of their type
            for k in {cores[h].ctype for h in unit}:
                while sum(1 for h in unit if cores[h].ctype == k and h not in pinned) > len(pool[k]):
                    victims = sorted(h for h, s in pinned.items() if cores[h].ctype == k and h not in unit)
                    if not victims:
                        break
                    pool[k].append(pinned.pop(victims[rng.randrange(len(victims))]))
    free = {k: len(v) for k, v in pool.items()}
    passes = _passes(prog, cores, seg_units, rng, knobs, off, pinned, free)
    if d.pass_budget is not None and len(passes) > d.pass_budget:
        passes = _passes(prog, cores, seg_units, rng, dict(knobs, cut=0.0), off, pinned, free)
        if len(passes) > d.pass_budget:
            raise GoldenRefusal("PassBudgetRefused", f"{len(passes)} passes > budget {d.pass_budget}")
    slot_of: Dict[int, int] = {}
    for _, hids in passes:
        taken: set = set()
        for h in hids:
            if h in pinned:
                continue
            k = cores[h].ctype
            avail = [s for s in pool[k] if s not in taken]
            if "H10" in off:
                stolen = [s for s in pinned.values() if s not in taken and starts[k] <= s < starts[k] + d.core_types[k].count]
                avail = stolen if stolen and rng.random() < knobs["prefer_pinned_slot"] else avail + stolen
            slot_of[h] = rng.choice(avail)
            taken.add(slot_of[h])
    finish(prog, cores, rng)
    return Packed(cores, passes, pinned, slot_of, rng.randint(1, batch), at)


def identity_pack(prog: Program, batch: int) -> Packed:
    """One core per layer with its whole fan-in in canonical order, one pass per segment, all pinned."""
    cores: List[HardCore] = []
    for node in prog.nodes:
        if node.kind != "neural":
            continue
        core = HardCore(len(cores), -1, prog.eta()[node.nid], core_timing(prog, node.nid),
                        core_scalar(prog, node.nid), False)
        core.axons = [int(w) for w in prog.fanin(node.nid)]
        wires = frozenset(core.axons)
        core.slots = [(node.nid, n, -1) for n in range(node.size)]
        core.slot_wires = [wires] * node.size
        core.frag_scalars, core.frag_latencies = [core.scalar], [core.latency]
        cores.append(core)
    finish(prog, cores, None)
    passes = []
    for seg in sorted({c.segment for c in cores}):
        passes.append((seg, tuple(c.hid for c in cores if c.segment == seg)))
    return Packed(cores, passes, {c.hid: c.hid for c in cores}, {}, batch, len(cores))
