"""Logical integer programs of the X3a model level: the network N, its declaration and its fused timing.

A ``Program`` is a DAG of nodes over integer wires, numbered globally (node order, then index):

- ``input``: ``size`` wires whose per-window value is drawn per sample (a count in ``[0, T]``, a TTFS
  level in ``[0, T]`` or an integer value in ``[0, 2**act_bits - 1]``);
- ``neural``: a layer of ``size`` neurons; its fan-in is the concatenation of its sources' wires in
  ascending node order, which is also the canonical event order (a wire's rank is its id); integer
  weights ``[fan_in, size]``, an integer bias per neuron and one core-scalar parameter per layer
  (``theta`` for spiking semantics, the requantization ``shift`` in the value domain);
- ``host``: an exact integer host operation (``add``, ``max``, ``relay``) over equal-size sources.

Segments follow the host depth eta (plan D16 Lemma 1): a neural node's eta is the largest
``eta(source) + [source is host]``, a host node's is its largest source's. A neural source of equal eta
is an in-segment wire (read from the producer's buffer of the previous cycle); every other source is a
segment input (re-encoded at segment entry from its window value).

Fused timing (pass-transparency memo section 4, U1 design section 2): within a segment, streamed cores
run ALAP: a producer runs at its deepest in-segment consumer's latency minus one, a sink at its longest
path from the segment inputs. Synchronized and value semantics order cores by ASAP depth (the group).
Nothing here imports the toolchain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

STREAMED_LIF = "streamed_lif"
SYNC_LIF = "sync_lif"
SYNC_TTFS = "sync_ttfs"
VALUE = "value"
SEMANTICS = (STREAMED_LIF, SYNC_LIF, SYNC_TTFS, VALUE)
HOST_OPS = ("add", "max", "relay")
#: Declaration fields added after the x3a-golden-program/1 records; written only when not at their default,
#: so every earlier record keeps its bytes.
_LATE_DEFAULTS = {"encoder": "even", "value_law": "relu_shift"}


@dataclass(frozen=True)
class CoreType:
    axons: int
    neurons: int
    count: int


@dataclass(frozen=True)
class Declaration:
    """What the chip computes (semantics, window, comparator, reset, membrane) and what it holds."""

    semantics: str
    T: int = 4
    comparator: str = "<="
    reset: str = "subtract"
    granularity: str = "per_cycle"
    membrane_bits: Optional[int] = None
    act_bits: int = 8
    core_types: Tuple[CoreType, ...] = (CoreType(64, 64, 32),)
    pass_budget: Optional[int] = None
    accumulator: str = "int64"
    partial_sum_bits: int = 4
    encoder: str = "even"
    value_law: str = "relu_shift"

    @property
    def hi(self) -> int:
        """The largest window value of a wire: T for counts and levels, 2**act_bits - 1 for values."""
        return (1 << self.act_bits) - 1 if self.semantics == VALUE else self.T


@dataclass
class Node:
    nid: int
    kind: str
    size: int
    sources: Tuple[int, ...] = ()
    op: str = ""
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    theta: int = 1
    shift: int = 0


@dataclass
class Program:
    decl: Declaration
    nodes: List[Node]
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # -- wires --------------------------------------------------------------------------------------
    def base(self, nid: int) -> int:
        if "base" not in self._cache:
            offsets, at = [], 0
            for node in self.nodes:
                offsets.append(at)
                at += node.size
            self._cache["base"] = (offsets, at)
        return self._cache["base"][0][nid]

    @property
    def n_wires(self) -> int:
        self.base(0)
        return self._cache["base"][1]

    def wires(self, nid: int) -> np.ndarray:
        start = self.base(nid)
        return np.arange(start, start + self.nodes[nid].size, dtype=np.int64)

    def fanin(self, nid: int) -> np.ndarray:
        """The global wire ids a neural node reads, in canonical (ascending) order."""
        node = self.nodes[nid]
        return np.concatenate([self.wires(s) for s in node.sources]) if node.sources else np.zeros(0, np.int64)

    def owner(self) -> np.ndarray:
        """Wire id to the node that produces it."""
        if "owner" not in self._cache:
            self._cache["owner"] = np.concatenate([np.full(n.size, n.nid, np.int64) for n in self.nodes])
        return self._cache["owner"]

    # -- segments and timing ------------------------------------------------------------------------
    def eta(self) -> List[int]:
        if "eta" not in self._cache:
            eta: List[int] = []
            for node in self.nodes:
                if node.kind == "input":
                    eta.append(0)
                elif node.kind == "host":
                    eta.append(max(eta[s] for s in node.sources))
                else:
                    eta.append(max(eta[s] + (self.nodes[s].kind == "host") for s in node.sources))
            self._cache["eta"] = eta
        return self._cache["eta"]

    def in_segment(self, nid: int, src: int) -> bool:
        """Whether ``src`` feeds neural ``nid`` inside one neural segment (a buffer read, not an encode)."""
        return self.nodes[src].kind == "neural" and self.eta()[src] == self.eta()[nid]

    def segments(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for node in self.nodes:
            if node.kind == "neural":
                out.setdefault(self.eta()[node.nid], []).append(node.nid)
        return out

    def depth(self) -> Dict[int, int]:
        """ASAP depth inside the segment: 0 for a layer reading only segment inputs."""
        if "depth" not in self._cache:
            depth: Dict[int, int] = {}
            for node in self.nodes:
                if node.kind == "neural":
                    inner = [depth[s] for s in node.sources if self.in_segment(node.nid, s)]
                    depth[node.nid] = 1 + max(inner) if inner else 0
            self._cache["depth"] = depth
        return self._cache["depth"]

    def latency(self) -> Dict[int, int]:
        """Fused streamed latency: ALAP, a producer at its deepest in-segment consumer minus one."""
        if "latency" not in self._cache:
            consumers: Dict[int, List[int]] = {}
            for node in self.nodes:
                for s in node.sources:
                    if node.kind == "neural" and self.in_segment(node.nid, s):
                        consumers.setdefault(s, []).append(node.nid)
            lat: Dict[int, int] = {}
            for node in reversed(self.nodes):
                if node.kind == "neural":
                    cons = consumers.get(node.nid)
                    lat[node.nid] = max(lat[c] for c in cons) - 1 if cons else self.depth()[node.nid]
            self._cache["latency"] = lat
        return self._cache["latency"]

    # -- serialization ------------------------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        decl = {k: v for k, v in self.decl.__dict__.items() if _LATE_DEFAULTS.get(k, object()) != v}
        decl["core_types"] = [ct.__dict__ for ct in self.decl.core_types]
        nodes = []
        for n in self.nodes:
            row: Dict[str, Any] = {"nid": n.nid, "kind": n.kind, "size": n.size, "sources": list(n.sources)}
            if n.kind == "host":
                row["op"] = n.op
            if n.kind == "neural":
                row.update(weights=n.weights.tolist(), bias=n.bias.tolist(), theta=n.theta, shift=n.shift)
            nodes.append(row)
        return {"schema": "x3a-golden-program/1", "declaration": decl, "nodes": nodes}

    @staticmethod
    def from_json(doc: Dict[str, Any]) -> "Program":
        decl = dict(doc["declaration"])
        decl["core_types"] = tuple(CoreType(**ct) for ct in decl["core_types"])
        nodes = []
        for row in doc["nodes"]:
            node = Node(row["nid"], row["kind"], row["size"], tuple(row["sources"]), row.get("op", ""),
                        theta=row.get("theta", 1), shift=row.get("shift", 0))
            if "weights" in row:
                node.weights = np.array(row["weights"], np.int64).reshape(-1, row["size"])
                node.bias = np.array(row["bias"], np.int64)
            nodes.append(node)
        return Program(Declaration(**decl), nodes)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()


def canonical_json(obj: Any) -> str:
    """One byte form for every record the screen digests (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
