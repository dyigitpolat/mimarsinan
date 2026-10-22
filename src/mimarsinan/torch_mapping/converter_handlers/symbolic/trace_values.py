"""Seeded concrete traces of a traced graph at batches 1, 2 and 3, and what they show of each node's value.

The traces run the graph natively (torch's kernels and every call outside the declared set, on real tensors of real
batches): they are samples, never proofs. A layout violation observed in them is a fact at that batch
(``batch_layout.py``). The symbolic-batch pass, which runs for a pure program only (``purity.py``), takes from them the
value of a node it does not evaluate symbolically (a kernel without a symbolic form, a node whose input it could not
evaluate), and the batch axes a kernel made concrete
(``sample``): an axis whose extent equals the batch in every trace carries the batch, one whose extent is the same in
every trace is static, any other one varies. No random state is read.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.fx as fx
from torch.utils import _pytree as pytree

from mimarsinan.torch_mapping.conversion_probe import guarded

__all__ = [
    "BATCH", "TRACE_BATCHES", "VARIES", "SampledTensor", "Trace", "graph_device", "placeholder_metas",
    "rng_neutral", "sample", "trace", "trace_shapes",
]

TRACE_BATCHES = (1, 2, 3)
"""The seeded concrete traces: the stand-in of a kernel whose FakeTensor run raised, the cross-check of every extent a
kernel rule states (with the run at the hint, two batches at least, review round 16), and a fact (a layout violation)
at a batch. [corrected in review round 14: batch 1 is no longer outside the symbolic pass, whose domain holds every
batch B >= 1; a seeded batch-1 trace still has to agree with a static extent]"""

Shapes = Dict[fx.Node, Tuple[int, ...]]


class _Marker:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


BATCH = _Marker("BATCH")
"""An axis whose extent equals the batch in every trace."""

VARIES = _Marker("VARIES")
"""An axis extent or a scalar that differs between the traces (other than as the batch)."""


@dataclass(frozen=True)
class _Tensor:
    shape: Tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class SampledTensor:
    """A tensor leaf across the traces: per axis ``BATCH``, a static extent or ``VARIES``; its traced dtype; and the
    batch-1 extents (the hints of the axes that vary)."""

    axes: Tuple[Any, ...]
    dtype: torch.dtype
    first: Tuple[int, ...]


@dataclass
class Trace:
    """One seeded run: every tensor value's shape, and every value's structure (tensor leaves as shape and dtype,
    Python scalars as themselves, any other leaf as ``None``)."""

    batch: int
    shapes: Shapes = field(default_factory=dict)
    values: Dict[fx.Node, Tuple[Any, Tuple[Any, ...]]] = field(default_factory=dict)


def _leaf(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _Tensor(tuple(value.shape), value.dtype)
    return value if type(value) in (int, float, bool) else None


class _Recorder(fx.Interpreter):
    """Runs the graph and records every value, leaving node metadata untouched."""

    def __init__(self, gm: fx.GraphModule, record: Trace):
        super().__init__(gm)
        self.record = record

    def run_node(self, n: fx.Node) -> Any:
        value = super().run_node(n)
        if isinstance(value, torch.Tensor):
            self.record.shapes[n] = tuple(value.shape)
        leaves, spec = pytree.tree_flatten(value)
        self.record.values[n] = (spec, tuple(_leaf(leaf) for leaf in leaves))
        return value


def placeholder_metas(gm: fx.GraphModule) -> Optional[list]:
    metas = [node.meta.get("tensor_meta") for node in gm.graph.nodes if node.op == "placeholder"]
    if not metas or any(meta is None or not hasattr(meta, "shape") or len(meta.shape) == 0 for meta in metas):
        return None
    return metas


def graph_device(gm: fx.GraphModule) -> torch.device:
    state = next(itertools.chain(gm.parameters(), gm.buffers()), None)
    return state.device if state is not None else torch.device("cpu")


def rng_neutral(gm: fx.GraphModule):
    return torch.random.fork_rng(devices=[graph_device(gm)] if graph_device(gm).type == "cuda" else [])


def _trace_inputs(gm: fx.GraphModule, batch: int) -> Optional[list]:
    metas = placeholder_metas(gm)
    if metas is None:
        return None
    device, generator = graph_device(gm), torch.Generator().manual_seed(batch)
    inputs = []
    for meta in metas:
        shape = (batch, *tuple(meta.shape)[1:])
        if meta.dtype.is_floating_point:
            inputs.append(torch.randn(shape, dtype=meta.dtype, generator=generator).to(device))
        else:
            inputs.append(torch.zeros(shape, dtype=meta.dtype, device=device))
    return inputs


def trace(gm: fx.GraphModule, batch: int) -> Trace:
    """Every value of ``gm`` at ``batch`` (seeded inputs); the values after a failing node have none."""
    record = Trace(batch)
    inputs = _trace_inputs(gm, batch)
    if inputs is not None:
        with rng_neutral(gm), torch.no_grad():
            guarded(lambda: _Recorder(gm, record).run(*inputs))
    return record


def trace_shapes(gm: fx.GraphModule, batch: int) -> Shapes:
    """Every tensor value's shape when ``gm`` runs at ``batch``; the values after a failing node have none."""
    return trace(gm, batch).shapes


def _axis(extents: Tuple[int, ...], batches: Tuple[int, ...]) -> Any:
    if extents == batches:
        return BATCH
    return extents[0] if len(set(extents)) == 1 else VARIES


def _sampled_leaf(leaves: Tuple[Any, ...], batches: Tuple[int, ...]) -> Optional[Any]:
    if all(isinstance(leaf, _Tensor) for leaf in leaves):
        if len({len(leaf.shape) for leaf in leaves}) > 1:
            return None
        axes = tuple(_axis(extents, batches) for extents in zip(*(leaf.shape for leaf in leaves)))
        return SampledTensor(axes, leaves[0].dtype, leaves[0].shape)
    if all(type(leaf) in (int, float, bool) for leaf in leaves) and len({type(leaf) for leaf in leaves}) == 1:
        return leaves[0] if len(set(leaves)) == 1 else (VARIES, leaves[0])
    return None


def sample(node: fx.Node, traces: Tuple[Trace, ...]) -> Optional[Tuple[Any, Tuple[Any, ...]]]:
    """``node``'s value across the traces: its structure and per leaf a ``SampledTensor``, a scalar the traces agree
    on, or ``(VARIES, batch-1 value)``; ``None`` when a trace lacks it, the structure or a tensor's rank differs, or a
    leaf is neither a tensor nor a Python scalar."""
    found = [record.values[node] for record in traces if node in record.values]
    if not found or len(found) < len(traces) or len({str(spec) for spec, _ in found}) > 1:
        return None
    batches = tuple(record.batch for record in traces)
    leaves = [_sampled_leaf(column, batches) for column in zip(*(leaves for _, leaves in found))]
    if any(leaf is None for leaf in leaves):
        return None
    return found[0][0], tuple(leaves)
