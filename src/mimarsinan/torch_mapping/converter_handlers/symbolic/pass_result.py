"""The symbolic-batch pass's result: every traced value's shape in the affine batch domain, the opaque batch values and
the values they reach (``symbolic_batch`` computes it, for a pure program only; review round 14)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Optional, Set, Tuple, Union

import torch.fx as fx

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import Affine, Opaque

__all__ = [
    "DATA_DEPENDENT", "Decision", "Extent", "OpaqueBatchValueError", "SYMBOLIC_HINT", "SymbolicPass", "SymbolicShape",
    "closure", "first_decisions",
]

SYMBOLIC_HINT = 7
"""The batch symbol's hint: the batch at which torch's kernels run in the pass and the rules are cross-checked."""

Extent = Union[int, Affine, Opaque]
"""An axis extent in the domain: a batch-free ``int``, an affine form of the batch, or an ``Opaque`` value."""

DATA_DEPENDENT = object()
"""The ``origin`` of a data-dependent extent (``Opaque``, its reason the size's name): not a batch decision."""

STATEMENT = "static shape proof: every batch-dependent size in the affine batch domain (a*B + c, B >= 1)"
"""The certificate's shape statement of a pure program."""


@dataclass(frozen=True)
class SymbolicShape:
    """A value's shape for every batch B >= 1, extent by extent; ``data_dependent``: a data-dependent size lies
    upstream."""

    dims: Tuple[Extent, ...]
    data_dependent: bool


@dataclass(frozen=True)
class Decision:
    """A node whose value the affine batch domain does not compute: ``opaque`` (a scalar operation outside the domain,
    or a kernel size no declared batch rule states: ``detail``) or ``failed`` (its symbolic run on a batch-dependent
    operand raised ``detail``); ``operator`` names it. Its value is computed on the host at run time."""

    node: fx.Node
    operator: str
    kind: str
    detail: str


@dataclass(frozen=True)
class SymbolicPass:
    """The batch symbol (``None``: no pass ran), every tensor value's ``SymbolicShape``, ``decided`` (per node, the
    first decision it is computed from, itself included), ``shape_decided`` (per node, the decision its per-sample
    shape or scalar value depends on), ``batch_axis_decided`` (per pure reshape, the decision its axis 0 depends on),
    the unproved values, and the first opaque node (``opaque``, traced order)."""

    batch: Optional[object]
    shapes: Dict[fx.Node, SymbolicShape] = field(default_factory=dict)
    decided: Dict[fx.Node, Decision] = field(default_factory=dict)
    shape_decided: Dict[fx.Node, Decision] = field(default_factory=dict)
    batch_axis_decided: Dict[fx.Node, Decision] = field(default_factory=dict)
    unproved: FrozenSet[fx.Node] = frozenset()
    opaque: Optional[Decision] = None

    @property
    def static_proof(self) -> bool:
        """The pass ran: every shape it holds is the affine batch domain's for every batch B >= 1."""
        return self.batch is not None

    def proved(self, node: fx.Node) -> bool:
        """``node``'s shape holds for every batch B >= 1 and every input."""
        symbolic = self.shapes.get(node)
        return (self.static_proof and symbolic is not None and not symbolic.data_dependent
                and node not in self.unproved)

    def batch_decision(self, node: fx.Node) -> Optional[Decision]:
        """The first decision ``node`` is computed from, through any data flow; ``None`` when none is."""
        return self.decided.get(node)

    def shape_decision(self, node: fx.Node) -> Optional[Decision]:
        """The decision ``node``'s per-sample shape (a tensor) or value (a scalar) depends on; ``None`` when none."""
        return self.shape_decided.get(node)

    def batch_axis_decision(self, node: fx.Node) -> Optional[Decision]:
        """The decision a pure reshape's axis 0 (the batch axis) depends on; ``None`` when it is provably B."""
        return self.batch_axis_decided.get(node)

    def proof_statement(self) -> str:
        """What the program's shapes are proved by, as the certificate records it, with its first opaque node."""
        if self.batch is None:
            return "no static shape proof: the graph has no batched input"
        if self.opaque is None:
            return STATEMENT
        return (f"{STATEMENT}; opaque batch values computed on the host at run time (first: {self.opaque.operator}, "
                f"FX node {self.opaque.node.name!r})")


def closure(gm: fx.GraphModule, roots: Set[fx.Node]) -> FrozenSet[fx.Node]:
    """``roots`` and every node computed from one of them."""
    found: Set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node in roots or found.intersection(node.all_input_nodes):
            found.add(node)
    return frozenset(found)


def first_decisions(gm: fx.GraphModule, own: Mapping[fx.Node, Decision]) -> Dict[fx.Node, Decision]:
    """Per node, the first decision (traced order) among its own and its inputs' (every node, run or not)."""
    order, decided = {n: i for i, n in enumerate(gm.graph.nodes)}, {}
    for node in gm.graph.nodes:
        found = [decided[arg] for arg in node.all_input_nodes if arg in decided] + ([own[node]] if node in own else [])
        if found:
            decided[node] = min(found, key=lambda decision: order[decision.node])
    return decided


class OpaqueBatchValueError(ToolchainRefusal, ValueError):
    """Stage 2 of a program with an opaque batch value: the value program fixes every per-sample shape at conversion,
    and the affine batch domain does not compute this value (review round 14)."""

    def __init__(self, node: str, operator: str, detail: str):
        self.node_name, self.operator, self.detail = node, operator, detail
        super().__init__(
            f"{operator} (FX node {node!r}): {detail}; the value program fixes every per-sample shape at conversion "
            "and is exact only where every batch-dependent value is in the affine batch domain, so it is refused"
        )
