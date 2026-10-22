"""Contraction: the host op of a product of two activations (``matmul``, ``@``, ``bmm``, two-operand ``einsum``).

It evaluates the traced call itself, bit for bit the native op, through ``ComputeAdapter``'s argument template, and
declares the interface the capability table (class D) and dynamic-operand lowering (N2) read: the einsum spec, the
operand roles, each operand's per-sample shape and its forward MACs. No quantization hook: N2 owns the operand grid.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from mimarsinan.mapping.support.call_template import ArgTemplate
from mimarsinan.mapping.support.compute_modules import ComputeAdapter

ACTIVATION = "activation"

#: The products of two activations, by the callable the traced graph applies.
PRODUCT_FUNCTIONS: Tuple[Any, ...] = (torch.matmul, operator.matmul, torch.bmm, torch.einsum, torch.Tensor.matmul,
                                      torch.Tensor.bmm)


def matmul_spec(rank_a: int, rank_b: int) -> str:
    """The einsum form of ``matmul`` at these operand ranks (batch axes broadcast as ``...``)."""
    left, right = ("k" if rank_a == 1 else "...mk"), ("k" if rank_b == 1 else "...kn")
    out = "...m" if rank_b == 1 and rank_a > 1 else "...n" if rank_a == 1 and rank_b > 1 else "" if rank_a == 1 \
        else "...mn"
    return f"{left},{right}->{out}"


def _labels(term: str, rank: int) -> Tuple[str, ...]:
    """One label per axis: its letter, or for an axis an ellipsis covers, its distance from the ellipsis's end."""
    if "..." not in term:
        return tuple(term)
    head, tail = term.split("...")
    covered = rank - len(head) - len(tail)
    return (*head, *(f"...{covered - 1 - i}" for i in range(covered)), *tail)


def _summed_extent(spec: str, shapes: Sequence[Tuple[int, ...]]) -> int:
    """The product of the extents of every axis an einsum sums over (in an operand, not in its output)."""
    inputs, explicit, output = spec.replace(" ", "").partition("->")
    extents: Dict[str, int] = {}
    for term, shape in zip(inputs.split(","), shapes):
        for label, extent in zip(_labels(term, len(shape)), shape):
            extents[label] = max(extents.get(label, 1), int(extent))
    if not explicit:  # torch's implicit output: the letters used once, and the ellipsis axes
        output = "".join(sorted(c for c in set(inputs) if c.isalpha() and inputs.count(c) == 1)) + "..."
    kept = {c for c in output if c.isalpha()} | ({label for label in extents if label.startswith("...")}
                                                 if "..." in output else set())
    return math.prod(extent for label, extent in extents.items() if label not in kept)


class Contraction(ComputeAdapter):
    """A product of two activations as one host op: the traced call evaluated as traced, with its declared
    interface. ``operand_shapes`` are per sample (batch axis stripped); ``forward_macs()`` is per sample."""

    @property
    def display_name(self) -> str:
        return "Contraction"

    def __init__(self, fn, einsum_spec: str, operand_shapes: Sequence[Tuple[int, ...]], macs: int, *,
                 extra_args: Sequence[Any] = (), kwargs: Optional[Mapping[str, Any]] = None,
                 arg_template: Optional[ArgTemplate] = None) -> None:
        super().__init__(fn, extra_args=extra_args, kwargs=kwargs, arg_template=arg_template)
        self.einsum_spec = str(einsum_spec)
        self.operand_roles = (ACTIVATION, ACTIVATION)
        self.operand_shapes = tuple(tuple(int(d) for d in shape) for shape in operand_shapes)
        self._macs = int(macs)

    def forward_macs(self) -> int:
        return self._macs


def contraction_macs(spec: str, operand_shapes: Sequence[Tuple[int, ...]], output_shape: Tuple[int, ...]) -> int:
    """Forward MACs of one sample: every output element times the extent summed over."""
    return math.prod(int(d) for d in output_shape) * _summed_extent(spec, operand_shapes)
