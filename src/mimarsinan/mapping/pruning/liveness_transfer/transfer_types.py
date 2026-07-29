"""Transfer primitives: kinds, the ``LivenessTransfer`` relation, shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Sequence, Set

from mimarsinan.mapping.ir import ComputeOp

__all__ = [
    "TRANSFER_ELEMENTWISE_1TO1",
    "TRANSFER_INDEX_BIJECTION",
    "TRANSFER_REGION_REDUCE",
    "TRANSFER_OPAQUE",
    "LivenessTransfer",
    "OPAQUE_TRANSFER",
]

TRANSFER_ELEMENTWISE_1TO1 = "elementwise_1to1"
TRANSFER_INDEX_BIJECTION = "index_bijection"
TRANSFER_REGION_REDUCE = "region_reduce"
TRANSFER_OPAQUE = "opaque"


@dataclass(frozen=True)
class LivenessTransfer:
    """One op's liveness relation in flat coordinates (opaque = no relation).

    - ``out_to_ins[o]`` — output ``o`` is CONSTANT ZERO iff every input in
      the region is eliminated (requires ``act(0) == 0`` — forward);
    - ``in_to_outs[i]`` — input ``i`` is UNUSED iff every output covering it
      is unused downstream (pure use-analysis — backward).
    """

    kind: str
    out_to_ins: Mapping[int, FrozenSet[int]] = field(default_factory=dict)
    in_to_outs: Mapping[int, FrozenSet[int]] = field(default_factory=dict)

    @property
    def is_opaque(self) -> bool:
        return self.kind == TRANSFER_OPAQUE


OPAQUE_TRANSFER = LivenessTransfer(kind=TRANSFER_OPAQUE)


def _relation_transfer(
    kind: str, out_to_ins: Mapping[int, FrozenSet[int]], n_inputs: int
) -> LivenessTransfer:
    in_to_outs: Dict[int, Set[int]] = {i: set() for i in range(n_inputs)}
    for o, region in out_to_ins.items():
        for i in region:
            in_to_outs[i].add(o)
    return LivenessTransfer(
        kind=kind,
        out_to_ins=dict(out_to_ins),
        in_to_outs={i: frozenset(v) for i, v in in_to_outs.items()},
    )


def _identity_transfer(kind: str, n: int) -> LivenessTransfer:
    return _relation_transfer(kind, {j: frozenset({j}) for j in range(n)}, n)


def _flat_size(shape: Sequence[int] | None) -> int | None:
    if shape is None:
        return None
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _module_of(op: ComputeOp):
    return (getattr(op, "params", None) or {}).get("module")


def _is_zero_preserving_or_opaque(op: ComputeOp) -> bool:
    """The certificate registry check, demoted to a predicate: unknown host
    activations are OPAQUE at transfer time (the certificate is where they
    fail loud).

    Imported lazily: the certificate package pulls in the propagation kernels
    (for ``certify_cascade_equivalence``), which consume this package.
    """
    from mimarsinan.mapping.pruning.certificate.errors import (
        CascadeCertificatePreconditionError,
    )
    from mimarsinan.mapping.pruning.certificate.zero_preserving import (
        is_zero_preserving_host_op,
    )
    try:
        return bool(is_zero_preserving_host_op(op))
    except CascadeCertificatePreconditionError:
        return False
