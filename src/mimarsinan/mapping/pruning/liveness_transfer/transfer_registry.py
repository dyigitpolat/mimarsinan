"""Per-op ``LivenessTransfer`` derivation: three exact classes + opaque default.

Implemented classes: ELEMENTWISE_1TO1 (zero-preserving activations with
matching flat widths, reusing the CHECKED certificate registry),
INDEX_BIJECTION (flatten/reshape/view/permute/transpose — the exact index
map derived by a double distinct-value probe of the op's own execution
seam), REGION_REDUCE (avg/max pool receptive fields, see
``transfer_region``). LayerNorm, softmax, attention, and every multi-input
join stay OPAQUE; unknown op types are opaque — never an error at transfer
time (the certificate is where unknown host ACTIVATIONS fail loud).
"""

from __future__ import annotations

from typing import FrozenSet, List

import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.pruning.boundary_policy import (
    _computeop_relays_deadness,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_policy import (
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    require_computeop_liveness_transfers,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_region import (
    REGION_REDUCE_HOST_OP_TYPES,
    derive_region_reduce,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
    OPAQUE_TRANSFER,
    TRANSFER_ELEMENTWISE_1TO1,
    TRANSFER_INDEX_BIJECTION,
    TRANSFER_OPAQUE,
    TRANSFER_REGION_REDUCE,
    LivenessTransfer,
    _flat_size,
    _identity_transfer,
    _is_zero_preserving_or_opaque,
    _module_of,
    _relation_transfer,
)

__all__ = [
    "TRANSFER_ELEMENTWISE_1TO1",
    "TRANSFER_INDEX_BIJECTION",
    "TRANSFER_REGION_REDUCE",
    "TRANSFER_OPAQUE",
    "LivenessTransfer",
    "OPAQUE_TRANSFER",
    "derive_liveness_transfer",
]

# Elementwise zero-preserving module types (subset of the certificate
# ZERO_PRESERVING_HOST_OP_TYPES that is also positionally 1:1).
_ELEMENTWISE_HOST_OP_TYPES: FrozenSet[type] = frozenset({
    nn.Identity,
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.Dropout,
    nn.Dropout2d,
})

_INDEX_BIJECTION_OP_TYPE_NAMES = frozenset({
    "flatten", "reshape", "view", "permute", "transpose",
})


def _derive_elementwise(op: ComputeOp, n_in: int) -> LivenessTransfer:
    n_out = _flat_size(getattr(op, "output_shape", None))
    if n_out is not None and n_out != n_in:
        return OPAQUE_TRANSFER
    if not _is_zero_preserving_or_opaque(op):
        return OPAQUE_TRANSFER
    return _identity_transfer(TRANSFER_ELEMENTWISE_1TO1, n_in)


def _derive_index_bijection(op: ComputeOp, n_in: int) -> LivenessTransfer:
    """Derive the exact flat index map by double distinct-value probing of the
    op's own execution seam; any failure or disagreement falls to opaque."""
    n_out = _flat_size(getattr(op, "output_shape", None))
    if n_out is None or n_out != n_in or _module_of(op) is None:
        return OPAQUE_TRANSFER

    probes = (
        [float(v + 1) for v in range(n_in)],
        [float(2 * n_in - v) for v in range(n_in)],
    )
    maps: List[List[int]] = []
    for values in probes:
        x = torch.tensor([values], dtype=torch.float64)
        try:
            with torch.no_grad():
                y = op.probe_on_gathered(x)
        except (RuntimeError, TypeError, ValueError, IndexError, KeyError):
            return OPAQUE_TRANSFER
        out_values = y.flatten().tolist()
        if len(out_values) != n_out:
            return OPAQUE_TRANSFER
        value_to_in = {v: i for i, v in enumerate(values)}
        out_to_in: List[int] = []
        for v in out_values:
            src = value_to_in.get(float(v))
            if src is None:
                return OPAQUE_TRANSFER
            out_to_in.append(src)
        if len(set(out_to_in)) != n_in:
            return OPAQUE_TRANSFER
        maps.append(out_to_in)
    if maps[0] != maps[1]:
        return OPAQUE_TRANSFER
    return _relation_transfer(
        TRANSFER_INDEX_BIJECTION,
        {o: frozenset({i}) for o, i in enumerate(maps[0])},
        n_in,
    )


def derive_liveness_transfer(
    op: ComputeOp,
    *,
    policy: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
) -> LivenessTransfer:
    """Derive the liveness transfer of one host ComputeOp (never raises for
    unknown ops — the conservative default is OPAQUE)."""
    policy = require_computeop_liveness_transfers(policy)
    n_in = int(len(op.input_sources.flatten()))
    if n_in == 0:
        return OPAQUE_TRANSFER

    if policy == COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY:
        if _computeop_relays_deadness(op):
            return _identity_transfer(TRANSFER_ELEMENTWISE_1TO1, n_in)
        return OPAQUE_TRANSFER

    params = getattr(op, "params", None) or {}
    if params.get("input_shapes") is not None:
        # Multi-input join (residual add, concat, attention): any live branch
        # keeps every output live — opaque, an honest barrier.
        return OPAQUE_TRANSFER

    module = _module_of(op)
    if module is None:
        if _computeop_relays_deadness(op):
            return _identity_transfer(TRANSFER_ELEMENTWISE_1TO1, n_in)
        return OPAQUE_TRANSFER

    if type(module) in _ELEMENTWISE_HOST_OP_TYPES:
        return _derive_elementwise(op, n_in)
    if (
        type(module) is nn.Flatten
        or str(getattr(op, "op_type", "")).lower() in _INDEX_BIJECTION_OP_TYPE_NAMES
    ):
        return _derive_index_bijection(op, n_in)
    if type(module) in REGION_REDUCE_HOST_OP_TYPES:
        return derive_region_reduce(op, n_in)
    return OPAQUE_TRANSFER
