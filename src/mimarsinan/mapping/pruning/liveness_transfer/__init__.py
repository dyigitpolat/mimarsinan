"""Per-op liveness-transfer framework [W4b]: how deadness crosses host ComputeOps.

- ``transfer_policy`` — the ``computeop_liveness_transfers`` config axis
  (``full`` default / ``identity_only`` kill-switch = the pre-W4b relay).
- ``transfer_registry`` — the per-op ``LivenessTransfer`` derivation:
  ELEMENTWISE_1TO1 (checked zero-preserving activations), INDEX_BIJECTION
  (probe-derived flatten/reshape/permute maps), REGION_REDUCE (pool
  receptive fields), and the conservative OPAQUE default for everything
  else (LayerNorm/softmax/attention/joins/unknown — never an error).
- ``transfer_index`` — composition through op chains into the graph-level
  maps the propagation kernels (closure, cascade, depth replay) consume.
"""

from mimarsinan.mapping.pruning.liveness_transfer.transfer_index import (
    ComputeOpTransferIndex,
    build_computeop_transfer_index,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_policy import (
    COMPUTEOP_LIVENESS_TRANSFERS_FULL,
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
    COMPUTEOP_LIVENESS_TRANSFERS_KEY,
    COMPUTEOP_LIVENESS_TRANSFERS_MODES,
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    require_computeop_liveness_transfers,
    resolve_computeop_liveness_transfers,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_registry import (
    OPAQUE_TRANSFER,
    TRANSFER_ELEMENTWISE_1TO1,
    TRANSFER_INDEX_BIJECTION,
    TRANSFER_OPAQUE,
    TRANSFER_REGION_REDUCE,
    LivenessTransfer,
    derive_liveness_transfer,
)

__all__ = [
    "COMPUTEOP_LIVENESS_TRANSFERS_KEY",
    "COMPUTEOP_LIVENESS_TRANSFERS_FULL",
    "COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY",
    "COMPUTEOP_LIVENESS_TRANSFERS_MODES",
    "DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS",
    "require_computeop_liveness_transfers",
    "resolve_computeop_liveness_transfers",
    "TRANSFER_ELEMENTWISE_1TO1",
    "TRANSFER_INDEX_BIJECTION",
    "TRANSFER_REGION_REDUCE",
    "TRANSFER_OPAQUE",
    "LivenessTransfer",
    "OPAQUE_TRANSFER",
    "derive_liveness_transfer",
    "ComputeOpTransferIndex",
    "build_computeop_transfer_index",
]
