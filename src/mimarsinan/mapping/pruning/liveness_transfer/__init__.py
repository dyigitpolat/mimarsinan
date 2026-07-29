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

[W4b-2] The same framework carries the CONSTANT LATTICE, which generalizes
"dead" (CONST(0)) to "carries a known value every timestep":

- ``constant_policy`` — the ``elimination_constant_folding`` axis
  (``full`` default / ``off`` kill-switch, forced off by ``identity_only``)
  plus the chip-domain exactness gate for non-zero constants;
- ``constant_lattice`` — ``TOP > CONST(c)`` with monotone descent and the
  single line-value query every rule shares;
- ``constant_transfer`` — the generic ComputeOp forward rule (execute the
  op's own seam under determinism / batch / dtype / support probes);
- ``constant_core`` — the exact NeuralCore fold (CONST rows onto the core's
  existing constant carrier) and the CONST column rule that makes a
  bias-only core an ordinary constant producer.
"""

from mimarsinan.mapping.pruning.liveness_transfer.constant_core import (
    ConstantCarrier,
    CoreConstantFacts,
    derive_core_constants,
    resolve_constant_carrier,
)
from mimarsinan.mapping.pruning.liveness_transfer.constant_lattice import (
    ConstantLattice,
    ConstantLatticeError,
    source_constant,
)
from mimarsinan.mapping.pruning.liveness_transfer.constant_policy import (
    DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    ELIMINATION_CONSTANT_FOLDING_FULL,
    ELIMINATION_CONSTANT_FOLDING_KEY,
    ELIMINATION_CONSTANT_FOLDING_MODES,
    ELIMINATION_CONSTANT_FOLDING_OFF,
    domain_admits_nonzero_constants,
    effective_constant_folding,
    require_elimination_constant_folding,
    resolve_elimination_constant_folding,
)
from mimarsinan.mapping.pruning.liveness_transfer.constant_transfer import (
    derive_constant_outputs,
)
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
    "ConstantCarrier",
    "ConstantLattice",
    "ConstantLatticeError",
    "CoreConstantFacts",
    "DEFAULT_ELIMINATION_CONSTANT_FOLDING",
    "ELIMINATION_CONSTANT_FOLDING_FULL",
    "ELIMINATION_CONSTANT_FOLDING_KEY",
    "ELIMINATION_CONSTANT_FOLDING_MODES",
    "ELIMINATION_CONSTANT_FOLDING_OFF",
    "derive_constant_outputs",
    "derive_core_constants",
    "domain_admits_nonzero_constants",
    "effective_constant_folding",
    "require_elimination_constant_folding",
    "resolve_constant_carrier",
    "resolve_elimination_constant_folding",
    "source_constant",
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
