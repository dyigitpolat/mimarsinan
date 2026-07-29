"""Cascade equivalence certificate (W1): per-instance proof that structured elimination preserved semantics."""

from mimarsinan.mapping.pruning.certificate.cascade_certificate import (
    CascadeEquivalenceCertificate,
    certify_cascade_equivalence,
)
from mimarsinan.mapping.pruning.certificate.seed_reference import (
    check_shared_bank_union_rule,
)
from mimarsinan.mapping.pruning.certificate.dyadic_grid import (
    assert_dyadic_exactness_grid,
    snap_ir_graph_to_dyadic_grid,
)
from mimarsinan.mapping.pruning.certificate.errors import (
    CascadeCertificateError,
    CascadeCertificatePreconditionError,
)
from mimarsinan.mapping.pruning.certificate.zero_preserving import (
    NON_ZERO_PRESERVING_HOST_OP_TYPES,
    ZERO_PRESERVING_HOST_OP_TYPES,
    assert_zero_preserving_preconditions,
    derive_cols_with_implicit_source,
    is_zero_preserving_host_op,
    op_outputs_fully_constant,
)

__all__ = [
    "CascadeCertificateError",
    "CascadeCertificatePreconditionError",
    "CascadeEquivalenceCertificate",
    "NON_ZERO_PRESERVING_HOST_OP_TYPES",
    "ZERO_PRESERVING_HOST_OP_TYPES",
    "assert_dyadic_exactness_grid",
    "assert_zero_preserving_preconditions",
    "certify_cascade_equivalence",
    "check_shared_bank_union_rule",
    "derive_cols_with_implicit_source",
    "is_zero_preserving_host_op",
    "op_outputs_fully_constant",
    "snap_ir_graph_to_dyadic_grid",
]
