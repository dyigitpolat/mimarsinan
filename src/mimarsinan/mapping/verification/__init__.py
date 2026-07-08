"""Layout verification, hardware suggester, wizard services."""

from mimarsinan.mapping.verification.onchip_majority import (
    DEFAULT_ONCHIP_FLOOR,
    DEFAULT_ONCHIP_MAJORITY,
    OnchipMajorityError,
    OnchipOpsBreakdown,
    OnchipParamBreakdown,
    assert_onchip_majority_or_raise,
    compute_onchip_fraction,
    compute_onchip_ops_fraction,
    count_host_ops,
    count_host_params,
)
from mimarsinan.mapping.verification.onchip_fraction import (
    OnchipFractionEstimate,
    OnchipValidityReport,
    ValidityVerdict,
    assert_onchip_majority_estimate_or_raise,
    assert_onchip_validity_or_raise,
    classify_validity,
    estimate_onchip_fraction,
)
from mimarsinan.mapping.verification.capacity import (
    CapacityEstimate,
    CapacityExceededError,
    estimate_cores_needed,
)

__all__ = [
    "OnchipMajorityError",
    "OnchipOpsBreakdown",
    "OnchipParamBreakdown",
    "assert_onchip_majority_or_raise",
    "compute_onchip_fraction",
    "compute_onchip_ops_fraction",
    "count_host_ops",
    "count_host_params",
    "OnchipFractionEstimate",
    "OnchipValidityReport",
    "ValidityVerdict",
    "assert_onchip_majority_estimate_or_raise",
    "assert_onchip_validity_or_raise",
    "classify_validity",
    "estimate_onchip_fraction",
    "DEFAULT_ONCHIP_FLOOR",
    "DEFAULT_ONCHIP_MAJORITY",
    "CapacityEstimate",
    "CapacityExceededError",
    "estimate_cores_needed",
]
