"""Layout verification, hardware suggester, wizard services."""

from mimarsinan.mapping.verification.onchip_majority import (
    OnchipMajorityError,
    OnchipParamBreakdown,
    assert_onchip_majority_or_raise,
    compute_onchip_fraction,
    count_host_params,
)
from mimarsinan.mapping.verification.onchip_fraction import (
    OnchipFractionEstimate,
    ValidityVerdict,
    assert_onchip_majority_estimate_or_raise,
    classify_validity,
    estimate_onchip_fraction,
)

__all__ = [
    "OnchipMajorityError",
    "OnchipParamBreakdown",
    "assert_onchip_majority_or_raise",
    "compute_onchip_fraction",
    "count_host_params",
    "OnchipFractionEstimate",
    "ValidityVerdict",
    "assert_onchip_majority_estimate_or_raise",
    "classify_validity",
    "estimate_onchip_fraction",
]
