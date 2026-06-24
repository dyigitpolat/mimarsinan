"""Layout verification, hardware suggester, wizard services."""

from mimarsinan.mapping.verification.onchip_majority import (
    OnchipMajorityError,
    OnchipParamBreakdown,
    assert_onchip_majority_or_raise,
    compute_onchip_fraction,
    count_host_params,
)

__all__ = [
    "OnchipMajorityError",
    "OnchipParamBreakdown",
    "assert_onchip_majority_or_raise",
    "compute_onchip_fraction",
    "count_host_params",
]
