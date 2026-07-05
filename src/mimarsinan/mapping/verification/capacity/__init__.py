"""Static placement-capacity diagnostic (E4): hard-core count without placement."""

from mimarsinan.mapping.verification.capacity.dryrun import (
    PackFeasibility,
    dryrun_pack_feasible,
)
from mimarsinan.mapping.verification.capacity.estimate import (
    PACKER_DIVERGENCE_MARGIN,
    CapacityEstimate,
    CapacityExceededError,
    estimate_cores_needed,
)

__all__ = [
    "CapacityEstimate",
    "CapacityExceededError",
    "PACKER_DIVERGENCE_MARGIN",
    "estimate_cores_needed",
    "PackFeasibility",
    "dryrun_pack_feasible",
]
