"""Greedy bin-packing for softcores into hardcores."""

from mimarsinan.mapping.packing.canonical import (
    HardCoreLike,
    HardT,
    SoftCoreLike,
    SoftT,
    canonical_fuse_hardcores,
    canonical_is_mapping_possible,
    canonical_split_softcore,
    pick_best_softcore,
)
from mimarsinan.mapping.packing.greedy import greedy_pack_softcores

__all__ = [
    "HardCoreLike",
    "HardT",
    "SoftCoreLike",
    "SoftT",
    "canonical_fuse_hardcores",
    "canonical_is_mapping_possible",
    "canonical_split_softcore",
    "greedy_pack_softcores",
    "pick_best_softcore",
]
