"""On-chip attention / LayerNorm mappability frontier: realizable LayerNorm centering plus realizability verdicts for non-mappable attention/LN sub-ops."""

from mimarsinan.mapping.onchip_attention.attention_mappability import (
    AffineRealizability,
    affine_realizability_report,
)
from mimarsinan.mapping.onchip_attention.onchip_layernorm import (
    OnchipLayerNormCentering,
    build_centering_matrix,
)

__all__ = [
    "AffineRealizability",
    "affine_realizability_report",
    "OnchipLayerNormCentering",
    "build_centering_matrix",
]
