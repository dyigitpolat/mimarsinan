"""Convenience wrapper to map a model representation to an IRGraph."""

from __future__ import annotations

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.ir_mapping_class import IRMapping

def map_model_to_ir(
    model_representation,
    q_max: float = 1.0,
    firing_mode: str = "Default",
    max_axons: int | None = None,
    max_neurons: int | None = None,
    allow_coalescing: bool = False,
    hardware_bias: bool = False,
) -> IRGraph:
    """Convenience wrapper around ``IRMapping.map``."""
    return IRMapping(
        q_max=q_max,
        firing_mode=firing_mode,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    ).map(model_representation)
