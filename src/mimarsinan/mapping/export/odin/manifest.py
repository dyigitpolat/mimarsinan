"""The ODIN export manifest: the evidence artifact naming the law, the order, the gates."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.export.odin.feasibility import count_ceiling

EXPORT_FORMAT_VERSION = 1

#: The event-order contract every consumer of this export must implement.
ORDERING_VERSION = "canonical-event-order-v1"


def build_manifest(
    soma_law: SomaLaw,
    bounds: Dict[Tuple[int, int], int],
    geometry_rows: List[Dict[str, Any]],
    thetas: Dict[str, int],
    *,
    weight_bits: int,
    weight_sign_granularity: str,
    sign_expansion: int,
    effective_max_axons: int,
    cores_exported: int,
) -> Dict[str, Any]:
    """Every value here is one the export ALREADY checked; nothing is echoed unvalidated."""
    per_core_max: Dict[str, int] = {}
    for (core_index, _neuron), value in bounds.items():
        key = str(core_index)
        per_core_max[key] = max(per_core_max.get(key, 0), value)
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "soma_law": {
            "firing_mode": soma_law.firing_mode,
            "thresholding_mode": soma_law.thresholding_mode,
            "firing_granularity": soma_law.firing_granularity,
            "membrane_arithmetic": soma_law.membrane_arithmetic,
            "membrane_bits": int(soma_law.membrane_bits),
            "bias_slot": soma_law.bias_slot,
        },
        "ordering": {
            "version": ORDERING_VERSION,
            "bias_slot": soma_law.bias_slot,
            "slots": "ascending",
            "adjacency": "one slot's multiplicity is delivered adjacently",
            "row_pair": "logical slot a occupies physical rows (2a, 2a+1)",
        },
        "geometry": {
            "sign_expansion": int(sign_expansion),
            "cores": geometry_rows,
        },
        "emission_bounds": {
            "ceiling": count_ceiling(soma_law),
            "max": max(bounds.values(), default=0),
            "per_core_max": per_core_max,
        },
        "feasibility": {
            "gates": {
                "theta_ceiling": True,
                "weight_magnitude_range": True,
                "sign_granularity": True,
                "fan_in": True,
                "emission_bound": True,
            },
            "theta_per_core": thetas,
            "weight_bits": int(weight_bits),
            "weight_sign_granularity": weight_sign_granularity,
            "effective_max_axons": int(effective_max_axons),
            "cores_exported": int(cores_exported),
        },
    }
