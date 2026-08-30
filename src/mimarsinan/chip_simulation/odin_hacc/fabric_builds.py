"""Which fabric a bundle's passes are built for — the ONE place that chooses.

The choice is made on the CLAIMS a bundle carries (``ChipConfig.bundle_claims``),
which is the same predicate the shipped reader dispatches its AER wording on. A
crossbar that signs a whole PRE-synaptic row is the vendored core and nothing
else; a synapse cell that signs itself is a generated one, and a generated core
only exists as a NAMED chip configuration — so those claims must resolve to one
rather than describing a geometry no cosimulation ever proved.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from mimarsinan.chip_simulation.odin_fpga.chip_selection import chip_config_for
from mimarsinan.chip_simulation.odin_hacc.pass_build import PassBuild
from mimarsinan.chip_simulation.odin_hacc.variant_pass_build import VariantPassBuild
from mimarsinan.chip_simulation.odin_rtl.reference import CycleTrace
from mimarsinan.chip_simulation.soma_axes import PER_SYNAPSE_SIGN


def pass_builds(mapping: Any, traces: Sequence[CycleTrace], *, weight_bits: int,
                effective_max_axons: int, weight_sign_granularity: str,
                soma_law: Any, membrane_init: int) -> List[Any]:
    """One build per core, on the fabric these claims identify."""
    if str(weight_sign_granularity) == PER_SYNAPSE_SIGN:
        chip = chip_config_for(
            weight_bits=int(weight_bits),
            weight_sign_granularity=str(weight_sign_granularity),
            effective_max_axons=int(effective_max_axons),
            membrane_bits=int(soma_law.membrane_bits))
        return [
            VariantPassBuild(mapping, index, traces[0], spec=chip.core_spec,
                             membrane_init=int(membrane_init))
            for index in range(len(mapping.cores))
        ]
    return [
        PassBuild(mapping, index, traces[0], weight_bits=int(weight_bits),
                  effective_max_axons=int(effective_max_axons),
                  soma_law=soma_law,
                  weight_sign_granularity=str(weight_sign_granularity),
                  membrane_init=int(membrane_init))
        for index in range(len(mapping.cores))
    ]
