"""Band algebra + the component tables the absolute pricer prices from.

Every band here is positive and every operation monotone, so corner arithmetic is
elementwise — except inversion and division, whose corners FLIP (the documented
dividing-coefficient discipline).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.provenance import Band


def scale_band(band: Band, factor: float, basis: str) -> Band:
    """``band`` × a non-negative scalar quantity."""
    if factor < 0:
        raise ValueError(f"quantities are non-negative, got {factor}")
    return Band(band.low * factor, band.nominal * factor, band.high * factor, basis)


def mul_bands(a: Band, b: Band, basis: str) -> Band:
    """Elementwise product of two positive bands (monotone, corners aligned)."""
    return Band(a.low * b.low, a.nominal * b.nominal, a.high * b.high, basis)


def add_bands(bands: Sequence[Band], basis: str) -> Band:
    if not bands:
        raise ValueError("nothing to sum")
    return Band(
        sum(b.low for b in bands),
        sum(b.nominal for b in bands),
        sum(b.high for b in bands),
        basis,
    )


def invert_band(band: Band, basis: str) -> Band:
    """1/band — corners FLIP so the result stays a (low, nominal, high) band."""
    if band.low <= 0:
        raise ValueError(f"cannot invert a band touching zero: {band}")
    return Band(1.0 / band.high, 1.0 / band.nominal, 1.0 / band.low, basis)


def div_band(a: Band, b: Band, basis: str) -> Band:
    """a / b with flipped divisor corners (the dividing-coefficient discipline)."""
    return mul_bands(a, invert_band(b, basis), basis)


#: Optional energy components: (constant, quantity factors, component label).
#: Priced when the constant survives supersession AND every factor is present;
#: quantity-present-but-constant-undeclared components are NAMED on the headline.
ENERGY_COMPONENTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("e_adc_conversion", ("adc_conversions",), "adc"),
    ("e_row_drive", ("boundary_events",), "row_drive"),
    ("e_neuron_update", ("neurons_used", "timesteps"), "neuron_update"),
    ("e_leak_per_neuron_step", ("neurons_used", "timesteps"), "leak"),
    ("e_intra_tile_packet", ("noc_intra_tile_packets",), "intra_tile"),
    ("e_inter_tile_hop", ("noc_total_hops",), "inter_tile"),
    ("e_sync_barrier", ("sync_count",), "sync"),
)

#: Decomposed area components, priced only when no area aggregate supersedes them.
AREA_COMPONENTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("area_per_cell", ("cells_physical",), "cells"),
    ("area_per_cell_per_weight_bit", ("cells_physical", "weight_bits"), "cell_bits"),
    ("area_per_row_driver", ("axons_physical",), "row_drivers"),
    ("area_per_adc", ("adc_count",), "adcs"),
    ("area_per_neuron_logic", ("neurons_physical",), "neuron_logic"),
    ("area_per_router", ("tiles",), "routers"),
    ("area_per_tile_fixed", ("tiles",), "tile_fixed"),
)

#: Steady-state latency adders beyond t_cycle × latency_steps.
LATENCY_COMPONENTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("t_sync_barrier", ("sync_count",), "sync"),
    ("t_core_init", ("segment_cores",), "core_init"),
)


def quantity_product(quantities: Quantities, keys: Sequence[str]) -> Optional[float]:
    """The product of the named quantities, or None when any is absent."""
    product = 1.0
    for key in keys:
        if not quantities.has(key):
            return None
        product *= quantities.get(key).value
    return product


def programming_payload_bytes(
    quantities: Quantities, physics: PlatformPhysics
) -> Optional[Tuple[float, str]]:
    """Reprogrammed payload bytes: weights + connectivity at the declared wire width.

    Returns ``(bytes, basis note)`` or None when the byte census is absent. The
    connectivity contribution needs the target's declared entry width; without it the
    weight bytes still price, and the note says what was left out.
    """
    if not quantities.has("reprogrammed_bytes"):
        return None
    total = quantities.get("reprogrammed_bytes").value
    note = "weight payload bytes"
    if quantities.has("connectivity_entries"):
        entries = quantities.get("connectivity_entries").value
        if physics.has("bytes_per_connectivity_entry"):
            width = physics.band("bytes_per_connectivity_entry")
            total += entries * width.nominal
            note = "weight payload + connectivity entries at the declared wire width"
        elif entries > 0:
            note = (
                "weight payload bytes; connectivity entries present but "
                "bytes_per_connectivity_entry is undeclared, so they are unpriced"
            )
    return total, note
