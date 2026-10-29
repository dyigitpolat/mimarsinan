"""Band algebra + the component tables the absolute pricer prices from.

Every band here is positive and every operation monotone, so corner arithmetic is
elementwise — except inversion and division, whose corners FLIP (the documented
dividing-coefficient discipline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.compute_technology import share_physics
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


@dataclass(frozen=True)
class DerivedFactor:
    """A component factor read off several census quantities: ``prod(over) / prod(per)``.

    ``over[0]`` is the census that carries the work: absent, there is no work to name. Present and
    positive while another input is absent (or the divisor is 0), the factor is WITHHELD with the
    reason, so its component is named unpriced on its headline, never dropped.
    """

    label: str
    over: Tuple[str, ...]
    per: Tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.label

    def value(self, quantities: Quantities) -> Optional[float]:
        work, per = quantity_product(quantities, self.over), quantity_product(quantities, self.per)
        if work is None or per is None or (per == 0 and work):
            return None
        return work / per if per else 0.0

    def withheld(self, quantities: Quantities) -> Optional[str]:
        lead = self.over[0]
        if not quantities.has(lead):
            return quantities.withheld_reason(lead)
        if quantities.get(lead).value == 0:
            return None
        for key in self.over[1:] + self.per:
            if not quantities.has(key):
                return quantities.withheld_reason(key) or f"the {lead} census states no {key}"
        if quantity_product(quantities, self.per) == 0:
            return f"{lead} > 0 on {' x '.join(self.per)} = 0"
        return None


Factor = Union[str, DerivedFactor]

#: [Lanes] The lane block's factors, read only from a capability census (none by default).
LANES = DerivedFactor("lanes_per_tile x tiles", over=("lanes_per_tile", "tiles"))
LANE_OPS_PER_LANE = DerivedFactor("lane_ops / (lanes_per_tile x tiles)", over=("lane_ops",),
                                  per=("lanes_per_tile", "tiles"))

#: Optional energy components: (constant, quantity factors, component label).
#: Priced when the constant survives supersession AND every factor is present;
#: quantity-present-but-constant-undeclared components are NAMED on the headline.
ENERGY_COMPONENTS: Tuple[Tuple[str, Tuple[Factor, ...], str], ...] = (
    ("e_adc_conversion", ("adc_conversions",), "adc"),
    ("e_row_drive", ("boundary_events",), "row_drive"),
    ("e_neuron_update", ("neurons_used", "timesteps"), "neuron_update"),
    ("e_leak_per_neuron_step", ("neurons_used", "timesteps"), "leak"),
    ("e_intra_tile_packet", ("noc_intra_tile_packets",), "intra_tile"),
    ("e_inter_tile_hop", ("noc_total_hops",), "inter_tile"),
    ("e_sync_barrier", ("sync_count",), "sync"),
    # [E4] Per INFERENCE, not per program load: a pass resets the cores it uses
    # every time it runs, which is exactly when t_core_init is charged too.
    ("e_core_init", ("segment_cores",), "core_init"),
    # [E3] The intra-segment pass boundary is crossed TWICE per inference, and
    # the two crossings ride different paths: the host reads emissions back off
    # the chip, then re-injects a train over the DMA channel programming uses.
    # Charged separately, with their own constants; neither is the buffer
    # figure (carried_raster_bytes), which sizes what is HELD in between.
    ("e_dma_per_byte", ("carry_in_bytes",), "carry_inject"),
    ("e_readout_per_byte", ("carry_out_bytes",), "carry_readout"),
    # [U-F4] A declared on-chip carry buffer's accesses: a wavefront group's own values, written once and read once.
    ("e_carry_buffer_per_byte", ("carry_onchip_bytes",), "carry_buffer"),
    # [CH-D2] A value-domain program's inter-core values, per byte-hop.
    ("e_noc_value_per_byte_hop", ("noc_value_bytes",), "noc_value"),
    # Element operations on arithmetic lanes (present only when a capability census supplies them).
    ("e_lane_op", ("lane_ops",), "lane"),
)

#: [U-F4] The declared on-chip carry buffer's capacity quantity and the constants priced on it: its static power
#: (over the priced latency) and its area (beside the compute footprint, ``CHIP_AREA_ADDERS``).
BUFFER_CAPACITY = "carry_buffer_capacity_bytes"
BUFFER_STATIC = "p_carry_buffer_static_per_byte"
_BUFFER_CONSTANTS = ((BUFFER_STATIC, "the buffer's static power per declared byte"),
                     ("area_carry_buffer_per_byte", "the buffer's area per declared byte"))


def carry_buffer_refusals(quantities: Any, physics: Any,
                          unstated: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """[U-F4] The declared on-chip carry buffer refuses the energy by name where the program does not fit it (the
    carry census withheld with the reason) or the target states no access energy for it, and the energy (its static
    power) or the chip area by name where the target states no constant for the declared capacity. [U-G] A constant
    the chip's carry-buffer table leaves unstated at the declared capacity refuses with the table's reason
    (``unstated``, ``PlatformPhysics.at_carry_buffer``)."""
    refused: Dict[str, str] = {}
    unstated = unstated or {}
    constant, key = "e_carry_buffer_per_byte", "carry_onchip_bytes"
    if key in getattr(quantities, "withheld", {}):
        refused[constant] = quantities.withheld[key]
    elif quantities.has(key) and quantities.get(key).value and not physics.has(constant):
        refused[constant] = unstated.get(constant) or (
            f"the declared on-chip carry buffer holds carried bytes, and the target declares no {constant} (the "
            "buffer's access energy, with its source)")
    capacity = quantities.get(BUFFER_CAPACITY).value if quantities.has(BUFFER_CAPACITY) else 0
    for name, what in _BUFFER_CONSTANTS:
        if capacity and not physics.has(name):
            refused[name] = unstated.get(name) or (f"the chip declares an on-chip carry buffer of {int(capacity)} B, "
                                                   f"and the target declares no {name} ({what}, with its source)")
    return refused


def carry_buffer_static_watts(quantities: Quantities, physics: PlatformPhysics) -> Optional[Band]:
    """[U-F4] The declared on-chip carry buffer's static power on its declared capacity, None without both."""
    if not (physics.has(BUFFER_STATIC) and quantities.has(BUFFER_CAPACITY)):
        return None
    return scale_band(physics.band(BUFFER_STATIC), quantities.get(BUFFER_CAPACITY).value,
                      f"{BUFFER_STATIC} x {BUFFER_CAPACITY}")


#: [TS6] The per-byte program-load energies, in the order a byte incurs them:
#: (constant, component label). Both multiply the SAME payload bytes and are
#: distinct physical costs — moving the byte onto the chip, then committing it to
#: the weight store — so a target declaring both pays both, and one declaring only
#: the number its paper published prices only that.
PROGRAMMING_BYTE_ENERGIES: Tuple[Tuple[str, str], ...] = (
    ("e_dma_per_byte", "dma"),
    ("e_program_per_byte", "commit"),
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

#: Area beside the compute footprint, on EITHER area path (an aggregate supersedes none of it).
CHIP_AREA_ADDERS: Tuple[Tuple[str, Tuple[Factor, ...], str], ...] = (
    ("area_per_lane", (LANES,), "lanes"),
    ("area_carry_buffer_per_byte", (BUFFER_CAPACITY,), "carry_buffer"),  # [U-F4] on the declared capacity
)

#: Steady-state latency adders beyond t_cycle × latency_steps.
LATENCY_COMPONENTS: Tuple[Tuple[str, Tuple[Factor, ...], str], ...] = (
    ("t_sync_barrier", ("sync_count",), "sync"),
    ("t_core_init", ("segment_cores",), "core_init"),
    # The standalone lane pass: a tile's lanes run in parallel.
    ("t_lane_op", (LANE_OPS_PER_LANE,), "lanes"),
)


def quantity_product(quantities: Quantities, keys: Sequence[Factor]) -> Optional[float]:
    """The product of the named quantities (or derived factors), or None when any is absent."""
    product = 1.0
    for key in keys:
        if isinstance(key, DerivedFactor):
            value = key.value(quantities)
            if value is None:
                return None
            product *= value
            continue
        if not quantities.has(key):
            return None
        product *= quantities.get(key).value
    return product


def factor_withheld(quantities: Quantities, factor: Factor) -> Optional[str]:
    """Why ``factor``'s work may exist although the view cannot size it, or None."""
    if isinstance(factor, DerivedFactor):
        return factor.withheld(quantities)
    return quantities.withheld_reason(factor)


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


def per_core_type_compute(split: Any, chip: PlatformPhysics, compute_constant: Optional[str], synaptic_events: float,
                          scale: float) -> Tuple[Optional[List[Band]], List[str], Optional[str]]:
    """[E-6] (bands, evidence, refusal reason): each core type's MAC events at its own e_mac
    (``quantities/compute_technology``), x ``scale`` (J to the headline's unit); refused by name where the plane
    states no split, the split differs from the MAC census, the chip prices its compute by a per-event aggregate,
    or a share's physics declares no e_mac."""
    if split.unavailable is not None:
        return None, [], f"per-core-type compute physics: {split.unavailable}"
    if compute_constant != "e_mac":
        return None, [], ("per-core-type compute physics prices e_mac per core type; the chip prices its compute "
                          "by the per-event aggregate e_synaptic_event_total, which is not split per core type")
    if abs(float(split.events) - float(synaptic_events)) > 0.5:
        return None, [], (f"per-core-type compute physics: the per-core-type MAC split ({split.events} events) "
                          f"differs from the synaptic_events census ({synaptic_events:g})")
    bands: List[Band] = []
    evidence: List[str] = []
    for share in split.shares:
        physics = share_physics(share, chip)
        if not physics.has("e_mac"):
            return None, [], f"per-core-type compute physics: {share.label} declares no e_mac"
        value = physics.band("e_mac")
        bands.append(scale_band(value, share.events * scale, f"e_mac of {share.label} x its MAC events"))
        evidence.append(f"compute: e_mac of {share.label} [{value.basis}] x {share.events} MAC events")
    return bands, evidence, None
