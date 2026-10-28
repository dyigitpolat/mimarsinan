"""The value-domain quantity rule: one evaluation per stage, MAC events, carry at the value grid.

A value-domain (MVM) program computes each committed cell once per inference and carries values,
not spike rasters, across its pass boundaries. The resolved platform names the value domain
(``core_semantics`` rides it exactly for value-domain runs), so both pricing planes read one
surface and every spiking quantity is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.write_census import stage_latency
from mimarsinan.deployment_record.quantities.spec import QuantityValue
from mimarsinan.mapping.noc.value_traffic import stage_value_noc_wire_hops
from mimarsinan.mapping.support.period_census import natural
from mimarsinan.mapping.support.schedule.pass_cut import raster_bytes

#: The resolved-surface key naming the value domain, and the value grid beside it.
SEMANTICS_KEY = "core_semantics"
VALUE_BITS_KEY = "activation_bits"
#: Spike-raster quantities a value-domain program never produces.
RASTER_ONLY = ("carried_raster_bytes",)
#: Why a value carry or value NoC census that is not zero has no size without the declared grid.
NO_VALUE_GRID = "activation_bits undeclared: the value carry has no grid"
NO_VALUE_NOC_GRID = "activation_bits undeclared: the value NoC traffic has no grid"
#: Why a sealed record states no value NoC census (it seals no ``value_noc_wire_hops``: every archived record).
NO_RECORD_WIRES = "the sealed record keeps no wire census (KA-G3-1)"
#: [U-F4] The quantity of the carried bytes a declared on-chip carry buffer holds (each written once, read once), its
#: census key, and the platform component that declares the buffer (its capacity in bytes).
ONCHIP_CARRY, ONCHIP_CENSUS, CARRY_BUFFER_KEY = "carry_onchip_bytes", "onchip_bytes", "carry_buffer_bytes"
#: [U-F4] The quantity of the buffer's declared capacity, which its static power and area multiply.
BUFFER_CAPACITY = "carry_buffer_capacity_bytes"
#: [U-F4] Why a value-domain group's live carry has no size in a declared on-chip carry buffer without the grid.
NO_BUFFER_GRID = (f"activation_bits undeclared: the live values a wavefront group holds in the declared "
                  f"{CARRY_BUFFER_KEY} have no grid, so their fit is not decided")
_CARRY_KEYS = (("carry_out_bytes", "boundary_out_bytes"), ("carry_in_bytes", "boundary_in_bytes"),
               (ONCHIP_CARRY, ONCHIP_CENSUS))


class ValueDomainActivityError(ToolchainRefusal, ValueError):
    """A spiking switching-activity assumption declared on a value-domain program."""


class ValueDomainCensusError(ToolchainRefusal, ValueError):
    """A carry census that is not the dense crossing census the value rule inverts."""


def names_value_domain(platform: Mapping[str, Any]) -> bool:
    """Whether the resolved surface names a value-domain run (absent on every spiking run)."""
    semantics = platform.get(SEMANTICS_KEY)
    return semantics is not None and is_mvm_core_semantics(semantics)


def value_bits(platform: Mapping[str, Any]) -> Optional[int]:
    """The declared value grid (``activation_bits``), or None when the run declares none."""
    bits = platform.get(VALUE_BITS_KEY)
    return None if bits is None else natural(bits, VALUE_BITS_KEY, error=ValueDomainCensusError)


def record_value_noc_wire_hops(hybrid_mapping: Any, platform: Mapping[str, Any]) -> Optional[int]:
    """[KA-G3-1] The value-hops a value-domain build seals (``ScheduleRecord.value_noc_wire_hops``): the built
    stages' wire census on the resolved floorplan (``stage_value_noc_wire_hops``); None for a spiking run or a
    platform that resolves no floorplan."""
    cores_per_tile = int(platform.get("cores_per_tile_resolved", 0) or 0)
    rows = int(platform.get("tile_grid_rows_resolved", 0) or 0)
    if not names_value_domain(platform) or cores_per_tile <= 0 or rows <= 0:
        return None
    return stage_value_noc_wire_hops(hybrid_mapping.stages, cores_per_tile=cores_per_tile, mesh_height=rows)


def value_noc_reading(hops: Optional[int], bits: Optional[int]) -> Tuple[Optional[float], Optional[str]]:
    """``(noc_value_bytes, withheld reason)`` of a value NoC census, one rule on both planes: the byte-hops at the
    declared grid; zero traffic is zero whatever the grid; other traffic without the grid is withheld by name."""
    if hops is None:
        return None, None
    if bits:
        return hops * int(bits) / 8.0, None
    return (0.0, None) if hops == 0 else (None, NO_VALUE_NOC_GRID)


def refuse_spiking_activity(platform: Mapping[str, Any]) -> None:
    """A declared ``activity_factor`` > 0 is a spiking switching assumption: refused by name."""
    activity = float(platform.get("activity_factor", 0.0) or 0.0)
    if activity > 0.0:
        raise ValueDomainActivityError(
            f"activity_factor={activity} is declared on a value-domain program; it is a spiking "
            "switching assumption, and a value-domain core computes every committed cell once per "
            "inference (MAC events = committed cells), so the declaration is refused"
        )


def value_window(stage_latencies: Iterable[int]) -> int:
    """Executed steps of a value program: each stage evaluates once, its latency plus input delivery."""
    return sum(natural(latency, "stage latency", error=ValueDomainCensusError) + 1
               for latency in stage_latencies)


def crossing_elements(boundary_bytes: int, timesteps: int) -> int:
    """Carried elements behind a dense crossing census (``raster_bytes`` = width x ceil(T / 8)).

    The exact inverse of the census both planes seal, read per direction, so every train the census
    counts (a delayed slice's extra re-injection included) is counted here; a byte count the census
    cannot have produced refuses by name rather than being truncated.
    """
    steps = natural(timesteps, "carry timesteps", error=ValueDomainCensusError)
    total = natural(boundary_bytes, "carry boundary bytes", error=ValueDomainCensusError)
    per_element = (steps + 7) // 8
    if per_element == 0 or total % per_element:
        raise ValueDomainCensusError(
            f"carry boundary bytes {total} are not a dense crossing census at {steps} timesteps "
            f"({per_element} B per carried element)"
        )
    return total // per_element


def value_bytes(elements: int, bits: int) -> int:
    """``ceil(elements x bits / 8)``: carried values at the declared grid."""
    return -((natural(elements, "carried elements") * natural(bits, "value bits")) // -8)


def value_carry(
    carry: Optional[Mapping[str, Any]], timesteps: Optional[int], bits: Optional[int]
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """The value carry per direction at the value grid, and the directions it withholds.

    Without a carry census nothing is stated. A direction the census shows empty carries 0 bytes
    whatever the grid; a nonzero one needs ``activation_bits`` (and the window it was censused at),
    and without them it is withheld by name, so its carry term is named unpriced, never dropped.
    """
    if carry is None:
        return {}, {}
    sized: Dict[str, int] = {}
    withheld: Dict[str, str] = {}
    for key, census_key in _CARRY_KEYS:
        if census_key not in carry:  # [U-F4] a census of no on-chip carry states no such bytes
            continue
        total = natural(int(carry[census_key]), "carry boundary bytes", error=ValueDomainCensusError)
        if total == 0:
            sized[key] = 0
        elif bits is None or timesteps is None:
            withheld[key] = NO_VALUE_GRID if bits is None else "the carry census states no window"
        else:
            sized[key] = value_bytes(crossing_elements(total, int(timesteps)), bits)
    return sized, withheld


def declared_buffer_bytes(platform: Mapping[str, Any]) -> Optional[int]:
    """[U-F4] The declared on-chip carry buffer's capacity in bytes (``carry_buffer_bytes``), None when undeclared."""
    capacity = platform.get(CARRY_BUFFER_KEY)
    return None if capacity is None else natural(capacity, CARRY_BUFFER_KEY, error=ValueDomainCensusError)


def buffer_live_bytes(carry: Mapping[str, Any], platform: Mapping[str, Any]) -> Tuple[Optional[int], str]:
    """[U-F4] ``(bytes, what they are)`` of the largest live carry a wavefront group holds in a declared on-chip carry
    buffer, in the run's domain, one rule on both planes: a value-domain buffer holds the carried values at the value
    grid (``value_bytes`` at ``activation_bits``; None without the grid), a spiking buffer what the pass transfer
    buffers (the census's raster under verbatim, its window counts under collapse)."""
    if not names_value_domain(platform):
        return int(carry["onchip_peak_live_bytes"]), "the pass transfer's live rasters or window counts"
    if "onchip_peak_live_elements" not in carry:
        raise ValueDomainCensusError("the carry census states no onchip_peak_live_elements: a value-domain buffer "
                                     "sizes the live carried values, not the transfer's bytes")
    bits = value_bits(platform)
    if bits is None:
        return None, NO_BUFFER_GRID
    return value_bytes(int(carry["onchip_peak_live_elements"]), bits), f"live values at activation_bits {bits}"


def onchip_carry(carry: Optional[Mapping[str, Any]], platform: Mapping[str, Any]
                 ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """[U-F4] ``carry`` held as the platform declares it: without a declared ``carry_buffer_bytes`` every carried value
    crosses the boundary (the census unchanged); with one, a wavefront group's own values (``onchip_*``) stay in the
    buffer (their crossings leave the boundary census and enter ``onchip_bytes``), and the reason a group's live carry,
    sized in the run's domain (``buffer_live_bytes``), does not fit the buffer, which refuses the program by name. A
    census without wavefront groups is unchanged."""
    capacity = declared_buffer_bytes(platform)
    if carry is None or capacity is None or "onchip_out_bytes" not in carry:
        return (None if carry is None else dict(carry)), None
    need, what = buffer_live_bytes(carry, platform)
    held = {k: v for k, v in carry.items() if not k.startswith("onchip_peak_live")}
    out, inbound = int(held.pop("onchip_out_bytes")), int(held.pop("onchip_in_bytes"))
    held["boundary_out_bytes"] = int(held["boundary_out_bytes"]) - out
    held["boundary_in_bytes"] = int(held["boundary_in_bytes"]) - inbound
    held[ONCHIP_CENSUS] = out + inbound
    if need is None:
        return held, what
    refusal = None if need <= capacity else (
        f"a wavefront group holds {need} B ({what}) on chip, above the declared {CARRY_BUFFER_KEY} {capacity}: "
        "the program does not fit the on-chip carry buffer")
    return held, refusal


@dataclass(frozen=True)
class ValueDomainFacts:
    """The value rule's answers on one plane; None is absent, never zero."""

    mac_events: Optional[int]
    latency_steps: Optional[int]
    carry_out_bytes: Optional[int]
    carry_in_bytes: Optional[int]
    #: Quantities whose work may exist but which this plane cannot size, with the reason.
    withheld: Tuple[Tuple[str, str], ...] = ()
    #: [KA-G3-1] The record's value NoC byte-hops (the candidate plane states them on its view).
    noc_value_bytes: Optional[float] = None
    #: [U-F4] The value bytes a declared on-chip carry buffer holds (each written once, read once), else None.
    onchip_bytes: Optional[int] = None


def apply_value_rule(values: Dict[str, QuantityValue], facts: ValueDomainFacts) -> Dict[str, str]:
    """Replace a plane's spiking-rule quantities by the value rule's, in place; returns the
    quantities the rule withholds (``Quantities(withheld=)``)."""
    for key in RASTER_ONLY + ("carry_out_bytes", "carry_in_bytes", "latency_steps", ONCHIP_CARRY):
        values.pop(key, None)
    values["timesteps"] = QuantityValue(1.0, "static")
    for key, value in (("synaptic_events", facts.mac_events), ("latency_steps", facts.latency_steps),
                       ("carry_out_bytes", facts.carry_out_bytes), ("carry_in_bytes", facts.carry_in_bytes),
                       ("noc_value_bytes", facts.noc_value_bytes), (ONCHIP_CARRY, facts.onchip_bytes)):
        if value is not None:
            values[key] = QuantityValue(float(value), "static")
    return {key: reason for key, reason in facts.withheld if key not in values}


def record_value_facts(record: Any) -> ValueDomainFacts:
    """The value rule on a sealed record: MAC events = committed cells, the window, value carry."""
    segments = record.schedule.segments()
    spans = [stage_latency([c.static_delay_levels for c in segment.cores]) for segment in segments]
    latencies = None if any(span is None for span in spans) else [int(span) for span in spans if span is not None]
    carry = record.schedule.carry
    # A sealed schedule that carries nothing seals no carry census: a known zero. [U-F4] held as the chip declares it.
    held, refusal = onchip_carry(record_carry_census(record.schedule), record.identity.platform)
    if refusal is not None and held is not None:  # the program does not fit the buffer: no on-chip carry is sized
        held.pop(ONCHIP_CENSUS, None)
    sized, withheld = value_carry(held, None if carry is None else carry.timesteps, value_bits(record.identity.platform))
    if refusal is not None:
        withheld[ONCHIP_CARRY] = refusal
    sealed = getattr(record.schedule, "value_noc_wire_hops", None)
    noc_value, noc_withheld = value_noc_reading(sealed, value_bits(record.identity.platform))
    if sealed is None and any(len(segment.cores) > 1 for segment in segments):
        noc_withheld = NO_RECORD_WIRES
    if noc_withheld is not None:
        withheld["noc_value_bytes"] = noc_withheld
    return ValueDomainFacts(
        mac_events=sum(c.cells_used for s in segments for i, c in enumerate(s.cores)
                       if i not in (s.dark_core_indices or ())),  # [U-F4] a dark core is loaded, never evaluated
        latency_steps=None if latencies is None else value_window(latencies),
        carry_out_bytes=sized.get("carry_out_bytes"), carry_in_bytes=sized.get("carry_in_bytes"),
        withheld=tuple(sorted(withheld.items())), noc_value_bytes=noc_value, onchip_bytes=sized.get(ONCHIP_CARRY),
    )


def record_carry_census(schedule: Any) -> Dict[str, int]:
    """A sealed schedule's carry census as the planes' census states it: its boundary crossings and, [U-F4] where it
    runs wavefront groups, the share each group holds itself (summed; the peaks, in bytes and in values, the largest
    group's)."""
    carry = schedule.carry
    census = {"boundary_out_bytes": 0 if carry is None else int(carry.boundary_out_bytes),
              "boundary_in_bytes": 0 if carry is None else int(carry.boundary_in_bytes)}
    groups = getattr(schedule, "wavefronts", None) or ()
    if groups:
        census.update(onchip_out_bytes=sum(g.onchip_out_bytes for g in groups),
                      onchip_in_bytes=sum(g.onchip_in_bytes for g in groups),
                      onchip_peak_live_bytes=max(g.onchip_peak_live_bytes for g in groups),
                      onchip_peak_live_elements=max(g.onchip_peak_live_elements for g in groups))
    return census


def domain_bytes_of(platform: Mapping[str, Any], timesteps: Optional[int]) -> Optional[Any]:
    """How a boundary's carried elements are sized in this run's domain, or None when unknown:
    values at the declared grid, else the dense spike raster at the window."""
    if names_value_domain(platform):
        bits = value_bits(platform)
        return None if bits is None else (lambda elements: value_bytes(elements, bits))
    return None if timesteps is None else (lambda elements: raster_bytes(elements, int(timesteps)))
