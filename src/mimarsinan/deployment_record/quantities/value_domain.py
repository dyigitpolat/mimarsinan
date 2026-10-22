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
from mimarsinan.mapping.support.period_census import natural

#: The resolved-surface key naming the value domain, and the value grid beside it.
SEMANTICS_KEY = "core_semantics"
VALUE_BITS_KEY = "activation_bits"
#: Spike-raster quantities a value-domain program never produces.
RASTER_ONLY = ("carried_raster_bytes",)
#: Why a value carry or value NoC census that is not zero has no size without the declared grid.
NO_VALUE_GRID = "activation_bits undeclared: the value carry has no grid"
NO_VALUE_NOC_GRID = "activation_bits undeclared: the value NoC traffic has no grid"
#: Why a sealed record states no value NoC census (KA-G3-1: no wire census is sealed).
NO_RECORD_WIRES = "the sealed record keeps no wire census (KA-G3-1)"
_CARRY_KEYS = (("carry_out_bytes", "boundary_out_bytes"), ("carry_in_bytes", "boundary_in_bytes"))


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
        total = natural(int(carry[census_key]), "carry boundary bytes", error=ValueDomainCensusError)
        if total == 0:
            sized[key] = 0
        elif bits is None or timesteps is None:
            withheld[key] = NO_VALUE_GRID if bits is None else "the carry census states no window"
        else:
            sized[key] = value_bytes(crossing_elements(total, int(timesteps)), bits)
    return sized, withheld


@dataclass(frozen=True)
class ValueDomainFacts:
    """The value rule's answers on one plane; None is absent, never zero."""

    mac_events: Optional[int]
    latency_steps: Optional[int]
    carry_out_bytes: Optional[int]
    carry_in_bytes: Optional[int]
    #: Quantities whose work may exist but which this plane cannot size, with the reason.
    withheld: Tuple[Tuple[str, str], ...] = ()


def apply_value_rule(values: Dict[str, QuantityValue], facts: ValueDomainFacts) -> Dict[str, str]:
    """Replace a plane's spiking-rule quantities by the value rule's, in place; returns the
    quantities the rule withholds (``Quantities(withheld=)``)."""
    for key in RASTER_ONLY + ("carry_out_bytes", "carry_in_bytes", "latency_steps"):
        values.pop(key, None)
    values["timesteps"] = QuantityValue(1.0, "static")
    for key, value in (("synaptic_events", facts.mac_events), ("latency_steps", facts.latency_steps),
                       ("carry_out_bytes", facts.carry_out_bytes), ("carry_in_bytes", facts.carry_in_bytes)):
        if value is not None:
            values[key] = QuantityValue(float(value), "static")
    return {key: reason for key, reason in facts.withheld if key not in values}


def record_value_facts(record: Any) -> ValueDomainFacts:
    """The value rule on a sealed record: MAC events = committed cells, the window, value carry."""
    segments = record.schedule.segments()
    spans = [stage_latency([c.static_delay_levels for c in segment.cores]) for segment in segments]
    latencies = None if any(span is None for span in spans) else [int(span) for span in spans if span is not None]
    carry = record.schedule.carry
    # A sealed schedule that carries nothing seals no carry census: a known zero.
    sized, withheld = value_carry(
        {"boundary_out_bytes": 0 if carry is None else carry.boundary_out_bytes,
         "boundary_in_bytes": 0 if carry is None else carry.boundary_in_bytes},
        None if carry is None else carry.timesteps, value_bits(record.identity.platform),
    )
    if any(len(segment.cores) > 1 for segment in segments):
        withheld["noc_value_bytes"] = NO_RECORD_WIRES
    return ValueDomainFacts(
        mac_events=sum(c.cells_used for s in segments for c in s.cores),
        latency_steps=None if latencies is None else value_window(latencies),
        carry_out_bytes=sized.get("carry_out_bytes"), carry_in_bytes=sized.get("carry_in_bytes"),
        withheld=tuple(sorted(withheld.items())),
    )
