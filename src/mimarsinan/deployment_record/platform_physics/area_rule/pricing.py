"""[U-AREA] The pricer's rule path (W3): chip_area_mm2 of a description that declares an ``area_rule`` block.

``price_absolute`` (cost/absolute/metrics.py, ``price_area``) asks ``rule_area`` first. A physics without a block
returns None and keeps R5's path whole. A block of complete declared terms keeps R5's path for its value (its
components, adders and outside part) and names only its size against a published one (ruling R3). Any other block
prices its core grid and outside part by the rule here; the adders (lanes, carry buffer) stay R5's, beside it.

- The census's core types: the declared (or census-proved) core types (``validity.CoreGeometry``: device rows x
  columns x count), one census core each; a census that proves only its totals (``validity.CensusCells``, or no
  geometry) is one pooled core whose terms are linear in its totals (cells_physical, neurons_physical), a per-core
  ceiling on it a named refusal (anchor.anchor_ratio, typerule.type_core_um2).
- The weight bits: the census's ``weight_bits``, else the description's cell bits, flagged.
- A converter kind resolves at the run's readout bits (``readout_law.adc_bits``, review N-6): a kind whose area
  depends on bits refuses by name where the run declares none or bits outside its published ones; a kind whose
  sources state no bits prices at any bits with that flag.
- The flags (identity, beyond or below the published chip, off geometry, off word width, weight bits not stated, the
  readings beside the band) go into the term's basis and into ``AbsolutePricing.flags``, computed at read and never
  sealed.

This module reads no pricing context: it takes the physics, the quantities and the judged geometry, so the platform
physics package stays below the pricer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import anchor, typerule, units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import equation as core_rule
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door, remembered
from mimarsinan.deployment_record.platform_physics.area_rule.core.fields import CENSUS
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import COUNT, POSITIVE, judged
from mimarsinan.deployment_record.platform_physics.area_rule.declaration import AREA_RULE_AXES
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, read_ids
from mimarsinan.deployment_record.platform_physics.area_rule.registry import load_registry
from mimarsinan.deployment_record.platform_physics.area_rule.rule import (
    AreaFlag, CensusCore, converter_area, resolve_kind, type_file)

#: The census totals a pooled census core is read from.
POOLED_TOTALS = ("cores_physical", "cells_physical", "neurons_physical")


@dataclass(frozen=True)
class RuleArea:
    """The rule path's answer: ``band`` (low, nominal, high) in mm^2 of the core grid with its outside part and its
    ``basis``; ``band`` None where R5's path prices (complete terms), ``basis`` then the departures it names; or a
    named ``refusal``. ``declared_parts``: the outside part is R5's (the chip components' sum or a sourced scalar)."""

    band: Optional[Tuple[float, float, float]]
    basis: str
    flags: Tuple[AreaFlag, ...]
    refusal: Optional[str] = None
    declared_parts: bool = False


def _value(quantities: Any, key: str) -> Optional[float]:
    return quantities.get(key).value if quantities.has(key) else None


@door
def census_types(quantities: Any, geometry: Any, weight_bits: Optional[float],
                 own_core: Any = None) -> Tuple[Tuple[float, CensusCore], ...]:
    """The census core types ``(count, core)``: the declared or proved core types; else one pooled core from the
    census's totals (a census judged under ``CENSUS_PROVED`` refuses it by name: its band's edges rest on the census's
    mean core); else, where the caller reads the census as declared (no geometry judged: ``UNJUDGED`` or a census
    that states no core geometry), its cores at the description's own core (``own_core``, its validity geometry). Each
    census core is built from the judge's returned values (Python ints and floats: the fourth delta round, MA-C1)."""
    if isinstance(geometry, (tuple, list)):
        return tuple((float(judged("{}'s core types[{}].count", core.count, COUNT, CENSUS, i)), CensusCore(
            judged("{}'s core types[{}].axons", core.axons, COUNT, CENSUS, i),
            judged("{}'s core types[{}].neurons", core.neurons, COUNT, CENSUS, i), weight_bits))
            for i, core in enumerate(geometry))
    if getattr(geometry, "strict", False):
        raise AreaRuleRefusal("the census pools core types and its caller judges only what it proves "
                              "(CENSUS_PROVED): the area band's edges rest on its mean core; pass the declared core "
                              "types")
    cores, cells, neurons = (_value(quantities, key) for key in POOLED_TOTALS)
    if cores and cells is not None and neurons:  # its mean core judged where it is made
        rows, cols = (judged(f"{AREA_RULE_AXES[u.FIRST]}'s pooled census: its mean core's {name}", value, POSITIVE)
                      for name, value in (("rows", cells / neurons), ("columns", neurons / cores)))
        return ((cores, CensusCore(rows, cols, weight_bits, pooled=True)),)
    if geometry is None and cores and own_core is not None:
        return ((cores, CensusCore(judged("validity.geometry.axons", own_core.axons, COUNT), judged(
            "validity.geometry.neurons", own_core.neurons, COUNT), weight_bits, read_as_declared=True)),)
    raise AreaRuleRefusal("the area rule prices the census's core types: pass the declared core types, or a census "
                          f"that states {', '.join(POOLED_TOTALS)}")


def _census_flags(types: Tuple[Tuple[float, CensusCore], ...]) -> List[AreaFlag]:
    flags = []
    if any(census.pooled for _n, census in types):
        flags.append(AreaFlag("pooled_census", "a pooled census: the nominal is linear in its totals (cells, "
                                               "neurons); the band's edges are read at its mean core"))
    if any(census.read_as_declared for _n, census in types):
        flags.append(AreaFlag("read_as_declared", "the census states no core geometry and no cells (read as declared): "
                                                  "its cores are read at the description's own core"))
    return flags


def _converter_flags(registry: Any, block: Any, readout_bits: Optional[float]) -> List[AreaFlag]:
    """N-6: the block's converter kind (an anchor's declared one; a type rule's declared or default one) resolved at
    the run's readout bits, a named refusal where its area needs bits the run does not declare; made once per frozen
    block and bits (the fourth delta round: core/door.remembered)."""
    def made() -> Tuple[AreaFlag, ...]:
        declared = (block.converter or {}).get("kind")
        kind = declared if block.anchor else resolve_kind(registry, block.type, declared)
        if kind is None or converter_area(registry, kind, readout_bits).bits_stated:
            return ()
        note = registry.converter_kind(kind).get("bits_note") or "its sources state no bits"
        return (AreaFlag("converter_bits_not_stated", f"converter kind {kind}: priced at any bits ({note})"),)
    return list(remembered(registry, "converter flags", (block,), made, readout_bits))


def _evidence(registry: Any, block: Any) -> str:
    """The evidence the price rests on, by id: an anchor's published figures, else its type's table (and the
    registry's evidence file behind it)."""
    if block.anchor:
        ids = sorted({i for key in ("core_mm2", "die_mm2", "n_pub") if block.anchor.get(key)
                      for i in read_ids(str(block.anchor[key]))})
        return f"evidence {', '.join(ids)}; the {type_file(registry, block.type)} table"
    return f"the {type_file(registry, block.type)} table; node {block.node_nm:g} nm"


def _complete(registry: Any, block: Any, quantities: Any) -> RuleArea:
    """Complete declared terms (R1, Table I rows): R5's own path prices them; a published size flags the size."""
    flags = [AreaFlag("complete_terms", "complete declared terms: R5's own path, unchanged")]
    cores = _value(quantities, "cores_physical")
    if block.published_size and cores is not None:  # MA-D1: the published size judged as a count, as at load
        flags += anchor.size_flags(cores, core_rule.cited(registry, "area_rule.published_size.n_pub",
                                                          block.published_size.get("n_pub"), core_rule.COUNT))
    named = [f.reason for f in flags if f.kind != "complete_terms"]
    return RuleArea(None, f"; flags: {'; '.join(named)}" if named else "", tuple(flags), declared_parts=True)


def rule_area(physics: Any, quantities: Any, geometry: Any, readout_bits: Optional[float] = None
              ) -> Optional[RuleArea]:
    """The rule path's answer for ``physics`` on the census, or None where the physics declares no block (R5's own
    path, whose input the rule never reads); a named refusal of the door or of the rule is the answer's refusal."""
    if getattr(physics, "area_rule", None) is None:
        return None
    try:
        return rule_price(physics, quantities, geometry, readout_bits)
    except AreaRuleRefusal as refused:
        return RuleArea(None, "", (), refusal=str(refused))


@door(physics="physics with a block")
def rule_price(physics: Any, quantities: Any, geometry: Any, readout_bits: Optional[float] = None) -> RuleArea:
    """The rule path's price of a physics that declares a block, its whole input judged at the door first."""
    block = physics.area_rule
    registry = load_registry()
    if block.terms == "complete":
        return _complete(registry, block, quantities)
    flags = _converter_flags(registry, block, readout_bits)
    weight_bits = _value(quantities, "weight_bits")
    withheld = quantities.withheld_reason("weight_bits") if weight_bits is None else None
    if withheld:  # a census that withholds its weight bits (core types of several widths) names why
        flags.append(AreaFlag("weight_bits_withheld", f"weight bits withheld by the census ({withheld}): read at "
                                                      "the description's cell"))
    validity = physics.validity.geometry
    types = census_types(quantities, geometry, weight_bits, validity)
    flags += _census_flags(types)
    # the price's band is judged where it is made, its order with it (rule.AreaPrice, mi-K1)
    price = typerule.price_block_types(registry, block, types, None if validity is None else validity.cell_bits,
                                       readout_bits)
    flags += list(price.flags) + [AreaFlag(name, f"a reading beside the band, never an edge ({name})", value)
                                  for name, value in price.readings.items()
                                  if not any(f.kind == name for f in price.flags)]
    form = f"anchor ({block.anchor['scope']})" if block.anchor else "type rule"
    basis = (f"area_rule {block.type} {form}, outside {block.outside}: the core grid of the census's core types "
             f"[{_evidence(registry, block)}]; flags: {'; '.join(f.reason for f in flags)}")
    return RuleArea((price.low, price.nominal, price.high), basis, tuple(flags),
                    declared_parts=block.outside == "declared_parts")
