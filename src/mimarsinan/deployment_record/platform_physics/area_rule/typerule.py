"""[U-AREA] A type rule: a description priced from its type's table with no anchor of its own (design 2.1), through the
one core equation of every type (``core.equation.core_terms``, re-exported here with its named helpers), which the
published points' held-out estimate and an anchor's own memory share price through as well (MA-3 of the build's
verify, MA-R1 and mi-R3 of the fix round's re-verify: one core-area rule).

The core: A_core = (A_mem + A_conv) / s(type), A_mem = N x a_cell / eta(type) on the type's scaled count (the one
count form, ``core.equation.scaled_counts``: stored bits on 'bits' terms, cells on 'cells_converters' terms), A_conv
= N_conv x a_conv; a memory a point states beside its core is the array part of its named type's equation. The cell:
the declared kind (its own printed units read as inputs), else the type's default kind, else the type's default factor
over the logic-rule 6T; the converter: the declaration's own printed unit, else its kind or the type's default at the
run's bits (rule E). The chip: the sum over census core types of N_t x A_core,t / gamma (the outside field's gamma: the
type's, or 1 for declared parts). The band: nominal x [L, H] of the type, its low edge widened to a declared reading's
x L of the reading's type where that lies lower (ruling R2). Beside the band stands the fixed-outside reading (MA-3):
the core grid plus the largest published outside part of the dies the type's gamma averages, never an edge, a named
refusal where the type's table states none (the fix round's MA-1).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import anchor, units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import equation as core_rule
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door, remembered
from mimarsinan.deployment_record.platform_physics.area_rule.core.equation import (
    Counts, Declared, census_counts, converter_um2, core_cell_um2, core_terms, scaled_counts, type_eta)
from mimarsinan.deployment_record.platform_physics.area_rule.declaration import declared_cell_flags, declared_of
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.rule import (
    AreaFlag, AreaPrice, CensusCore, band_factors, held_outside_part, outside_gamma, type_file, type_gamma, type_value)


@door(reading="reading or absent")
def type_core_um2(registry: Any, type_name: str, block: Any, census: CensusCore, weight_bits: float,
                  readout_bits: Optional[float], reading: Optional[dict] = None) -> float:
    """One core of the type rule (um^2) on the census core; ``reading`` (a declared reading of another type) takes
    that type's default cell and its own declared bits per neuron."""
    declared = declared_of(registry, block, reading is not None)
    organization = reading if reading is not None else block.organization
    counts = core_rule.census_counts(registry, type_name, declared, organization, census, weight_bits,
                                     core_rule.ORGANIZATION if reading is None else core_rule.READING)
    return core_rule.core_terms(registry, type_name, declared, counts, readout_bits).core_um2


@door
def reads_declared_cell(registry: Any, block: Any) -> bool:
    """mi-D1: whether the type rule's cell (its type's, or its reading's) read the description's declared cell; made
    once per frozen block (the fourth delta round: core/door.remembered)."""
    if block.bitcell_um2 is None:  # no declared cell: none read (and no cell resolved for it)
        return False
    reading = (block.reading_low_edge or {}).get("type")
    return remembered(registry, "reads declared cell", (block,), lambda: any(
        core_rule.resolved_cell(registry, t, declared_of(registry, block, of_reading)).reads_declared
        for t, of_reading in ((block.type, False), (reading, True)) if t is not None))


@door
def price_type_rule(registry: Any, block: Any, cores: float, census: CensusCore, cell_bits: Optional[float] = None,
                    readout_bits: Optional[float] = None) -> AreaPrice:
    return price_type_rule_types(registry, block, ((cores, census),), cell_bits, readout_bits)


@door
def price_type_rule_types(registry: Any, block: Any, types: Sequence[Tuple[float, CensusCore]],
                          cell_bits: Optional[float] = None, readout_bits: Optional[float] = None) -> AreaPrice:
    """chip_area_mm2 (mm^2) of census core types ``(count, core)`` under the block's type rule."""
    flags: List[AreaFlag] = [AreaFlag("projection", "a type rule: the type's table, no anchor of its own")]
    reading_type = (block.reading_low_edge or {}).get("type")
    grid = reading_grid = u.NONE
    for cores, census in types:
        weight_bits = census.weight_bits
        if weight_bits is None:
            if cell_bits is None:
                raise AreaRuleRefusal("a type rule needs the census's weight bits or the description's cell bits")
            weight_bits = cell_bits
            if not any(f.kind == "weight_bits_not_stated" for f in flags):
                flags.append(AreaFlag("weight_bits_not_stated", f"weight bits not stated: read at {cell_bits:g} bits"))
        grid += cores * type_core_um2(registry, block.type, block, census, weight_bits, readout_bits)
        if reading_type is not None:
            reading_grid += cores * type_core_um2(registry, reading_type, block, census, weight_bits, readout_bits,
                                                  dict(block.reading_low_edge))
    gamma = outside_gamma(block.outside, type_gamma(registry, block.type))
    nominal = grid / u.UM2_PER_MM2 / gamma
    low_factor, high_factor = band_factors(registry, block.type)
    readings = {}
    low = nominal * low_factor
    if reading_type is not None:
        readings[f"{type_file(registry, reading_type).split('/')[u.SECOND]}_reading"] = \
            reading_grid / u.UM2_PER_MM2 / gamma
        # the reading widens the band below the type's own low edge, never above it (a census whose neuron state
        # outweighs its cells puts the reading above the nominal: the type's own edge then stays the low edge)
        low = min(low, reading_grid / u.UM2_PER_MM2 / gamma * band_factors(registry, reading_type)[u.FIRST])
        flags.append(AreaFlag("reading_low_edge", f"the {reading_type} reading is the band's low edge "
                                                  f"({(block.reading_low_edge or {}).get('ruling', '')})"))
    flags += declared_cell_flags(registry, block) if reads_declared_cell(registry, block) else []
    try:
        held = held_outside_part(registry, block.type)
    except AreaRuleRefusal as refused:  # MA-1: the reading is a named refusal beside the band; the price stands
        flags.append(AreaFlag("fixed_outside_reading_refused", f"the fixed-outside reading: {refused}"))
        return AreaPrice(nominal=nominal, low=low, high=nominal * high_factor, flags=tuple(flags), readings=readings)
    readings["fixed_outside_reading"] = grid / u.UM2_PER_MM2 + held.value
    flags.append(AreaFlag("fixed_outside_reading", "beside the band, never an edge: the core grid plus the largest "
                                                   "published outside part of the dies the type's gamma averages",
                          readings["fixed_outside_reading"]))
    return AreaPrice(nominal=nominal, low=low, high=nominal * high_factor,
                     fixed_reading=readings["fixed_outside_reading"], flags=tuple(flags), readings=readings)


@door
def price_block(registry: Any, block: Any, cores: float, census: CensusCore, cell_bits: Optional[float] = None,
                readout_bits: Optional[float] = None) -> AreaPrice:
    """chip_area_mm2 of a block at ``cores`` of one core type (``price_block_types`` with one type)."""
    return price_block_types(registry, block, ((cores, census),), cell_bits, readout_bits)


@door
def price_block_types(registry: Any, block: Any, types: Sequence[Tuple[float, CensusCore]],
                      cell_bits: Optional[float] = None, readout_bits: Optional[float] = None) -> AreaPrice:
    """chip_area_mm2 of a block over census core types ``(count, core)``: an anchor scales its published core
    (anchor.py); a type rule prices its terms; complete terms are R5's own path (not priced here)."""
    if block.anchor:
        return anchor.price_anchor_types(anchor.anchor_law(registry, block), types)
    if block.terms == "type_rule":
        return price_type_rule_types(registry, block, types, cell_bits, readout_bits)
    raise AreaRuleRefusal(f"area_rule terms {block.terms!r}: the description's own declared terms price it (R5's "
                          "path), not the rule")


#: The one core equation and its named helpers (core/equation.py) and a block's declaration (declaration.py).
__all__ = ["Counts", "Declared", "census_counts", "converter_um2", "core_cell_um2", "core_terms", "declared_of",
           "price_block", "price_block_types", "price_type_rule", "price_type_rule_types", "reads_declared_cell",
           "scaled_counts", "type_core_um2", "type_eta", "type_value"]
