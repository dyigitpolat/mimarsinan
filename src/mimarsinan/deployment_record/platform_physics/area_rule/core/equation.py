"""[U-AREA] The one core equation of every type and its named helpers (design revision 3, section 2.1; the build's
verify MA-3 and the fix round's re-verify MA-R1 and mi-R3: one core-area rule, the class swept).

    q_bits  = cells x stored bits per cell x ceil(w / weight bits per cell) + columns x bits per neuron
    q_cells = cells x ceil(w / weight bits per cell);  q_converters = ceil(columns / m)
    A_core  = (A_mem + A_conv) / s(type),  A_mem = N x a_cell / eta(type),  A_conv = N_conv x a_conv

Every count of a core is ``scaled_counts`` (a type rule's census, an anchor's census and its own core, the anchor's
wider-word reading, its own memory share); every core's terms are ``core_terms`` (a type rule's core, a published
point's core, a memory a point states beside its core, an anchor's own memory share), on the cell ``core_cell_um2``,
the converter ``converter_um2`` and the eta ``type_eta`` resolve. A count an organization cannot state is a named
refusal, never a KeyError. Every quantity the count form and the equation read from a declaration is judged as a
quantity of its kind by one judge, ``judged`` (the second delta round, MA-D1; core/judge.py), at load and at the price;
every public function here is a door (core/door.py, the third delta round), and the counts and the cell it makes are
judged where they are made.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple, Type

from mimarsinan.deployment_record.platform_physics.area_rule import rule as _rule, units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door, remembered
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import (  # noqa: F401 - the judge's names here
    CELL, COUNT, COUNT_OR_ZERO, NONNEGATIVE, POSITIVE, SHARE, WEIGHTS, Kind, judged)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import (
    AreaRuleRefusal, ExpressionRefusal, HeldOut, read_ids)
from mimarsinan.deployment_record.platform_physics.area_rule.rule import (
    BITS_TERMS, CensusCore, ConverterRefusal, CoreTerms, cell_area, core_area, type_value)

#: The kind of every field the count form reads off an organization or a reading (the code's reads here and in
#: anchor.py; test_area_rule_judge.py checks the table against them): a field the count form comes to read takes its
#: kind here, and the judge then judges it at load and at the price.
QUANTITIES = {"rows": COUNT, "cols": COUNT, "weight_bits_per_cell": COUNT, "columns_per_converter": COUNT,
              "cells_per_weight": COUNT, "stored_bits_per_cell": COUNT_OR_ZERO, "bits_per_neuron": COUNT_OR_ZERO,
              "term_weights": WEIGHTS}
#: An anchor's published figures, expressions over the evidence, and their kinds.
ANCHOR_FIGURES = {"core_mm2": POSITIVE, "die_mm2": POSITIVE, "n_pub": COUNT}
ORGANIZATION, READING = "area_rule.organization", "area_rule.reading_low_edge"


def judged_fields(mapping: Any, place: str) -> Mapping[str, Any]:
    """An organization or a reading at ``place``: an object (or absent), each ``QUANTITIES`` field it states judged."""
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise AreaRuleRefusal(f"{place} {mapping!r} is not an object")
    for name, kind in QUANTITIES.items():
        if name in mapping:
            judged("{}.{}", mapping[name], kind, place, name)
    return mapping


def cited(registry: Any, field: str, expression: Any, kind: Kind, cited_as: Optional[str] = "a published figure",
          refusal: Type[AreaRuleRefusal] = AreaRuleRefusal) -> float:
    """A quantity stated as an expression (``cited_as``: a figure, a cell, a size; None: a split, which may read the
    registry alone): a text reading evidence ids, evaluated to its ``kind`` and judged (``refusal``: the class)."""
    try:
        ids = read_ids(expression) if isinstance(expression, str) else ()
        if cited_as is not None and not ids:
            raise refusal(f"{field} {expression!r} reads no evidence entry: {cited_as} is cited by id, never typed")
        value = registry.evaluate(expression)
    except ExpressionRefusal as failed:  # the registry's own arithmetic names its failure (evidence.evaluate)
        raise refusal(f"{field} {expression!r} is not an expression the registry evaluates ({failed.cause})") from None
    except HeldOut as held:  # a declared figure the registry's view holds out is no price
        raise refusal(f"{field} {expression!r} reads an entry the registry holds out ({held})") from None
    except AreaRuleRefusal:
        raise
    except (SyntaxError, ArithmeticError, LookupError, TypeError, ValueError) as caught:  # its arithmetic fails
        raise refusal(f"{field} {expression!r} is not an expression the registry evaluates "
                      f"({type(caught).__name__}: {caught})") from None
    try:
        return judged("{} {!r} =", value, kind, field, expression)
    except AreaRuleRefusal as refused:
        raise refusal(str(refused)) from None


@door
def grid_share(n_pub: float, core_mm2: float, die_mm2: float) -> float:
    """An own-die anchor's grid share of its die, N_pub x A_core / D_pub, a share (else a negative outside part)."""
    return judged("area_rule.anchor: the grid share n_pub x core_mm2 / die_mm2", n_pub * core_mm2 / die_mm2, SHARE)


class Declared(NamedTuple):
    """What a description (or a published point) declares for its core: its node, its own cell (um^2), cell kind,
    converter kind and the converter's own printed unit (an evidence id)."""

    node_nm: float
    cell_kind: Optional[str] = None
    bitcell_um2: Optional[float] = None
    converter_kind: Optional[str] = None
    own_unit: Optional[str] = None


class Counts(NamedTuple):
    """A core's counts on its type's scaled terms: the array (stored bits or cells), its converters, and a memory a
    point states beside its core (bits, the type whose memory rule prices it)."""

    array: float
    converters: float = u.NONE
    beside: Optional[Tuple[float, str]] = None


@door
def cells_per_weight(organization: Optional[Mapping[str, Any]], weight_bits: Optional[float]) -> int:
    """ceil(w / weight bits per cell), at least one (M1(b): the ceiling, never a silent fit); one cell where the weight
    bits are not stated or the organization states no cell."""
    org = organization or {}
    if weight_bits is None or "weight_bits_per_cell" not in org:
        return int(u.UNITY)
    cap = org["weight_bits_per_cell"]
    return max(int(u.UNITY), math.ceil(float(weight_bits) / cap))


@door
def scaled_counts(form: str, organization: Optional[Mapping[str, Any]], rows: float, cols: float,
                  weight_bits: Optional[float], pooled: bool = False, converts: bool = False) -> Dict[str, float]:
    """The one count form of a core of ``rows`` x ``cols`` cells at ``weight_bits`` on the scaled terms ``form``:
    'bits' on bits terms, else 'cells', and 'converters' where the core ``converts`` (linear in a pooled census's
    columns where m = 1, a named refusal where m > 1). An organization that states no cell reads a w-bit weight as one
    cell of w stored bits and one converter per column; one that states its cell in part is a named refusal."""
    org = organization or {}
    k = cells_per_weight(org, weight_bits)
    if form in BITS_TERMS:
        if "stored_bits_per_cell" in org:
            stored = org["stored_bits_per_cell"]
        elif "weight_bits_per_cell" in org:
            raise AreaRuleRefusal("area_rule.organization states weight_bits_per_cell and no stored_bits_per_cell: "
                                  "a count of stored bits needs the bits each cell stores")
        elif weight_bits is None:
            raise AreaRuleRefusal("a count of stored bits needs the census's weight bits or the organization's cell")
        else:
            stored = weight_bits
        counts = {"bits": rows * cols * stored * k + cols * org.get("bits_per_neuron", u.NONE)}
    else:
        counts = {"cells": rows * cols * k}
    if converts:
        m = org.get("columns_per_converter", u.UNITY)
        if pooled and m != u.UNITY:
            raise AreaRuleRefusal("converter count: the census pools core types and the description converts every "
                                  f"{m:g} columns per converter; pass the declared core types")
        # a pooled census's mean core may hold a fraction of a column: one converter per column is linear in it
        counts["converters"] = cols if pooled else math.ceil(cols / m)
    for term, count in counts.items():  # judged where made: a count the floats hold
        judged("the count form's {} on {!r} x {!r}", count, NONNEGATIVE, term, rows, cols)
    return counts


@door(converts="flag or absent")
def own_counts(form: str, organization: Optional[Mapping[str, Any]], converts: Optional[bool] = None
               ) -> Dict[str, float]:
    """An organization's own core in the one count form (its own rows, columns and cell); converters on cells terms
    unless ``converts`` says otherwise; a named refusal where it states no rows or columns."""
    org = organization or {}
    for key in ("rows", "cols"):
        if key not in org:
            raise AreaRuleRefusal(f"area_rule.organization states no {key}: its own core's count needs it")
    return scaled_counts(form, org, org["rows"], org["cols"], org.get("weight_bits_per_cell"),
                         converts=form not in BITS_TERMS if converts is None else converts)


@door
def anchor_counts(registry: Any, type_name: str, organization: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """An anchor's own core on its type's scaled terms, its organization judged (at load and at the price): its
    quantities each of its kind, counted on its type's terms, its term weights on those terms with a positive sum over
    positive own counts."""
    form = registry.type_table(type_name)["scaled_terms"]
    org = judged_fields(organization, ORGANIZATION)
    if org.get("terms") != form:
        raise AreaRuleRefusal(f"area_rule.organization.terms {org.get('terms')!r} is not the {type_name} type's "
                              f"scaled terms {form!r}: an anchor counts on its type's terms")
    return weighted_counts(form, org)


@door
def weighted_counts(form: str, organization: Mapping[str, Any]) -> Dict[str, float]:
    """An anchor's own core on the scaled terms ``form``, its term weights on those terms with a positive sum over
    positive own counts."""
    org = organization
    own = own_counts(form, org)
    weights = org.get("term_weights") or {}
    if not weights or set(weights) - set(own) or not sum(weights.values()) > u.NONE or any(
            weights[t] > u.NONE and not own[t] > u.NONE for t in weights):
        raise AreaRuleRefusal(f"area_rule.organization.term_weights {dict(weights)!r}: the anchor's ratio weighs its "
                              f"type's terms {sorted(own)} by non-negative weights of a positive sum, each weighted "
                              "term with a positive own count")
    return own


@door
def census_counts(registry: Any, type_name: str, declared: Declared, organization: Any, census: CensusCore,
                  weight_bits: float, place: str = ORGANIZATION) -> Counts:
    """A census core's counts on its type's scaled terms (the one count form), converters where a kind resolves; the
    organization (or the reading at ``place``) judged, the core's array count a count above 0."""
    organization = judged_fields(organization, place)
    converts = _rule.resolve_kind(registry, type_name, declared.converter_kind) is not None
    q = scaled_counts(registry.type_table(type_name)["scaled_terms"], organization, census.rows, census.cols,
                      weight_bits, census.pooled, converts)
    array = judged("{}: a core's count on the {} type's terms", q["bits"] if "bits" in q else q["cells"], POSITIVE,
                   place, type_name)
    return Counts(array, q.get("converters", u.NONE))


@door
def type_eta(registry: Any, type_name: str) -> float:
    """A type's array efficiency: its table's eta, else 1 (a type whose array figure holds its periphery)."""
    return type_value(registry, type_name, "eta") if "eta" in registry.type_table(type_name) else u.UNITY


class Cell(NamedTuple):
    """A core's resolved cell: its area (um^2) and whether it read the description's declared cell (mi-D1)."""

    um2: float
    reads_declared: bool


@door
def resolved_cell(registry: Any, type_name: str, declared: Declared) -> Cell:
    """The core's cell: the declared kind, else the type's default kind, else its default factor x k_L x sigma; a kind
    whose evidence the view holds out (a chip's own printed unit, strict) gives way to the next. It read a declared
    cell where it rests on sigma (``rule.reads_sigma``), never on an F2 or a node-flat kind. Its area is judged where it
    is made, a cell's area above 0 (MA-K2: an F2 cell beyond the floats is a named refusal naming the node and kind).
    Made once per type and declaration (the fourth delta round: core/door.remembered)."""
    return remembered(registry, "cell", (), lambda: _resolved_cell(registry, type_name, declared), type_name, declared)


def _resolved_cell(registry: Any, type_name: str, declared: Declared) -> Cell:
    table = registry.type_table(type_name)
    stated = declared.bitcell_um2 is not None
    for kind, own in ((declared.cell_kind, True), (table.get("default_cell_kind"), False)):
        if kind is not None:
            try:
                area = cell_area(registry, kind, declared.node_nm, declared_um2=declared.bitcell_um2, declared=own)
            except HeldOut:
                continue
            return Cell(judged("the {} cell at {!r} nm", area, CELL, kind, declared.node_nm),
                        stated and _rule.reads_sigma(registry, kind))
    area = type_value(registry, type_name, "default_cell_factor") * _rule.logic_rule_factor(registry) * \
        registry.sigma(declared.node_nm, (), declared.bitcell_um2)
    return Cell(judged("the {} type's default cell at {!r} nm", area, CELL, type_name, declared.node_nm), stated)


@door
def core_cell_um2(registry: Any, type_name: str, declared: Declared) -> float:
    """The core's cell (um^2): the area of ``resolved_cell``."""
    return resolved_cell(registry, type_name, declared).um2


@door
def converter_um2(registry: Any, type_name: str, declared: Declared, readout_bits: Optional[float]) -> float:
    """One converter's unit area: the declaration's own printed unit (held out in strict mode), else its kind or the
    type's default kind at the run's bits (rule E); made once per type, declaration and bits."""
    def made() -> float:
        if declared.own_unit is not None:
            try:
                return registry.declared_entry(declared.own_unit)
            except HeldOut:
                pass
        kind = _rule.resolve_kind(registry, type_name, declared.converter_kind)
        return _rule.converter_area(registry, kind, readout_bits).value
    return remembered(registry, "converter unit", (), made, type_name, declared, readout_bits)


@door
def core_terms(registry: Any, type_name: str, declared: Declared, counts: Counts, readout_bits: Optional[float] = None,
               unit: Optional[float] = None) -> CoreTerms:
    """The one core equation of every type (``unit``: a reading's converter unit in place of the resolved one)."""
    cell = core_cell_um2(registry, type_name, declared)
    eta = type_eta(registry, type_name)
    share = type_value(registry, type_name, "s")
    conv = u.NONE
    if counts.converters:
        try:
            conv = unit if unit is not None else converter_um2(registry, type_name, declared, readout_bits)
        except AreaRuleRefusal as refused:  # mi-R3: the array part priced before it is the one equation's own
            raise ConverterRefusal(str(refused), core_area(counts.array, cell, eta, share).array_um2) from None
    terms = core_area(counts.array, cell, eta, share, n_conv=counts.converters, conv_um2=conv)
    if counts.beside is not None:  # mi-R3: a memory beside the core is the array part of its named type's equation
        bits, beside_type = counts.beside
        memory = core_terms(registry, beside_type, Declared(declared.node_nm), Counts(bits)).array_um2
        terms = terms._replace(core_um2=terms.core_um2 + memory)
    return terms


__all__ = ["ANCHOR_FIGURES", "CELL", "COUNT", "COUNT_OR_ZERO", "Cell", "Counts", "Declared", "Kind", "QUANTITIES",
           "POSITIVE", "SHARE", "WEIGHTS", "anchor_counts", "cells_per_weight", "census_counts", "cited",
           "converter_um2", "core_cell_um2", "core_terms", "grid_share", "judged", "judged_fields", "own_counts",
           "resolved_cell", "scaled_counts", "type_eta", "weighted_counts"]
