"""[U-AREA] Anchors: a description priced from its own published core, scaled off its geometry, word width and size
(design revision 3, sections 2.7, 2.8 and 4; review M1, m5, N-2, N-3, N-12, N-13; the check's minor G).

- The ratio per core type: rho = sum_t w_t q_t(census) / q_t(anchor) / sum_t w_t over the type's scaled terms, a w-bit
  weight taking ceil(w / weight bits per cell) cells; census rows are device rows (a weight row of a differential
  pair is cells_per_weight device rows, N-2); a pooled census prices linear terms and refuses a per-core ceiling.
  Every count is the one count form (``core.equation.scaled_counts``) and the own memory share the one core equation
  (``core.equation.core_terms``), the anchor's organization judged on its type's terms (the delta round, MA-R1).
- The core band (m5): the whole core scaled (the type rule's form) and the anchor's own scaled share s_anchor scaled
  with its logic held; the nominal is the held-logic form where the block declares its logic held per core (N-13),
  else the whole-core form; an off-word point's wider-word reading (stored bits per cell plus the extra weight bits,
  N-12) is named and is the low edge where it lies below; where s_anchor needs a cell rule B refuses, the held-logic
  edge is a named refusal and the nominal prices (N-3).
- The size (R3): proportional nominal; the other edge holds the outside part at its published size: a published
  die's remainder for an own-die anchor, 0 for an anchor without a published die (minor G).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import equation as core_rule
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door, remembered
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import COUNT, POSITIVE, judged, judged_band
from mimarsinan.deployment_record.platform_physics.area_rule.declaration import (
    SCOPES, declared_cell, declared_cell_flags, declared_of)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, HeldOut
from mimarsinan.deployment_record.platform_physics.area_rule.rule import (
    BITS_TERMS, AreaFlag, AreaPrice, CensusCore, outside_gamma, type_gamma)

#: The flag kinds a census decides (T-12 and T-14 compare them).
CENSUS_FLAG_KINDS = ("anchor_identity", "beyond_published", "below_published", "off_geometry", "off_word_width",
                     "weight_bits_not_stated")
__all__ = ["AnchorLaw", "CENSUS_FLAG_KINDS", "CensusCore", "CoreBand", "OwnShare", "Ratio", "anchor_core_band",
           "anchor_law", "anchor_ratio", "device_rows", "own_die_gamma", "price_anchor", "price_anchor_types",
           "size_flags", "wider_word_ratio"]


class Ratio(NamedTuple):
    ratio: float
    cells_per_weight: int
    counts: Mapping[str, float]
    anchor_counts: Mapping[str, float]


@door(organization="anchor organization")
def device_rows(organization: Mapping[str, Any], weight_rows: int) -> int:
    """N-2: the device rows of ``weight_rows`` weight rows (a signed weight a differential pair of cells)."""
    return int(weight_rows * organization.get("cells_per_weight", int(u.UNITY)))


@door(organization="anchor organization")
def anchor_ratio(organization: Mapping[str, Any], census: CensusCore) -> Ratio:
    """A_rule(census core) / A_rule(anchor core) for one core type (review M1): both counts in the one count form
    (``core.equation.scaled_counts``) on the organization's terms, which ``anchor_law`` judges equal to its type's;
    the ratio judged where it is made, a size above 0."""
    org = organization
    form = org["terms"]
    k = core_rule.cells_per_weight(org, census.weight_bits)
    q = core_rule.scaled_counts(form, org, census.rows, census.cols, census.weight_bits, census.pooled,
                                form not in BITS_TERMS)
    qa = core_rule.own_counts(form, org)
    weights = org["term_weights"]
    ratio = sum(weights[t] * q[t] / qa[t] for t in weights) / sum(weights.values())
    return Ratio(judged("the anchor's ratio of the census core", ratio, POSITIVE), k, q, qa)


@door(organization="anchor organization")
def wider_word_ratio(organization: Mapping[str, Any], census: CensusCore) -> Optional[float]:
    """N-12: a weight wider than the cell read as one wider word (stored bits per cell plus the extra weight bits),
    on 'bits' terms with stored bits per cell (the one count form of a cell that holds the whole weight); None where
    the weight fits the cell."""
    org, w = organization, census.weight_bits
    if org["terms"] not in BITS_TERMS or w is None or w <= org["weight_bits_per_cell"] or \
            "stored_bits_per_cell" not in org:
        return None
    stored = org["stored_bits_per_cell"] + (w - org["weight_bits_per_cell"])
    wider = dict(org, stored_bits_per_cell=stored, weight_bits_per_cell=w)
    q = core_rule.scaled_counts(org["terms"], wider, census.rows, census.cols, w)["bits"]
    qa = core_rule.own_counts(org["terms"], org)["bits"]
    weights = org["term_weights"]
    return judged("the anchor's wider-word ratio", weights["bits"] * q / qa / sum(weights.values()), POSITIVE)


class CoreBand(NamedTuple):
    nominal: float
    low: float
    high: float
    held_logic: Optional[float]


@door
def anchor_core_band(core_mm2: float, ratio: float, own_share: Optional[float], logic_held: bool = False,
                     wider: Optional[float] = None) -> CoreBand:
    """The per-core band off the anchor's geometry (m5), continuous: zero width at ratio 1; judged where it is made."""
    whole = core_mm2 * ratio
    if own_share is None:  # N-3: the held-logic edge is refused by name; the whole-core form prices
        if logic_held and ratio != u.UNITY:
            raise AreaRuleRefusal("the anchor holds its logic per core and its own share is refused")
        band = CoreBand(whole, whole if wider is None else min(whole, wider), whole, None)
    else:
        held = core_mm2 * (u.UNITY + own_share * (ratio - u.UNITY))
        nominal = held if logic_held else whole
        low = min(whole, held) if wider is None else min(whole, held, wider)
        band = CoreBand(nominal, low, max(whole, held), held)
    judged_band("the anchor's core band", band.low, band.nominal, band.high)
    return band


class OwnShare(NamedTuple):
    value: Optional[float]
    basis: str
    refusal: Optional[str]


@dataclass(frozen=True)
class AnchorLaw:
    type: str
    scope: str
    outside: str
    node_nm: float
    core_mm2: float
    n_pub: float
    gamma: float
    organization: Mapping[str, Any]
    own_share: OwnShare
    logic_held_per_core: bool
    die_mm2: Optional[float] = None
    gamma_from: str = ""
    #: MA-2: the flag of a declared cell the anchor's own memory share reads (its citation in the reason).
    cell_flags: Tuple[AreaFlag, ...] = ()


def _figure(registry: Any, anchor: Mapping[str, Any], name: str) -> float:
    """An anchor's published figure, an expression over the evidence judged as a quantity of its kind (MA-D1)."""
    return core_rule.cited(registry, f"area_rule.anchor.{name}", anchor.get(name), core_rule.ANCHOR_FIGURES[name])


@door
def own_die_gamma(n_pub: float, core_mm2: float, die_mm2: float) -> float:
    """An own-die anchor's core-grid share of its published die, judged as a share (``core.equation.grid_share``)."""
    return core_rule.grid_share(n_pub, core_mm2, die_mm2)


def _own_share(registry: Any, block: Any, core_mm2: float, declared: Any) -> OwnShare:
    """s_anchor (either basis) judged as a share above 0 and at most 1 (MA-D1): outside it N-3's path, the share
    refused by name and the held-logic edge with it."""
    share = _share_of(registry, block, core_mm2, declared)
    try:
        return share if share.value is None else share._replace(
            value=core_rule.judged(f"area_rule.anchor.own_share ({share.basis})", share.value, core_rule.SHARE))
    except AreaRuleRefusal as refused:
        return OwnShare(None, share.basis, str(refused))


def _share_of(registry: Any, block: Any, core_mm2: float, declared: Any) -> OwnShare:
    """s_anchor: a published split, or the anchor's own memory share, which is the type rule's memory term of its own
    organization over its core (MA-R1: ``core.equation.core_terms`` on its own memory count, its declared cell and
    cell kind)."""
    spec = core_rule.judged_fields(block.anchor.get("own_share"), "area_rule.anchor.own_share")
    basis = spec.get("basis", "")
    try:
        if basis == "published_split":
            if "expression" not in spec:
                raise AreaRuleRefusal("area_rule.anchor.own_share basis published_split states no expression")
            return OwnShare(core_rule.cited(registry, "area_rule.anchor.own_share.expression", spec["expression"],
                                            core_rule.SHARE, None), basis, None)
        if basis == "own_memory_share":
            form = registry.type_table(block.type)["scaled_terms"]
            (memory,) = core_rule.own_counts(form, block.organization, converts=False).values()
            terms = core_rule.core_terms(registry, block.type, declared, core_rule.Counts(memory))
            return OwnShare((terms.array_um2 + terms.state_um2) / (core_mm2 * u.UM2_PER_MM2), basis, None)
    except (AreaRuleRefusal, HeldOut) as refused:
        return OwnShare(None, basis, str(refused))
    raise AreaRuleRefusal(f"area_rule.anchor.own_share basis {basis!r} is not published_split or own_memory_share")


@door(block="anchor block")
def anchor_law(registry: Any, block: Any) -> AnchorLaw:
    """The anchor's published core, size, die and share, read from the evidence by the block's expressions, each judged
    as a quantity of its kind (MA-D1: ``core.equation.cited``, at load and here); made once per frozen block (the
    fourth delta round: core/door.remembered)."""
    return remembered(registry, "anchor law", (block,), lambda: _anchor_law(registry, block))


def _anchor_law(registry: Any, block: Any) -> AnchorLaw:
    a = block.anchor
    if a.get("scope") not in SCOPES:
        raise AreaRuleRefusal(f"area_rule.anchor.scope {a.get('scope')!r} is not one of {list(SCOPES)}")
    n_pub = _figure(registry, a, "n_pub")
    die = _figure(registry, a, "die_mm2") if a.get("die_mm2") is not None else None
    if block.outside == "own_die" and die is None:
        raise AreaRuleRefusal("area_rule.anchor: outside own_die states no die_mm2")
    if a.get("core_mm2") is not None:
        core = _figure(registry, a, "core_mm2")
        gamma, gamma_from = ((own_die_gamma(n_pub, core, die), "the anchor's own die") if die is not None
                             and block.outside == "own_die" else (type_gamma(registry, block.type),
                                                                  f"the {block.type} type's gamma"))
    else:  # a die published without its core: the core is the type's core-grid share of the die (flagged)
        if die is None:
            raise AreaRuleRefusal("area_rule.anchor states neither core_mm2 nor die_mm2")
        gamma, gamma_from = type_gamma(registry, block.type), f"the {block.type} type's gamma (no published core)"
        core = gamma * die / n_pub
    if block.outside == "declared_parts":  # the outside field decides: declared parts sit beside the grid
        gamma, gamma_from = outside_gamma(block.outside, gamma), "declared parts (gamma 1)"
    core_rule.anchor_counts(registry, block.type, block.organization)  # MA-R1: its count, or a named refusal
    cell = declared_cell(registry, block)  # MA-2: a typed cell is a named refusal of the price, never swallowed
    share = _own_share(registry, block, core, declared_of(registry, block))
    # mi-D1: the declared cell's flag only where the own memory share priced on a cell that read it
    reads_cell = cell is not None and share.basis == "own_memory_share" and share.value is not None \
        and core_rule.resolved_cell(registry, block.type, declared_of(registry, block)).reads_declared
    return AnchorLaw(type=block.type, scope=a["scope"], outside=block.outside, node_nm=block.node_nm,
                     core_mm2=core, n_pub=n_pub, gamma=gamma, organization=block.organization, own_share=share,
                     logic_held_per_core=bool(a.get("logic_held_per_core")), die_mm2=die, gamma_from=gamma_from,
                     cell_flags=tuple(declared_cell_flags(registry, block)) if reads_cell else ())


def _type_flags(law: AnchorLaw, census: CensusCore, ratio: Ratio) -> List[AreaFlag]:
    """The departures one census core type names (its geometry and word; the size is the whole chip's)."""
    org, flags = law.organization, []
    if (census.rows, census.cols) != (org["rows"], org["cols"]):
        basis = ("logic held per core (the anchor's published figures fix it): the held-logic form is the nominal"
                 if law.logic_held_per_core else
                 "the whole core scales (the type rule's form; the anchor's figures do not fix its logic per core)")
        flags.append(AreaFlag("off_geometry", f"off geometry: {census.rows:g} x {census.cols:g} against the anchor's "
                                              f"{org['rows']} x {org['cols']}; {basis}"))
    if census.weight_bits is None:
        flags.append(AreaFlag("weight_bits_not_stated", "weight bits not stated: read at the anchor's cell"))
    elif ratio.cells_per_weight > int(u.UNITY):
        flags.append(AreaFlag("off_word_width", f"off word width: {census.weight_bits:g}-bit weights over "
                                                f"{org['weight_bits_per_cell']}-bit cells take {ratio.cells_per_weight}"
                                                " cells"))
    return flags


@door
def size_flags(n: float, n_pub: float) -> List[AreaFlag]:
    """Ruling R3: a chip of another size than its published one is priced and flagged, never refused."""
    if n == n_pub:
        return []
    where = "beyond" if n > n_pub else "below"
    return [AreaFlag(f"{where}_published", f"{where} the published chip: a design study at another size")]


@door(census="census core or absent")
def price_anchor(law: AnchorLaw, n: float, census: Optional[CensusCore] = None) -> AreaPrice:
    """chip_area_mm2 (mm^2) of ``n`` cores of the census core type under an anchor's law (none: its own core, made of
    the values the judge returns, the fourth delta round's MA-C1)."""
    org = law.organization
    own = census or CensusCore(*(judged("{}.{}", org[k], COUNT, core_rule.ORGANIZATION, k)
                                 for k in ("rows", "cols", "weight_bits_per_cell")))
    return price_anchor_types(law, ((n, own),))


@door
def price_anchor_types(law: AnchorLaw, types: Sequence[Tuple[float, CensusCore]]) -> AreaPrice:
    """chip_area_mm2 (mm^2) of a chip of census core types ``(count, core)`` under an anchor's law: the grid is the
    sum of each type's count x its core band; the outside part, the size and the identity are the whole chip's."""
    org, a, flags = law.organization, law.core_mm2, []
    grid = low = high = wider_reading = u.NONE  # the core grid's nominal, low and high (mm^2), the N-12 reading
    at_point, wider_seen, ratios = True, False, []
    for n_t, census in types:
        ratio = anchor_ratio(org, census)
        ratios.append(ratio)
        wider = wider_word_ratio(org, census) if ratio.cells_per_weight > int(u.UNITY) else None
        band = anchor_core_band(a, ratio.ratio, law.own_share.value, law.logic_held_per_core,
                                None if wider is None else a * wider)
        grid, low, high = grid + n_t * band.nominal, low + n_t * band.low, high + n_t * band.high
        flags += _type_flags(law, census, ratio)
        if wider is not None:
            wider_reading, wider_seen = wider_reading + n_t * a * wider, True
        if law.own_share.value is None and ratio.ratio != u.UNITY:
            flags.append(AreaFlag("band_edge_refused", f"the held-logic band edge needs the anchor's own share: "
                                                       f"{law.own_share.refusal}"))
        at_point = at_point and (census.rows, census.cols) == (org["rows"], org["cols"]) \
            and ratio.cells_per_weight == int(u.UNITY)
    n = sum(n_t for n_t, _c in types)
    flags += size_flags(n, law.n_pub)
    if not flags:
        flags.append(AreaFlag("anchor_identity", "anchor identity: its published size, geometry and word"))
    if wider_seen:
        flags.append(AreaFlag("wider_word_reading", "the weight read as one wider word (stored bits per cell plus the "
                                                    "extra weight bits), its core grid bare (N-12)", wider_reading))
    flags += law.cell_flags
    g, n_pub = law.gamma, law.n_pub
    held_part = n_pub * a * (u.UNITY / g - u.UNITY) if law.outside == "own_die" else u.NONE
    nominal = grid / g
    fixed_reading = grid + held_part
    edges = [nominal, fixed_reading, low / g, high / g, low + held_part, high + held_part]
    if law.scope != "die" and n == n_pub:
        edges.append(low)  # a core or macro published without its die: the bare core is the low edge
    if law.scope == "die" and n == n_pub and at_point:
        edges = [nominal]  # a die anchor at its own point: an identity, zero width
    one = ratios[u.FIRST] if len(ratios) == int(u.UNITY) else None
    return AreaPrice(nominal=nominal, low=min(edges), high=max(edges), fixed_reading=fixed_reading,
                     flags=tuple(flags), ratio=None if one is None else one.ratio,
                     cells_per_weight=None if one is None else one.cells_per_weight)
