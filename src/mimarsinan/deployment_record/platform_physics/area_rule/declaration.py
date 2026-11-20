"""[U-AREA] What a description declares: the optional ``area_rule`` block beside ``converter`` and ``carry_buffer``
(design revision 3, section 2.6), its schema and its judge (named refusals).

    "area_rule": {"type": T1 | T2_digital | T2_analog | T3 | T4, "terms": type_rule | complete,
                  "outside": declared_parts | own_die | type,
                  "node": {"node_nm": ..., "bitcell_um2": <expression> (a declared cell, cited by id)},
                  "anchor": {"scope": die | core | macro, "core_mm2": <expression>, "die_mm2": <expression>,
                             "n_pub": <expression>, "own_share": {"basis": ..., "expression": ...},
                             "logic_held_per_core": {"citation": [...]}},
                  "organization": {"rows", "cols", "weight_bits_per_cell", "stored_bits_per_cell", "bits_per_neuron",
                                   "cells_per_weight", "columns_per_converter", "terms", "term_weights", "citation"},
                  "cell_kind": ..., "converter": {"kind": ..., "per_core": ..., "own_unit": <evidence id>},
                  "reading_low_edge": {"type": ..., "ruling": ..., ...},
                  "published_size": {"n_pub": <expression>, "citation": ...}, "drops": [...], "note": ...}

An anchor's figures and a declared cell are expressions over the evidence (registry.py), never numbers typed into a
description; a price that rests on a declared cell carries the flag ``declared_cell`` with its citation (the fix round's
MA-2, design 2.5 and 2.6). ``declared_of`` is a block's declaration as the one core equation reads it
(core/equation.py); an anchor's organization is judged by the one count form on its type's terms (the delta round,
MA-R1). Every quantity a block states is judged as a quantity of its kind by the one judge, ``core.equation.judged``,
here at load and again at the price (the second delta round, MA-D1), through the one door (core/door.py, the third
delta round): ``judge_block`` is the one function the loader calls, and every door calls it on its block.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from mimarsinan.deployment_record.platform_physics.area_components import FIXED
from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import door as _door, equation as core_rule, fields
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door
from mimarsinan.deployment_record.platform_physics.area_rule.core.frozen import freeze
from mimarsinan.deployment_record.platform_physics.area_rule.core.fields import (  # noqa: F401 - the block's names
    OUTSIDE, OWN_SHARE_BASES, SCOPES, TERMS)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, read_ids
from mimarsinan.deployment_record.platform_physics.area_rule.rule import (  # noqa: F401 - the axes, read here too
    AREA_RULE_AXES as AREA_RULE_AXES, AreaFlag, outside_rule)

#: The profile field that holds the block.
AREA_RULE = "area_rule"
ANCHOR_FIGURES = tuple(core_rule.ANCHOR_FIGURES)
ORGANIZATION_REQUIRED = ("rows", "cols", "weight_bits_per_cell", "terms", "term_weights", "citation")
#: The block's fields that hold declared quantities: each an object or absent (MA-D1).
HOLDERS = ("node", "anchor", "organization", "converter", "reading_low_edge", "published_size")


class AreaRuleDeclarationError(AreaRuleRefusal):
    """A description's area_rule block the loader declines, by the name of what it states wrong."""


@dataclass(frozen=True)
class AreaRuleBlock:
    type: str
    terms: str
    outside: str
    node: Optional[Mapping[str, Any]] = None
    anchor: Optional[Mapping[str, Any]] = None
    organization: Optional[Mapping[str, Any]] = None
    cell_kind: Optional[str] = None
    converter: Optional[Mapping[str, Any]] = None
    reading_low_edge: Optional[Mapping[str, Any]] = None
    #: The published chip's core count of a block without an anchor (complete terms): the size its flags read (R3).
    published_size: Optional[Mapping[str, Any]] = None
    drops: Tuple[str, ...] = ()
    note: str = ""
    source: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in HOLDERS:  # MA-D1: a field that holds declared quantities is an object, or absent
            if getattr(self, name) is not None and not isinstance(getattr(self, name), Mapping):
                raise AreaRuleDeclarationError(f"area_rule.{name} {getattr(self, name)!r} is not an object")
        # the fourth delta round: frozen through and through, the door's memo keys its identity
        for name in HOLDERS + ("type", "terms", "outside", "cell_kind"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        if not isinstance(self.drops, tuple) or not all(isinstance(d, str) for d in self.drops) or \
                not isinstance(self.note, str):  # the block's own text (to_dict) reads them
            raise AreaRuleDeclarationError(f"area_rule drops {self.drops!r:.80} and note {self.note!r:.80}: texts")

    @property
    def node_nm(self) -> float:
        if not self.node or "node_nm" not in self.node:
            raise AreaRuleDeclarationError("area_rule.node states no node_nm")
        return float(core_rule.judged("area_rule.node.node_nm", self.node["node_nm"], core_rule.POSITIVE))

    @property
    def bitcell_um2(self) -> Any:
        """The declared cell as stated (an expression over evidence ids; ``declared_cell`` reads and judges it)."""
        return (self.node or {}).get("bitcell_um2")

    def to_dict(self) -> Mapping[str, Any]:
        out: Dict[str, Any] = {"type": self.type, "terms": self.terms, "outside": self.outside}
        for name in ("node", "anchor", "organization", "cell_kind", "converter", "reading_low_edge",
                     "published_size"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.drops:
            out["drops"] = list(self.drops)
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AreaRuleBlock":
        if not isinstance(data, Mapping):
            raise AreaRuleDeclarationError("area_rule must be an object")
        allowed = {"type", "terms", "outside", "node", "anchor", "organization", "cell_kind", "converter",
                   "reading_low_edge", "published_size", "drops", "note"}
        if unknown := sorted(set(data) - allowed):
            raise AreaRuleDeclarationError(f"area_rule declares unknown fields {unknown}")
        for name in ("type", "terms", "outside"):
            if name not in data:
                raise AreaRuleDeclarationError(f"area_rule states no {name}")
        return cls(type=str(data["type"]), terms=str(data["terms"]), outside=str(data["outside"]),
                   node=data.get("node"), anchor=data.get("anchor"), organization=data.get("organization"),
                   cell_kind=data.get("cell_kind"), converter=data.get("converter"),
                   reading_low_edge=data.get("reading_low_edge"), published_size=data.get("published_size"),
                   drops=tuple(data.get("drops") or ()),
                   note=str(data.get("note") or ""), source=dict(data))


def block_digest(block: Optional[AreaRuleBlock]) -> str:
    """The digest that names a block (the fidelity gate compares chip area only on equal digests)."""
    if block is None:
        return "none"
    text = json.dumps(block.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:u.DIGEST_HEX_DIGITS]


class DeclaredCell(NamedTuple):
    value: float
    ids: Tuple[str, ...]
    expression: str


@door
def declared_cell(registry: Any, block: Any) -> Optional[DeclaredCell]:
    """MA-2: the description's own cell (``node.bitcell_um2``, um^2), judged as an anchor's figures are: an expression
    that reads evidence ids and evaluates to a positive area; a typed number is a named refusal. None where the block
    declares no cell. Every site that reads the description's node cell reads it here (a type rule's cells, an anchor's
    own memory share). Made once per frozen block (the fourth delta round: core/door.remembered)."""
    def made() -> Optional[DeclaredCell]:
        stated = block.bitcell_um2
        if stated is None:
            return None
        value = core_rule.cited(registry, "area_rule.node.bitcell_um2", stated, core_rule.CELL, "a declared cell",
                                AreaRuleDeclarationError)
        return DeclaredCell(value, read_ids(stated), stated)
    return _door.remembered(registry, "declared cell", (block,), made)


@door
def declared_of(registry: Any, block: Any, reading: bool = False) -> core_rule.Declared:
    """A block's declaration as the one core equation reads it (a reading of another type takes that type's default
    cell), made once per frozen block."""
    def made() -> core_rule.Declared:
        converter, cell = block.converter or {}, declared_cell(registry, block)
        return core_rule.Declared(block.node_nm, None if reading else block.cell_kind, None if cell is None
                                  else cell.value, converter.get("kind"), converter.get("own_unit"))
    return _door.remembered(registry, "declared of", (block,), made, reading)


@door
def declared_cell_flags(registry: Any, block: Any) -> List[AreaFlag]:
    """The flag of a price that rests on a declared cell, its citation in the reason."""
    cell = declared_cell(registry, block)
    if cell is None:
        return []
    return [AreaFlag("declared_cell", f"the description's declared cell: node.bitcell_um2 = {cell.expression} "
                                      f"({cell.value:g} um^2, evidence {', '.join(cell.ids)}), in place of the node "
                                      f"series' cell at {block.node_nm:g} nm", cell.value)]


@door
def required_cell_kind(registry: Any, kind: str) -> Optional[str]:
    """N-5: the cell kind a converter kind is priced beside (its DAC is that cell array), None where any is."""
    return registry.converter_kind(kind).get("requires_cell_kind")


def _judge_anchor(block: AreaRuleBlock, registry: Any) -> None:
    anchor, organization = block.anchor or {}, block.organization or {}
    if anchor.get("scope") not in SCOPES:
        raise AreaRuleDeclarationError(f"area_rule.anchor.scope {anchor.get('scope')!r} is not one of {list(SCOPES)}")
    figures = {}
    for name in ANCHOR_FIGURES:
        if anchor.get(name) is None:
            if name == "n_pub" or (name == "die_mm2" and block.outside == "own_die"):
                raise AreaRuleDeclarationError(f"area_rule.anchor states no {name}")
            continue
        figures[name] = core_rule.cited(registry, f"area_rule.anchor.{name}", anchor[name],  # MA-D1: as priced
                                        core_rule.ANCHOR_FIGURES[name], refusal=AreaRuleDeclarationError)
    if block.outside == "own_die" and "core_mm2" in figures:
        core_rule.grid_share(figures["n_pub"], figures["core_mm2"], figures["die_mm2"])
    if missing := [k for k in ORGANIZATION_REQUIRED if k not in organization]:
        raise AreaRuleDeclarationError(f"area_rule.organization states no {missing}")
    if not organization["citation"]:
        raise AreaRuleDeclarationError("area_rule.organization cites nothing")
    try:  # MA-R1: the anchor's own core counts on its type's terms, as at the price
        core_rule.anchor_counts(registry, block.type, organization)
    except AreaRuleRefusal as refused:
        raise AreaRuleDeclarationError(str(refused)) from None
    share = core_rule.judged_fields(anchor.get("own_share"), "area_rule.anchor.own_share")
    if share.get("basis") not in OWN_SHARE_BASES or (share["basis"] == "published_split" and not share.get(
            "expression")):
        raise AreaRuleDeclarationError(f"area_rule.anchor.own_share {dict(share)!r}: a basis of "
                                       f"{list(OWN_SHARE_BASES)}, a published split with its expression")


def judge_block(block: AreaRuleBlock, registry: Any, *, fixed_stated: Optional[Mapping[str, Optional[float]]] = None,
                converter_table: bool = False, validity: Any = None) -> None:
    """A named refusal for anything the block states wrong, alone or beside the description's other rows, and for
    the description's validity geometry the count form reads (MA-K1): the one function (core/door.judged_input), as
    the loader calls it; a price's door judges the block and the validity geometry by the same judges, once per frozen
    input (the rows beside the block, R5's, which the rule never reads, at every entry they have: judge_beside)."""
    _door.judged_input({"block": block, "validity": validity, "beside": (fixed_stated, converter_table)},
                       registry=registry, memo=False)


def judged_block(block: AreaRuleBlock, registry: Any) -> None:
    """The block's judge (the door's table calls it, at load and at the price): its own rules by name, then every
    field of every holder by the judge's table (core/fields.py: mi-K2's held logic among them)."""
    if not isinstance(block.type, str) or block.type not in registry.types:
        raise AreaRuleDeclarationError(f"area_rule.type {block.type!r} is not one of {list(registry.types)}")
    if block.terms not in TERMS:
        raise AreaRuleDeclarationError(f"area_rule.terms {block.terms!r} is not one of {list(TERMS)}")
    if block.outside not in OUTSIDE:
        raise AreaRuleDeclarationError(f"area_rule.outside {block.outside!r} is not one of {list(OUTSIDE)}")
    if block.terms == "complete" and block.outside != "declared_parts":
        raise AreaRuleDeclarationError("area_rule: complete terms take their declared parts (outside declared_parts)")
    if block.anchor is not None:
        _judge_anchor(block, registry)
    core_rule.judged_fields(block.organization, core_rule.ORGANIZATION)  # MA-D1 (d): a type rule's too
    if block.reading_low_edge is not None:  # MA-D1 (d): the reading's quantities and its type, as at the price
        core_rule.judged_fields(block.reading_low_edge, core_rule.READING)
        registry.type_file(block.reading_low_edge.get("type"))
    if block.published_size is not None:
        if block.anchor is not None or block.published_size.get("n_pub") is None:
            raise AreaRuleDeclarationError("area_rule.published_size states the published chip's core count of a "
                                           "block without an anchor, n_pub cited by id (an anchor states anchor.n_pub)")
        core_rule.cited(registry, "area_rule.published_size.n_pub", block.published_size["n_pub"], core_rule.COUNT,
                        refusal=AreaRuleDeclarationError)
    if (block.terms == "type_rule" or block.anchor is not None) and not (block.node or {}).get("node_nm"):
        raise AreaRuleDeclarationError("area_rule: a type rule or an anchor names its node (node.node_nm)")
    if (block.node or {}).get("node_nm") is not None:
        block.node_nm  # noqa: B018 - MA-D1: the node judged as a size, as at the price
    declared_cell(registry, block)  # MA-2: a declared cell is cited by id, at load as at the price
    if block.cell_kind is not None:
        registry.cell_kind(block.cell_kind)
    converter = block.converter or {}
    if converter.get("kind") is not None:
        required = required_cell_kind(registry, converter["kind"])
        if required is not None and block.cell_kind != required:
            raise AreaRuleDeclarationError(
                f"converter kind {converter['kind']} requires cell kind {required} (its DAC is that cell array); the "
                f"description declares {block.cell_kind}")
    if converter.get("own_unit") is not None:
        registry.evidence.entry(converter["own_unit"])
    fields.judged_holder(registry, AREA_RULE, block, "block")  # every field the rule reads, by the judge's table


def judge_beside(block: AreaRuleBlock, *, fixed_stated: Optional[Mapping[str, Optional[float]]] = None,
                 converter_table: bool = False) -> None:
    """What the block states beside the description's other rows, at every entry (the profile, an override, a store
    restatement, a run's converter table): an area_global_fixed beside gamma ("the outside part is stated twice",
    M2) and a converter area beside the block's converter kind ("the converter area is stated twice", review N-6)."""
    kind = (block.converter or {}).get("kind")
    if kind is not None and converter_table:
        raise AreaRuleDeclarationError(f"the converter area is stated twice: the block's converter kind {kind} and an "
                                       "area_per_adc (a flat constant or a converter table row)")
    if block.outside != "declared_parts" and fixed_stated:
        outside_rule(block.outside, u.UNITY, dict(fixed_stated))


def converter_area_stated(constants: Mapping[str, Any], table: Any) -> bool:
    """[U-AREA N-6] Whether a converter's area is stated beside the block: a flat area_per_adc or a converter table
    row's (``table``; the loader's reading at load, the physics judge's at the price: one reading)."""
    return "area_per_adc" in constants or any("area_per_adc" in row for row in (table.rows.values()
                                                                                 if table is not None else ()))


def beside_of(physics: Any) -> Tuple[Mapping[str, Optional[float]], bool]:
    """The rows beside the block as the loader reads them (loader.py profile_from_dict): the physics' own outside part
    (area_global_fixed's nominal, its profile's row) and whether a converter's area is stated beside the block."""
    fixed = physics.constants.get(FIXED)
    return {"profile": None if fixed is None else fixed.nominal}, converter_area_stated(physics.constants,
                                                                                       physics.converter)
