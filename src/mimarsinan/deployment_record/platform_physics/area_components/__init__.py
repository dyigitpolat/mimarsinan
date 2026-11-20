"""[U-A] Chip area from declared components (owner instruction of 2026-10-28), as data.

A component (``<name>.json`` beside this module) is one physical block with a sourced area at its own node and a
declared node-scaling rule (``node_scaling.json``): the rule takes the area from the component's node to the chip's
(``validity.technology_node_nm``), and every scaled value names its component, source node, rule and factor.

A chip description declares two kinds of use:

* ``chip_components`` (a profile's list): the blocks that do not grow with the core count, each with its count (the
  host link PHY by link type and lane count, a memory PHY where the chip owns a DRAM interface, the global
  controller, clock generation, pads). Their sum at the chip's node IS the chip's ``area_global_fixed`` (derived);
  a scalar declared beside them is refused by name, so nothing is counted twice.
* a per-core area row naming a component (``{"component": <name>, "note": ...}``): the row's value is the
  component's area at the chip's node, in the row's unit (a router per NoC tile, for example).

A component of unknown name or rule, an unsourced component (an area must carry published, datasheet, derived or
measured evidence), a component at another node beside a chip that declares none, and a link PHY whose count is not
its link's lane count, are each refused by name (``AreaComponentError``). ``fixed_area_refusal`` is the pricer's
rule for ``area_global_fixed``: the sum, else a sourced scalar, else the chip area is refused by name (an
``estimated`` scalar prices only on a projection, a profile whose every constant is an estimate by declaration)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.common.refusal.format_version import is_version
from mimarsinan.deployment_record.platform_physics.constants import constant_from_declaration, spec_for
from mimarsinan.deployment_record.units import AREA, to_canonical, unit_for

AREA_COMPONENT_FORMAT_VERSION = 1
#: What a component is: chip-level blocks (``CHIP_LEVEL``) and the per-core blocks a row may name.
CHIP_LEVEL = ("link_phy", "memory_phy", "controller", "clock", "pads")
KINDS = CHIP_LEVEL + ("router", "tile_logic")
#: What one counted instance of a component is.
PER = ("lane", "channel", "link", "instance", "router")
#: Evidence an area component may carry: a source, never a bare estimate.
SOURCED = frozenset({"published", "datasheet", "derived", "measured"})
FIXED = "area_global_fixed"
_FIELDS = frozenset({"format_version", "name", "display_name", "kind", "per", "technology_node_nm", "scaling",
                     "link_family", "area"})
_DECL_FIELDS = frozenset({"component", "count", "role", "link", "note"})
_ATTACH = re.compile(r"^(?P<family>.*\S)\s+x(?P<lanes>\d+)$")


class AreaComponentError(ToolchainRefusal, ValueError):
    """An area component or its declaration that is unknown, unsourced, unscalable or counted twice."""


@dataclass(frozen=True)
class ScalingRule:
    """Area at node ``n_to`` = area at ``n_from`` x (n_to / n_from) ** exponent; the band takes the bracket's two
    exponents' factors, ordered, so it holds in either direction of the node ratio."""

    name: str
    exponent: float
    bracket: Tuple[float, float]
    basis: str

    def factors(self, n_from: float, n_to: float) -> Tuple[float, float, float]:
        ratio = float(n_to) / float(n_from)
        ends = sorted(ratio ** e for e in self.bracket)
        return ends[0], ratio ** self.exponent, ends[1]


@dataclass(frozen=True)
class AreaComponent:
    name: str
    display_name: str
    kind: str
    per: str
    technology_node_nm: float
    scaling: str
    #: The declaration this component's area holds at: its band (unit, evidence and source beside it).
    area: Mapping[str, Any]
    #: A link PHY's link family ("PCIe Gen3"): a chip's link declaration must name it.
    link_family: Optional[str] = None

    def band_um2(self) -> Tuple[float, float, float]:
        value = constant_from_declaration(FIXED, {**self.area, "unit": self.area.get("unit", "mm^2")})
        scale = to_canonical(1.0, value.unit) / to_canonical(1.0, "um^2")
        return value.low * scale, value.nominal * scale, value.high * scale


@dataclass(frozen=True)
class ChipComponent:
    """One chip-level component a profile declares: which, how many, its role and (a link PHY) its link."""

    component: str
    count: int
    role: str = ""
    #: A link PHY's link declaration: ``{"host" or "peer": <the other end>, "attach": "<family> x<n>"}`` (n lanes of a
    #: lane PHY, n links of a link PHY).
    link: Optional[Mapping[str, str]] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"component": self.component, "count": self.count, "role": self.role,
                **({"link": dict(self.link)} if self.link else {}), "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChipComponent":
        unknown = set(data) - _DECL_FIELDS
        if unknown or "component" not in data or int(data.get("count", 0)) < 1:
            raise AreaComponentError(f"chip component {dict(data)!r}: fields {sorted(_DECL_FIELDS)}, a component "
                                     f"name and a count of at least 1 (unknown {sorted(unknown)})")
        return cls(component=str(data["component"]), count=int(data["count"]), role=str(data.get("role", "")),
                   link=dict(data["link"]) if data.get("link") else None, note=str(data.get("note", "")))


def _here() -> Path:
    return Path(__file__).parent


def component_names() -> Tuple[str, ...]:
    return tuple(sorted(p.stem for p in _here().glob("*.json") if p.stem != "node_scaling"))


@lru_cache(maxsize=None)
def scaling_rules() -> Mapping[str, ScalingRule]:
    data = json.loads((_here() / "node_scaling.json").read_text(encoding="utf-8"))
    return {name: ScalingRule(name, float(r["exponent"]), (float(r["bracket"][0]), float(r["bracket"][1])),
                              str(r["basis"])) for name, r in data["rules"].items()}


@lru_cache(maxsize=None)
def load_component(name: str) -> AreaComponent:
    """The shipped component ``name``, refused by name when unknown, malformed, unscalable or unsourced."""
    path = _here() / f"{name}.json"
    if name == "node_scaling" or not path.is_file():
        raise AreaComponentError(f"area component {name!r} is not shipped; known: {list(component_names())}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) - _FIELDS or not is_version(data.get("format_version"), AREA_COMPONENT_FORMAT_VERSION) \
            or data.get("name") != name:  # [U-HOST3] the version rule's one reading
        raise AreaComponentError(f"area component {path.name}: fields {sorted(data)}, format_version "
                                 f"{data.get('format_version')!r}, name {data.get('name')!r}")
    if data.get("kind") not in KINDS or data.get("per") not in PER or data.get("scaling") not in scaling_rules():
        raise AreaComponentError(f"area component {name!r}: kind {data.get('kind')!r} (one of {list(KINDS)}), per "
                                 f"{data.get('per')!r} (one of {list(PER)}), scaling {data.get('scaling')!r} (one of "
                                 f"{sorted(scaling_rules())})")
    component = AreaComponent(name=name, display_name=str(data["display_name"]), kind=str(data["kind"]),
                              per=str(data["per"]), technology_node_nm=float(data["technology_node_nm"]),
                              scaling=str(data["scaling"]), area=dict(data["area"]),
                              link_family=data.get("link_family"))
    check_sourced(component)
    return component


def check_sourced(component: AreaComponent) -> None:
    """An area component carries a source (``SOURCED`` evidence with its citation, derivation or measurement)."""
    kind = str(component.area.get("evidence_kind", "estimated"))
    if kind not in SOURCED:
        raise AreaComponentError(f"area component {component.name!r} is unsourced (evidence {kind!r}): a component's "
                                 f"area must be {sorted(SOURCED)}, with its source")
    if unit_for(str(component.area.get("unit", "mm^2"))).dimension != AREA:
        raise AreaComponentError(f"area component {component.name!r}: unit {component.area.get('unit')!r} is no area")
    component.band_um2()  # the vocabulary's judge: band order and the evidence's required field


def scaled_um2(component: AreaComponent, node: Optional[float]) -> Tuple[Tuple[float, float, float], str]:
    """The component's area band at ``node`` (um^2) and the note naming its node, rule and factor."""
    low, nominal, high = component.band_um2()
    home = component.technology_node_nm
    if node is not None and float(node) == home:
        return (low, nominal, high), f"{component.name} at its own node {home:g} nm, unscaled"
    if node is None:
        raise AreaComponentError(f"area component {component.name!r} holds at {home:g} nm, priced beside physics that "
                                 f"declares no technology node: its {component.scaling} rule has no target node")
    rule = scaling_rules()[component.scaling]
    f_low, f_nom, f_high = rule.factors(home, float(node))
    note = (f"{component.name} at {home:g} nm scaled to {float(node):g} nm by rule {rule.name} "
            f"(exponent {rule.exponent:g}, band {rule.bracket[0]:g} to {rule.bracket[1]:g}): factors {f_low:.4g} / "
            f"{f_nom:.4g} / {f_high:.4g}")
    return (low * f_low, nominal * f_nom, high * f_high), note


def _check_link(decl: ChipComponent, component: AreaComponent) -> None:
    if component.kind != "link_phy":
        if decl.link:
            raise AreaComponentError(f"chip component {decl.component!r} is no link PHY but declares a link")
        return
    attach = str((decl.link or {}).get("attach", ""))
    match = _ATTACH.match(attach)
    if match is None or match.group("family") != component.link_family:
        raise AreaComponentError(f"link PHY {decl.component!r} serves {component.link_family!r}; the chip's link "
                                 f"declaration attach {attach!r} names no '<family> x<lanes>' of that family")
    expected = int(match.group("lanes"))  # x<n>: n of the component's unit (lanes of a lane PHY, links of a link)
    if decl.count != expected:
        raise AreaComponentError(f"link PHY {decl.component!r} is counted per {component.per}: its link {attach!r} "
                                 f"needs count {expected}, the chip declares {decl.count}")


def chip_components_of(data: Sequence[Mapping[str, Any]]) -> Tuple[ChipComponent, ...]:
    """A profile's ``chip_components``, each judged: a shipped chip-level component, its link where a link PHY."""
    out = tuple(ChipComponent.from_dict(d) for d in data)
    for decl in out:
        component = load_component(decl.component)
        if component.kind not in CHIP_LEVEL:
            raise AreaComponentError(f"chip component {decl.component!r} is a {component.kind!r} block, not a "
                                     f"chip-level one ({list(CHIP_LEVEL)}); a per-core block is named by its row")
        _check_link(decl, component)
    return out


def fixed_area_declaration(decls: Sequence[ChipComponent], node: Optional[float]) -> Dict[str, Any]:
    """The chip's ``area_global_fixed`` as the derived sum of its chip-level components at ``node`` (mm^2)."""
    total, parts, sources = [0.0, 0.0, 0.0], [], []
    for decl in decls:
        component = load_component(decl.component)
        (low, nominal, high), note = scaled_um2(component, node)
        for i, v in enumerate((low, nominal, high)):
            total[i] += decl.count * v / 1e6
        parts.append(f"{decl.count} x {component.display_name} [{decl.role}] = {decl.count * nominal / 1e6:.6g} mm^2 "
                     f"nominal ({note})")
        sources.append(f"{component.name}: {component.area.get('citation') or component.area.get('evidence_kind')}")
    return {"low": total[0], "nominal": total[1], "high": total[2], "unit": "mm^2", "evidence_kind": "derived",
            "citation": "; ".join(sources),
            "derivation": "[U-A] the sum of the chip's declared chip_components at its node: " + "; ".join(parts),
            "note": " ".join(d.note for d in decls if d.note)}


def component_row(key: str, data: Mapping[str, Any], node: Optional[float]) -> Dict[str, Any]:
    """A per-core area row that names a component: the component's area at ``node`` in the row's unit."""
    spec = spec_for(key)
    if spec.dimension != AREA or set(data) - {"component", "note"}:
        raise AreaComponentError(f"{key}: a row naming a component is an area row declaring only 'component' and "
                                 f"'note', got {sorted(data)}")
    component = load_component(str(data["component"]))
    if component.kind in CHIP_LEVEL:
        raise AreaComponentError(f"{key}: component {component.name!r} is chip-level ({component.kind}); it is "
                                 f"declared among chip_components, not per core")
    (low, nominal, high), note = scaled_um2(component, node)
    scale = to_canonical(1.0, "um^2") / to_canonical(1.0, spec.display_unit)
    return {"low": low * scale, "nominal": nominal * scale, "high": high * scale, "unit": spec.display_unit,
            "evidence_kind": "derived", "citation": str(component.area.get("citation", "")),
            "derivation": f"[U-A] the area component {component.display_name}: {note}; its source: "
                          f"{component.area.get('citation') or component.area.get('derivation') or component.area.get('note')}",
            "note": str(data.get("note", ""))}


def resolve_area_declarations(constants: Mapping[str, Any], chip: Sequence[Mapping[str, Any]],
                              node: Optional[float]) -> Tuple[Dict[str, Any], Tuple[ChipComponent, ...]]:
    """A profile's constants with every component-named row resolved and, where it declares chip components, its
    ``area_global_fixed`` as their sum; a scalar beside the sum is refused by name (counted twice)."""
    out = {k: (component_row(k, v, node) if isinstance(v, Mapping) and "component" in v else v)
           for k, v in constants.items()}
    decls = chip_components_of(chip)
    if decls:
        if FIXED in out:
            raise AreaComponentError(f"{FIXED} is declared beside chip_components: the chip-level area is their "
                                     f"sum, so a scalar beside it would count it twice")
        out[FIXED] = fixed_area_declaration(decls, node)
    return out, decls


def fixed_area_refusal(physics: Any) -> Optional[str]:
    """Why the chip area cannot carry ``area_global_fixed`` (None when it can): absent, or an unsourced estimate on a
    chip described as anything but a projection, where the chip declares no chip components."""
    if not physics.has(FIXED):
        return (f"the chip declares neither chip_components nor {FIXED}: its chip-level area (host link and memory "
                f"PHYs, controller, clock, pads) is unknown; declare its chip_components (each a sourced area at its "
                f"node with a node-scaling rule) or a sourced {FIXED}")
    value = physics.constants[FIXED]
    kind = physics.validity.measurement_kind
    if value.evidence_kind == "estimated" and not value.overridden and kind != "projection":
        return (f"{FIXED} is an unsourced estimate on a chip described as {kind!r}, not a projection, and the chip "
                f"declares no chip_components: declare them (each a sourced area at its node with a node-scaling "
                f"rule) or a sourced {FIXED}")
    return None
