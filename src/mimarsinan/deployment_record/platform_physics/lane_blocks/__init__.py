"""Shipped lane blocks (W08): the area, energy and time per lane of an output-stage lane, as data.

A chip that declares lanes names its block (``onchip_arithmetic.lane_block``); the resolved physics then carries the
block's G3 lane rows (``area_per_lane``, ``e_lane_op``, ``t_lane_op``), each note prefixed with the block's name. A
profile's own lane row, or an operator override, is never forked: the block adds only the rows the profile lacks,
and an override block still replaces them by name.

A block declares the node and supply its numbers hold at and one scaling rule per dimension (area ``node^2``,
energy ``node*V^2`` or ``node``, time ``none``); the resolver applies it against the physics' own declared node and
supply and names the factor in each row's note. A block priced beside physics that declares no node, or whose rule
cannot reach the target, is refused by name: a 45 nm block is never priced unscaled beside a 22 nm chip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

LANE_BLOCK_FORMAT_VERSION = 1
#: The G3 lane rows a block may declare (``platform_physics/vocabulary.py``).
LANE_ROWS = ("area_per_lane", "e_lane_op", "t_lane_op")
_FIELDS = frozenset({"format_version", "name", "display_name", "classes", "columns_per_lane", "source",
                     "technology_node_nm", "supply_v", "scaling", "constants"})
#: The scaling rules a block may declare per dimension, and the dimension each lane row scales by.
SCALING_RULES = {"area": ("node^2",), "energy": ("node*V^2", "node"), "time": ("none", "node")}
ROW_DIMENSION = {"area_per_lane": "area", "e_lane_op": "energy", "t_lane_op": "time"}


class LaneBlockError(ToolchainRefusal, ValueError):
    """A lane block that is unknown, malformed, or forks a constant the profile already declares."""


@dataclass(frozen=True)
class LaneBlock:
    name: str
    display_name: str
    classes: Tuple[str, ...]
    columns_per_lane: int
    constants: Mapping[str, Mapping[str, Any]]
    #: The node and supply the block's numbers hold at, and its rule per dimension (``SCALING_RULES``).
    technology_node_nm: Optional[float] = None
    supply_v: Optional[float] = None
    scaling: Mapping[str, str] = field(default_factory=dict)


def lane_block_names() -> Tuple[str, ...]:
    return tuple(sorted(path.stem for path in Path(__file__).parent.glob("*.json")))


def load_lane_block(name: str) -> LaneBlock:
    """The shipped block ``name``, refused by name when unknown or malformed."""
    path = Path(__file__).with_name(f"{name}.json")
    if not path.is_file():
        raise LaneBlockError(f"lane_block {name!r} is no shipped lane block; known: {list(lane_block_names())}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) - _FIELDS or data.get("format_version") != LANE_BLOCK_FORMAT_VERSION or data.get("name") != name:
        raise LaneBlockError(f"lane block {path.name}: fields {sorted(data)}, format_version "
                             f"{data.get('format_version')!r}, name {data.get('name')!r}")
    constants = data.get("constants") or {}
    unknown = sorted(set(constants) - set(LANE_ROWS))
    if unknown:
        raise LaneBlockError(f"lane block {name!r} declares {unknown}, which are no lane rows {list(LANE_ROWS)}")
    scaling = dict(data.get("scaling") or {})
    bad = {dim: rule for dim, rule in scaling.items() if rule not in SCALING_RULES.get(dim, ())}
    if bad or (scaling and data.get("technology_node_nm") is None):
        raise LaneBlockError(f"lane block {name!r}: scaling {scaling} at node {data.get('technology_node_nm')!r}; "
                             f"rules per dimension {SCALING_RULES}, a node declared")
    return LaneBlock(name=name, display_name=str(data["display_name"]), classes=tuple(data["classes"]),
                     columns_per_lane=int(data["columns_per_lane"]), constants=constants,
                     technology_node_nm=None if data.get("technology_node_nm") is None
                     else float(data["technology_node_nm"]),
                     supply_v=None if data.get("supply_v") is None else float(data["supply_v"]), scaling=scaling)


@dataclass(frozen=True)
class TargetNode:
    """The node and supply of the physics a block is priced beside (its validity), the scaling's target."""

    technology_node_nm: Optional[float] = None
    supply_v: Optional[float] = None


def _factor(block: LaneBlock, dimension: str, target: TargetNode) -> Tuple[float, str]:
    """``(factor, note)`` taking ``dimension``'s numbers from the block's node and supply to ``target``'s."""
    rule = block.scaling.get(dimension)
    node, home = target.technology_node_nm, block.technology_node_nm
    if node is not None and home is not None and float(node) == float(home):
        there, own = target.supply_v, block.supply_v
        if rule != "node*V^2" or there is None or own is None or float(there) == float(own):
            return 1.0, f"at the target's node {float(node):g} nm, unscaled"
        volts = (float(there) / float(own)) ** 2  # node ratio 1: the supply still scales energy
        return volts, f"at the target's node {float(node):g} nm; scaled by V^2 = ({there:g}/{own:g})^2 = {volts:.4g}"
    if rule == "none":
        return 1.0, f"{dimension} unscaled (rule none)"
    if rule is None or node is None or home is None:
        raise LaneBlockError(f"lane block {block.name!r} holds at {home} nm with {dimension} rule {rule!r}, priced beside "
                             f"physics at {node} nm: a block at another (or an undeclared) node needs a declared "
                             f"scaling rule and a target node; declare the lane rows (platform_physics_overrides) or a "
                             f"profile with a technology node")
    ratio = float(node) / float(home)
    if rule == "node^2":
        return ratio ** 2, f"scaled by (node ratio)^2 = ({node:g}/{home:g})^2 = {ratio ** 2:.4g}"
    if rule == "node":
        return ratio, f"scaled by node ratio {node:g}/{home:g} = {ratio:.4g}"
    if target.supply_v is None or block.supply_v is None:  # node*V^2 without the target's supply
        return ratio, (f"scaled by node ratio {node:g}/{home:g} = {ratio:.4g}; the target declares "
                       f"no supply, so V^2 is not scaled")
    volts = (float(target.supply_v) / float(block.supply_v)) ** 2
    return ratio * volts, (f"scaled by node x V^2 = ({node:g}/{home:g}) x "
                           f"({target.supply_v:g}/{block.supply_v:g})^2 = {ratio * volts:.4g}")


def _scaled(value: Mapping[str, Any], factor: float) -> Dict[str, Any]:
    out = dict(value)
    for key in ("low", "nominal", "high"):
        if out.get(key) is not None:
            out[key] = float(out[key]) * factor
    return out


def lane_block_rows(name: Optional[str], declared: Mapping[str, Any],
                    target: TargetNode = TargetNode()) -> Dict[str, Dict[str, Any]]:
    """The block's rows the physics does not declare yet, as override-shaped declarations scaled to ``target`` by the
    block's rule, with the block's name and the factor in every note (``declared``: the profile's and the overrides'
    keys)."""
    if name is None:
        return {}
    block = load_lane_block(name)
    rows: Dict[str, Dict[str, Any]] = {}
    for key, value in block.constants.items():
        if key in declared:
            continue
        factor, how = _factor(block, ROW_DIMENSION[key], target)
        node = "" if block.technology_node_nm is None else f"{block.technology_node_nm:g} nm block"
        scale_note = f"({node}; {how})" if how else ""
        rows[key] = {**_scaled(value, factor),
                     "note": " ".join(filter(None, (f"[lane block {name}]", scale_note, value.get("note"))))}
    return rows
