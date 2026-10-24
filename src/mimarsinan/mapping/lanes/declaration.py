"""The lane declaration ``onchip_arithmetic`` (WI-5e): which lane classes a value-domain chip's cores execute.

Absent, the chip has no lanes and every program is the MVM contract's, byte for byte. Declared, the classes are
a subset of the lane subset this tree builds (arith_toggle.md §3.1): ``L`` (the clamp law: ReLU as the law's
clamp on an unsigned grid), ``E`` (a fused add: one on-chip operand entering the law with its own multiplier) and
``P`` (an average pool as a requantizing reduction); the toggle's other classes are named and refused. Lanes run
only on the integer path (``readout_law`` declared): a float value program on a lane chip keeps its host ops and
says so (:func:`lanes_executed`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Mapping, Optional

from mimarsinan.common.refusal import ToolchainRefusal

#: The lane classes this tree realizes, in the toggle's order.
CLAMP, ADD, POOL = "L", "E", "P"
BUILT_LANE_CLASSES = (CLAMP, ADD, POOL)
#: The toggle's classes no lowering here realizes yet (affine, comparator, wiring, input law): refused by name.
RESERVED_LANE_CLASSES = ("A", "C", "W", "I")
LANE_KEY = "onchip_arithmetic"


class LaneDeclarationError(ToolchainRefusal, ValueError):
    """An ``onchip_arithmetic`` declaration this tree cannot realize."""


@dataclass(frozen=True)
class LaneDeclaration:
    """The declared lane classes and the lane block a chip prices them with: lanes per core and the shipped block
    whose constants (area, energy and time per lane) the physics takes (W08)."""

    classes: FrozenSet[str]
    lanes_per_core: Optional[int] = None
    lane_block: Optional[str] = None

    @property
    def any(self) -> bool:
        return bool(self.classes)

    def declares(self, lane_class: str) -> bool:
        return lane_class in self.classes


def parse_lane_declaration(value: Any) -> Optional[LaneDeclaration]:
    """The declaration ``value`` names, or ``None`` when absent; refused by name where it cannot be realized."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or "classes" not in value:
        raise LaneDeclarationError(f"{LANE_KEY} must be an object with a classes list, got {value!r}")
    classes = value["classes"]
    if not isinstance(classes, (list, tuple)) or not all(isinstance(c, str) for c in classes):
        raise LaneDeclarationError(f"{LANE_KEY}.classes must be a list of class letters, got {classes!r}")
    if len(set(classes)) != len(classes):
        raise LaneDeclarationError(f"{LANE_KEY}.classes names a class twice: {list(classes)}")
    reserved = sorted(set(classes) & set(RESERVED_LANE_CLASSES))
    if reserved:
        raise LaneDeclarationError(f"{LANE_KEY} declares lane classes {reserved}, which no lowering here realizes "
                                   f"(built: {list(BUILT_LANE_CLASSES)})")
    unknown = sorted(set(classes) - set(BUILT_LANE_CLASSES))
    if unknown:
        raise LaneDeclarationError(f"{LANE_KEY} declares unknown lane classes {unknown}; known: "
                                   f"{list(BUILT_LANE_CLASSES)}")
    if classes and CLAMP not in classes:
        raise LaneDeclarationError(f"{LANE_KEY} declares {sorted(classes)} without {CLAMP}: every lane writes "
                                   "through the clamp law, so E and P need L")
    lanes = value.get("lanes_per_core")
    if lanes is not None and (isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1):
        raise LaneDeclarationError(f"{LANE_KEY}.lanes_per_core must be an integer of at least 1, got {lanes!r}")
    block = value.get("lane_block")
    if block is not None and (not isinstance(block, str) or not block):
        raise LaneDeclarationError(f"{LANE_KEY}.lane_block must name a shipped lane block, got {block!r}")
    if not classes and (lanes is not None or block is not None):
        raise LaneDeclarationError(f"{LANE_KEY} declares no lane class but a lane block: a chip without lanes "
                                   "prices none")
    return LaneDeclaration(classes=frozenset(classes), lanes_per_core=lanes, lane_block=block)


def lane_declaration(config: Mapping[str, Any]) -> Optional[LaneDeclaration]:
    """The declaration a config or resolved platform carries (``None``: no lanes)."""
    return parse_lane_declaration(config.get(LANE_KEY))


def lanes_executed(config: Mapping[str, Any]) -> Optional[LaneDeclaration]:
    """The declaration when its lanes run: classes declared and the program on the integer path."""
    declaration = lane_declaration(config)
    if declaration is None or not declaration.any or config.get("readout_law") is None:
        return None
    return declaration


def declared_lane_block(config: Mapping[str, Any]) -> Optional[str]:
    """The lane block a chip with lanes prices (W08), on or off the integer path; ``None`` without lanes."""
    declaration = lane_declaration(config)
    return declaration.lane_block if declaration is not None and declaration.any else None
