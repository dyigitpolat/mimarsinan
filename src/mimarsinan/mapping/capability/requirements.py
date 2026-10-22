"""The one table from op kind to capability class: what an operator needs before it can run on a core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Mapping

from mimarsinan.common.refusal import ToolchainRefusal

#: Capability classes of the ladder (scout_network.md §3.3; L A E P C W are the arithmetic toggle's, arith_toggle.md §3.1).
CAPABILITY_CLASSES: Mapping[str, str] = {
    "G": "grouped or depthwise convolution lowering",
    "S": "static-linear lowering of an operator with no crossbar form today",
    "D": "dynamic-operand matrix product (both operands activations)",
    "L": "output-stage clamp law",
    "A": "per-channel affine folded into a law",
    "E": "elementwise add of aligned on-chip tensors",
    "P": "sum reduction (average pooling, token mean)",
    "C": "max reduction (max pooling)",
    "W": "wiring completion (slices, stack, padding, constant-token concat)",
    "X": "elementwise product of two on-chip tensors",
    "T": "table law (smooth activations)",
    "R": "row reduction (softmax, norms)",
}

#: The arithmetic-toggle lane: one law and one reduction per output element (arith_toggle.md §3.2-§3.4).
LANE_CLASSES = frozenset("LAEPC")

#: Placement rules: a crossbar kind is lowered to cores once its classes are declared; a lane kind is absorbed
#: into an on-chip producer (classes declared and every producer on chip); a wiring kind becomes gather wiring
#: (transparent off chip); a host-only kind is a host op at every level.
CROSSBAR, LANE, WIRING, HOST_ONLY = "crossbar", "lane", "wiring", "host"
PLACEMENTS = frozenset({CROSSBAR, LANE, WIRING, HOST_ONLY})

#: Element operations of an on-chip op: one per output element, one per input element (reductions), or none.
OUTPUT_ELEMENTS, INPUT_ELEMENTS, NO_WORK = "output", "input", "none"
WORK_RULES = frozenset({OUTPUT_ELEMENTS, INPUT_ELEMENTS, NO_WORK})


class UnknownOpKindError(ToolchainRefusal, KeyError):
    """An op kind (or an operator) the requirement table has no row for; the census refuses rather than guess."""


@dataclass(frozen=True)
class Requirement:
    """One table row: the classes a kind needs, how it is placed, its element work, and why it is host today."""

    kind: str
    classes: frozenset
    placement: str
    work: str
    reason: str
    #: A boundary the census states (with the kind's host MACs) where the host placement is by design (§16 decision 4).
    host_boundary: str = ""
    #: The kind is a product with MACs (every crossbar kind, and a torch leaf): the MAC SSOT counting it 0 is a gap.
    carries_macs: bool = False

    def met_by(self, declared: AbstractSet[str]) -> bool:
        return self.placement != HOST_ONLY and self.classes <= declared

    @property
    def is_lane(self) -> bool:
        return self.placement == LANE and bool(self.classes) and self.classes <= LANE_CLASSES


def _row(kind: str, classes: str, placement: str, work: str, reason: str, host_boundary: str = "",
         carries_macs: bool = False) -> Requirement:
    return Requirement(kind, frozenset(classes), placement, work, reason, host_boundary,
                       carries_macs or placement == CROSSBAR)


REQUIREMENTS: Mapping[str, Requirement] = {r.kind: r for r in (
    _row("mm", "", CROSSBAR, NO_WORK, "static matrix product: a crossbar core at every level"),
    _row("mmg", "G", CROSSBAR, NO_WORK, "grouped or depthwise convolution: no core lowering until N5",
         host_boundary="grouped convolution on the host until N5"),
    _row("mms", "S", CROSSBAR, NO_WORK, "static-weight linear operator with no crossbar lowering today (N6)"),
    _row("dmm", "D", CROSSBAR, NO_WORK, "product of two activations: no dynamic-operand core until N2"),
    _row("clamp", "L", LANE, OUTPUT_ELEMENTS, "clamp law outside the MVM contract until the arithmetic toggle (N3)"),
    _row("affine", "A", LANE, OUTPUT_ELEMENTS, "standalone affine outside the MVM contract until N3"),
    _row("add", "E", LANE, OUTPUT_ELEMENTS, "add of two activations outside the MVM contract until N3"),
    _row("pool", "P", LANE, INPUT_ELEMENTS, "sum reduction outside the MVM contract until N3"),
    _row("maxp", "C", LANE, INPUT_ELEMENTS, "max reduction outside the MVM contract until N3"),
    _row("prod", "X", LANE, OUTPUT_ELEMENTS, "product of two on-chip tensors needs an operand lane (N4)"),
    _row("smooth", "T", LANE, OUTPUT_ELEMENTS, "smooth activation needs a table law (N4)"),
    _row("hswish", "LX", LANE, OUTPUT_ELEMENTS, "HardSwish is a clamp times its input (N3 and N4)"),
    _row("softmax", "RTX", LANE, OUTPUT_ELEMENTS, "softmax needs a row reduction, a table law and a product (N4)"),
    _row("norm", "RTXA", LANE, OUTPUT_ELEMENTS, "normalization needs a row reduction and a data-dependent scale (N4)"),
    _row("wire+", "W", WIRING, NO_WORK, "a wiring completion the gather does not express today (N6)"),
    _row("leaf", "", HOST_ONLY, NO_WORK,
         "a torch leaf module (attention, recurrent or encoder layer) stays one host op until N1 decomposes it",
         carries_macs=True),
    _row("data_division", "", HOST_ONLY, NO_WORK, "division by an activation is not a law of on-chip integers"),
    _row("payload_write", "", HOST_ONLY, NO_WORK,
         "host-side write of a dynamic operand into its crossbar (the design note's bank_program op)"),
)}


def requirement_of(kind: str) -> Requirement:
    """The table row of ``kind``, or a refusal by name: the census never guesses a requirement."""
    try:
        return REQUIREMENTS[kind]
    except KeyError:
        raise UnknownOpKindError(
            f"op kind {kind!r} has no row in the requirement table (mapping/capability/requirements.py); "
            f"known kinds: {sorted(REQUIREMENTS)}"
        ) from None
