"""[U-J5 L-2] Which ends of the host-chip link a host's link energy row counts, corner by corner, and the boundary its
priced terms name.

No row of the vocabulary prices the chip's own end of the link (``hosts.store``'s rule: a crossing byte pays the link
and host memory, never the chip's backing-store rows). Whether a host's ``e_link_per_byte`` leaves that end out is
the host description's fact, and it may differ by corner (the Jetson description counts the host's PHY end alone at
its low and nominal ends and both ends at its high end). The description declares it (``link_ends``:
``{"rows": {"e_link_per_byte": {"low": [...], "nominal": [...], "high": [...]}}, "basis": ...}``, each corner's ends
among ``ENDS``); its block row carries the declaration (``LINK_ENDS_FIELD``) into the resolved physics, so a record
reprices with it. A link term names an end unpriced only at the corners where the row does not count it
(``link_end``); a physics that declares no ends for the row keeps U-J4's text, the chip's end unpriced at every corner.
A declaration on another row, or one that does not state each corner's ends among ``ENDS`` once each, refuses by
name."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

LINK_ENDS_FIELD = "link_ends"
HOST_END, CHIP_END = "host", "chip"
ENDS = (HOST_END, CHIP_END)
CORNERS = ("low", "nominal", "high")
#: The vocabulary rows that price the link's PHY ends per byte.
LINK_END_ROWS = ("e_link_per_byte",)
#: [U-J4] The boundary where no ends are declared: the chip's own end, unpriced at every corner.
CHIP_END_UNPRICED = ("boundary: the chip's own end of the host-chip link is unpriced (no row of the vocabulary prices "
                     "it: a crossing byte pays the link and host memory, never the chip's backing-store rows)")
_WHOSE = {HOST_END: "the host's own end", CHIP_END: "the chip's own end"}
_ALONE = {HOST_END: "the host's end alone", CHIP_END: "the chip's end alone"}


def link_ends_refusal(row: str, ends: Any) -> Optional[str]:
    """Why a ``link_ends`` declaration of ``row`` does not stand, or None."""
    if row not in LINK_END_ROWS:
        return f"{row}: {LINK_ENDS_FIELD} is stated only on a row pricing the link's ends ({list(LINK_END_ROWS)})"
    if not isinstance(ends, Mapping) or set(ends) != set(CORNERS):
        return (f"{row}: {LINK_ENDS_FIELD} states the ends the row counts at each corner {list(CORNERS)}, got "
                f"{ends!r}")
    for corner in CORNERS:
        value = ends[corner]
        if (not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value)
                or set(value) - set(ENDS)):
            return (f"{row}: {LINK_ENDS_FIELD} at the {corner} corner states {value!r}: the ends it counts, among "
                    f"{list(ENDS)}, each once")
    return None


def judged_link_ends(row: str, ends: Any, where: str) -> Dict[str, Tuple[str, ...]]:
    """The declaration as per-corner ends in ``ENDS`` order; refused by name (``ValueError``) where it fails."""
    if why := link_ends_refusal(row, ends):
        raise ValueError(f"{where}: {why}")
    return {corner: tuple(end for end in ENDS if end in ends[corner]) for corner in CORNERS}


def _corners(names: Tuple[str, ...]) -> str:
    joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return f"the {joined} corner{'s' if len(names) > 1 else ''}"


def _counted(ends: Tuple[str, ...]) -> str:
    return "both ends" if set(ends) == set(ENDS) else _ALONE[ends[0]]


def link_end(physics: Any, row: str) -> str:
    """What a link row's priced term states beside its basis: for the link's energy, each end its row leaves out at the
    corners where it does (the declared ``link_ends``), or U-J4's text at every corner where none is declared;
    nothing for any other row, or where the row counts both ends at every corner."""
    if row not in LINK_END_ROWS:
        return ""
    declared: Optional[Mapping[str, Tuple[str, ...]]] = (getattr(physics, "link_ends", None) or {}).get(row)
    if declared is None:
        return f"; {CHIP_END_UNPRICED}"
    parts = []
    for end in (CHIP_END, HOST_END):
        left = tuple(corner for corner in CORNERS if end not in declared[corner])
        if not left:
            continue
        counted = "; ".join(f"{_counted(declared[c])} at {_corners((c,))}" for c in CORNERS if c not in left)
        parts.append(f"; boundary: {_WHOSE[end]} of the host-chip link is unpriced at {_corners(left)} (the host "
                     f"description's {row} counts {_counted(declared[left[0]])} there"
                     + (f"; it counts {counted}" if counted else "") + "; no row of the vocabulary prices that end: a "
                     "crossing byte pays the link and host memory, never the chip's backing-store rows)")
    return "".join(parts)


def link_ends_from_declaration(data: Any, where: str) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """[chain 5b budget layout B-2] ``PlatformPhysics.from_dict``'s ``link_ends``: each row's declaration judged
    (``judged_link_ends``); none declared, empty."""
    return {str(row): judged_link_ends(str(row), ends, where) for row, ends in (data or {}).items()}


def link_ends_to_dict(link_ends: Mapping[str, Mapping[str, Tuple[str, ...]]]) -> Dict[str, Any]:
    """[chain 5b budget layout B-1] ``PlatformPhysics.to_dict``'s ``link_ends`` entry: present only where a row
    declares its ends."""
    return ({"link_ends": {row: {corner: list(e) for corner, e in ends.items()}
                           for row, ends in sorted(link_ends.items())}} if link_ends else {})
