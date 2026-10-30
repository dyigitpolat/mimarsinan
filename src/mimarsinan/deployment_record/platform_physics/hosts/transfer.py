"""[U-J5 L-1] Where a chip's per-byte transfer rows price: its backing store, or a chip-internal buffer.

The vocabulary's per-byte transfer rows (``TRANSFER_ROWS``: ``e_dma_per_byte`` in, ``e_readout_per_byte`` out) price
the chip's backing-store channel: the bytes it moves between its store and its cores. Some chip descriptions declare
such a row for a buffer inside the chip that every arriving byte passes (isaac_like: a byte read through the tile's
eDRAM buffer, ``shafiee2016isaac`` Table I). The chip description says which (``transfer_rows``:
``{"<row>": {"place": "chip_buffer", "citation" | "derivation" | "note": <source>}}``); a row whose place it does not
declare is the backing store's, the vocabulary's meaning, so every profile that declares none prices as before.

The link-crossing rule of ``hosts.store`` (U-J4) and the host-owned store's rule (U-J3) drop only a backing-store row:
a byte that crosses the host-chip link, or lives in the host's memory, never touches the chip's own store channel. A
chip-internal buffer row is paid by every byte that enters that buffer, crossing the link or not (``chip_buffer``).
Without the link the two places price alike: the row is the chip's one channel either way. A declaration of a row
outside the transfer rows, of a row the chip does not declare, of an unknown place or field, or without a source,
refuses by name (``TransferRowError``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.schema.serde import strict_kwargs

TRANSFER_FIELD = "transfer_rows"
#: The chip's per-byte transfer rows (the vocabulary's): a byte moved onto the chip, a byte read off it.
TRANSFER_ROWS = ("e_dma_per_byte", "e_readout_per_byte")
BACKING_STORE, CHIP_BUFFER = "backing_store", "chip_buffer"
PLACES = (BACKING_STORE, CHIP_BUFFER)
_SOURCE_FIELDS = ("citation", "derivation", "note")
#: The basis a chip-buffer row's term carries where the bytes it prices cross the link (or come from a host store).
BUFFER_BASIS = (" (a chip-internal buffer every arriving byte passes, the chip description's transfer_rows: paid "
                "whether the byte crosses the host-chip link or not)")


class TransferRowError(ToolchainRefusal, ValueError):
    """A ``transfer_rows`` declaration outside its closed form, or of a row the chip does not declare."""


@dataclass(frozen=True)
class TransferPlace:
    """Where one transfer row prices (``PLACES``), with its source."""

    place: str
    citation: str = ""
    derivation: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"place": self.place, **{key: getattr(self, key) for key in _SOURCE_FIELDS if getattr(self, key)}}


def transfer_rows_from_declaration(data: Any, where: str) -> Dict[str, TransferPlace]:
    """The declaration ``data`` judged (None: no row declared); an ill-formed one refuses by name."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TransferRowError(f"{where}: {TRANSFER_FIELD} maps a transfer row to its place, got {data!r}")
    out: Dict[str, TransferPlace] = {}
    for row, entry in data.items():
        if row not in TRANSFER_ROWS:
            raise TransferRowError(f"{where}: {TRANSFER_FIELD} names {row!r}, not one of the transfer rows "
                                   f"{list(TRANSFER_ROWS)}")
        if not isinstance(entry, Mapping) or "place" not in entry:
            raise TransferRowError(f"{where}: {TRANSFER_FIELD} {row!r} states its place ({list(PLACES)}) with its "
                                   f"source, got {entry!r}")
        if unknown := sorted(set(entry) - {"place", *_SOURCE_FIELDS}):
            raise TransferRowError(f"{where}: {TRANSFER_FIELD} {row!r} fields {unknown} are not among "
                                   f"{['place', *_SOURCE_FIELDS]}")
        if entry["place"] not in PLACES:
            raise TransferRowError(f"{where}: {TRANSFER_FIELD} {row!r} place {entry['place']!r} is not one of "
                                   f"{list(PLACES)}")
        source = {key: str(entry.get(key) or "") for key in _SOURCE_FIELDS}
        if not any(source.values()):
            raise TransferRowError(f"{where}: {TRANSFER_FIELD} {row!r} states no source (a citation, derivation or "
                                   "note): where a row prices is a declaration with its source, never assumed")
        out[str(row)] = TransferPlace(place=str(entry["place"]), **source)
    return out


def judged_transfer_rows(rows: Mapping[str, TransferPlace], constants: Mapping[str, Any], where: str) -> None:
    """Each row a ``transfer_rows`` declaration places is one the chip declares."""
    if missing := sorted(row for row in rows if row not in constants):
        raise TransferRowError(f"{where}: {TRANSFER_FIELD} places {missing}, which the chip declares no value of")


def chip_buffer(physics: Any, row: str) -> bool:
    """Whether ``row`` prices a chip-internal buffer every arriving byte passes (declared, with its value)."""
    place = (getattr(physics, TRANSFER_FIELD, None) or {}).get(row)
    return place is not None and place.place == CHIP_BUFFER and physics.has(row)


def transfer_rows_to_dict(transfer_rows: Mapping[str, TransferPlace]) -> Dict[str, Any]:
    """[chain 5b budget layout B-1] ``PlatformPhysics.to_dict``'s ``transfer_rows`` entry: present only where a
    row is placed."""
    return ({"transfer_rows": {row: place.to_dict() for row, place in sorted(transfer_rows.items())}}
            if transfer_rows else {})


# [chain 5b budget layout B-3] CL2's external component, moved verbatim from ``platform_physics.profile``
# (which re-exports it): a component beside the chip that a run declares, such as the SRAM-CIM macro's
# backing store with its transfer row ``e_dma_per_byte``.
@dataclass(frozen=True)
class ExternalComponent:
    """[CL2] A component the priced system needs and the profile does not describe (an SRAM-CIM macro's backing
    store): a run declares its ``constants`` beside the profile, or nothing absolute is read."""

    name: str
    constants: Tuple[str, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.name.isspace():
            raise ValueError("ExternalComponent needs a name")
        if not self.constants:
            raise ValueError(f"external component {self.name!r} names no constant")
        for key in self.constants:
            spec_for(key)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "constants": list(self.constants), "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalComponent":
        kwargs = strict_kwargs(cls, data)
        kwargs["constants"] = tuple(kwargs.get("constants") or ())
        return cls(**kwargs)
