"""[U-J3] Where a chip's backing store lives: the chip's own DRAM through its memory PHY, or the host's memory over the
host-chip link.

A chip description states it (the profile's ``backing_store``: ``{"chip": {<source>}}`` or ``{"host": {<source>}}``, the
source a citation, derivation or note), and a run may restate it as a declared departure (a ``backing_store`` entry
among its ``platform_physics_overrides``, a named block of ``resolve.merge_override_blocks``, ``store_block``): the
restatement is marked overridden and travels with the resolved physics every record seals. A declaration stating both
stores or none, or a field outside its closed set, or no source, refuses by name (``BackingStoreError``). The chip's
area components agree with it where the chip declares them (``area_components``): a memory PHY exactly when the store
is the chip's, and a host-owned store is reached over a link PHY to the host. A host-owned restatement drops the
chip's memory PHYs, so its ``area_global_fixed`` is the sum of the rest; a chip-owned one needs the chip's own; a host
attach row (a chip channel fed over the host link, ``hosts.description.host_attach_block``) on a chip-owned store
refuses by name: its write port reads the chip's own DRAM.

The store takes the store traffic: what a program moves between its passes and its weights' home, the pass carry
(``carry_in_bytes``, ``carry_out_bytes``) and the write law's backing-store payload (a period's ``payload_bytes`` over
K), and a program load's payload. On a chip-owned store the chip's rows price it alone (the carry's re-injection and
read-back ``e_dma_per_byte`` and ``e_readout_per_byte``, the write transfer ``e_dma_per_byte``, the chip's write port,
the memory PHY's area); on a host-owned store the host's rows do (``LINK_ROWS``: the link and host memory per
byte, the slower of their per-byte times, one transfer latency per carried pass boundary and per written pass), and
none of the chip's store rows. What the program itself sends across the boundary (the segment I/O at the network's
input and output and at host operators, host-sourced operand payloads) crosses the link on either store. Where the
physics prices the host-chip link and states no store, store traffic refuses by name (it would be priced on both
stores or on neither, ``cost.host_terms.store_refusal``); a physics without the link and without a declaration prices
it on the chip's rows as it always has."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.area_components import load_component
from mimarsinan.deployment_record.schema.provenance import Band

STORE_FIELD = "backing_store"
CHIP, HOST = "chip", "host"
STORES = (CHIP, HOST)
#: A store's source fields; a declaration's top level holds one store, a block's note and the departure mark.
_SOURCE_FIELDS = frozenset({"citation", "derivation", "note"})
_TOP_FIELDS = frozenset(STORES) | {"note", "overridden"}
#: The edge host's link and host-memory rows (the vocabulary's HOST group): a host-owned store's rows too.
LINK_ROWS = ("e_link_per_byte", "t_link_per_byte", "t_link_transfer", "e_host_mem_per_byte", "t_host_mem_per_byte")
#: The store traffic a view states as quantities: the pass carry, both directions.
CARRY_QUANTITIES = ("carry_in_bytes", "carry_out_bytes")
#: The chip's store rows of the energy table: the carry's re-injection and read-back (``formulas.ENERGY_COMPONENTS``
#: labels), which a host-owned store does not charge.
CHIP_CARRY_LABELS = ("carry_inject", "carry_readout")
_MEMORY_PHY, _LINK_PHY = "memory_phy", "link_phy"
#: The write law's note for a host-owned store's write transfer (``PlatformPhysics.store_transfer_note``).
TRANSFER_NOTE = ("the write transfer streams from the host-owned store over the host-chip link: priced there "
                 "(energy_host_link_mj, energy_host_mem_mj), refused by name without the link")


class BackingStoreError(ToolchainRefusal, ValueError):
    """A backing-store declaration stating both stores or none, an unknown field or no source, or one the chip's
    area components or a host attach row contradict."""


@dataclass(frozen=True)
class BackingStore:
    """Where the chip's backing store lives (``CHIP`` or ``HOST``), its source, and whether a run restated it."""

    store: str
    citation: str = ""
    derivation: str = ""
    note: str = ""
    #: A run's restatement (a declared departure from the chip description).
    overridden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {self.store: {"citation": self.citation, "derivation": self.derivation, "note": self.note},
                **({"overridden": True} if self.overridden else {})}


def store_from_declaration(data: Any, where: str, *, overridden: bool = False) -> BackingStore:
    """The declaration ``data`` judged: one store with its source; both, none, an unknown field or no source refuse."""
    if not isinstance(data, Mapping):
        raise BackingStoreError(f"{where}: {STORE_FIELD} is a mapping of one store ({list(STORES)}) to its source")
    if unknown := sorted(set(data) - _TOP_FIELDS):
        raise BackingStoreError(f"{where}: {STORE_FIELD} fields {unknown} are not among {sorted(_TOP_FIELDS)}")
    stated = [store for store in STORES if store in data]
    if len(stated) > 1:
        raise BackingStoreError(f"{where}: {STORE_FIELD} states both stores {stated}: a chip's backing store is its "
                                "own DRAM or the host's memory, never both")
    if not stated:
        raise BackingStoreError(f"{where}: {STORE_FIELD} states no store: declare {list(STORES)}")
    source = data[stated[0]]
    if not isinstance(source, Mapping) or set(source) - _SOURCE_FIELDS:
        raise BackingStoreError(f"{where}: {STORE_FIELD} {stated[0]!r} states its source as {sorted(_SOURCE_FIELDS)}, "
                                f"got {source!r}")
    text = {key: str(source.get(key) or "") for key in sorted(_SOURCE_FIELDS)}
    block = str(data.get("note") or "")
    if not any(text.values()) and not (overridden and block):
        raise BackingStoreError(f"{where}: {STORE_FIELD} {stated[0]!r} states no source (a citation, derivation or "
                                "note): where a store lives is a declaration with its source, never assumed")
    note = " ".join(part for part in (block, text["note"]) if part)
    return BackingStore(store=stated[0], citation=text["citation"], derivation=text["derivation"], note=note,
                        overridden=overridden or bool(data.get("overridden")))


def store_block(store: str, note: str, citation: str = "") -> Dict[str, Dict[str, Any]]:
    """A run's named block restating the store (``resolve.merge_override_blocks`` merges it): a declared departure."""
    return {STORE_FIELD: {store: {"citation": citation, "note": note}}}


def _kind(decl: Any) -> str:
    return load_component(decl.component).kind


def _kinds(components: Sequence[Any]) -> Dict[str, Tuple[Any, ...]]:
    out: Dict[str, Tuple[Any, ...]] = {}
    for decl in components:
        out[_kind(decl)] = out.get(_kind(decl), ()) + (decl,)
    return out


def judged_store(store: Optional[BackingStore], components: Sequence[Any], where: str) -> None:
    """The chip's area components agree with its store where it declares them: a memory PHY exactly when the store
    is the chip's, and a host-owned store over a link PHY to the host; refused by name otherwise."""
    if not components:
        return
    kinds = _kinds(components)
    memory = [d.component for d in kinds.get(_MEMORY_PHY, ())]
    if store is None and memory:
        raise BackingStoreError(f"{where}: the chip declares a memory PHY {memory} and no {STORE_FIELD}: a memory PHY "
                                "is the chip's own DRAM interface, so declare its store chip-owned (with its source)")
    if store is not None and store.store == CHIP and not memory:
        raise BackingStoreError(f"{where}: the store is chip-owned, and the chip's components declare no memory "
                                "PHY: a chip's own DRAM is reached through its memory PHY (declare it among "
                                "chip_components)")
    if store is not None and store.store == HOST:
        if memory:
            raise BackingStoreError(f"{where}: the store is host-owned, and the chip declares a memory PHY {memory}: "
                                    "its area would price a DRAM interface the chip does not use")
        if not any((d.link or {}).get(HOST) for d in kinds.get(_LINK_PHY, ())):
            raise BackingStoreError(f"{where}: the store is host-owned, and the chip's components declare no link PHY "
                                    "to the host: a host-owned store is reached over the host-chip link")


def restated_store(components: Tuple[Any, ...], declaration: Any, where: str) -> Tuple[BackingStore, Tuple[Any, ...]]:
    """A run's restatement: the store it declares (marked overridden) and the chip components that store keeps (a
    host-owned store drops the chip's memory PHYs)."""
    new = store_from_declaration(declaration, where, overridden=True)
    if new.store == HOST:
        components = tuple(d for d in components if _kind(d) != _MEMORY_PHY)
    return new, components


def attach_refusal(store: Optional[BackingStore], fed: Sequence[str]) -> Optional[str]:
    """Why chip channel rows fed over the host link (``fed``: non-host rows naming a host) do not stand, or None."""
    if store is None or store.store != CHIP or not fed:
        return None
    return (f"{sorted(fed)}: a chip channel fed over the host-chip link (a host attach row) on a chip whose "
            f"{STORE_FIELD} is chip-owned: its write port reads the chip's own DRAM, never the host link")


def prices_host_link(physics: Any) -> bool:
    """Whether ``physics`` declares the host-chip link or host memory (an edge host block)."""
    return physics is not None and any(physics.has(key) for key in LINK_ROWS)


def owner(physics: Any) -> Optional[str]:
    """The store the physics declares (``CHIP``, ``HOST``) or None."""
    store = getattr(physics, STORE_FIELD, None)
    return None if store is None else store.store


def host_owned(physics: Any) -> bool:
    return owner(physics) == HOST


def store_refusal(ctx: Any) -> Optional[str]:
    """Why a pricing view's store traffic cannot be priced, or None: beside the host-chip link a chip that states no
    store (the traffic would be priced on both stores or on neither), or a host-owned store without the link. ``ctx``
    states ``physics``, ``quantities`` and the write law's crossing (``write_link_bytes``: its ``store`` part)."""
    store, link, q = owner(ctx.physics), prices_host_link(ctx.physics), ctx.quantities
    if store == CHIP or (store is None and not link) or (store == HOST and link):
        return None
    crossing = getattr(ctx, "write_link_bytes", None)
    traffic = [kind for kind, there in (
        ("the pass carry", any((q.has(k) and q.get(k).value) or q.withheld_reason(k) for k in CARRY_QUANTITIES)),
        ("the write law's backing-store payload", bool(crossing and crossing.store))) if there]
    if not traffic:
        return None
    if store is None:
        return (f"{STORE_FIELD}: the physics prices the host-chip link and the chip states no backing store, so "
                f"{' and '.join(traffic)} would be priced on both stores or on neither; declare where it lives "
                f"({CHIP}: the chip's own DRAM through its memory PHY; {HOST}: the host's memory over the link)")
    return (f"{STORE_FIELD}: the chip's store is host-owned (the host's memory over the host-chip link), and the "
            f"physics declares no host-chip link ({', '.join(LINK_ROWS)}): {' and '.join(traffic)} cannot be priced")


def chip_store_table(physics: Any, table: Any) -> Any:
    """An energy table without the chip's own store rows of the carry where the store is host-owned."""
    return table if owner(physics) != HOST else tuple(row for row in table if row[2] not in CHIP_CARRY_LABELS)


def _ends(fn, *bands: Band) -> Tuple[float, float, float]:
    return tuple(fn(*(getattr(b, end) for b in bands)) for end in ("low", "nominal", "high"))  # type: ignore


def link_stage(physics: Any) -> Optional[Band]:
    """A host-owned store's per-byte time: the slower of its declared per-byte stages (the pipelined bound, as the
    link prices every crossing byte), None where it declares neither."""
    stages = [key for key in ("t_link_per_byte", "t_host_mem_per_byte") if physics.has(key)]
    if not stages:
        return None
    return Band(*_ends(max, *(physics.band(key) for key in stages)),
                f"the slower of {' and '.join(stages)} per byte (the host-owned store's stage)")


def store_byte_energy(physics: Any) -> Optional[Band]:
    """A host-owned store's energy per byte: the link plus host memory where declared, None where neither is."""
    rows = [key for key in ("e_link_per_byte", "e_host_mem_per_byte") if physics.has(key)]
    if not rows:
        return None
    return Band(*_ends(lambda *v: sum(v), *(physics.band(key) for key in rows)),
                " + ".join(rows) + " (the host-owned store's bytes)")


def slower_port(physics: Any, port: Band) -> Band:
    """The chip's per-byte write port on its store: the port itself, or on a host-owned store the slower of the port
    and the store's stage (the payload streams from host memory over the link into it)."""
    stage = link_stage(physics) if host_owned(physics) else None
    if stage is None:
        return port
    return Band(*_ends(max, port, stage), f"the slower of the chip's port [{port.basis}] and {stage.basis}")
