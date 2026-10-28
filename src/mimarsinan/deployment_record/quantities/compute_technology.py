"""Per-core-type compute technology (E-6, N2 follow-up F5): the MAC events of each core type at its own e_mac.

A core type declares ``compute_physics``, a shipped physics profile whose compute row (``COMPUTE_TECHNOLOGY_KEYS``:
``e_mac``) prices that core type's MAC events; a core type naming none takes the chip's. The split reads a plane's
passes (``write_census.ProgramPasses``, one rule on every plane): per physical tile its committed cells, static and
operand (``core_cells`` and ``producer_core_cells``), grouped by the tile's core geometry, which is the value rule's MAC
events per core type (``value_domain``: MAC events = committed cells). A chip whose core types name no compute physics
states no split (``compute_split`` is None), so its energy is priced byte for byte as before.

**Declared deployment model.** Only a value-domain chip has MAC events per tile; a spiking chip's synaptic-event
census states no per-core-type split, so per-core-type compute physics on it refuses the energy by name. Two core
types of one geometry are told apart by geometry only (as the write census pools them), so they must name one compute
physics or refuse by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import compute_physics_profile
from mimarsinan.deployment_record.quantities.value_domain import names_value_domain

#: The resolved surface's key holding each named compute physics as sealed (``PlatformPhysics.to_dict``).
COMPUTE_PHYSICS_RESOLVED = "compute_physics_resolved"
CoreType = Tuple[int, int]


@dataclass(frozen=True)
class ComputeShare:
    """One core geometry's MAC events and the compute physics they are priced at (None: the chip's)."""

    core_type: CoreType
    events: int
    compute_physics: Optional[str]
    #: The copy the run sealed of the named physics (None: unsealed, resolved by name at pricing, or the chip's).
    physics: Optional[PlatformPhysics] = None

    @property
    def label(self) -> str:
        return f"core type {self.core_type[0]} x {self.core_type[1]} ({self.compute_physics or 'the chip physics'})"


@dataclass(frozen=True)
class ComputeSplit:
    """A plane's MAC events per core type, or why the plane cannot state them (``unavailable``, the refusal's
    reason)."""

    shares: Tuple[ComputeShare, ...] = ()
    unavailable: Optional[str] = None

    @property
    def events(self) -> int:
        return sum(share.events for share in self.shares)


def _declared(platform: Mapping[str, Any]) -> Tuple[Dict[CoreType, Optional[str]], Optional[str]]:
    """{geometry: its compute physics name or None}, or the refusal of two names on one geometry."""
    named: Dict[CoreType, set] = {}
    for ct in platform.get("cores") or ():
        if "max_axons" not in ct or "max_neurons" not in ct:
            continue
        kind = (int(ct["max_axons"]), int(ct["max_neurons"]))
        named.setdefault(kind, set()).add(ct.get("compute_physics") or None)
    clash = sorted(f"{k[0]} x {k[1]}" for k, names in named.items() if len(names) > 1)
    if clash:
        return {}, (f"core types of one geometry ({', '.join(clash)}) name different compute physics, which the "
                    "plane's passes (typed by geometry) cannot tell apart")
    return {k: next(iter(names)) for k, names in named.items()}, None


def compute_split(platform: Mapping[str, Any], passes: Optional[Any]) -> Optional[ComputeSplit]:
    """The plane's MAC events per core type with each type's compute physics; None when no core type names one (the
    chip's one e_mac prices every MAC, unchanged)."""
    cores = platform.get("cores") or ()
    if not any(ct.get("compute_physics") for ct in cores):
        return None
    if not names_value_domain(platform):
        return ComputeSplit(unavailable=(
            "per-core-type compute physics prices the value domain's MAC events (committed cells) per core type; a "
            "spiking chip's synaptic_events census states no per-core-type split"))
    declared, clash = _declared(platform)
    if clash is not None:
        return ComputeSplit(unavailable=clash)
    if passes is None:
        return ComputeSplit(unavailable="the plane states no pass structure, so it has no per-core-type MAC split")
    events: Dict[CoreType, int] = {kind: 0 for kind in declared}
    only = next(iter(declared)) if len(declared) == 1 else None
    for row in getattr(passes, "rows", ()):
        types = row.core_types
        if types is None and only is None:
            return ComputeSplit(unavailable=("the plane's passes do not type their cores, so its MAC events are not "
                                             "attributed to the declared core types"))
        producer = row.producer_core_cells or (0,) * len(row.core_cells)
        for index, cells in enumerate(row.core_cells):
            kind = only if types is None else (int(types[index][0]), int(types[index][1]))
            if kind is None or kind not in events:
                tile = "an untyped tile" if kind is None else f"a tile of {kind[0]} x {kind[1]}"
                return ComputeSplit(unavailable=f"{tile} is no declared core type, so its MAC events have no compute "
                                                "physics")
            events[kind] += int(cells) + int(producer[index])
    sealed = platform.get(COMPUTE_PHYSICS_RESOLVED) or {}
    shares = []
    for kind in sorted(events):
        name = declared[kind]
        physics = PlatformPhysics.from_dict(sealed[name]) if name is not None and name in sealed else None
        shares.append(ComputeShare(core_type=kind, events=events[kind], compute_physics=name, physics=physics))
    return ComputeSplit(shares=tuple(shares))


def share_physics(share: ComputeShare, chip: PlatformPhysics) -> PlatformPhysics:
    """The physics a share's MAC events are priced at: the chip's, else the sealed copy, else the shipped profile."""
    if share.compute_physics is None:
        return chip
    if share.physics is not None:
        return share.physics
    return compute_physics_profile(share.compute_physics)


def split_digest(split: Optional[ComputeSplit]) -> Optional[Sequence[Tuple[CoreType, int, Optional[str]]]]:
    """A comparable reading of a split (geometry, events, physics name) for records and tests."""
    if split is None:
        return None
    return tuple((share.core_type, share.events, share.compute_physics) for share in split.shares)
