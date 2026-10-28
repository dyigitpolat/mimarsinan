"""Per-core-type write technology (N2b, owner decision 4): what each core type's cells are written under.

A core type declares ``write_physics``, a shipped physics profile whose write rows (``WRITE_TECHNOLOGY_KEYS``) price
its cells, beside its own ``write_granularity`` and ``write_units_in_parallel``. The write law then prices each write
technology's share of a plane's period census by G3's one law and joins the shares (``cost/absolute/writes.py``). A
chip whose core types share one write technology states none (``core_technologies`` is None), so its census is
priced as one, byte for byte.

**Declared deployment model.** The census places cores by geometry (``write_census.core_capacity_of`` pools the types
of one geometry), so two core types of one geometry are one technology or refuse by name. The technologies share the
chip's write ports: a pass's write walls add across technologies before the exposure rule, and so do the write
energies; the transfer channel (``e_dma_per_byte``, ``t_program_per_byte``) stays the chip's. The lifetime is the
shortest technology's and the floor verdict the worst.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.write_census import CoreType, PassRow, ProgramPasses, WriteCensus
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

#: The resolved surface's key holding each named write physics as sealed (``PlatformPhysics.to_dict``).
WRITE_PHYSICS_RESOLVED = "write_physics_resolved"


class WriteTechnologyError(ToolchainRefusal, ValueError):
    """Write technologies the census cannot tell apart (two on one geometry), or operand core types of two
    technologies under the stationary choice's price."""


def _kind(kind: CoreType) -> str:
    return f"{kind[0]} x {kind[1]}"


@dataclass(frozen=True)
class CoreTechnology:
    """One core geometry's write technology: its write physics (None: the chip's) and write declarations."""

    core_type: CoreType
    count: int
    #: The named write physics (None: the chip's) and the copy the run sealed of it (``write_physics_resolved``;
    #: None where unsealed: the pricer resolves the name, ``platform_physics.resolve.write_physics_profile``).
    write_physics: Optional[str]
    physics: Optional[PlatformPhysics]
    write_granularity: Optional[str]
    write_units_in_parallel: int
    operand: bool = False

    @property
    def key(self) -> Tuple[Any, ...]:
        return (self.write_physics, self.write_granularity, self.write_units_in_parallel)

    @property
    def label(self) -> str:
        return f"core type {_kind(self.core_type)} ({self.write_physics or 'the chip physics'})"


def core_technologies(platform: Mapping[str, Any]) -> Optional[Tuple[CoreTechnology, ...]]:
    """Per declared geometry, its write technology; None where every core type writes by one technology without a
    write physics of its own (the chip's one law). Refuses by name two technologies on one geometry (a write physics
    no profile ships refuses where it is resolved: at the platform's resolution, or at pricing)."""
    cores = [dict(ct) for ct in platform.get("cores") or ()]
    keys = [(ct.get("write_physics"), ct.get("write_granularity"), int(ct.get("write_units_in_parallel", 1)))
            for ct in cores]
    if not any(key[0] for key in keys) and len(set(keys)) <= 1:
        return None
    if any("max_axons" not in ct or "max_neurons" not in ct for ct in cores):
        return None  # no core type is known to hold any tile: the period census refuses by name, every write term too
    sealed = platform.get(WRITE_PHYSICS_RESOLVED)
    by_kind: Dict[CoreType, CoreTechnology] = {}
    for ct, (name, granularity, units) in zip(cores, keys):
        kind = (int(ct["max_axons"]), int(ct["max_neurons"]))
        copy = None if name is None or not sealed or name not in sealed else PlatformPhysics.from_dict(sealed[name])
        tech = CoreTechnology(kind, int(ct.get("count", 1)), name, copy, granularity, units,
                              ct.get("operand") is not None)
        prior = by_kind.get(kind)
        if prior is not None and prior.key != tech.key:
            raise WriteTechnologyError(
                f"two core types of geometry {_kind(kind)} declare different write technologies ({prior.key} and "
                f"{tech.key}); the period census pools cores by geometry, so a write technology must differ in "
                "geometry")
        by_kind[kind] = tech if prior is None else replace(prior, count=prior.count + tech.count,
                                                            operand=prior.operand or tech.operand)
    return tuple(by_kind[kind] for kind in sorted(by_kind))


def technology_groups(technologies: Sequence[CoreTechnology]) -> Tuple[Tuple[CoreTechnology, ...], ...]:
    """The geometries that share one write technology, grouped in geometry order."""
    groups: Dict[Tuple[Any, ...], List[CoreTechnology]] = {}
    for tech in technologies:
        groups.setdefault(tech.key, []).append(tech)
    return tuple(tuple(group) for group in groups.values())


def technology_policy(policy: Any, group: Sequence[CoreTechnology]) -> Any:
    """The chip's write policy at one technology's write declarations and cores (``write_law.WritePolicy``)."""
    return replace(policy, write_granularity=group[0].write_granularity,
                   write_units_in_parallel=group[0].write_units_in_parallel, mixed_core_writes=None,
                   core_capacity=tuple(sorted((t.core_type, t.count) for t in group)),
                   physical_cores=sum(t.count for t in group), core_technologies=None)


def operand_technology(policy: Any) -> Optional[CoreTechnology]:
    """The write technology of the operand core types, which hold every operand region; None where the chip states
    one technology. Operand core types of two technologies refuse by name: no one physics prices their regions."""
    technologies = getattr(policy, "core_technologies", None)
    if not technologies:
        return None
    operand = [t for t in technologies if t.operand]
    if len({t.key for t in operand}) > 1:
        raise WriteTechnologyError(
            "operand core types declare different write technologies (" + "; ".join(t.label for t in operand) +
            "): the stationary choice's price places operand regions on no one of them")
    return operand[0] if operand else None


@dataclass(frozen=True)
class TechnologyWrites:
    """One write technology's share of a plane's period census: the census restricted to the cores of its core
    types, with what ``price_write_terms`` reads beside it."""

    group: Tuple[CoreTechnology, ...]
    policy: Any
    census: PassPeriodCensus
    written_extents: Optional[Tuple[Tuple[Tuple[int, int], ...], ...]]
    most_written: Optional[int]
    producer_slots: Optional[Tuple[Tuple[int, int], ...]]
    pass_core_types: Optional[Tuple[Tuple[Tuple[CoreType, int], ...], ...]]

    @property
    def technology(self) -> CoreTechnology:
        return self.group[0]

    @property
    def label(self) -> str:
        return ", ".join(t.label for t in self.group)


@dataclass(frozen=True)
class TechnologySplit:
    """A plane's census split by write technology, or why it cannot be."""

    parts: Tuple[TechnologyWrites, ...] = ()
    refused: Optional[str] = None


def _producer_share(row: PassRow, types: Sequence[CoreType], kinds: set) -> Tuple[int, Optional[str]]:
    if not row.producer_cells:
        return 0, None
    if row.producer_core_cells is not None:
        return sum(c for c, kind in zip(row.producer_core_cells, types) if kind in kinds), None
    holders = {types[i] for i in row.producer_cores}
    if holders and holders <= kinds:
        return row.producer_cells, None
    if holders and not holders & kinds:
        return 0, None
    return 0, ("a pass writes operand regions on cores of several write technologies (or on no stated core) and "
               "states no payload cells per core, so their writes cannot be priced per write technology")


def split_by_technology(passes: ProgramPasses, written: WriteCensus, policy: Any, *,
                        weight_bits: int) -> TechnologySplit:
    """The census of each write technology: the full census (``period_census``, one rule over the chip) restricted
    to the cores of that technology's core types, pass by pass."""
    groups = technology_groups(policy.core_technologies)
    one = policy.core_technologies[0].core_type  # an untyped row on a chip of one write technology: all its cores
    if any(row.core_types is None for row in passes.rows) and len(groups) > 1:
        return TechnologySplit(refused=("the plane states no core type of its cores on a chip of several write "
                                        "technologies, so its writes cannot be priced per core type"))
    if len(written.written_cores) != len(passes.rows):
        return TechnologySplit(refused="the census states no rewritten cores per pass, so it cannot be split")
    typed = [row.core_types or (one,) * len(row.core_cells) for row in passes.rows]
    parts: List[TechnologyWrites] = []
    for group in groups:
        kinds = {t.core_type for t in group}
        rows: List[PassCensus] = []
        extents: List[Tuple[Tuple[int, int], ...]] = []
        for position, (row, types, full) in enumerate(zip(passes.rows, typed, written.census.passes)):
            mine = [i for i, kind in enumerate(types) if kind in kinds]
            cores = written.written_cores[position]
            rewritten = [i for i in cores if types[i] in kinds]
            producer, refused = _producer_share(row, types, kinds)
            if refused is not None:
                return TechnologySplit(refused=refused)
            rows.append(PassCensus(
                position=full.position, segment_index=full.segment_index, resident=full.resident, cores=len(mine),
                resident_cores=len(mine) - len(rewritten),
                written_cells_backing_store=sum(row.core_cells[i] for i in rewritten),
                written_cells_producer=producer,
                payload_bytes=sum(row.core_bytes(i, weight_bits) for i in rewritten),  # cells and sign register
                compute_steps=full.compute_steps, operand_only_cores=len(set(rewritten) & row.operand_only_cores)))
            if written.written_extents is not None:
                extents.append(tuple(extent for i, extent in zip(cores, written.written_extents[position])
                                     if types[i] in kinds))
        slots = written.slot_writes
        typed_slots = written.type_producer_slots
        parts.append(TechnologyWrites(
            group=group, policy=technology_policy(policy, group),
            census=PassPeriodCensus(passes=tuple(rows), label=f"{written.census.label}; the cores of "
                                                              f"{', '.join(t.label for t in group)}"),
            written_extents=None if written.written_extents is None else tuple(extents),
            most_written=None if slots is None else max(
                (w for slot in slots if slot.core_type in kinds for w in slot.writes), default=0),
            producer_slots=None if typed_slots is None else tuple(sorted(
                {pair for kind, pairs in typed_slots if kind in kinds for pair in pairs})),
            pass_core_types=None if written.pass_core_types is None else tuple(
                tuple((kind, n) for kind, n in types_of if kind in kinds) for types_of in written.pass_core_types),
        ))
    return TechnologySplit(parts=tuple(parts))


def shadow_room_refusal(capacity: Optional[Sequence[Tuple[CoreType, int]]], fraction: float,
                        pass_types: Optional[Sequence[Sequence[Tuple[CoreType, int]]]],
                        pass_cores: Sequence[int]) -> Optional[str]:
    """Why overlapped writes cannot hide behind compute, or None. While a pass computes, the next pass's
    tiles are written into the shadow share s of each core type's declared cores, so every pass computes
    on at most floor(count_t x (1 - s)) cores of each type t (s read exactly as declared)."""
    declared = dict(capacity or ())
    if not declared:
        return "overlapped writes keep a shadow share of each core type, and the chip declares no core types"
    if pass_types is None and len(declared) > 1:
        return (f"overlapped writes keep a shadow share of each core type, and the plane states no core type of "
                f"its cores on a chip of {len(declared)} core types: the shadow room is undecidable")
    if pass_types is None:
        pass_types = [((next(iter(declared)), cores),) for cores in pass_cores]
    for kind, count in sorted(declared.items()):
        room = int(count * (1 - Fraction(repr(float(fraction)))))
        widest = max((dict(types).get(kind, 0) for types in pass_types), default=0)
        if widest > room:
            return (f"overlapped writes keep shadow_capacity_fraction {fraction} of each core type's declared "
                    f"cores: a pass computes on {widest} cores of type {_kind(kind)}, more than the {room} of its "
                    f"{count} left to compute (the program must be planned at count_t x (1 - s))")
    return None
