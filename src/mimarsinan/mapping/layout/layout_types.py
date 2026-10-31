from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, List, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.packing.canonical import declared_core_roles as _declared_core_roles

from mimarsinan.mapping.packing.infeasibility_proofs import VERDICT_FEASIBLE
from mimarsinan.mapping.packing.placement_cost import note_resident_bank
from mimarsinan.mapping.platform.core_residency import BASIS_VALUES


class CoreRoleGeometryError(ToolchainRefusal, ValueError):
    """An operand core type and a static core type of one geometry: the planner's pool cannot tell them apart."""


@dataclass(frozen=True)
class LayoutSoftCoreSpec:
    """Shape-only representation of a deployable soft core."""

    input_count: int
    output_count: int

    residency_class_id: int = 0
    # Which rule produced residency_class_id. Ids from different bases are not comparable, so a
    # mapping that mixes them is a defect rather than a merge decision; see core_residency.
    residency_basis: str = BASIS_VALUES
    latency_tag: Optional[int] = None
    segment_id: Optional[int] = None

    # The shared weight matrix this core reads (``None`` = it owns its weights).
    # Bank-clustered scheduling and every bank-composition question are asked of
    # THIS field; without it the mapper's sharing map died at the layout boundary.
    bank_id: Optional[int] = None
    # Real IR-layer identity: which source perceptron emitted this core (``None``
    # for relays/psum cores that belong to no layer). Consumers key layer rollups
    # on it instead of splitting the core NAME on separator conventions.
    perceptron_index: Optional[int] = None

    name: Optional[str] = None
    # N2: the role of the soft core (``packing.canonical.OPERAND_ROLE`` for a dynamic operand bank's instance).
    core_role: Optional[str] = None
    # [W04] The IR core's coalescing group (``None`` = none): the packer never neuron-splits a group's core and the
    # planner keeps the group in one pass, on this plane as on the builder's.
    coalescing_group_id: Optional[int] = None
    # [U-CG] The IR core's role in its group (``master``: it holds its whole fan-in on fused cores). A group whose
    # members are all ``master`` shares no partial sum, so the planner may split it between members
    # (``schedule_budget.pass_atoms``).
    coalescing_role: Optional[str] = None
    # [F2] Of ``input_count``, the axons that hold a fused add's operand rows (W08: placed, zero crossbar weights, the
    # lane's port): packed as axons, counted as no programmed cell (``CoreOccupancy``); 0 off the builder's lane cores.
    operand_axons: int = 0

    def get_input_count(self) -> int:
        return int(self.input_count)

    def get_output_count(self) -> int:
        return int(self.output_count)

    @property
    def weight_bank_id(self) -> Optional[int]:
        """[W04] The bank under the name the packer reads on every soft core: a bank-backed spec is bank-affine."""
        return self.bank_id

    @property
    def area(self) -> int:
        return int(self.input_count) * int(self.output_count)


@dataclass
class LayoutHardCoreType:
    max_axons: int
    max_neurons: int
    count: int
    # N2: the soft-core roles this type's cores accept (None: every role), ``packing.canonical.declared_core_roles``.
    accepted_roles: Optional[frozenset] = None


@dataclass
class LayoutHardCoreInstance:
    """Shape-only approximation of a hardware core instance."""

    axons_per_core: int
    neurons_per_core: int

    available_axons: int = field(init=False)
    available_neurons: int = field(init=False)

    residency_class_id: Optional[int] = None
    latency_tag: Optional[int] = None
    accepted_roles: Optional[frozenset] = None

    unusable_space: int = 0
    used_area: int = 0
    softcore_count: int = 0
    #: [W04] Per placed spec (bank, axons, neurons, axon offset, neuron offset), as ``HardCore`` records placements.
    regions: List[Tuple] = field(default_factory=list, compare=False, repr=False)
    #: A fused core's member axons in order (``HardCore.fused_component_axons``); None for one physical core.
    component_axons: Optional[Tuple[int, ...]] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        self.available_axons = int(self.axons_per_core)
        self.available_neurons = int(self.neurons_per_core)

    def get_input_count(self) -> int:
        return int(self.axons_per_core)

    def get_output_count(self) -> int:
        return int(self.neurons_per_core)

    @property
    def capacity(self) -> int:
        return int(self.axons_per_core) * int(self.neurons_per_core)

    def add_softcore(self, softcore: LayoutSoftCoreSpec) -> None:
        in_c = int(softcore.get_input_count())
        out_c = int(softcore.get_output_count())

        if in_c > self.available_axons or out_c > self.available_neurons:
            raise ValueError("Softcore does not fit into this hardcore instance")

        axon_offset = self.axons_per_core - self.available_axons
        neuron_offset = self.neurons_per_core - self.available_neurons

        # Packing-inefficiency formula must stay identical to HardCore.add_softcore().
        self.unusable_space += (neuron_offset * in_c) + (axon_offset * out_c)
        self.regions.append((getattr(softcore, "bank_id", None), in_c, out_c, axon_offset, neuron_offset))

        self.available_axons -= in_c
        self.available_neurons -= out_c

        self.used_area += int(in_c) * int(out_c)
        self.softcore_count += 1

        if self.residency_class_id is None:
            self.residency_class_id = int(softcore.residency_class_id)

        if self.latency_tag is None:
            self.latency_tag = softcore.latency_tag
        note_resident_bank(self, softcore)  # [W04] as HardCore.add_softcore: later specs of the bank prefer this core


@dataclass(frozen=True)
class LayoutCoreSnapshot:
    """Per-used-core summary captured after packing."""

    axons_per_core: int
    neurons_per_core: int
    used_axons: int
    used_neurons: int
    used_area: int
    softcore_count: int
    #: A fused core's member axons in order; None for one physical core.
    component_axons: Optional[Tuple[int, ...]] = field(default=None, compare=False)

    @property
    def capacity(self) -> int:
        return self.axons_per_core * self.neurons_per_core

    @property
    def tiles(self) -> int:
        """The physical cores this used core occupies: a fused core counts its members."""
        return len(self.component_axons) if self.component_axons else 1

    @property
    def wasted_axons(self) -> int:
        return self.axons_per_core - self.used_axons

    @property
    def wasted_neurons(self) -> int:
        return self.neurons_per_core - self.used_neurons


def core_types_from_declaration(cores) -> "list[LayoutHardCoreType]":
    """The declared chip's hardcore types, as every layout consumer reads them (with their N2 core roles)."""
    from_roles = _declared_core_roles(list(cores))
    return [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
            accepted_roles=roles,
        )
        for ct, roles in zip(cores, from_roles)
    ]


def core_pool_types(core_types) -> "tuple[list, Callable[[tuple], LayoutHardCoreInstance]]":
    """(the pool's ``(geometry, count)`` list, the instance factory): an instance carries its type's roles; two
    types of one geometry and different roles cannot share a geometry-keyed pool and refuse by name."""
    roles: dict = {}
    for t in core_types:
        key = (int(t.max_axons), int(t.max_neurons))
        if roles.setdefault(key, t.accepted_roles) != t.accepted_roles:
            raise CoreRoleGeometryError(
                f"two core types of geometry {key[0]} x {key[1]} accept different roles; the shape-only planner "
                "pools cores by geometry, so an operand core type must differ in geometry from a static one")
    return ([((int(t.max_axons), int(t.max_neurons)), int(t.count)) for t in core_types],
            lambda key: LayoutHardCoreInstance(axons_per_core=key[0], neurons_per_core=key[1],
                                               accepted_roles=roles.get(key)))


@dataclass(frozen=True)
class LayoutPackingResult:
    feasible: bool
    cores_used: int

    total_capacity: int
    used_area: int

    unused_area_total: int
    avg_unused_area_per_core: float

    unusable_space_total: int = 0
    avg_unusable_space_per_core: float = 0.0

    error: Optional[str] = None
    used_core_softcore_counts: Optional[Tuple[int, ...]] = None
    used_core_snapshots: Optional[Tuple[LayoutCoreSnapshot, ...]] = None

    coalesced_fragment_count: int = 0
    split_fragment_count: int = 0

    coalescing_group_sizes: Optional[Tuple[int, ...]] = None

    split_counts_per_sc: Optional[Tuple[int, ...]] = None

    # (origin softcore index, hardcore index) per placed unit, present only
    # when the pack was asked to collect placements (the NoC estimator input);
    # split/coalescing fragments repeat their origin index.
    placements: Optional[Tuple[Tuple[int, int], ...]] = None
    # [W04] With placements: per used core, its placed regions (``LayoutHardCoreInstance.regions``).
    core_regions: Optional[Tuple[FrozenSet[Tuple], ...]] = None
    # [E-3] Packed by builder segment (``pack_layout(runs=...)``): per segment in program order, its cores, which are
    # consecutive in ``used_core_snapshots``; None for one co-pack.
    segment_cores: Optional[Tuple[int, ...]] = None

    # [TS4] Why the pack ended this way, additive beside ``feasible``: a refusal
    # is either PROVEN (``infeasibility_proofs``, scoped to this one pack — a
    # scheduled program reuses cores and is a different question) or the greedy
    # engine's, which some other placement order may still take.
    verdict: str = VERDICT_FEASIBLE


