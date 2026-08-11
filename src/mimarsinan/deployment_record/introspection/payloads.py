"""The introspection payloads: frozen, versioned answers about a deployment.

An optimizer (LLM-driven or classical) may only ask questions the artifact can
answer, in shapes it can rely on. Every payload therefore carries its own
``payload``/``payload_version`` and refuses to load under another name or
version — evolution is a version bump plus an explicit migration, exactly the
``CostRecord.from_dict`` discipline the record schema follows. Consumers depend
on these types and NOTHING else; no consumer reaches into ``mapping`` internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.schema.serde import strict_kwargs, tuple_of
from mimarsinan.deployment_record.schema.utilization import LayoutStatsRecord

INTROSPECTION_FORMAT_VERSION = 1


@dataclass(frozen=True)
class IntrospectionRow:
    """One JSON-safe row of a payload table."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntrospectionRow":
        kwargs = strict_kwargs(cls, data)
        # A row's only sequence fields are tuples (rows are frozen and hashable),
        # and JSON has no tuple — so a decoded list is one, always.
        return cls(**{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in kwargs.items()
        })


@dataclass(frozen=True)
class IntrospectionPayload:
    """One named, versioned answer; ``ROWS`` names the fields holding row tables."""

    NAME: ClassVar[str] = ""
    VERSION: ClassVar[int] = 1
    ROWS: ClassVar[Mapping[str, type]] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload": self.NAME,
            "payload_version": self.VERSION,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntrospectionPayload":
        payload = dict(data)
        name = payload.pop("payload", None)
        version = payload.pop("payload_version", None)
        if name != cls.NAME:
            raise ValueError(
                f"{cls.__name__} cannot load payload {name!r}; it serves "
                f"{cls.NAME!r}"
            )
        if version != cls.VERSION:
            raise ValueError(
                f"payload {cls.NAME!r} is version {cls.VERSION}, got {version!r} "
                f"— migrate explicitly, never tolerate silently"
            )
        kwargs = strict_kwargs(cls, payload)
        for field_name, row_type in cls.ROWS.items():
            kwargs[field_name] = tuple_of(row_type.from_dict, kwargs[field_name])
        return cls(**kwargs)


@dataclass(frozen=True)
class SoftcoreRow(IntrospectionRow):
    """One deployable soft core: its shape and both of its identities."""

    index: int
    name: Optional[str]
    input_count: int
    output_count: int
    area: int
    residency_class_id: Optional[int]
    residency_basis: Optional[str]
    latency_tag: Optional[int]
    segment_id: Optional[int]
    bank_id: Optional[int]
    perceptron_index: Optional[int]


@dataclass(frozen=True)
class SoftcoresPayload(IntrospectionPayload):
    NAME: ClassVar[str] = "softcores"
    VERSION: ClassVar[int] = 1
    ROWS: ClassVar[Mapping[str, type]] = {"softcores": SoftcoreRow}

    softcores: Tuple[SoftcoreRow, ...] = ()

    @property
    def count(self) -> int:
        return len(self.softcores)


@dataclass(frozen=True)
class LayerRollupRow(IntrospectionRow):
    """One source layer's cores, keyed by REAL identity, not by a name convention."""

    perceptron_index: Optional[int]
    layer: str
    softcore_count: int
    total_area: int
    max_input_count: int
    max_output_count: int
    residency_class_count: int
    latency_tag_count: int
    segment_count: int
    bank_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class LayerRollupPayload(IntrospectionPayload):
    NAME: ClassVar[str] = "layer_rollup"
    VERSION: ClassVar[int] = 1
    ROWS: ClassVar[Mapping[str, type]] = {"layers": LayerRollupRow}

    layers: Tuple[LayerRollupRow, ...] = ()


@dataclass(frozen=True)
class BankRow(IntrospectionRow):
    """One shared weight bank and the degree to which it is shared."""

    bank_id: int
    softcore_count: int
    perceptron_index: Optional[int]
    max_input_count: Optional[int]
    max_output_count: Optional[int]
    rows: Optional[int] = None
    cols: Optional[int] = None
    params: Optional[int] = None
    segment_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class BankCompositionPayload(IntrospectionPayload):
    NAME: ClassVar[str] = "bank_composition"
    VERSION: ClassVar[int] = 1
    ROWS: ClassVar[Mapping[str, type]] = {"banks": BankRow}

    banks: Tuple[BankRow, ...] = ()
    unbanked_softcore_count: int = 0


@dataclass(frozen=True)
class PlacementRow(IntrospectionRow):
    """One soft core placed on one physical core, in one pass of one segment."""

    ir_node_id: int
    segment_index: int
    pass_index: int
    hard_core_index: int
    axon_offset: int
    neuron_offset: int
    axons: int
    neurons: int
    perceptron_index: Optional[int]
    weight_bank_id: Optional[int]


@dataclass(frozen=True)
class PlacementPayload(IntrospectionPayload):
    NAME: ClassVar[str] = "placement"
    VERSION: ClassVar[int] = 1
    ROWS: ClassVar[Mapping[str, type]] = {"placements": PlacementRow}

    placements: Tuple[PlacementRow, ...] = ()


@dataclass(frozen=True)
class SegmentPassRow(IntrospectionRow):
    """One neural segment's share of the program."""

    segment_index: int
    passes: Optional[int]
    softcore_count: Optional[int]
    programming: Optional[str] = None


@dataclass(frozen=True)
class SchedulePayload(IntrospectionPayload):
    """The pass structure — under the policy the platform declares."""

    NAME: ClassVar[str] = "schedule"
    VERSION: ClassVar[int] = 1
    ROWS: ClassVar[Mapping[str, type]] = {"segments": SegmentPassRow}

    schedule_policy: str = "pool"
    max_schedule_passes: Optional[int] = None
    pass_count: int = 0
    sync_count: int = 0
    max_cores_per_pass: Optional[int] = None
    segments: Tuple[SegmentPassRow, ...] = ()
    reprogram_passes: Optional[int] = None
    reuse_passes: Optional[int] = None
    compute_op_count: Optional[int] = None


@dataclass(frozen=True)
class CapabilitiesPayload(IntrospectionPayload):
    """Every capability bit the platform declares — the complete set, or none."""

    NAME: ClassVar[str] = "capabilities"
    VERSION: ClassVar[int] = 1

    bits: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilitiesPayload":
        payload = super().from_dict(data)
        assert isinstance(payload, cls)
        return cls(bits=dict(payload.bits))


@dataclass(frozen=True)
class LayoutStatsPayload(IntrospectionPayload):
    """The full layout-verification census, typed by the record's own mirror."""

    NAME: ClassVar[str] = "layout_stats"
    VERSION: ClassVar[int] = 1

    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def typed(self) -> LayoutStatsRecord:
        """The stats as the record's typed mirror (drift-guarded by its own tests)."""
        return LayoutStatsRecord.from_dict(self.stats)
