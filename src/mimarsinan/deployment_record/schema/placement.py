"""Softcore→hardcore placement, weight banks, and the tile floorplan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.schema.serde import (
    int_pair,
    optional,
    require_choice,
    strict_kwargs,
    tuple_of,
)

FLOORPLAN_DERIVATIONS = frozenset({"declared", "derived"})


@dataclass(frozen=True)
class SoftcorePlacementRecord:
    """A pure read of one soft-core placement onto a hard core."""

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
    bank_axon_range: Optional[Tuple[int, int]]
    bank_neuron_range: Optional[Tuple[int, int]]
    split_group_id: Optional[int]
    split_fragment_index: Optional[int]
    coalescing_group_id: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SoftcorePlacementRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["bank_axon_range"] = optional(int_pair, kwargs["bank_axon_range"])
        kwargs["bank_neuron_range"] = optional(int_pair, kwargs["bank_neuron_range"])
        return cls(**kwargs)


@dataclass(frozen=True)
class BankRecord:
    """One weight bank and its sharing degree across placements."""

    bank_id: int
    rows: int
    cols: int
    params: int
    placement_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BankRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class FloorplanRecord:
    """The chip mesh geometry the cores were laid out on."""

    mesh_width: int
    mesh_height: int
    cores_per_tile: int
    derivation: str

    def __post_init__(self) -> None:
        require_choice(
            "FloorplanRecord", "derivation", self.derivation, FLOORPLAN_DERIVATIONS
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FloorplanRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class TileRecord:
    """One NoC-free hard-core group and its mesh position."""

    tile_index: int
    x: int
    y: int
    core_indices: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TileRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["core_indices"] = tuple(int(c) for c in kwargs["core_indices"])
        return cls(**kwargs)


@dataclass(frozen=True)
class PlacementRecord:
    """The placement fragment: softcores, banks, floorplan (None without SANA-FE), tiles."""

    softcores: Tuple[SoftcorePlacementRecord, ...]
    banks: Tuple[BankRecord, ...]
    floorplan: Optional[FloorplanRecord]
    tiles: Tuple[TileRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlacementRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["softcores"] = tuple_of(
            SoftcorePlacementRecord.from_dict, kwargs["softcores"]
        )
        kwargs["banks"] = tuple_of(BankRecord.from_dict, kwargs["banks"])
        kwargs["floorplan"] = optional(FloorplanRecord.from_dict, kwargs["floorplan"])
        kwargs["tiles"] = tuple_of(TileRecord.from_dict, kwargs["tiles"])
        return cls(**kwargs)
