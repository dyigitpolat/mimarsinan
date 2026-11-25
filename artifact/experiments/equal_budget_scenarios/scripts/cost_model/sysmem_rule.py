""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Tuple

COMPONENTS = ("store", "buffer", "working")
KIND = {"store": "weights", "buffer": "activations", "working": "activations"}
BUDGETS = ("chip", "system")
BITS_PER_MB = 8_000_000


class SystemMemoryRefusal(ValueError):
    ""


@dataclass(frozen=True)
class Placement:
    name: str
    on_chip: bool
    admits: Tuple[str, ...]
    density_mm2_per_bit: Optional[float]
    density_source: str
    energy_row: Optional[str]
    energy_source: str
    refusals: Mapping[str, str] = field(default_factory=dict)
    excluded: Tuple[str, ...] = ()
    excluded_reason: str = ""


@dataclass(frozen=True)
class Table:
    ""
    name: str
    placements: Mapping[str, Placement]
    conventions: Tuple[str, ...] = ("runtime",)
    refused_conventions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Setting:
    ""
    name: str
    store_convention: str = "runtime"
    store: str = "dram"
    buffer: str = "dram"
    working: str = "chip"

    def placement_of(self, component: str) -> str:
        return {"store": self.store, "buffer": self.buffer, "working": self.working}[component]

    def check(self, table: Table) -> None:
        if self.store_convention not in table.conventions:
            why = table.refused_conventions.get(self.store_convention, "not a convention the table declares")
            raise SystemMemoryRefusal(f"{self.name}: store convention {self.store_convention!r} refused by name: {why}")
        for c in COMPONENTS:
            p = self.placement_of(c)
            if p not in table.placements:
                raise SystemMemoryRefusal(f"{self.name}: {c} placement {p!r} is not a declared placement "
                                          f"({sorted(table.placements)})")
            row = table.placements[p]
            if c not in row.admits:
                why = row.refusals.get(c, f"the {p} placement does not admit the {c}")
                raise SystemMemoryRefusal(f"{self.name}: the {c} placed at {p!r} is refused by name: {why}")


@dataclass(frozen=True)
class Census:
    ""
    resident: bool
    tiles_total: int
    tiles_rewritten: int
    tile_bits: int
    value_bits: int
    largest_cut: Optional[Callable[[], int]]
    peak_live: int


def capacity_bits(component: str, census: Census, K: int, convention: str) -> int:
    if component == "store":
        if convention != "runtime":
            raise SystemMemoryRefusal(f"store convention {convention!r}: no capacity rule")
        return int(census.tiles_rewritten) * int(census.tile_bits)
    if component == "buffer":
        if census.resident:
            return 0
        if census.largest_cut is None:
            raise SystemMemoryRefusal("a multi-pass deployment states no carried-cut census: its batch buffer cannot be "
                                      "sized")
        return int(K) * int(census.largest_cut()) * int(census.value_bits)
    if component == "working":
        return int(census.peak_live) * int(census.value_bits)
    raise KeyError(component)


@dataclass(frozen=True)
class Component:
    name: str
    kind: str
    capacity_bits: int
    placement: str
    on_chip: bool
    status: str
    area_mm2: Optional[float]
    note: str = ""

    @property
    def megabytes(self) -> float:
        return self.capacity_bits / BITS_PER_MB


def component(c: str, census: Census, K: int, setting: Setting, table: Table) -> Component:
    p = table.placements[setting.placement_of(c)]
    bits = capacity_bits(c, census, K, setting.store_convention)
    if c in p.excluded:
        return Component(c, KIND[c], bits, p.name, p.on_chip, "excluded", None, p.excluded_reason)
    area = None if p.density_mm2_per_bit is None else bits * p.density_mm2_per_bit
    if p.on_chip:
        if area is None:
            raise SystemMemoryRefusal(f"the {c} is placed on the chip, whose placement declares no area per bit")
        return Component(c, KIND[c], bits, p.name, True, "counted", area, p.density_source)
    return Component(c, KIND[c], bits, p.name, False, "beside", area,
                     p.density_source if area is not None else f"not priced: {p.density_source}")


@dataclass
class Area:
    ""
    tiles_mm2: float
    chip_mm2: float
    counted: Tuple[Component, ...]
    excluded: Tuple[Component, ...]
    _beside: Callable[[], Tuple[Component, ...]]
    _cache: Optional[Tuple[Component, ...]] = None

    @property
    def beside(self) -> Tuple[Component, ...]:
        if self._cache is None:
            self._cache = self._beside()
        return self._cache

    @property
    def beside_bits(self) -> int:
        return sum(c.capacity_bits for c in self.beside)

    @property
    def beside_megabytes(self) -> float:
        return self.beside_bits / BITS_PER_MB

    @property
    def beside_mm2(self) -> Optional[float]:
        if any(c.area_mm2 is None for c in self.beside):
            return None
        return sum(c.area_mm2 for c in self.beside)

    def budget_area(self, budget: str) -> float:
        ""
        if budget == "chip":
            return self.chip_mm2
        if budget == "system":
            beside = self.beside_mm2
            if beside is None:
                raise SystemMemoryRefusal("a system-area budget needs every memory beside the chip priced: "
                                          + "; ".join(c.note for c in self.beside if c.area_mm2 is None))
            return self.chip_mm2 + beside
        raise SystemMemoryRefusal(f"budget {budget!r} not in {BUDGETS}")


def deployment_area(tiles_mm2: float, census: Census, K: int, setting: Setting, table: Table) -> Area:
    ""
    setting.check(table)
    on = [c for c in COMPONENTS if table.placements[setting.placement_of(c)].on_chip]
    off = [c for c in COMPONENTS if c not in on]
    comps = [component(c, census, K, setting, table) for c in on]
    counted = tuple(x for x in comps if x.status == "counted")
    chip = tiles_mm2
    for x in counted:
        chip = chip + x.area_mm2
    return Area(tiles_mm2=tiles_mm2, chip_mm2=chip, counted=counted,
                excluded=tuple(x for x in comps if x.status == "excluded"),
                _beside=lambda: tuple(component(c, census, K, setting, table) for c in off))


def energy_rows(setting: Setting, table: Table) -> Dict[str, Optional[str]]:
    ""
    setting.check(table)
    return {c: table.placements[setting.placement_of(c)].energy_row for c in COMPONENTS}
