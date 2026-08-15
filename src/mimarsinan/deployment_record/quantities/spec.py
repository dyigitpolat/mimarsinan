"""The quantity catalog: every priceable multiplicand, declared once with its meaning.

A quantity is a number a deployment produces (a count, a byte total, a wall time) that
a physics constant multiplies. The catalog is closed so a pricing formula can never
reference a number nobody produces, and ABSENCE is meaningful throughout: a quantity a
view cannot answer is missing, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Tuple

from mimarsinan.deployment_record.units import (
    DATA,
    DIMENSIONLESS,
    DIMENSIONS,
    TIME,
    unit_for,
)

#: measured = read off an executed deployment; static = a fact of the declared shape
#: or packing; modeled = rests on a stated assumption (and says so downstream).
PROVENANCE_KINDS = frozenset({"measured", "static", "modeled"})

_COUNT = (DIMENSIONLESS, "1")

#: (key, (dimension, unit), doc) — declaration order is report order.
_ROWS: Tuple[Tuple[str, Tuple[str, str], str], ...] = (
    # --- structural: the chip and what the mapping occupies ---------------------
    ("cells_used", _COUNT,
     "Weight cells the mapping actually occupies, post-compaction, as mapped."),
    ("cells_physical", _COUNT,
     "Weight cells the DECLARED chip provides (its parameter capacity over every "
     "declared core) — area prices the chip, not the cores a mapping allocated."),
    ("cores_physical", _COUNT,
     "Cores the DECLARED chip provides (the whole-chip static-power and area "
     "multiplicand)."),
    ("cores_allocated", _COUNT,
     "Hard cores the mapping allocates."),
    ("neurons_used", _COUNT, "Neuron slots the mapping drives."),
    ("neurons_physical", _COUNT,
     "Neuron slots the DECLARED chip provides, over every declared core."),
    ("axons_used", _COUNT, "Axon (row) slots the mapping drives."),
    ("axons_physical", _COUNT,
     "Axon (row) slots the DECLARED chip provides, over every declared core."),
    ("tiles", _COUNT, "NoC tiles of the resolved floorplan."),
    ("weight_bits", (DIMENSIONLESS, "bit"), "Declared weight precision of the run."),
    ("macs", _COUNT,
     "As-mapped MAC sites: every mapped instance's cells count, so a weight bank "
     "replicated across cores counts once per replica — the ENERGY multiplicand, "
     "because replicas really fire. Distinct from the logical census total_macs."),
    # --- program: passes, payloads, barriers -------------------------------------
    ("pass_count", _COUNT, "Passes of the deployed schedule."),
    ("sync_count", _COUNT, "Chip-wide synchronization barriers of the schedule."),
    ("reprogram_passes", _COUNT, "Passes that reprogram core weights."),
    ("reprogrammed_bytes", (DATA, "B"),
     "Programming payload bytes over REPROGRAM passes only — a resident pass sends "
     "nothing (the resident-payload discipline)."),
    ("connectivity_entries", _COUNT,
     "Connectivity (axon source span) entries programmed over reprogram passes only."),
    ("segment_cores", _COUNT,
     "Sum of core counts over EVERY pass — each pass resets its cores' neuron state, "
     "resident or not (the core-init multiplicand)."),
    ("reprogrammed_cores", _COUNT,
     "Sum of core counts over reprogram passes only (the per-core programming "
     "overhead multiplicand)."),
    ("carried_raster_bytes", (DATA, "B"),
     "Payload crossing INTRA-SEGMENT pass boundaries per inference, sized under "
     "the run's sealed transfer discipline: raster bits under verbatim, window "
     "counts under collapse. A pass that halves cores but doubles this is not "
     "a free win."),
    ("carry_peak_live_bytes", (DATA, "B"),
     "Peak concurrently-live carried payload over the segment's boundaries — the "
     "buffer a scheduled chip must provision. Wires with disjoint live ranges "
     "share it. A capacity fact for constraints; deliberately not an energy "
     "multiplicand."),
    # --- dynamics: what an execution did ------------------------------------------
    ("timesteps", _COUNT, "Timesteps of one inference window (S)."),
    ("latency_steps", _COUNT,
     "End-to-end latency in timesteps: the sum of per-segment executed timesteps "
     "for one sample — what t_cycle converts to seconds."),
    ("total_spikes", _COUNT,
     "Spike EMISSIONS counted by the simulator — not synapse arrivals."),
    ("boundary_events", _COUNT,
     "Value/spike events crossing observed stage boundaries (gate-reduced totals)."),
    ("synaptic_events", _COUNT,
     "Spike arrivals at active synapses — one event is one MAC on an event-driven "
     "chip. No producer seals this yet: at candidate time it is MODELED from the "
     "declared activity factor; at record time it is absent until an event census "
     "is sealed."),
    ("noc_total_packets", _COUNT, "All NoC packets of the execution."),
    ("noc_intra_tile_packets", _COUNT, "Packets delivered within their source tile."),
    ("noc_inter_tile_packets", _COUNT, "Packets that left their source tile."),
    ("noc_total_hops", _COUNT,
     "Total link traversals: the sum of per-link packet counts over the NoC "
     "(derived from the sealed link loads; one packet crossing three links is "
     "three hops)."),
    # --- periphery: producers land with the conversion models (C6) ----------------
    ("adc_conversions", _COUNT,
     "Analog-to-digital conversions of the execution, produced by the target's "
     "declared conversion model; identically zero on fully digital targets."),
    ("adc_count", _COUNT,
     "ADC instances the chip needs (columns / sharing factor), produced by the "
     "target's declared conversion model."),
    # --- host / the logical split --------------------------------------------------
    ("host_ops_s", (TIME, "s"),
     "Measured wall time of host ComputeOps for one sample (per-pass normalized)."),
    ("total_params", _COUNT, "Logical parameter census of the deployed model."),
    ("onchip_params", _COUNT, "Logical parameters realized on chip cores."),
    ("host_params", _COUNT, "Logical parameters held by host ComputeOps."),
    ("total_macs", _COUNT,
     "Logical forward-MAC census of one inference — replication-free, the host/chip "
     "SPLIT census (distinct from the as-mapped energy multiplicand macs)."),
    ("onchip_macs", _COUNT, "Logical forward MACs realized on chip."),
    ("host_macs", _COUNT, "Logical forward MACs executed host-side."),
)


@dataclass(frozen=True)
class QuantitySpec:
    """One catalog row: what the number is, in what unit, and what it means."""

    key: str
    dimension: str
    unit: str
    doc: str

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ValueError(f"{self.key}: unknown dimension {self.dimension!r}")
        if unit_for(self.unit).dimension != self.dimension:
            raise ValueError(
                f"{self.key}: unit {self.unit!r} is not a {self.dimension} unit"
            )


def _build() -> Dict[str, QuantitySpec]:
    specs: Dict[str, QuantitySpec] = {}
    for key, (dimension, unit), doc in _ROWS:
        if key in specs:
            raise ValueError(f"quantity {key!r} is declared twice")
        specs[key] = QuantitySpec(key=key, dimension=dimension, unit=unit, doc=doc)
    return specs


QUANTITY_SPECS: Mapping[str, QuantitySpec] = _build()


def quantity_spec(key: str) -> QuantitySpec:
    """The declaration for ``key``, or a loud error naming the catalog."""
    try:
        return QUANTITY_SPECS[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a catalog quantity; the catalog is {sorted(QUANTITY_SPECS)}"
        ) from None


@dataclass(frozen=True)
class QuantityValue:
    """One produced number and how it was obtained."""

    value: float
    provenance: str

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_KINDS:
            raise ValueError(
                f"QuantityValue.provenance must be one of {sorted(PROVENANCE_KINDS)}, "
                f"got {self.provenance!r}"
            )


class Quantities:
    """A view's produced quantities. Absence is meaningful — there is no default."""

    def __init__(self, values: Mapping[str, QuantityValue]) -> None:
        for key in values:
            quantity_spec(key)
        # Catalog order, so reports and comparisons are stable.
        self._values: Dict[str, QuantityValue] = {
            key: values[key] for key in QUANTITY_SPECS if key in values
        }

    def has(self, key: str) -> bool:
        quantity_spec(key)
        return key in self._values

    def get(self, key: str) -> QuantityValue:
        quantity_spec(key)
        if key not in self._values:
            raise KeyError(
                f"this view produces no {key!r}; the terms that need it are "
                f"unavailable, and no default may stand in for a census"
            )
        return self._values[key]

    def missing(self, keys: Iterable[str]) -> Tuple[str, ...]:
        return tuple(key for key in keys if not self.has(key))

    def keys(self) -> Iterator[str]:
        return iter(self._values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Quantities) and self._values == other._values

    def __repr__(self) -> str:
        return f"Quantities({sorted(self._values)})"
