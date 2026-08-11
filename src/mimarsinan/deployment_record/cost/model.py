"""The parameterized cost model over a sealed :class:`DeploymentRecord`.

Every modeled number is the program's EXISTING per-phase cost model
(``chip_simulation.weight_reuse_cost_model``) applied to REAL record
quantities — the segment census, the programmed payload, the sync census —
never to module constants. Measured record quantities pass through untouched
and unbanded; modeled terms carry the ``(low, nominal, high)`` band and its
written basis; derived terms combine the two and say which.

Two disciplines are structural here:

* **The record's own modeled terms are not consumed.** ``energy.breakdown``
  entries with ``kind="modeled"`` and ``timing.latency.programming_s`` /
  ``sync_s`` are a producer's model output; this model re-derives those terms
  out of record quantities, so passing them through would double-count them.
* **NoC transport is never charged twice.** Measured ``sim_time_s`` already
  contains it, and the model REFUSES a record whose latency note does not say
  so (:func:`_require_no_double_count_note`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from mimarsinan.deployment_record.cost.coefficients import (
    BYTES_PER_CONNECTIVITY_ENTRY,
    CORE_INIT,
    E_DMA_PER_BYTE_MJ,
    E_SYNC_BARRIER_MJ,
    PROGRAMMING_BANDWIDTH_BYTES_PER_S,
    SYNC_BARRIER_S,
    CoreInitCoefficients,
    banded,
    corner_value,
    sync_energy_mj,
)
from mimarsinan.deployment_record.cost.combine import (
    band_of,
    derived,
    flat_band,
    measured,
    modeled,
    sum_bands,
)
from mimarsinan.deployment_record.cost.segments import segment_costs
from mimarsinan.deployment_record.cost.terms import (
    CostTerm,
    DeploymentCostReport,
    SegmentInitCost,
    find_term,
)
from mimarsinan.deployment_record.schema import Band, DeploymentRecord

_NOC_NOTE_TOKENS = ("noc", "double-count")

_TOTAL_ENERGY_BASIS = (
    "measured SANA-FE total + modeled programming payload, per-core init and "
    "sync barriers; the measured plane terms decompose the measured total and "
    "are never added on top of it"
)
_TOTAL_LATENCY_BASIS = (
    "PER SAMPLE (one program traversal): measured compute (SANA-FE sim_time_s "
    "for one sample, NoC transport already inside) + measured host-op walls "
    "normalized per pass + modeled programming, per-core init and sync; no "
    "term adds NoC hops a second time and no whole-run total joins unscaled"
)
_THROUGHPUT_BASIS = (
    "steady-state samples/s = 1 / per-sample latency total; no cross-sample "
    "pipelining is assumed, so the band inverts the latency band"
)
_MODEL_NOTE = (
    "modeled terms are weight_reuse_cost_model coefficients applied to record "
    "quantities; record-carried modeled terms are re-derived here, not passed "
    "through, so nothing is double-counted"
)


def _require_no_double_count_note(note: str) -> None:
    """The record must SAY that measured time already carries NoC transport."""
    lowered = note.lower()
    missing = [token for token in _NOC_NOTE_TOKENS if token not in lowered]
    if missing:
        raise ValueError(
            f"timing.latency.note must state the no-double-count discipline "
            f"(missing {missing}); the model refuses to add transport terms on "
            f"top of a record that does not declare that sim_time_s already "
            f"includes NoC hops. note={note!r}"
        )


def _sum_segment_bands(
    segments: Sequence[SegmentInitCost], name: str, *, basis: str
) -> Band:
    """Σ over segments of one named per-segment band."""
    return sum_bands([band_of(segment.term(name)) for segment in segments], basis=basis)


@dataclass(frozen=True)
class DeploymentCostModel:
    """The cost surfaces over a record, parameterized by its coefficient bands."""

    core_init: CoreInitCoefficients = CORE_INIT
    bytes_per_connectivity_entry: Band = BYTES_PER_CONNECTIVITY_ENTRY
    programming_bandwidth_bytes_per_s: Band = PROGRAMMING_BANDWIDTH_BYTES_PER_S
    sync_barrier_s: Band = SYNC_BARRIER_S

    def segment_initialization(
        self, record: DeploymentRecord
    ) -> Tuple[SegmentInitCost, ...]:
        """Per-segment initialization cost: reset constant + programming payload."""
        return segment_costs(
            record,
            core_init=self.core_init,
            bytes_per_connectivity_entry=self.bytes_per_connectivity_entry,
            programming_bandwidth_bytes_per_s=self.programming_bandwidth_bytes_per_s,
        )

    def energy(self, record: DeploymentRecord) -> Tuple[CostTerm, ...]:
        """Measured passthrough terms + the modeled programming/init/sync terms."""
        record_energy = record.energy
        if record_energy is None:
            raise ValueError(
                "the cost model requires the record's energy fragment; a "
                "SANA-FE-disabled run carries no measured energy to cost"
            )
        segments = self.segment_initialization(record)
        measured_total = float(record_energy.total_energy_mj)
        terms: List[CostTerm] = [
            measured("measured_total_mj", "mJ", measured_total, "energy.total_energy_mj")
        ]
        terms += [
            measured(
                f"measured_{term.name}_mj", "mJ", float(term.mj),
                f"energy.breakdown[{term.name}].mj",
            )
            for term in record_energy.breakdown
            if term.kind == "measured"
        ]
        modeled_terms = (
            modeled(
                "modeled_programming_mj", "mJ",
                _sum_segment_bands(
                    segments, "payload_energy_mj",
                    basis=f"sum over scheduled segments; {E_DMA_PER_BYTE_MJ.basis}",
                ),
                "sum of segment payload_energy_mj",
            ),
            modeled(
                "modeled_core_init_mj", "mJ",
                _sum_segment_bands(
                    segments, "reset_energy_mj",
                    basis=(
                        f"sum over scheduled segments; "
                        f"{self.core_init.energy_mj.basis}"
                    ),
                ),
                "sum of segment reset_energy_mj",
            ),
            modeled(
                "modeled_sync_mj", "mJ",
                banded(
                    lambda corner: sync_energy_mj(record.schedule.sync_count, corner),
                    basis=E_SYNC_BARRIER_MJ.basis,
                ),
                "schedule.sync_count x weight_reuse_cost_model sync channel",
            ),
        )
        total = sum_bands(
            [band_of(term) for term in modeled_terms]
            + [flat_band(measured_total, _TOTAL_ENERGY_BASIS)],
            basis=_TOTAL_ENERGY_BASIS,
        )
        return tuple(terms) + modeled_terms + (
            derived(
                "total_mj", "mJ", total.nominal,
                "measured_total_mj + modeled programming/core-init/sync",
                band=total,
            ),
        )

    def latency(self, record: DeploymentRecord) -> Tuple[CostTerm, ...]:
        """Programming/init/sync (modeled) + compute and host ops (measured)."""
        decomposition = record.timing.latency
        _require_no_double_count_note(decomposition.note)
        compute_s = decomposition.compute_sim_time_s
        if compute_s is None:
            raise ValueError(
                "the cost model requires measured compute latency "
                "(timing.latency.compute_sim_time_s); a record without the "
                "SANA-FE per-segment census has no compute time to decompose"
            )
        segments = self.segment_initialization(record)
        programming = _sum_segment_bands(
            segments, "programming_time_s",
            basis=(
                f"sum over scheduled segments; "
                f"{self.programming_bandwidth_bytes_per_s.basis}"
            ),
        )
        core_init = _sum_segment_bands(
            segments, "reset_time_s",
            basis=f"sum over scheduled segments; {self.core_init.time_s.basis}",
        )
        sync = banded(
            lambda corner: record.schedule.sync_count
            * corner_value(self.sync_barrier_s, corner),
            basis=self.sync_barrier_s.basis,
        )
        terms: List[CostTerm] = [
            modeled("programming_s", "s", programming, "sum of segment programming_time_s"),
            modeled("core_init_s", "s", core_init, "sum of segment reset_time_s"),
            measured("compute_s", "s", float(compute_s), "timing.latency.compute_sim_time_s"),
        ]
        measured_s = float(compute_s)
        host_ops_s = decomposition.host_ops_s
        if host_ops_s is not None:
            # compute_sim_time_s is a per-sample census; the raw host wall covers
            # the whole run. Only the per-pass normalization is commensurable.
            per_pass = decomposition.host_ops_s_per_pass
            if per_pass is None:
                raise ValueError(
                    "measured host-op walls carry no per pass normalization "
                    "(timing.latency.host_ops_s_per_pass is None because some "
                    "timed op has no invocation count); the whole-run total "
                    "cannot join a per-sample decomposition and a guessed "
                    "divisor would be a proxy presented as a measurement"
                )
            measured_s += float(per_pass)
            terms.append(
                measured(
                    "host_ops_s", "s", float(per_pass),
                    "timing.latency.host_ops_s_per_pass",
                )
            )
        terms.append(modeled("sync_s", "s", sync, "schedule.sync_count x sync_barrier_s"))
        total = sum_bands(
            [programming, core_init, sync, flat_band(measured_s, _TOTAL_LATENCY_BASIS)],
            basis=_TOTAL_LATENCY_BASIS,
        )
        terms.append(
            derived(
                "total_s", "s", total.nominal,
                "measured compute + host ops, modeled programming/core-init/sync",
                band=total,
            )
        )
        return tuple(terms)

    def area(self, record: DeploymentRecord) -> Tuple[CostTerm, ...]:
        """Cores used, occupancy, waste and fragmentation — from utilization."""
        crossbar = record.utilization.crossbar
        layout = record.utilization.layout
        occupancy = float(crossbar.cell_occupancy)
        return (
            measured("cores_used", "cores", float(crossbar.cores_allocated),
                     "utilization.crossbar.cores_allocated"),
            measured("cell_occupancy", "fraction", occupancy,
                     "utilization.crossbar.cell_occupancy"),
            derived("cell_waste_fraction", "fraction", 1.0 - occupancy,
                    "1 - utilization.crossbar.cell_occupancy"),
            measured("unused_area_cells", "cells", float(layout.unused_area_total),
                     "utilization.layout.unused_area_total"),
            measured("unusable_space_cells", "cells", float(crossbar.unusable_space),
                     "utilization.crossbar.unusable_space"),
            measured("fragmentation_pct", "percent", float(layout.fragmentation_pct),
                     "utilization.layout.fragmentation_pct"),
        )

    def throughput(self, record: DeploymentRecord) -> Tuple[CostTerm, ...]:
        """Samples/s from the latency decomposition (the band inverts)."""
        return self._throughput(self.latency(record))

    def _throughput(self, latency_terms: Sequence[CostTerm]) -> Tuple[CostTerm, ...]:
        total = band_of(find_term(latency_terms, "total_s"))
        if total.low <= 0.0:
            raise ValueError(
                f"per-sample latency total must be positive to invert, got "
                f"low={total.low}"
            )
        band = Band(
            low=1.0 / total.high,
            nominal=1.0 / total.nominal,
            high=1.0 / total.low,
            basis=_THROUGHPUT_BASIS,
        )
        return (
            derived("samples_per_s", "samples/s", band.nominal,
                    "1 / DeploymentCostModel.latency(total_s)", band=band),
        )

    def evaluate(self, record: DeploymentRecord) -> DeploymentCostReport:
        """The whole cost surface, JSON-safe, every term banded and sourced."""
        latency_terms = self.latency(record)
        return DeploymentCostReport(
            segments=self.segment_initialization(record),
            energy=self.energy(record),
            latency=latency_terms,
            area=self.area(record),
            throughput=self._throughput(latency_terms),
            notes=(record.timing.latency.note, _MODEL_NOTE),
        )
