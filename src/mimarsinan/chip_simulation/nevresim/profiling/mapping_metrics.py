"""Connectivity metrics for HardCoreMapping / ChipModel profiling."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.code_generation.cpp_chip_model import ChipModel, SpikeSource
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources


@dataclass(frozen=True)
class MappingConnectivityMetrics:
    core_count: int
    axons_per_core: int
    neurons_per_core: int
    total_axon_slots: int
    max_spans_per_core: int
    total_spans: int
    span_compression_ratio: float
    cross_core_source_count: int
    unique_cross_core_pairs: int
    input_source_count: int
    off_source_count: int

    def as_dict(self) -> dict:
        return {
            "core_count": self.core_count,
            "axons_per_core": self.axons_per_core,
            "neurons_per_core": self.neurons_per_core,
            "total_axon_slots": self.total_axon_slots,
            "max_spans_per_core": self.max_spans_per_core,
            "total_spans": self.total_spans,
            "span_compression_ratio": self.span_compression_ratio,
            "cross_core_source_count": self.cross_core_source_count,
            "unique_cross_core_pairs": self.unique_cross_core_pairs,
            "input_source_count": self.input_source_count,
            "off_source_count": self.off_source_count,
        }


def _is_cross_core(src: SpikeSource) -> bool:
    return not src.is_input_ and not src.is_off_ and not src.is_always_on_


def metrics_from_hardcore_mapping(hcm: HardCoreMapping) -> MappingConnectivityMetrics:
    core_count = len(hcm.cores)
    axons_per_core = hcm.axons_per_core
    neurons_per_core = hcm.neurons_per_core
    total_spans = 0
    max_spans = 0
    cross_core = 0
    input_count = 0
    off_count = 0
    pairs: set[tuple[int, int]] = set()

    for core in hcm.cores:
        spans = compress_spike_sources(core.axon_sources)
        total_spans += len(spans)
        max_spans = max(max_spans, len(spans))
        for src in core.axon_sources:
            if src.is_input_:
                input_count += 1
            elif src.is_off_:
                off_count += 1
            elif _is_cross_core(src):
                cross_core += 1
                pairs.add((int(src.core_), int(src.neuron_)))

    total_axon_slots = core_count * axons_per_core
    ratio = float(total_axon_slots) / float(total_spans) if total_spans else 0.0

    return MappingConnectivityMetrics(
        core_count=core_count,
        axons_per_core=axons_per_core,
        neurons_per_core=neurons_per_core,
        total_axon_slots=total_axon_slots,
        max_spans_per_core=max_spans,
        total_spans=total_spans,
        span_compression_ratio=ratio,
        cross_core_source_count=cross_core,
        unique_cross_core_pairs=len(pairs),
        input_source_count=input_count,
        off_source_count=off_count,
    )


def metrics_from_chip_model(chip: ChipModel) -> MappingConnectivityMetrics:
    total_spans = sum(len(con.get_spans()) for con in chip.connections)
    max_spans = chip._max_spans_per_core()  # noqa: SLF001 — profiling helper
    cross_core = 0
    input_count = 0
    off_count = 0
    pairs: set[tuple[int, int]] = set()

    for con in chip.connections:
        for src in con.axon_sources:
            if src.is_input_:
                input_count += 1
            elif src.is_off_:
                off_count += 1
            elif _is_cross_core(src):
                cross_core += 1
                pairs.add((int(src.core_), int(src.neuron_)))

    total_axon_slots = chip.core_count * chip.axon_count
    ratio = float(total_axon_slots) / float(total_spans) if total_spans else 0.0

    return MappingConnectivityMetrics(
        core_count=chip.core_count,
        axons_per_core=chip.axon_count,
        neurons_per_core=chip.neuron_count,
        total_axon_slots=total_axon_slots,
        max_spans_per_core=max_spans,
        total_spans=total_spans,
        span_compression_ratio=ratio,
        cross_core_source_count=cross_core,
        unique_cross_core_pairs=len(pairs),
        input_source_count=input_count,
        off_source_count=off_count,
    )
