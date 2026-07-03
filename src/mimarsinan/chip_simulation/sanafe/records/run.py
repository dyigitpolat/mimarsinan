"""SANA-FE run-record dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from mimarsinan.chip_simulation.ttfs.ttfs_recorder import TtfsRunRecord

from mimarsinan.chip_simulation.recording.spike_recorder import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)
from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown
from mimarsinan.chip_simulation.sanafe.records.hardware import (
    SanafeArchGeometry,
    SanafeCascadePoint,
    SanafeConnectivityEdge,
    SanafeCoreRecord,
    SanafeCriticalCore,
    SanafeCycleEnergyPoint,
    SanafeNocLink,
    SanafeNocLinkLoad,
    SanafeTileRecord,
)

@dataclass
class SanafeCoreDiff:
    """Per-core parity-gate delta (HCM expected vs SANA-FE actual).

    Deltas are absolute: positive means SANA-FE over-reported, negative under.
    """

    core_index: int
    input_delta_sum: int
    output_delta_sum: int


@dataclass
class SanafeSegmentRecord:
    """Per-neural-``HybridStage`` record produced by one ``chip.sim()`` call."""

    stage_index: int
    stage_name: str
    schedule_segment_index: Optional[int]
    schedule_pass_index: Optional[int]

    timesteps_executed: int
    sim_time_s: float
    energy: SanafeEnergyBreakdown
    spikes: int
    packets_sent: int
    neurons_updated: int
    neurons_fired: int

    seg_input_rates: np.ndarray
    seg_input_spike_count: np.ndarray
    seg_output_spike_count: np.ndarray
    per_core: List[SanafeCoreRecord] = field(default_factory=list)

    per_tile: List[SanafeTileRecord] = field(default_factory=list)
    per_neuron_spike_counts: Optional[np.ndarray] = None
    per_neuron_spike_trace: Optional[np.ndarray] = None
    per_neuron_potential_trace: Optional[np.ndarray] = None
    message_trace: Optional[List[Dict[str, Any]]] = None
    arch_geometry: Optional["SanafeArchGeometry"] = None
    noc_links: List["SanafeNocLink"] = field(default_factory=list)
    noc_link_load: List["SanafeNocLinkLoad"] = field(default_factory=list)
    cycle_energy: List["SanafeCycleEnergyPoint"] = field(default_factory=list)
    cascade: List["SanafeCascadePoint"] = field(default_factory=list)
    critical_cores: List["SanafeCriticalCore"] = field(default_factory=list)
    connectivity: List["SanafeConnectivityEdge"] = field(default_factory=list)
    hcm_diff: List["SanafeCoreDiff"] = field(default_factory=list)
    contract_ttfs_cores: List[Any] = field(default_factory=list)
    contract_ttfs_seg_output: Optional[np.ndarray] = None
    noc_traffic_per_cycle: List[List[List[int]]] = field(default_factory=list)
    tile_packets_per_cycle: List[Dict[int, int]] = field(default_factory=list)
    inter_tile_packets: int = 0
    intra_tile_packets: int = 0
    input_path_packets: int = 0
    cross_tile_connectivity_edges: int = 0
    chip_spike_count: int = 0
    lif_spike_count: int = 0
    spike_trace_parse_skipped: int = 0
    spike_capture_warning: Optional[str] = None
    mapped_cross_tile_axons: int = 0
    ttfs_contract_active_cores: int = 0
    ttfs_hardware_active_cores: int = 0
    ttfs_event_active_cores: int = 0
    ttfs_activation_event_mismatch_count: int = 0


@dataclass
class SanafeRunRecord:
    """All SANA-FE output for one sample.

    ``segments`` is keyed by ``stage_index`` to match HCM's ``RunRecord``;
    sparse / non-contiguous keys are preserved end-to-end.
    """

    arch_preset: str
    arch_name: str
    sample_index: int
    T: int
    segments: Dict[int, SanafeSegmentRecord] = field(default_factory=dict)
    compute_outputs: Dict[int, np.ndarray] = field(default_factory=dict)

    aggregate_energy: SanafeEnergyBreakdown = field(
        default_factory=SanafeEnergyBreakdown.zero
    )
    aggregate_sim_time_s: float = 0.0
    total_spikes: int = 0
    total_packets: int = 0

    def to_hcm_subset(self) -> RunRecord:
        """Project to ``spike_recorder.RunRecord`` for parity diffing.

        Drops energy/latency/NoC/trace fields; keeps the spike-count layers plus
        the bookkeeping fields the diff cause-suggestion uses.
        """
        out = RunRecord(sample_index=self.sample_index, T=self.T)
        for stage_index, seg in self.segments.items():
            out.segments[stage_index] = SegmentSpikeRecord(
                stage_index=seg.stage_index,
                stage_name=seg.stage_name,
                schedule_segment_index=seg.schedule_segment_index,
                schedule_pass_index=seg.schedule_pass_index,
                seg_input_rates=seg.seg_input_rates,
                seg_input_spike_count=seg.seg_input_spike_count,
                seg_output_spike_count=seg.seg_output_spike_count,
                cores=[
                    CoreSpikeCounts(
                        core_index=c.core_index,
                        n_in_used=c.n_axons_used,
                        n_out_used=c.n_neurons,
                        core_latency=c.core_latency,
                        has_hardware_bias=c.has_hardware_bias,
                        n_always_on_axons=c.n_always_on_axons,
                        input_spike_count=c.input_spike_count,
                        output_spike_count=c.output_spike_count,
                    )
                    for c in seg.per_core
                ],
            )
        for k, v in self.compute_outputs.items():
            out.compute_outputs[k] = v
        return out

    def _ttfs_record_from_segments(
        self,
        *,
        spiking_mode: str,
        core_source: str,
    ) -> "TtfsRunRecord":
        """Build ``TtfsRunRecord`` from per-segment TTFS core lists.

        ``core_source`` is ``"hardware"`` (``output_activation``/trace) or
        ``"contract"`` (``contract_ttfs_*`` from ``TtfsAnalyticalExecutor``).
        """
        from mimarsinan.chip_simulation.ttfs.ttfs_recorder import (
            SegmentTtfsRecord,
            TtfsRunRecord,
        )

        out = TtfsRunRecord(
            sample_index=self.sample_index,
            simulation_length=self.T,
            spiking_mode=spiking_mode,
        )
        for stage_index, seg in self.segments.items():
            if core_source == "contract":
                cores = list(seg.contract_ttfs_cores)
                seg_out = seg.contract_ttfs_seg_output
                if seg_out is None:
                    seg_out = np.zeros(0, dtype=np.float64)
            else:
                from mimarsinan.chip_simulation.ttfs.ttfs_recorder import (
                    CoreTtfsActivations,
                )

                cores = []
                for c in seg.per_core:
                    if c.output_activation is None:
                        continue
                    cores.append(CoreTtfsActivations(
                        core_index=c.core_index,
                        n_out_used=c.n_neurons,
                        output_activation=np.asarray(
                            c.output_activation, dtype=np.float64,
                        ),
                    ))
                seg_out = np.zeros(0, dtype=np.float64)
            out.segments[stage_index] = SegmentTtfsRecord(
                stage_index=seg.stage_index,
                stage_name=seg.stage_name,
                schedule_segment_index=seg.schedule_segment_index,
                schedule_pass_index=seg.schedule_pass_index,
                seg_output=np.asarray(seg_out, dtype=np.float64),
                cores=cores,
            )
        for k, v in self.compute_outputs.items():
            out.compute_outputs[k] = v
        return out

    def to_ttfs_hardware_subset(self, *, spiking_mode: str = "ttfs") -> "TtfsRunRecord":
        """TTFS activations from plugin somas via ``potential_trace`` readout."""
        return self._ttfs_record_from_segments(
            spiking_mode=spiking_mode, core_source="hardware",
        )

    def to_ttfs_contract_subset(self, *, spiking_mode: str = "ttfs") -> "TtfsRunRecord":
        """TTFS activations from ``TtfsAnalyticalExecutor`` on segment inputs."""
        return self._ttfs_record_from_segments(
            spiking_mode=spiking_mode, core_source="contract",
        )

    def to_ttfs_subset(self, *, spiking_mode: str = "ttfs") -> "TtfsRunRecord":
        """Alias for hardware TTFS parity (plugin ``potential_trace``)."""
        return self.to_ttfs_hardware_subset(spiking_mode=spiking_mode)
