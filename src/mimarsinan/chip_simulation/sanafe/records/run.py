"""SANA-FE run-record dataclasses.

These records carry SANA-FE's *full* output for a single sample — per-tile
and per-core energy breakdowns, latency (``sim_time``), NoC packet counts,
optional per-neuron spike + potential traces, and the spike-count subset
that overlaps with HCM's ``spike_recorder.RunRecord``.

Why our own record types
------------------------
HCM's ``RunRecord`` is a spike-count-only shape sized for the diff-based
parity gate.  SANA-FE produces a richer, multi-dimensional output; bolting
the rich fields onto ``RunRecord`` would mix two simulators' contracts.
Instead, we keep the rich record native to SANA-FE and expose a single
lossless projection — :meth:`SanafeRunRecord.to_hcm_subset` — to bridge
back to the HCM shape when the parity gate runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

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

    Only populated when ``sanafe_simulation_step`` ran with parity
    checking on and the diff was non-empty; otherwise the floorplan
    diff overlay is disabled.  ``input_delta`` and ``output_delta``
    are absolute deltas; positive means SANA-FE over-reported, negative
    means under-reported.
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

    # Parity-gate fields ------------------------------------------------------
    seg_input_rates: np.ndarray            # (1, seg_in_size) float32
    seg_input_spike_count: np.ndarray      # (seg_in_size,) int64
    seg_output_spike_count: np.ndarray     # (seg_out_size,) int64
    per_core: List[SanafeCoreRecord] = field(default_factory=list)

    # Rich-only fields --------------------------------------------------------
    per_tile: List[SanafeTileRecord] = field(default_factory=list)
    per_neuron_spike_counts: Optional[np.ndarray] = None
    per_neuron_spike_trace: Optional[np.ndarray] = None       # (n_neurons, T)
    per_neuron_potential_trace: Optional[np.ndarray] = None   # (n_neurons, T)
    message_trace: Optional[List[Dict[str, Any]]] = None
    # GUI floorplan + NoC overlay ---------------------------------------------
    arch_geometry: Optional["SanafeArchGeometry"] = None
    noc_links: List["SanafeNocLink"] = field(default_factory=list)
    # NoC link-level congestion (per single mesh edge, XY-routed).
    noc_link_load: List["SanafeNocLinkLoad"] = field(default_factory=list)
    # Reconstructed per-cycle energy breakdown for the waterfall chart.
    cycle_energy: List["SanafeCycleEnergyPoint"] = field(default_factory=list)
    # Per-(cycle, depth) firing counts for the cascade-timeline view.
    cascade: List["SanafeCascadePoint"] = field(default_factory=list)
    # Per-cycle critical-core series for the critical-core highlight.
    critical_cores: List["SanafeCriticalCore"] = field(default_factory=list)
    # Static connectivity edges (independent of activity) for the overlay.
    connectivity: List["SanafeConnectivityEdge"] = field(default_factory=list)
    # Optional HCM↔SF parity-gate deltas, attached by the pipeline step.
    hcm_diff: List["SanafeCoreDiff"] = field(default_factory=list)
    # TTFS contract layer: TtfsAnalyticalExecutor on segment inputs (no SANA-FE core).
    contract_ttfs_cores: List[Any] = field(default_factory=list)
    contract_ttfs_seg_output: Optional[np.ndarray] = None
    # Per-cycle compact NoC traffic for the animated playback view.
    # Each cycle is a list of ``[src_x, src_y, dst_x, dst_y, count]``;
    # empty when ``log_message_trace=False``.
    noc_traffic_per_cycle: List[List[List[int]]] = field(default_factory=list)
    # Per-cycle packet count per destination tile_index (intra+inter); GUI playback fallback.
    tile_packets_per_cycle: List[Dict[int, int]] = field(default_factory=list)
    # Message-trace taxonomy (non-placeholder events only).
    inter_tile_packets: int = 0
    intra_tile_packets: int = 0
    input_path_packets: int = 0
    cross_tile_connectivity_edges: int = 0
    # Spike-trace diagnostics (chip aggregate vs attributed LIF / input groups).
    chip_spike_count: int = 0
    lif_spike_count: int = 0
    spike_trace_parse_skipped: int = 0
    spike_capture_warning: Optional[str] = None
    # Expected vs observed NoC / TTFS activity (segment-level diagnostics).
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

        Drops energy, latency, NoC and trace fields; preserves the spike-count
        layers (segment input / per-core input / per-core output / segment
        output) plus the bookkeeping fields the diff cause-suggestion uses.
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
        # compute_outputs flow through verbatim (host-side ComputeOp outputs).
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

        ``core_source`` is ``"hardware"`` (``output_activation`` / trace) or
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
