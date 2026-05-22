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

from mimarsinan.chip_simulation.spike_recorder import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)


# Energy


@dataclass
class SanafeEnergyBreakdown:
    """Per-component energy in joules.

    SANA-FE reports ``total`` independently of the components so we store
    both: ``total_j`` is what SANA-FE returns; :meth:`components_sum` is
    the cross-check.  In practice they agree to numerical precision, but
    if SANA-FE ever bills overhead components we don't track, the gap is
    visible rather than papered over.
    """

    synapse_j: float
    dendrite_j: float
    soma_j: float
    network_j: float
    total_j: float

    def components_sum(self) -> float:
        return self.synapse_j + self.dendrite_j + self.soma_j + self.network_j

    def add(self, other: "SanafeEnergyBreakdown") -> "SanafeEnergyBreakdown":
        """Pure aggregation helper used by ``SanafeStepReport``."""
        return SanafeEnergyBreakdown(
            synapse_j=self.synapse_j + other.synapse_j,
            dendrite_j=self.dendrite_j + other.dendrite_j,
            soma_j=self.soma_j + other.soma_j,
            network_j=self.network_j + other.network_j,
            total_j=self.total_j + other.total_j,
        )

    @classmethod
    def zero(cls) -> "SanafeEnergyBreakdown":
        return cls(synapse_j=0.0, dendrite_j=0.0, soma_j=0.0,
                   network_j=0.0, total_j=0.0)

    @classmethod
    def from_sanafe_dict(cls, d: Dict[str, float]) -> "SanafeEnergyBreakdown":
        """Build from the ``results['energy']`` dict returned by ``chip.sim()``."""
        return cls(
            synapse_j=float(d.get("synapse", 0.0)),
            dendrite_j=float(d.get("dendrite", 0.0)),
            soma_j=float(d.get("soma", 0.0)),
            network_j=float(d.get("network", 0.0)),
            total_j=float(d.get("total", 0.0)),
        )


# Hardware-aligned records


@dataclass
class SanafeCoreRecord:
    """Per-``HardCore`` rich record.

    The spike-count fields (``input_spike_count``, ``output_spike_count``)
    plus the bookkeeping fields (``core_latency``, ``has_hardware_bias``,
    ``n_always_on_axons``) project losslessly back to a
    :class:`spike_recorder.CoreSpikeCounts` in :meth:`to_hcm_subset`.
    """

    core_index: int                 # HardCore index inside its HCM
    n_neurons: int                  # used neuron count for this core
    n_axons_used: int               # used axon count for this core
    core_latency: int               # propagated for parity diff context
    has_hardware_bias: bool
    n_always_on_axons: int
    spikes_fired: int
    input_spike_count: np.ndarray   # (n_axons_used,) int64
    output_spike_count: np.ndarray  # (n_neurons,) int64
    energy: SanafeEnergyBreakdown   # per-core energy estimate
    # Per-core 2D spike raster (n_neurons, T_eff) uint8 — slice of the
    # segment-wide trace.  Powers the "click a core → see its raster"
    # mini-view in the GUI; None when log_potential_trace=False or the
    # group wasn't logged.
    spike_raster: Optional[np.ndarray] = None
    output_activation: Optional[np.ndarray] = None
    # Spikes on ``core{N}_in`` / ``core{N}_on`` groups (input-path neurons).
    input_neuron_spikes_fired: int = 0


@dataclass
class SanafeTileRecord:
    """Per-SANA-FE-tile aggregate (energy / spikes / packets).

    Tiles are SANA-FE's outer grouping for cores.  This record is *not*
    used by the parity gate — it exists purely for the GUI / aggregation.

    ``mesh_x`` / ``mesh_y`` are the tile's coordinates in the
    architecture's NoC mesh (set when the arch synth records the
    geometry).  Defaults of ``-1`` mean "unknown" — the floorplan view
    falls back to row-major layout in that case.
    """

    tile_index: int
    cores: List[int]                # HardCore indices placed in this tile
    energy: SanafeEnergyBreakdown
    spikes_fired: int
    packets_sent: int
    mesh_x: int = -1
    mesh_y: int = -1


@dataclass
class SanafeNocLink:
    """Aggregated NoC traffic between a (src_tile, dst_tile) pair.

    Built from SANA-FE's ``message_trace`` after the segment finishes;
    one entry per **directed** tile pair that carried at least one
    real (non-placeholder) spike.  Powers the NoC-traffic overlay in
    the GUI floorplan view — the count / spikes / hops fields each
    feed a different colormap option.
    """

    src_tile: int
    dst_tile: int
    src_x: int
    src_y: int
    dst_x: int
    dst_y: int
    packet_count: int
    spike_count: int
    total_hops: int


@dataclass
class SanafeArchGeometry:
    """Lightweight 2D-mesh description for the GUI floorplan view.

    Captures only what the frontend needs to render cores in their
    physical positions: total mesh dimensions and per-tile (x, y)
    placement.  The per-tile core list lives on ``SanafeTileRecord``
    so each tile carries its own cores explicitly — no need to
    duplicate it here.
    """

    width: int
    height: int
    tiles_xy: List[List[int]] = field(default_factory=list)  # [(x, y)] indexed by tile_index


@dataclass
class SanafeNocLinkLoad:
    """Per-mesh-edge packet count for the NoC congestion heatmap.

    A "mesh edge" is a single hop between two physically-adjacent tiles
    in the NoC (north / east / south / west).  Aggregated by routing
    every packet through XY-routing — first travel along x, then along
    y — so the load on every intermediate edge is counted, not just
    the (src_tile, dst_tile) endpoints.
    """

    from_x: int
    from_y: int
    to_x: int
    to_y: int
    packet_count: int


@dataclass
class SanafeCycleEnergyPoint:
    """One row of the energy-waterfall: per-cycle event-driven energy split.

    Reconstructed from per-cycle event counts × preset constants — SANA-FE
    doesn't itself report per-cycle energy breakdowns, but we have the
    raw counts (spike trace + message trace) and the YAML constants,
    so we can produce a faithful breakdown for the GUI.
    """

    cycle: int
    synapse_j: float
    dendrite_j: float
    soma_j: float
    network_j: float
    total_j: float


@dataclass
class SanafeCascadePoint:
    """One row of the latency-cascade timeline: per-cycle firings per depth.

    ``depth`` is the HCM core-latency layer (depth-0 = input pool,
    depth-1 = first consumers, …).  The cascade-timeline view stacks
    one bar per depth and shows the cycle-by-cycle firing pattern, so
    users can see the cascade propagating through the network.
    """

    cycle: int
    depth: int
    firings: int


@dataclass
class SanafeCriticalCore:
    """Per-cycle critical-core: the core whose event load drove sim_time.

    Approximated by per-cycle event count (firings + incoming spikes)
    — SANA-FE's actual ``sim_time = max(neuron_processing,
    message_processing)`` is computed from the longest event chain,
    and the busiest core is the strongest proxy for that.
    """

    cycle: int
    core_index: int
    event_count: int


@dataclass
class SanafeConnectivityEdge:
    """Static connectivity edge: ``(src_core, dst_core)`` with summed |w|.

    Built from ``HardCore.axon_sources`` × ``core_matrix`` once per
    segment; doesn't depend on simulation activity.  Powers the
    "connectivity overlay" view that shows routing complexity even
    on idle networks.
    """

    src_core: int
    dst_core: int
    weight_sum_abs: float
    fan_count: int


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
        from mimarsinan.chip_simulation.ttfs_recorder import (
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
                from mimarsinan.chip_simulation.ttfs_recorder import (
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
