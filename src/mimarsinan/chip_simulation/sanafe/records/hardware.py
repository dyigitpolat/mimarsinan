from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown

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

