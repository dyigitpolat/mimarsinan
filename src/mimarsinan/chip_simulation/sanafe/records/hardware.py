from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown


@dataclass
class SanafeCoreRecord:
    """Per-``HardCore`` rich record.

    Spike-count + bookkeeping fields project losslessly to
    :class:`spike_recorder.CoreSpikeCounts` in :meth:`to_hcm_subset`.
    """

    core_index: int
    n_neurons: int
    n_axons_used: int
    core_latency: int
    has_hardware_bias: bool
    n_always_on_axons: int
    spikes_fired: int
    input_spike_count: np.ndarray
    output_spike_count: np.ndarray
    energy: SanafeEnergyBreakdown
    spike_raster: Optional[np.ndarray] = None
    output_activation: Optional[np.ndarray] = None
    input_neuron_spikes_fired: int = 0


@dataclass
class SanafeTileRecord:
    """Per-SANA-FE-tile aggregate (energy/spikes/packets); GUI-only, not the parity gate.

    ``mesh_x``/``mesh_y`` default ``-1`` (unknown → floorplan falls back to row-major).
    """

    tile_index: int
    cores: List[int]
    energy: SanafeEnergyBreakdown
    spikes_fired: int
    packets_sent: int
    mesh_x: int = -1
    mesh_y: int = -1


@dataclass
class SanafeNocLink:
    """Aggregated NoC traffic for one directed ``(src_tile, dst_tile)`` pair."""

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
    """Lightweight 2D-mesh description (dims + per-tile x,y) for the GUI floorplan."""

    width: int
    height: int
    tiles_xy: List[List[int]] = field(default_factory=list)


@dataclass
class SanafeNocLinkLoad:
    """Per-mesh-edge packet count (XY-routed) for the NoC congestion heatmap."""

    from_x: int
    from_y: int
    to_x: int
    to_y: int
    packet_count: int


@dataclass
class SanafeCycleEnergyPoint:
    """One energy-waterfall row: per-cycle energy split reconstructed from counts × preset."""

    cycle: int
    synapse_j: float
    dendrite_j: float
    soma_j: float
    network_j: float
    total_j: float


@dataclass
class SanafeCascadePoint:
    """One cascade-timeline row: per-cycle firings at HCM core-latency ``depth``."""

    cycle: int
    depth: int
    firings: int


@dataclass
class SanafeCriticalCore:
    """Per-cycle critical core (busiest by firings + incoming spikes), a sim_time proxy."""

    cycle: int
    core_index: int
    event_count: int


@dataclass
class SanafeConnectivityEdge:
    """Static ``(src_core, dst_core)`` edge with summed ``|w|`` (activity-independent)."""

    src_core: int
    dst_core: int
    weight_sum_abs: float
    fan_count: int

