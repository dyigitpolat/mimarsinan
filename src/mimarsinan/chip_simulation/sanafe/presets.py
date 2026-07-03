"""Per-event energy & latency presets for SANA-FE architecture synthesis."""

from __future__ import annotations

from typing import Dict, TypedDict


class PerEventEnergy(TypedDict):
    """SANA-FE-flavoured per-event energy (J) and latency (s) costs."""
    tile_hop_energy_j: float
    tile_hop_latency_s: float
    axon_in_energy_j: float
    axon_in_latency_s: float
    axon_out_energy_j: float
    axon_out_latency_s: float
    synapse_energy_j: float
    synapse_latency_s: float
    dendrite_energy_j: float
    dendrite_latency_s: float
    soma_access_energy_j: float
    soma_access_latency_s: float
    soma_update_energy_j: float
    soma_update_latency_s: float
    soma_spike_out_energy_j: float
    soma_spike_out_latency_s: float


# Loihi 1 reference numbers (Davies 2018; public).
LOIHI_PRESET: PerEventEnergy = {
    "tile_hop_energy_j":         3.5e-12,
    "tile_hop_latency_s":        5.0e-9,
    "axon_in_energy_j":          0.0,
    "axon_in_latency_s":         16.0e-9,
    "axon_out_energy_j":         111.0e-12,
    "axon_out_latency_s":        5.1e-9,
    "synapse_energy_j":          35.5e-12,
    "synapse_latency_s":         3.8e-9,
    "dendrite_energy_j":         0.0,
    "dendrite_latency_s":        0.0,
    "soma_access_energy_j":      51.2e-12,
    "soma_access_latency_s":     6.0e-9,
    "soma_update_energy_j":      21.6e-12,
    "soma_update_latency_s":     3.7e-9,
    "soma_spike_out_energy_j":   69.3e-12,
    "soma_spike_out_latency_s":  30.0e-9,
}

# TrueNorth reference numbers (Merolla 2014; bundled approximations).
TRUENORTH_PRESET: PerEventEnergy = {
    "tile_hop_energy_j":         5.0e-14,
    "tile_hop_latency_s":        4.0e-9,
    "axon_in_energy_j":          0.0,
    "axon_in_latency_s":         5.0e-9,
    "axon_out_energy_j":         3.0e-13,
    "axon_out_latency_s":        5.0e-9,
    "synapse_energy_j":          2.0e-13,
    "synapse_latency_s":         1.0e-9,
    "dendrite_energy_j":         0.0,
    "dendrite_latency_s":        0.0,
    "soma_access_energy_j":      1.0e-13,
    "soma_access_latency_s":     1.0e-9,
    "soma_update_energy_j":      2.5e-13,
    "soma_update_latency_s":     1.0e-9,
    "soma_spike_out_energy_j":   3.0e-13,
    "soma_spike_out_latency_s":  5.0e-9,
}

PRESETS: Dict[str, PerEventEnergy] = {
    "loihi": LOIHI_PRESET,
    "truenorth": TRUENORTH_PRESET,
}


# Hardware-unit names referenced by net_synth/arch_synth as strings — public surface, don't rename.
SOMA_LIF_NAME = "lif"
SOMA_TTFS_CONTINUOUS_NAME = "ttfs_continuous"
SOMA_TTFS_QUANTIZED_NAME = "ttfs_quantized"
SOMA_TTFS_CYCLE_NAME = "ttfs_cycle"
SOMA_TTFS_CASCADE_NAME = "ttfs_cascade"
SOMA_INPUT_RANGE_NAME = "inputs"
SYNAPSE_NAME = "dense_syn"
DENDRITE_NAME = "dend"
AXON_IN_NAME = "ax_in"
AXON_OUT_NAME = "ax_out"
