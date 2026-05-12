"""Per-event energy & latency presets for SANA-FE architecture synthesis.

Each preset is a dict of seven numbers — energy per event (J) for the
four pipeline stages SANA-FE bills (synapse, dendrite, soma, network) and
latency (s) for the three time-dominating stages (synapse, soma,
network).  ``build_architecture`` injects these into the per-core
attribute kwargs of ``sanafe.Architecture.create_core``.

The Loihi numbers are the public per-event costs cited in Davies et al.
2018 ("Loihi: a neuromorphic manycore processor with on-chip learning",
IEEE Micro).  TrueNorth numbers come from Merolla et al. 2014 ("A
million spiking-neuron integrated circuit ...", Science).  Both serve as
starting points; users can override with ``arch_preset="custom"`` and a
hand-tuned YAML.
"""

from __future__ import annotations

from typing import Dict, TypedDict


class PerEventEnergy(TypedDict):
    """SANA-FE per-event energy (J) and latency (s) costs."""
    synapse_energy_j: float
    dendrite_energy_j: float
    soma_energy_j: float
    network_energy_j: float
    synapse_latency_s: float
    soma_latency_s: float
    network_latency_s: float


# Loihi 1 reference numbers (Davies 2018, Tables I-II — approximate, public).
LOIHI_PRESET: PerEventEnergy = {
    "synapse_energy_j":  3.1e-12,
    "dendrite_energy_j": 1.3e-12,
    "soma_energy_j":     5.3e-12,
    "network_energy_j":  1.0e-12,
    "synapse_latency_s": 1.5e-9,
    "soma_latency_s":    2.0e-9,
    "network_latency_s": 5.0e-9,
}

# TrueNorth reference numbers (Merolla 2014; ICONS / SANA-FE bundled approximations).
TRUENORTH_PRESET: PerEventEnergy = {
    "synapse_energy_j":  2.0e-13,
    "dendrite_energy_j": 1.0e-13,
    "soma_energy_j":     2.5e-13,
    "network_energy_j":  5.0e-14,
    "synapse_latency_s": 1.0e-9,
    "soma_latency_s":    1.0e-9,
    "network_latency_s": 4.0e-9,
}


PRESETS: Dict[str, PerEventEnergy] = {
    "loihi": LOIHI_PRESET,
    "truenorth": TRUENORTH_PRESET,
}
