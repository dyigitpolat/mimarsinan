"""SANA-FE trace, energy, and connectivity analysis helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCycleEnergyPoint,
    SanafeEnergyBreakdown,
)

def _per_core_energy_sanafe(
    *,
    preset: Dict[str, float],
    n_neurons: int,
    T_active: int,
    T_eff: int,
    incoming_spikes: int,
    firings: int,
    packets_in: int,
    packets_out: int,
) -> SanafeEnergyBreakdown:
    """Per-core energy mirroring SANA-FE ``sim_calculate_core_energy``."""
    if n_neurons <= 0:
        return SanafeEnergyBreakdown.zero()
    syn = float(preset.get("synapse_energy_j", 0.0)) * int(incoming_spikes)
    dend = float(preset.get("dendrite_energy_j", 0.0)) * n_neurons * int(T_eff)
    soma = (
        float(preset.get("soma_access_energy_j", 0.0)) * n_neurons * int(T_eff)
        + float(preset.get("soma_update_energy_j", 0.0)) * n_neurons * int(T_active)
        + float(preset.get("soma_spike_out_energy_j", 0.0)) * int(firings)
    )
    net = (
        float(preset.get("axon_in_energy_j", 0.0)) * int(packets_in)
        + float(preset.get("axon_out_energy_j", 0.0)) * int(packets_out)
    )
    return SanafeEnergyBreakdown(
        synapse_j=syn, dendrite_j=dend, soma_j=soma, network_j=net,
        total_j=syn + dend + soma + net,
    )


def _per_core_packet_counts(
    message_trace: Any, *, n_cores: int, cores_per_tile: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-core packet in/out counts from the message trace."""
    pkts_in = np.zeros(max(n_cores, 1), dtype=np.int64)
    pkts_out = np.zeros(max(n_cores, 1), dtype=np.int64)
    if not message_trace:
        return pkts_in, pkts_out
    cpt = max(int(cores_per_tile), 1)
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            st = int(ev.get("src_tile_id", -1))
            sc = int(ev.get("src_core_id", -1))
            dt = int(ev.get("dest_tile_id", -1))
            dc = int(ev.get("dest_core_id", -1))
            if st >= 0 and sc >= 0:
                gid = st * cpt + sc
                if 0 <= gid < n_cores:
                    pkts_out[gid] += 1
            if dt >= 0 and dc >= 0:
                gid = dt * cpt + dc
                if 0 <= gid < n_cores:
                    pkts_in[gid] += 1
    return pkts_in, pkts_out


def _energy_share(total: SanafeEnergyBreakdown, *, n_cores: int) -> SanafeEnergyBreakdown:
    """Split run-wide energy evenly across cores."""
    if n_cores <= 0:
        return SanafeEnergyBreakdown.zero()
    f = 1.0 / float(n_cores)
    return SanafeEnergyBreakdown(
        synapse_j=total.synapse_j * f,
        dendrite_j=total.dendrite_j * f,
        soma_j=total.soma_j * f,
        network_j=total.network_j * f,
        total_j=total.total_j * f,
    )
def _compute_cycle_energy_breakdown(
    message_trace: Any,
    spike_trace: list,
    preset: Dict[str, float],
    hcm: Any,
) -> List[SanafeCycleEnergyPoint]:
    """Reconstruct per-cycle energy split from spike/message traces."""
    if not spike_trace and not message_trace:
        return []
    T_eff = max(
        len(spike_trace) if spike_trace else 0,
        len(message_trace) if message_trace else 0,
    )
    if T_eff <= 0:
        return []
    firings_per_cycle = np.zeros(T_eff, dtype=np.int64)
    if spike_trace:
        for c, evs in enumerate(spike_trace[:T_eff]):
            firings_per_cycle[c] = len(evs)
    pkt_per_cycle = np.zeros(T_eff, dtype=np.int64)
    hop_per_cycle = np.zeros(T_eff, dtype=np.int64)
    syn_per_cycle = np.zeros(T_eff, dtype=np.int64)
    dend_targets: List[set] = [set() for _ in range(T_eff)]
    if message_trace:
        for c, evs in enumerate(message_trace[:T_eff]):
            for ev in evs:
                if not isinstance(ev, dict) or ev.get("placeholder"):
                    continue
                pkt_per_cycle[c] += 1
                hop_per_cycle[c] += int(ev.get("hops", 0) or 0)
                syn_per_cycle[c] += int(ev.get("spikes", 0) or 0)
                dst_core = int(ev.get("dest_core_id", -1))
                dst_neuron = int(ev.get("dest_neuron_offset",
                                         ev.get("dest_axon_id", -1)))
                if dst_core >= 0 and dst_neuron >= 0:
                    dend_targets[c].add((dst_core, dst_neuron))
    total_live_neurons = 0
    if hcm is not None and getattr(hcm, "cores", None):
        for c in hcm.cores:
            np_used = int(c.neurons_per_core) - int(getattr(c, "available_neurons", 0))
            if np_used > 0:
                total_live_neurons += np_used
    out: List[SanafeCycleEnergyPoint] = []
    syn_e = float(preset.get("synapse_energy_j", 0.0))
    dend_e = float(preset.get("dendrite_energy_j", 0.0))
    soma_access_e = float(preset.get("soma_access_energy_j", 0.0))
    soma_update_e = float(preset.get("soma_update_energy_j", 0.0))
    soma_spike_e = float(preset.get("soma_spike_out_energy_j", 0.0))
    axon_in_e = float(preset.get("axon_in_energy_j", 0.0))
    axon_out_e = float(preset.get("axon_out_energy_j", 0.0))
    hop_e = float(preset.get("tile_hop_energy_j", 0.0))
    for c in range(T_eff):
        synapse_j = syn_e * int(syn_per_cycle[c])
        dendrite_j = dend_e * len(dend_targets[c])
        soma_j = (
            soma_access_e * total_live_neurons
            + soma_update_e * total_live_neurons
            + soma_spike_e * int(firings_per_cycle[c])
        )
        network_j = (
            axon_in_e * int(pkt_per_cycle[c])
            + axon_out_e * int(pkt_per_cycle[c])
            + hop_e * int(hop_per_cycle[c])
        )
        total = synapse_j + dendrite_j + soma_j + network_j
        out.append(SanafeCycleEnergyPoint(
            cycle=c,
            synapse_j=synapse_j, dendrite_j=dendrite_j,
            soma_j=soma_j, network_j=network_j, total_j=total,
        ))
    return out
