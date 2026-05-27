"""SANA-FE segment diagnostics and trace summarization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeConnectivityEdge,
    SanafeCoreRecord,
)


def _summarize_message_trace(message_trace: Any) -> Dict[str, int]:
    """Classify non-placeholder message-trace events."""
    inter = intra = input_path = 0
    if not message_trace:
        return {
            "inter_tile_packets": 0,
            "intra_tile_packets": 0,
            "input_path_packets": 0,
        }
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            src_t = int(ev.get("src_tile_id", -1))
            dst_t = int(ev.get("dest_tile_id", -1))
            gn = str(ev.get("src_neuron_group_id", ""))
            if gn.endswith("_in") or gn.endswith("_on"):
                input_path += 1
            if src_t >= 0 and dst_t >= 0 and src_t != dst_t:
                inter += 1
            elif src_t >= 0 and dst_t >= 0:
                intra += 1
    return {
        "inter_tile_packets": inter,
        "intra_tile_packets": intra,
        "input_path_packets": input_path,
    }


def _count_cross_tile_connectivity_edges(
    connectivity: List[SanafeConnectivityEdge],
    *,
    cores_per_tile: int,
) -> int:
    cpt = max(int(cores_per_tile), 1)
    return sum(
        1 for e in connectivity
        if e.src_core // cpt != e.dst_core // cpt
    )


def _build_spike_capture_warning(
    *,
    chip_spike_count: int,
    lif_spike_count: int,
    input_path_packets: int,
    spike_trace_parse_skipped: int,
    ttfs_hardware_active: int = 0,
    ttfs_event_active: int = 0,
    ttfs_mismatch_count: int = 0,
) -> Optional[str]:
    if ttfs_mismatch_count > 0:
        return (
            f"TTFS activations present on {ttfs_hardware_active:,} core(s) "
            f"(contract/hardware) but only {ttfs_event_active:,} core(s) emitted "
            f"hardware spike/message events ({ttfs_mismatch_count:,} mismatch). "
            "Inter-tile NoC routes require soma fired events; re-run after the "
            "TTFS event-emission fix."
        )
    if chip_spike_count <= 0:
        return None
    if lif_spike_count == 0 and input_path_packets > 0:
        return (
            "Chip reported "
            f"{chip_spike_count:,} spikes but LIF core groups logged none; "
            f"activity is on input-path neurons ({input_path_packets:,} NoC packets "
            "from coreN_in/coreN_on)."
        )
    if spike_trace_parse_skipped > 0 and lif_spike_count == 0:
        return (
            f"Spike trace had {spike_trace_parse_skipped:,} unparsed events; "
            "per-core spike counts may be incomplete."
        )
    if lif_spike_count == 0 and chip_spike_count > 0:
        return (
            f"Chip aggregate spikes={chip_spike_count:,} but no LIF group spikes "
            "were attributed in the trace."
        )
    return None


def _compute_ttfs_activity_diagnostics(
    contract_ttfs_cores: List[Any],
    per_core_records: List[SanafeCoreRecord],
) -> Dict[str, int]:
    """Compare TTFS contract activations, hardware readout, and spike events."""
    contract_by: Dict[int, np.ndarray] = {}
    for entry in contract_ttfs_cores:
        act = np.asarray(getattr(entry, "output_activation", []), dtype=np.float64)
        contract_by[int(entry.core_index)] = act

    per_by = {int(c.core_index): c for c in per_core_records}
    indices = set(contract_by.keys()) | set(per_by.keys())

    contract_active = hardware_active = event_active = mismatch = 0
    for ci in indices:
        c_act = contract_by.get(ci)
        rec = per_by.get(ci)
        c_has = c_act is not None and c_act.size > 0 and bool(np.any(c_act > 0))
        h_has = False
        if rec is not None and rec.output_activation is not None:
            h_act = np.asarray(rec.output_activation, dtype=np.float64)
            h_has = h_act.size > 0 and bool(np.any(h_act > 0))
        e_has = rec is not None and int(rec.spikes_fired) > 0
        if c_has:
            contract_active += 1
        if h_has:
            hardware_active += 1
        if e_has:
            event_active += 1
        if (c_has or h_has) and not e_has:
            mismatch += 1

    return {
        "ttfs_contract_active_cores": contract_active,
        "ttfs_hardware_active_cores": hardware_active,
        "ttfs_event_active_cores": event_active,
        "ttfs_activation_event_mismatch_count": mismatch,
    }


def _pack_potential_trace(potential_trace: Any) -> Optional[np.ndarray]:
    """Potential trace → ``(n_logged, T)``; ``None`` if empty or malformed."""
    if not potential_trace:
        return None
    arr = np.asarray(potential_trace, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return arr.T
