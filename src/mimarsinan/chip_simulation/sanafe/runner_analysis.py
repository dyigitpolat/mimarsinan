"""SANA-FE trace, energy, and connectivity analysis helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeConnectivityEdge,
    SanafeCriticalCore,
    SanafeCycleEnergyPoint,
    SanafeEnergyBreakdown,
    SanafeNocLink,
    SanafeNocLinkLoad,
)


_CORE_GROUP_RE = re.compile(r"^core(\d+)$")
_LIF_SPIKE_GROUP_RE = re.compile(r"^core(\d+)$")
_INPUT_SPIKE_GROUP_RE = re.compile(r"^core(\d+)_(in|on)$")


def _ttfs_potential_trace_group_names(chip: Any) -> List[str]:
    """``coreN`` group names in lex order (matches SANA-FE ``std::map`` iteration)."""
    groups = getattr(chip, "mapped_neuron_groups", None) or {}
    return [
        str(gn) for gn in sorted(groups.keys())
        if _CORE_GROUP_RE.match(str(gn))
    ]


def _read_ttfs_core_activations(
    chip: Any,
    core_idx: int,
    n_neurons: int,
    results: Dict[str, Any],
) -> np.ndarray:
    """Read per-neuron TTFS activations from ``potential_trace`` (plugin somas)."""
    out = np.zeros(n_neurons, dtype=np.float64)
    target = f"core{core_idx}"
    trace = results.get("potential_trace")
    if trace is None or len(trace) == 0:
        return out
    row = trace[-1]
    if row is None or len(row) == 0:
        return out
    groups = getattr(chip, "mapped_neuron_groups", None) or {}
    pos = 0
    for gn in _ttfs_potential_trace_group_names(chip):
        group = groups.get(gn)
        n_logged = len(group) if group is not None else 0
        if gn == target:
            n_copy = min(n_neurons, n_logged, len(row) - pos)
            if n_copy > 0:
                out[:n_copy] = np.asarray(row[pos : pos + n_copy], dtype=np.float64)
            return out
        pos += n_logged
    return out


def _group_name(group: Any) -> str:
    """Return a NeuronGroup name (``get_name()`` or ``.name``)."""
    if hasattr(group, "get_name"):
        return group.get_name()
    return group.name


def _group_name_to_size(net: Any) -> Dict[str, int]:
    """Build ``{group_name: size}`` from dict- or list-shaped ``net.groups``."""
    groups = net.groups
    if isinstance(groups, dict):
        return {name: len(g) for name, g in groups.items()}
    return {_group_name(g): len(g) for g in groups}


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


def _spike_event_group_and_index(event: Any) -> Optional[Tuple[str, int]]:
    """Parse a spike-trace entry (``NeuronAddress`` or ``group.idx`` string)."""
    gn = getattr(event, "group_name", None)
    if gn is not None:
        no = getattr(event, "neuron_offset", None)
        if no is None:
            return (str(gn), 0)
        try:
            return (str(gn), int(no))
        except (TypeError, ValueError):
            return (str(gn), 0)
    s = str(event)
    if "." not in s:
        return None
    group_name, idx_str = s.rsplit(".", 1)
    try:
        return (group_name, int(idx_str))
    except ValueError:
        return None


def _hardcore_index_from_spike_group(group_name: str) -> Optional[int]:
    """Map ``coreN``, ``coreN_in``, or ``coreN_on`` to HCM core index."""
    m = _INPUT_SPIKE_GROUP_RE.match(group_name)
    if m:
        return int(m.group(1))
    m = _LIF_SPIKE_GROUP_RE.match(group_name)
    if m:
        return int(m.group(1))
    return None


def _spike_trace_to_group_counts(
    spike_trace: list,
    *,
    group_sizes: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Tally spike-trace events into per-group counts; return (counts, parse_skipped)."""
    counts: Dict[str, np.ndarray] = {
        name: np.zeros(size, dtype=np.int64) for name, size in group_sizes.items()
    }
    parse_skipped = 0
    for events in spike_trace:
        for event in events:
            parsed = _spike_event_group_and_index(event)
            if parsed is None:
                parse_skipped += 1
                continue
            group_name, idx = parsed
            arr = counts.get(group_name)
            if arr is None or idx < 0 or idx >= arr.size:
                parse_skipped += 1
                continue
            arr[idx] += 1
    return counts, parse_skipped


def _lif_and_input_spike_totals(
    group_spike_counts: Dict[str, np.ndarray],
) -> Tuple[int, int]:
    """Sum LIF (``coreN``) vs input-path (``coreN_in`` / ``coreN_on``) spike counts."""
    lif_total = 0
    input_total = 0
    for group_name, arr in group_spike_counts.items():
        n = int(arr.sum())
        if n <= 0:
            continue
        if _INPUT_SPIKE_GROUP_RE.match(group_name):
            input_total += n
        elif _LIF_SPIKE_GROUP_RE.match(group_name):
            lif_total += n
    return lif_total, input_total


def _input_spikes_per_core(
    group_spike_counts: Dict[str, np.ndarray],
) -> Dict[int, int]:
    """Per-HCM-core spike count on input / always-on neuron groups."""
    out: Dict[int, int] = {}
    for group_name, arr in group_spike_counts.items():
        if not _INPUT_SPIKE_GROUP_RE.match(group_name):
            continue
        hci = _hardcore_index_from_spike_group(group_name)
        if hci is None:
            continue
        out[hci] = out.get(hci, 0) + int(arr.sum())
    return out


def _pack_spike_trace_matrix(
    spike_trace: list, groups: Any,
) -> Optional[np.ndarray]:
    """Spike trace → ``(sum_neurons, T)`` matrix; ``None`` if empty."""
    if not spike_trace:
        return None
    offsets: Dict[str, int] = {}
    cursor = 0
    iterable = (groups.items() if isinstance(groups, dict)
                else ((_group_name(g), g) for g in groups))
    for name, g in iterable:
        offsets[name] = cursor
        cursor += len(g)
    if cursor == 0:
        return None
    T = len(spike_trace)
    mat = np.zeros((cursor, T), dtype=np.uint8)
    for t, events in enumerate(spike_trace):
        for event in events:
            parsed = _spike_event_group_and_index(event)
            if parsed is None:
                continue
            group_name, idx = parsed
            if group_name not in offsets:
                continue
            row = offsets[group_name] + idx
            if 0 <= row < cursor:
                mat[row, t] = 1
    return mat


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


def _compute_tile_packets_per_cycle(
    message_trace: Any,
) -> List[Dict[int, int]]:
    """Per-cycle packet count per destination ``tile_id``."""
    if not message_trace:
        return []
    out: List[Dict[int, int]] = []
    for events in message_trace:
        bins: Dict[int, int] = {}
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            dt = int(ev.get("dest_tile_id", -1))
            if dt < 0:
                continue
            bins[dt] = bins.get(dt, 0) + 1
        out.append(bins)
    return out


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


def _flatten_message_trace(message_trace: Any) -> Optional[List[dict]]:
    """Flatten per-cycle message lists; drop placeholder entries."""
    if not message_trace:
        return None
    flat: List[dict] = []
    for events in message_trace:
        for ev in events:
            if isinstance(ev, dict):
                if ev.get("placeholder"):
                    continue
                flat.append({k: (float(v) if isinstance(v, float) else v)
                             for k, v in ev.items()})
    return flat or None


def _aggregate_noc_links(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLink]:
    """Aggregate cross-tile message trace into directed NoC links."""
    if not message_trace:
        return []
    bins: Dict[Tuple[int, int], Dict[str, int]] = {}
    src_coord: Dict[int, Tuple[int, int]] = {}
    dst_coord: Dict[int, Tuple[int, int]] = {}
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            src_t = int(ev.get("src_tile_id", -1))
            dst_t = int(ev.get("dest_tile_id", -1))
            if src_t < 0 or dst_t < 0 or src_t == dst_t:
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            src_coord[src_t] = (sx, sy)
            dst_coord[dst_t] = (dx, dy)
            slot = bins.setdefault(
                (src_t, dst_t),
                {"packets": 0, "spikes": 0, "hops": 0},
            )
            slot["packets"] += 1
            slot["spikes"] += int(ev.get("spikes", 0) or 0)
            slot["hops"] += int(ev.get("hops", 0) or 0)
    out: List[SanafeNocLink] = []
    for (src_t, dst_t), b in sorted(bins.items()):
        sx, sy = src_coord.get(src_t, (-1, -1))
        dx, dy = dst_coord.get(dst_t, (-1, -1))
        if geom is not None:
            if (sx < 0 or sy < 0) and 0 <= src_t < len(geom.tiles_xy):
                sx, sy = geom.tiles_xy[src_t]
            if (dx < 0 or dy < 0) and 0 <= dst_t < len(geom.tiles_xy):
                dx, dy = geom.tiles_xy[dst_t]
        out.append(SanafeNocLink(
            src_tile=src_t, dst_tile=dst_t,
            src_x=int(sx), src_y=int(sy),
            dst_x=int(dx), dst_y=int(dy),
            packet_count=int(b["packets"]),
            spike_count=int(b["spikes"]),
            total_hops=int(b["hops"]),
        ))
    return out


def _aggregate_noc_link_load(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLinkLoad]:
    """Per-mesh-edge packet load via XY routing."""
    if not message_trace:
        return []
    counts: Dict[Tuple[int, int, int, int], int] = {}
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            if sx < 0 or sy < 0 or dx < 0 or dy < 0:
                continue
            cx, cy = sx, sy
            step_x = 1 if dx > sx else -1 if dx < sx else 0
            step_y = 1 if dy > sy else -1 if dy < sy else 0
            while cx != dx:
                nx = cx + step_x
                k = (cx, cy, nx, cy)
                counts[k] = counts.get(k, 0) + 1
                cx = nx
            while cy != dy:
                ny = cy + step_y
                k = (cx, cy, cx, ny)
                counts[k] = counts.get(k, 0) + 1
                cy = ny
    out: List[SanafeNocLinkLoad] = []
    for (fx, fy, tx, ty), n in sorted(counts.items()):
        out.append(SanafeNocLinkLoad(
            from_x=fx, from_y=fy, to_x=tx, to_y=ty, packet_count=int(n),
        ))
    return out


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


def _build_neuron_to_core_map(
    net: Any, hcm: Any,
) -> Tuple[Dict[str, int], List[int]]:
    """Map spike-trace group names and global rows to HardCore indices."""
    group_to_core: Dict[str, int] = {}
    for core_idx, core in enumerate(hcm.cores):
        group_to_core[f"core{core_idx}"] = core_idx
        group_to_core[f"core{core_idx}_in"] = core_idx
        group_to_core[f"core{core_idx}_on"] = core_idx
    row_to_core: List[int] = []
    groups = net.groups
    iterable = (groups.items() if isinstance(groups, dict)
                else ((_group_name(g), g) for g in groups))
    for name, g in iterable:
        cidx = group_to_core.get(name, -1)
        row_to_core.extend([cidx] * len(g))
    return group_to_core, row_to_core


def _compute_cascade_timeline(
    spike_trace: list, *, net: Any, hcm: Any,
) -> List[SanafeCascadePoint]:
    """Bucket per-cycle firings by core latency depth."""
    if not spike_trace or hcm is None or not getattr(hcm, "cores", None):
        return []
    core_latency = [
        int(c.latency) if getattr(c, "latency", None) is not None else 0
        for c in hcm.cores
    ]
    group_to_core, _ = _build_neuron_to_core_map(net, hcm)
    out: List[SanafeCascadePoint] = []
    for cycle, evs in enumerate(spike_trace):
        bucket: Dict[int, int] = {}
        for ev in evs:
            parsed = _spike_event_group_and_index(ev)
            if parsed is None:
                continue
            gname, _ = parsed
            core_idx = group_to_core.get(gname, -1)
            if core_idx < 0 or core_idx >= len(core_latency):
                continue
            d = core_latency[core_idx]
            bucket[d] = bucket.get(d, 0) + 1
        for d, n in sorted(bucket.items()):
            out.append(SanafeCascadePoint(cycle=int(cycle), depth=int(d), firings=int(n)))
    return out


def _compute_critical_cores(
    spike_trace: list, message_trace: Any, *, net: Any, hcm: Any,
) -> List[SanafeCriticalCore]:
    """Per-cycle core with highest firings + incoming spike load."""
    if hcm is None or not getattr(hcm, "cores", None):
        return []
    n_cores = len(hcm.cores)
    T_eff = max(
        len(spike_trace) if spike_trace else 0,
        len(message_trace) if message_trace else 0,
    )
    if T_eff <= 0 or n_cores == 0:
        return []
    group_to_core, _ = _build_neuron_to_core_map(net, hcm)
    fires = np.zeros((n_cores, T_eff), dtype=np.int64)
    if spike_trace:
        for cycle, evs in enumerate(spike_trace[:T_eff]):
            for ev in evs:
                parsed = _spike_event_group_and_index(ev)
                if parsed is None:
                    continue
                gname, _ = parsed
                core_idx = group_to_core.get(gname, -1)
                if 0 <= core_idx < n_cores:
                    fires[core_idx, cycle] += 1
    incoming = np.zeros((n_cores, T_eff), dtype=np.int64)
    if message_trace:
        for cycle, evs in enumerate(message_trace[:T_eff]):
            for ev in evs:
                if not isinstance(ev, dict) or ev.get("placeholder"):
                    continue
                dt = int(ev.get("dest_tile_id", -1))
                dc = int(ev.get("dest_core_id", -1))
                if dt < 0 or dc < 0:
                    continue
                idx = dt * (1 + dc) + dc
                if 0 <= idx < n_cores:
                    incoming[idx, cycle] += int(ev.get("spikes", 0) or 0)
    score = fires + incoming
    out: List[SanafeCriticalCore] = []
    for cycle in range(T_eff):
        col = score[:, cycle]
        if col.sum() == 0:
            continue
        core_idx = int(col.argmax())
        out.append(SanafeCriticalCore(
            cycle=int(cycle), core_index=core_idx,
            event_count=int(col[core_idx]),
        ))
    return out


def _group_row_offsets(groups: Any) -> Dict[str, int]:
    """Group name → starting row (same order as ``_pack_spike_trace_matrix``)."""
    offsets: Dict[str, int] = {}
    cursor = 0
    iterable = (groups.items() if isinstance(groups, dict)
                else ((_group_name(g), g) for g in groups))
    for name, g in iterable:
        offsets[name] = cursor
        cursor += len(g)
    return offsets


def _compute_noc_traffic_per_cycle(message_trace: Any) -> List[List[List[int]]]:
    """Per-cycle ``[src_x, src_y, dst_x, dst_y, count]`` quintuples."""
    if not message_trace:
        return []
    out: List[List[List[int]]] = []
    for events in message_trace:
        bins: Dict[Tuple[int, int, int, int], int] = {}
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            if sx < 0 or sy < 0 or dx < 0 or dy < 0:
                continue
            if sx == dx and sy == dy:
                continue
            k = (sx, sy, dx, dy)
            bins[k] = bins.get(k, 0) + 1
        out.append([[fx, fy, tx, ty, n] for (fx, fy, tx, ty), n in bins.items()])
    return out


def _compute_connectivity_edges(hcm: Any) -> List[SanafeConnectivityEdge]:
    """Sum ``|weight|`` per ``(src_core, dst_core)`` from axon wiring."""
    if hcm is None or not getattr(hcm, "cores", None):
        return []
    bins: Dict[Tuple[int, int], Dict[str, float]] = {}
    for dst_idx, core in enumerate(hcm.cores):
        ax_per_core = int(core.axons_per_core)
        avail = int(getattr(core, "available_axons", 0))
        used_ax = max(ax_per_core - avail, 0)
        cm = getattr(core, "core_matrix", None)
        if cm is None or used_ax <= 0:
            continue
        for a in range(used_ax):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_input_", False) or getattr(src, "is_always_on_", False):
                continue
            src_core = int(src.core_)
            try:
                w_col = cm[a, :]
            except Exception:
                continue
            w_abs = float(np.abs(np.asarray(w_col, dtype=np.float64)).sum())
            if w_abs == 0.0:
                continue
            slot = bins.setdefault(
                (src_core, dst_idx), {"w": 0.0, "n": 0},
            )
            slot["w"] += w_abs
            slot["n"] += 1
    out: List[SanafeConnectivityEdge] = []
    for (src, dst), b in sorted(bins.items()):
        out.append(SanafeConnectivityEdge(
            src_core=int(src), dst_core=int(dst),
            weight_sum_abs=float(b["w"]), fan_count=int(b["n"]),
        ))
    return out
