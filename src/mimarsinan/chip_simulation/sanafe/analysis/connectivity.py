"""SANA-FE trace, energy, and connectivity analysis helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.analysis.trace import (
    _group_name,
    _spike_event_group_and_index,
)
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCascadePoint,
    SanafeConnectivityEdge,
    SanafeCriticalCore,
)


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
