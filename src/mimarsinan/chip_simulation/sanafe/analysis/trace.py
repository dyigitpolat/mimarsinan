"""SANA-FE trace, energy, and connectivity analysis helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
