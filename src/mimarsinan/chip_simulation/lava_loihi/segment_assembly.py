"""Per-core active-input assembly for host-scheduled Lava segment execution."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from mimarsinan.chip_simulation.lava_loihi.timing import _LAVA_DTYPE


def assemble_core_active_input(
    core: Any,
    *,
    core_idx: int,
    N: int,
    T: int,
    latency: int,
    used_ax: int,
    seg_input_logical: np.ndarray,
    core_buffer_spikes: Dict[int, np.ndarray],
) -> np.ndarray:
    """Gather one core's ``(used_ax, N, T)`` input window from segment input and upstream buffers."""
    active_input = np.zeros((used_ax, N, T), dtype=_LAVA_DTYPE)

    for sp in core.get_axon_source_spans():
        d0 = int(sp.dst_start)
        if d0 >= used_ax:
            continue
        end = min(int(sp.dst_end), used_ax)
        take = end - d0
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            active_input[d0:end, :, :] = 1.0
            continue
        if sp.kind == "input":
            s0 = int(sp.src_start)
            active_input[d0:end, :, :] = seg_input_logical[
                s0:s0 + take, :, latency:latency + T,
            ]
            continue

        src_core_id = int(sp.src_core)
        if src_core_id not in core_buffer_spikes:
            raise RuntimeError(
                f"Core {core_idx} depends on core {src_core_id}, "
                "but the source has not been scheduled yet."
            )
        s0 = int(sp.src_start)
        for local_cycle in range(T):
            src_cycle = latency + local_cycle - 1
            if src_cycle < 0:
                continue
            active_input[d0:end, :, local_cycle] = core_buffer_spikes[
                src_core_id
            ][s0:s0 + take, :, src_cycle]

    return active_input
