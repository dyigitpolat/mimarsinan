"""Single-spike (latched TTFS) per-cycle output decode, split from the reference loop."""

from __future__ import annotations

import torch


def single_spike_output_step(
    output_counts, out_arrival, output_spans, cores, *,
    cycle: int, T: int, buffers, input_spikes,
) -> None:
    """One cycle of the latched decode, moved VERBATIM from the reference loop.

    Count each latched source only within its own ``[src_lat, src_lat + T)``
    window, else shallow sources overcount and saturate.
    """
    for sp in output_spans:
        d0 = int(sp.dst_start)
        d1 = int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            if cycle < T:
                output_counts[:, d0:d1] += 1.0
            continue
        if sp.kind == "input":
            if cycle < T:
                torch.maximum(
                    out_arrival[:, d0:d1],
                    input_spikes[:, int(sp.src_start):int(sp.src_end)],
                    out=out_arrival[:, d0:d1],
                )
                output_counts[:, d0:d1] += out_arrival[:, d0:d1]
            continue
        src_lat = cores[int(sp.src_core)].latency
        if src_lat is None:
            continue
        if cycle < int(src_lat) or cycle >= int(src_lat) + T:
            continue
        torch.maximum(
            out_arrival[:, d0:d1],
            buffers[int(sp.src_core)][:, int(sp.src_start):int(sp.src_end)],
            out=out_arrival[:, d0:d1],
        )
        output_counts[:, d0:d1] += out_arrival[:, d0:d1]
