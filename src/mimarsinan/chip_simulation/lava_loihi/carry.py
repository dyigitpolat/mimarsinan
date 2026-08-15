"""lava's verbatim pass carry: the output-raster gather and the replay imports.

The lava twin of ``sanafe/runner/carry.py`` and ``simulation_runner/carry.py``:
lava is host-scheduled, so ``core_output_spikes`` already holds every core's
emissions host-side and extraction is a windowed gather — no lava process
changes. Replay rides the shared ``apply_carried_input`` on the encoded train.
"""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.spiking.hybrid.carry import (
    apply_carried_input,
    publish_carried_trains,
)

#: Re-exported so the segment runner reads gather + replay from one seam.
__all__ = ["_output_raster", "apply_carried_input", "publish_carried_trains"]


def _output_raster(
    seg, timing, core_output_spikes, seg_input_logical, T: int, N: int
) -> np.ndarray:
    """``(N, T, out_size)`` segment-output raster in PRODUCER-LOCAL time.

    The per-cycle twin of the ``seg_out_spikes`` gather above: same spans, but
    each core span windowed to its source core's ``[latency, latency + T)`` —
    which is where ``core_output_spikes`` holds that core's active emissions.
    """
    out_size = len(seg.output_sources)
    raster = np.zeros((N, T, out_size), dtype=np.uint8)
    for sp in compress_spike_sources(seg.output_sources):
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            raster[:, :, d0:d1] = 1
            continue
        if sp.kind == "input":
            raster[:, :, d0:d1] = (
                seg_input_logical[int(sp.src_start):int(sp.src_end), :, :T]
                .transpose(1, 2, 0).astype(np.uint8))
            continue
        core = seg.cores[int(sp.src_core)]
        latency = timing.core_latency(core)
        raster[:, :, d0:d1] = (
            core_output_spikes[int(sp.src_core)][
                int(sp.src_start):int(sp.src_end), :, latency:latency + T]
            .transpose(1, 2, 0).astype(np.uint8))
    return raster
