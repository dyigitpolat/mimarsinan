"""[calculus §16] the synchronized count-domain HCM reference executor."""

from __future__ import annotations

import torch

from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE
from mimarsinan.models.spiking.wire_semantics import lif_count_staircase


def run_neural_segment_counts(
    flow, input_spike_train, *, seg, T, batch_size, device,
) -> torch.Tensor:
    """The synchronized count-domain reference: per core, memb = W·counts +
    bias·T (+ V0·θ) and the emitted count is the strict staircase — bit-equal
    to two-window execution for ANY arrival (§16 theorem), and latch-correct
    for level-gapped consumers (counts have no windows)."""
    cores = seg["cores"]
    counts_in = input_spike_train.sum(dim=0)
    buffers = [
        torch.zeros(
            batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
            device=device, dtype=COMPUTE_DTYPE,
        )
        for c in cores
    ]
    signals = [
        torch.zeros(
            batch_size, max(int(c.axons_per_core - c.available_axons), 1),
            device=device, dtype=COMPUTE_DTYPE,
        )
        for c in cores
    ]
    v0 = float(getattr(flow, "lif_membrane_init", 0.0))
    order = sorted(
        (i for i, c in enumerate(cores) if c.latency is not None),
        key=lambda i: int(cores[i].latency or 0),
    )
    for i in order:
        # Always-on axons deliver one spike per cycle => count T.
        seg["axon_fill_plans"][i].apply(
            signals[i], input_spikes=counts_in, buffers=buffers,
            on_value=float(T),
        )
        total = torch.matmul(seg["core_params"][i], signals[i].T).T
        bias = seg["hw_biases"][i]
        if bias is not None:
            total = total + bias * float(T)
        theta = seg["thresholds"][i]
        if v0:
            total = total + v0 * theta
        stair = lif_count_staircase(
            total / float(T), theta, T, compare_mode=flow.thresholding_mode,
        )
        buffers[i] = stair * float(T) / torch.clamp(
            torch.as_tensor(theta, dtype=stair.dtype, device=stair.device),
            min=1e-12,
        )

    output_counts = torch.zeros(
        batch_size, len(seg["output_sources"]), device=device,
        dtype=COMPUTE_DTYPE,
    )
    for sp in seg["output_spans"]:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            output_counts[:, d0:d1] = float(T)
        elif sp.kind == "input":
            output_counts[:, d0:d1] = counts_in[:, int(sp.src_start):int(sp.src_end)]
        else:
            output_counts[:, d0:d1] = buffers[int(sp.src_core)][
                :, int(sp.src_start):int(sp.src_end)
            ]
    return output_counts
