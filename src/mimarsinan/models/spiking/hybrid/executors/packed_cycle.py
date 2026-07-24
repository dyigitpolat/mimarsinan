"""[cert-plan W1] stage-flat streaming executor: cycles loop, cores batched.

Same per-cycle physics as the per-core reference loop (the policy's
``advance`` is the shared kernel); only the charge layout changes. Cores are
bucketed by (latency, axons, neurons) and laid out latency-sorted, so each
cycle's active set is a run of whole buckets and latency gating needs no
masking. Window-gated output accumulation equals each producer's total
in-window fires, so outputs assemble once at the end. Bit-equal to the
reference loop (locked by test_packed_cycle_equivalence)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from mimarsinan.models.spiking.hybrid.membrane_readout import (
    stash_membrane_readout_correction,
)
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


@dataclass
class _Bucket:
    latency: int
    n_axons: int
    n_neurons: int
    core_indices: List[int]
    neuron_start: int
    neuron_end: int
    weights: torch.Tensor
    bias: "torch.Tensor | None"
    on_dst: "torch.Tensor | None"
    inp_dst: "torch.Tensor | None"
    inp_src: "torch.Tensor | None"
    buf_dst: "torch.Tensor | None"
    buf_src: "torch.Tensor | None"


@dataclass
class PackedStage:
    total_neurons: int
    buckets: List[_Bucket]
    theta_flat: torch.Tensor
    neuron_offset: Dict[int, int] = field(default_factory=dict)


def _index(values, device) -> "torch.Tensor | None":
    if not values:
        return None
    return torch.tensor(values, dtype=torch.long, device=device)


def build_packed_stage(seg: dict, device: torch.device) -> PackedStage:
    """Bucket the stage's stepable cores by (latency, axons, neurons)."""
    cores = seg["cores"]
    axon_spans = seg["axon_spans"]
    core_params = seg["core_params"]
    thresholds = seg["thresholds"]
    hw_biases = seg["hw_biases"]

    def _n_out(c) -> int:
        return max(int(c.neurons_per_core - c.available_neurons), 1)

    def _n_ax(c) -> int:
        return max(int(c.axons_per_core - c.available_axons), 1)

    keyed: Dict[tuple, List[int]] = {}
    for i, c in enumerate(cores):
        if c.latency is not None:
            keyed.setdefault((int(c.latency), _n_ax(c), _n_out(c)), []).append(i)

    drafts: List[dict] = []
    neuron_offset: Dict[int, int] = {}
    theta_parts: List[torch.Tensor] = []
    cursor = 0
    for (lat, n_ax, n_out), members in sorted(keyed.items()):
        start = cursor
        weights = torch.stack([core_params[i] for i in members])
        bias = None
        if any(hw_biases[i] is not None for i in members):
            bias = torch.stack([
                hw_biases[i] if hw_biases[i] is not None
                else torch.zeros(n_out, device=device, dtype=COMPUTE_DTYPE)
                for i in members
            ])
        theta_parts.append(torch.stack([
            torch.as_tensor(thresholds[i], dtype=COMPUTE_DTYPE, device=device)
            .expand(n_out).clone()
            for i in members
        ]).reshape(-1))
        on_dst: List[int] = []
        inp_dst: List[int] = []
        inp_src: List[int] = []
        buf_dst: List[int] = []
        buf_spans: List[tuple] = []
        for g, i in enumerate(members):
            neuron_offset[i] = start + g * n_out
            base = g * n_ax
            for sp in axon_spans[i]:
                if sp.kind == "off":
                    continue
                d0, d1 = int(sp.dst_start), int(sp.dst_end)
                if sp.kind == "on":
                    on_dst.extend(range(base + d0, base + d1))
                    continue
                if sp.kind == "input":
                    inp_dst.extend(range(base + d0, base + d1))
                    inp_src.extend(range(int(sp.src_start), int(sp.src_end)))
                    continue
                src_core = int(sp.src_core)
                if src_core < 0:
                    raise ValueError(
                        f"core-kind axon span with negative src_core "
                        f"{src_core} (mis-flagged input source?)"
                    )
                if cores[src_core].latency is None:
                    continue  # never stepped: reference reads zeros forever
                buf_dst.extend(range(base + d0, base + d1))
                buf_spans.append((src_core, int(sp.src_start), int(sp.src_end)))
        drafts.append(dict(
            lat=lat, n_ax=n_ax, n_out=n_out, members=members, start=start,
            weights=weights, bias=bias, on_dst=on_dst, inp_dst=inp_dst,
            inp_src=inp_src, buf_dst=buf_dst, buf_spans=buf_spans,
        ))
        cursor += len(members) * n_out

    buckets: List[_Bucket] = []
    for d in drafts:
        buf_src = [
            neuron_offset[src_core] + i
            for src_core, s0, s1 in d["buf_spans"]
            for i in range(s0, s1)
        ]
        buckets.append(_Bucket(
            latency=d["lat"], n_axons=d["n_ax"], n_neurons=d["n_out"],
            core_indices=d["members"], neuron_start=d["start"],
            neuron_end=d["start"] + len(d["members"]) * d["n_out"],
            weights=d["weights"], bias=d["bias"],
            on_dst=_index(d["on_dst"], device),
            inp_dst=_index(d["inp_dst"], device),
            inp_src=_index(d["inp_src"], device),
            buf_dst=_index(d["buf_dst"], device),
            buf_src=_index(buf_src, device),
        ))

    theta_flat = (
        torch.cat(theta_parts) if theta_parts
        else torch.zeros(0, device=device, dtype=COMPUTE_DTYPE)
    )
    return PackedStage(
        total_neurons=cursor, buckets=buckets, theta_flat=theta_flat,
        neuron_offset=neuron_offset,
    )


def run_neural_segment_packed(
    flow, input_spike_train, *, seg, stage, T, batch_size, device, policy,
    readout_corrections=None,
) -> torch.Tensor:
    """Stage-flat twin of the reference cycle loop (multi-spike, non-recording)."""
    packed = seg.get("packed")
    if packed is None:
        packed = build_packed_stage(seg, device)
        seg["packed"] = packed

    latency = seg["latency"]
    cycles = int(latency) + T
    output_sources = seg["output_sources"]
    output_spans = seg["output_spans"]

    state = policy.make_state(batch_size, max(packed.total_neurons, 1),
                              device, COMPUTE_DTYPE)
    membrane_init = float(getattr(flow, "lif_membrane_init", 0.0))
    if membrane_init and packed.total_neurons:
        state["memb"] += membrane_init * packed.theta_flat

    fires = torch.zeros(batch_size, max(packed.total_neurons, 1),
                        device=device, dtype=COMPUTE_DTYPE)
    counts = torch.zeros_like(fires)
    zeros_in = torch.zeros(batch_size, input_spike_train.shape[2],
                           device=device, dtype=COMPUTE_DTYPE)
    train = input_spike_train.to(COMPUTE_DTYPE)

    for cycle in range(cycles):
        # Two-phase, mirroring the reference loop: gather every active
        # bucket's signals against the PREVIOUS cycle's fires, then advance.
        staged = []
        for bucket in packed.buckets:
            if not (bucket.latency <= cycle < bucket.latency + T):
                continue
            group = len(bucket.core_indices)
            signals = torch.zeros(
                batch_size, group * bucket.n_axons,
                device=device, dtype=COMPUTE_DTYPE,
            )
            if bucket.on_dst is not None:
                signals.index_fill_(1, bucket.on_dst, 1.0)
            if bucket.inp_dst is not None and bucket.inp_src is not None:
                local = cycle - bucket.latency
                src = train[local] if 0 <= local < T else zeros_in
                signals.index_copy_(
                    1, bucket.inp_dst, src.index_select(1, bucket.inp_src))
            if bucket.buf_dst is not None and bucket.buf_src is not None:
                signals.index_copy_(
                    1, bucket.buf_dst, fires.index_select(1, bucket.buf_src))
            staged.append((bucket, signals))
        for bucket, signals in staged:
            group = len(bucket.core_indices)
            grouped = signals.reshape(batch_size, group, bucket.n_axons)
            charge = torch.einsum("gna,bga->bgn", bucket.weights, grouped)
            if bucket.bias is not None:
                charge = charge + bucket.bias
            contribution = charge.reshape(batch_size, -1)
            n0, n1 = bucket.neuron_start, bucket.neuron_end
            slice_state = {k: v[:, n0:n1] for k, v in state.items()}
            out = policy.advance(
                slice_state, contribution, packed.theta_flat[n0:n1],
                thresholding_mode=flow.thresholding_mode,
                output_dtype=COMPUTE_DTYPE,
            )
            fires[:, n0:n1] = out
            counts[:, n0:n1] += out

    output_counts = torch.zeros(
        batch_size, len(output_sources), device=device, dtype=COMPUTE_DTYPE)
    input_total = train[:T].sum(dim=0)
    for sp in output_spans:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            output_counts[:, d0:d1] = float(T)
            continue
        if sp.kind == "input":
            output_counts[:, d0:d1] = input_total[:, int(sp.src_start):int(sp.src_end)]
            continue
        src_core = int(sp.src_core)
        off = packed.neuron_offset.get(src_core)
        if off is None:
            continue
        output_counts[:, d0:d1] = counts[:, off + int(sp.src_start):off + int(sp.src_end)]

    if readout_corrections is not None and getattr(flow, "membrane_readout", False):
        # [C2] per-core membrane views over the flat state feed the same stash.
        cores = seg["cores"]
        per_core_states = []
        for i, c in enumerate(cores):
            n = max(int(c.neurons_per_core - c.available_neurons), 1)
            off = packed.neuron_offset.get(i)
            memb = (
                state["memb"][:, off:off + n] if off is not None
                else torch.zeros(batch_size, n, device=device, dtype=COMPUTE_DTYPE)
            )
            per_core_states.append({"memb": memb})
        stash_membrane_readout_correction(
            flow, seg=seg, stage=stage, output_counts=output_counts,
            output_spans=output_spans, neuron_states=per_core_states,
            thresholds=seg["thresholds"], single_spike=False,
            readout_corrections=readout_corrections,
        )
    return output_counts
