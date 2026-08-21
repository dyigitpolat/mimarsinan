"""Packed-bucket construction for the stage-flat streaming executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

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
