"""Chip-aligned NF forward — see ``src/mimarsinan/spiking/ARCHITECTURE.md``."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LIFActivation, run_cycle_accurate
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.spike_trains import uniform_spike_train


def _resolve_lif_for_perceptron(perceptron) -> Optional[LIFActivation]:
    return unwrap_lif_activation(getattr(perceptron, "activation", None))


def _set_cycle_accurate(lifs, mode: bool) -> None:
    for lif in lifs:
        if lif is not None:
            lif.set_cycle_accurate(mode)


def _safe_scale(scale, ref: torch.Tensor):
    if isinstance(scale, torch.Tensor):
        return scale.to(device=ref.device, dtype=ref.dtype).clamp(min=1e-12)
    return max(float(scale), 1e-12)


def chip_aligned_segment_forward(
    model: nn.Module, x: torch.Tensor, T: int,
    *, compute_min_recorder: dict | None = None,
) -> torch.Tensor:
    """Segment-aware chip-aligned NF forward (matches HCM ``_forward_rate``).

    Walks the mapper exec graph, keeping two representations per node: a per-cycle
    spike ``train`` (intra-segment perceptron cascade, like the chip core->core wire)
    and a ``rate`` (``count/T`` ∈ [0,1], the inter-segment value). Perceptrons run
    cycle-accurate (signed-IF) feeding off upstream trains; **ComputeOps run once on
    the decoded rate** (not per-cycle on spikes) and downstream perceptrons re-encode
    — exactly the decode->compute->re-encode HCM performs at each ComputeOp boundary.
    Encoding-layer perceptrons stay subsumed (rate mode + uniform encode).

    A ComputeOp node carrying a ``_negative_shift`` (per output channel) has it added
    to its rate before the [0,1] re-encode clamp, so a negative-producing boundary is
    lossless — the same shift HCM applies via ``node_output_shifts`` (the consuming
    perceptron's bias is pre-corrected by ``apply_negative_shift_bias``).

    ``compute_min_recorder``: when given, accumulates each ComputeOp node's per-channel
    output minimum (calibration for deriving the shift).
    """
    from spikingjelly.activation_based import functional
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    if not hasattr(model, "get_mapper_repr"):
        return run_cycle_accurate(model, x, T)
    mapper_repr = model.get_mapper_repr()
    if mapper_repr is None:
        return run_cycle_accurate(model, x, T)
    mapper_repr._ensure_exec_graph()
    exec_order = mapper_repr._exec_order
    deps_map = mapper_repr._deps

    all_lifs = [
        lif for p in mapper_repr.get_perceptrons()
        if (lif := _resolve_lif_for_perceptron(p)) is not None
    ]
    _set_cycle_accurate(all_lifs, False)
    functional.reset_net(model)

    node_train: dict = {}   # node -> (T, B, ...) per-cycle value (real, = spike*scale)
    node_rate: dict = {}    # node -> rate (real); ComputeOp inter-stage value

    def _is_perceptron(node) -> bool:
        return getattr(node, "perceptron", None) is not None

    def _forward(node, inputs: list):
        d = deps_map.get(node, [])
        if len(d) == 0:
            return node.forward(x)
        if len(d) == 1:
            return node.forward(inputs[0])
        return node.forward(tuple(inputs))

    def _train_of(dep):
        """Per-cycle train for ``dep``; encode (uniform, clamped) if only a rate exists."""
        t = node_train.get(dep)
        if t is not None:
            return t
        t = uniform_spike_train(node_rate[dep].clamp(0.0, 1.0), T)
        node_train[dep] = t
        return t

    try:
        for node in exec_order:
            d = deps_map.get(node, [])
            if isinstance(node, ComputeOpMapper):
                rate = _forward(node, [node_rate[dep] for dep in d])
                if compute_min_recorder is not None:
                    cur = rate.detach().amin(dim=0)
                    prev = compute_min_recorder.get(node)
                    compute_min_recorder[node] = (
                        cur if prev is None else torch.minimum(prev, cur)
                    )
                # Round-2a positive-domain shift: added to the ComputeOp's rate so it
                # propagates through downstream structural nodes to the consumer's
                # re-encode clamp; the consumer perceptron's baked bias compensates.
                shift = getattr(node, "_negative_shift", None)
                if shift is not None:
                    rate = rate + torch.as_tensor(shift, dtype=rate.dtype, device=rate.device)
                node_rate[node] = rate
                # boundary: leave node_train[node] unset (consumers re-encode the rate)
            elif _is_perceptron(node):
                p = node.perceptron
                lif = _resolve_lif_for_perceptron(p)
                scale = _safe_scale(getattr(lif, "activation_scale", 1.0), x)
                if getattr(p, "is_encoding_layer", False):
                    _set_cycle_accurate([lif], False)
                    rate_out = _forward(node, [node_rate[dep] for dep in d])
                    rate_norm = (rate_out / scale).clamp(0.0, 1.0)
                    node_rate[node] = rate_norm
                    node_train[node] = uniform_spike_train(rate_norm, T) * scale
                else:
                    _set_cycle_accurate([lif], True)
                    functional.reset_net(lif.if_node)
                    dep_trains = [_train_of(dep) for dep in d]
                    outs = [_forward(node, [dt[t] for dt in dep_trains]) for t in range(T)]
                    _set_cycle_accurate([lif], False)
                    train = torch.stack(outs, dim=0)
                    node_train[node] = train
                    node_rate[node] = (train / scale).mean(dim=0)
            else:
                # Structural (input / reshape / permute / concat): transparent. Carry a
                # train when every dep has one (per-cycle), and always carry a rate.
                if d and all(node_train.get(dep) is not None for dep in d):
                    dep_trains = [node_train[dep] for dep in d]
                    node_train[node] = torch.stack(
                        [_forward(node, [dt[t] for dt in dep_trains]) for t in range(T)],
                        dim=0,
                    )
                node_rate[node] = _forward(node, [node_rate[dep] for dep in d])
    finally:
        _set_cycle_accurate(all_lifs, False)

    out_node = mapper_repr.output_layer_mapper
    train = node_train.get(out_node)
    return train.mean(dim=0) if train is not None else node_rate[out_node]
