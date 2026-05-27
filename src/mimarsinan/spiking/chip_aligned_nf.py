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


def chip_aligned_nf_forward(
    model: nn.Module,
    x: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Drive ``model`` like the deployment chip does at structural boundaries.

    Encoding-layer perceptrons run once in rate mode; their outputs are
    uniform-encoded to ``(T, B, …)`` binary trains and substituted into the
    cycle-accurate pass for the rest of the graph. Falls back to
    :func:`run_cycle_accurate` when no encoding-layer perceptron is present.
    """
    from spikingjelly.activation_based import functional

    if not hasattr(model, "get_mapper_repr"):
        return run_cycle_accurate(model, x, T)
    mapper_repr = model.get_mapper_repr()
    if mapper_repr is None:
        return run_cycle_accurate(model, x, T)
    mapper_repr._ensure_exec_graph()

    exec_order = mapper_repr._exec_order
    deps_map = mapper_repr._deps

    encoding_indices: list[int] = []
    for i, node in enumerate(exec_order):
        perceptron = getattr(node, "perceptron", None)
        if perceptron is None:
            continue
        if getattr(perceptron, "is_encoding_layer", False):
            encoding_indices.append(i)

    if not encoding_indices:
        return run_cycle_accurate(model, x, T)

    encoding_perceptron_ids: set[int] = set()
    for i in encoding_indices:
        encoding_perceptron_ids.add(id(exec_order[i].perceptron))

    all_lifs: list[LIFActivation] = []
    non_encoding_lifs: list[LIFActivation] = []
    for p in mapper_repr.get_perceptrons():
        lif = _resolve_lif_for_perceptron(p)
        if lif is None:
            continue
        all_lifs.append(lif)
        if id(p) not in encoding_perceptron_ids:
            non_encoding_lifs.append(lif)

    _set_cycle_accurate(all_lifs, False)
    functional.reset_net(model)

    last_encoding_idx = max(encoding_indices)
    pre_nodes = exec_order[: last_encoding_idx + 1]
    rest_nodes = exec_order[last_encoding_idx + 1 :]

    def _run_node(node, values_dict, original_input):
        d = deps_map.get(node, [])
        if len(d) == 0:
            return node.forward(original_input)
        if len(d) == 1:
            return node.forward(values_dict[d[0]])
        return node.forward(tuple(values_dict[dep] for dep in d))

    values: dict = {}
    for node in pre_nodes:
        values[node] = _run_node(node, values, x)

    encoding_spike_trains: dict = {}
    for i in encoding_indices:
        node = exec_order[i]
        lif = _resolve_lif_for_perceptron(node.perceptron)
        scale = lif.activation_scale
        if isinstance(scale, torch.Tensor):
            safe_scale = scale.to(device=x.device, dtype=values[node].dtype).clamp(min=1e-12)
        else:
            safe_scale = max(float(scale), 1e-12)
        rate_norm = (values[node] / safe_scale).clamp(0.0, 1.0)
        encoding_spike_trains[node] = (uniform_spike_train(rate_norm, T), safe_scale)

    _set_cycle_accurate(non_encoding_lifs, True)
    functional.reset_net(model)
    try:
        per_cycle_outputs = []
        for t in range(T):
            cycle_values = dict(values)
            for node, (train, scale) in encoding_spike_trains.items():
                cycle_values[node] = train[t] * scale
            for node in rest_nodes:
                cycle_values[node] = _run_node(node, cycle_values, x)
            per_cycle_outputs.append(cycle_values[mapper_repr.output_layer_mapper])
        return torch.stack(per_cycle_outputs, dim=0).mean(dim=0)
    finally:
        _set_cycle_accurate(all_lifs, False)
