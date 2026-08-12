"""Who actually holds the host-side parameters — measured, never guessed.

The on-chip floor refusals used to end in a hardcoded parenthetical ("offloaded
encoding Linear/Conv, classifier readout, attention"), which named ops the model
may not have and used "offloaded" for host placement — the OPPOSITE of what
``encoding_layer_placement='offload'`` means. A refusal is only actionable if it
names THIS model's host units and their sizes, so both floor gates rank the real
contributors through here.

Vocabulary (config_schema/registry/entries_conversion.py, the SSOT):
``subsume`` runs the encoding layer HOST-side; ``offload`` maps it ON CHIP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.support.scale_wrapper import unwrap_scale_wrapper

_MAX_NAMED = 3


def _numel(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _shape_suffix(module: nn.Module) -> str:
    inner = unwrap_scale_wrapper(module)
    layer = getattr(inner, "layer", inner)
    if isinstance(layer, nn.Linear):
        return f" {layer.in_features}->{layer.out_features}"
    if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
        return f" {layer.in_channels}->{layer.out_channels}"
    return ""


def _label(module: nn.Module, *, is_encoder: bool) -> str:
    inner = unwrap_scale_wrapper(module)
    layer = getattr(inner, "layer", None)
    kind = type(layer).__name__ if layer is not None else type(inner).__name__
    role = "subsumed encoding layer" if is_encoder else "host op"
    return f"{role} {kind}{_shape_suffix(module)}"


def _ranked(units: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted(units, key=lambda pair: (-pair[1], pair[0]))


def host_contributors_from_flow(flow, host_unit_of) -> list[tuple[str, int]]:
    """``(label, params)`` per unique host unit of a mapper flow, largest first.

    ``host_unit_of`` is the caller's host-node decomposition, so this ranking is
    over exactly the units its host total summed.
    """
    mapper_repr = flow.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    seen: set[int] = set()
    units: list[tuple[str, int]] = []
    for node in mapper_repr._exec_order or []:
        unit = host_unit_of(node)
        if unit is None or id(unit) in seen:
            continue
        seen.add(id(unit))
        units.append(
            (_label(unit, is_encoder=bool(getattr(unit, "is_encoding_layer", False))),
             _numel(unit))
        )
    return _ranked(units)


def host_contributors_from_ir(ir_graph) -> list[tuple[str, int]]:
    """``(label, params)`` per unique host ComputeOp of a mapped IR graph.

    The IR twin of :func:`host_contributors_from_flow`; mirrors
    ``count_host_params``' deduping so the parts sum to the reported host total.
    A subsumed encoder reaches the IR as the Perceptron itself, so the encoder
    role is read off ``is_encoding_layer`` — the same field the flow side reads.
    """
    seen: set[int] = set()
    units: list[tuple[str, int]] = []
    for op in ir_graph.get_compute_ops():
        module = op.params.get("module")
        if module is None or not hasattr(module, "parameters"):
            bound = op.params.get("bound_tensors") or []
            count = int(sum(int(t.numel()) for t in bound if torch.is_tensor(t)))
            if count:
                units.append((f"host op {op.op_type} constants", count))
            continue
        unit = getattr(module, "perceptron", None) or module
        if id(unit) in seen:
            continue
        seen.add(id(unit))
        encoder = bool(getattr(unit, "is_encoding_layer", False))
        units.append((_label(unit, is_encoder=encoder), _numel(module)))
    return _ranked(units)


def subsumed_encoder_params(contributors) -> int:
    """Params held by the host units that are subsumed encoding layers."""
    return sum(count for label, count in contributors if label.startswith("subsumed"))


def describe_host_contributors(contributors, *, limit: int = _MAX_NAMED) -> str:
    """Human phrase naming the biggest host holders, e.g. ``"A (200960 params)"``."""
    if not contributors:
        return "no host-side unit holds parameters at all"
    named = contributors[:limit]
    parts = [f"{label} ({count} params)" for label, count in named]
    rest = len(contributors) - len(named)
    if rest > 0:
        parts.append(f"and {rest} smaller host unit{'s' if rest > 1 else ''}")
    return ", ".join(parts)


def describe_host_holders(contributors, *, limit: int = _MAX_NAMED) -> str:
    """The sentence a floor refusal uses to attribute its host parameters."""
    if not contributors:
        return "No host-side unit holds parameters at all"
    return f"The host side holds the majority: {describe_host_contributors(contributors, limit=limit)}"
