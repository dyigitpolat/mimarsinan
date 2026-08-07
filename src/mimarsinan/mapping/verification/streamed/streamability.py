"""Streamed-lif structural contract: host ops only as encode prefix / readout suffix."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.verification.onchip_fraction import (
    _build_flow,
    _exec_nodes,
    _host_unit,
    _onchip_unit,
)


class NotStreamableError(ValueError):
    """The architecture cannot deploy as an end-to-end streamed program."""


def _interior_host_ops(kinds: Sequence[Tuple[str, str]]) -> List[str]:
    """Host entries strictly inside the neural span. ``kinds`` is the ordered
    ``(name, "neural"|"host")`` classification (structural nodes excluded)."""
    neural_positions = [i for i, (_, kind) in enumerate(kinds) if kind == "neural"]
    if not neural_positions:
        raise NotStreamableError(
            "streamed lif requires an on-chip neural span, but the model maps "
            "no neural cores at all (everything runs host-side)."
        )
    first, last = neural_positions[0], neural_positions[-1]
    return [
        name for i, (name, kind) in enumerate(kinds)
        if kind == "host" and first < i < last
    ]


def _raise_not_streamable(offenders: List[str]) -> None:
    names = ", ".join(repr(n) for n in offenders)
    raise NotStreamableError(
        f"streamed lif requires ONE contiguous on-chip neural span — host "
        f"compute ops may only form an encode prefix and a readout suffix, "
        f"but {names} run(s) INTERIOR to the span. Remedies: re-architect the "
        f"model spiking-natively (e.g. strided conv instead of pooling), or "
        f"deploy the windowed semantics (spiking_variant='synchronized')."
    )


def assert_streamable_ir(ir_graph) -> None:
    """The authoritative gate on the mapped IR: every interior node between
    the first and last NeuralCore must itself be a NeuralCore."""
    kinds = [
        (node.name, "neural" if isinstance(node, NeuralCore) else "host")
        for node in ir_graph.nodes
    ]
    offenders = _interior_host_ops(kinds)
    if offenders:
        _raise_not_streamable(offenders)


def assert_streamable_model_or_raise(
    model, input_shape, num_classes, *, encoding_placement: str = "subsume",
) -> None:
    """Static fail-fast twin on the model SPEC (before pretraining): classifies
    the mapper exec order with the SAME host/on-chip units the on-chip-majority
    estimate uses."""
    flow = _build_flow(model, input_shape, num_classes, encoding_placement)
    kinds: List[Tuple[str, str]] = []
    for node in _exec_nodes(flow) or ():
        if _host_unit(node) is not None:
            kinds.append((getattr(node, "name", type(node).__name__), "host"))
        elif _onchip_unit(node) is not None:
            kinds.append((getattr(node, "name", type(node).__name__), "neural"))
    offenders = _interior_host_ops(kinds)
    if offenders:
        _raise_not_streamable(offenders)
