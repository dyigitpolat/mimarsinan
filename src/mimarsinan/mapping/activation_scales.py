"""IR-graph activation scale tables for TTFS / ComputeOp rescaling."""

from __future__ import annotations

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore


def perceptron_wrapped_activation_scale(module) -> float | None:
    """activation_scale on a Perceptron / PerceptronMapper held by a ComputeOp, or None."""
    if module is None:
        return None
    s = getattr(module, "activation_scale", None)
    if s is None:
        perceptron = getattr(module, "perceptron", None)
        if perceptron is not None:
            s = getattr(perceptron, "activation_scale", None)
    if s is None:
        return None
    try:
        return float(s.item() if hasattr(s, "item") else s)
    except (TypeError, ValueError):
        return None


def compute_node_output_scales(ir_graph: IRGraph) -> dict[int, float]:
    """Per-node output activation_scale for state-buffer / TTFS normalization."""
    scales: dict[int, float] = {}
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            s = node.activation_scale
            scales[node.id] = float(s.item() if hasattr(s, "item") else s)
        elif isinstance(node, ComputeOp):
            module = (node.params or {}).get("module")
            wrapped_scale = perceptron_wrapped_activation_scale(module)
            if wrapped_scale is not None:
                scales[node.id] = wrapped_scale
                continue
            src_scales: list[float] = []
            for src in node.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    src_scales.append(scales.get(src.node_id, 1.0))
            scales[node.id] = (
                sum(src_scales) / len(src_scales) if src_scales else 1.0
            )
    return scales


def compute_node_input_scales(ir_graph: IRGraph) -> dict[int, float]:
    """Input-rescale factors for ComputeOps (encoding path uses 1.0)."""
    out_scales = compute_node_output_scales(ir_graph)
    in_scales: dict[int, float] = {}
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            in_scales[node.id] = out_scales[node.id]
        elif isinstance(node, ComputeOp):
            src_scales: list[float] = []
            all_raw = True
            for src in node.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    all_raw = False
                    src_scales.append(out_scales.get(src.node_id, 1.0))
            if all_raw:
                in_scales[node.id] = 1.0
            else:
                in_scales[node.id] = (
                    sum(src_scales) / len(src_scales) if src_scales else 1.0
                )
    return in_scales
