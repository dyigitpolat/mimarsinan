"""IR-graph activation scale tables for TTFS / ComputeOp rescaling."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore

NodeScale = float | np.ndarray


def _coerce_node_scale(raw) -> NodeScale | None:
    """Normalize a raw activation_scale to a scalar float or a per-channel vector.

    A scalar / 1-element scale collapses to ``float`` (byte-identical scalar path);
    a multi-element 1-D scale surfaces as a contiguous ``float64`` vector.
    """
    if raw is None:
        return None
    if hasattr(raw, "detach"):
        raw = raw.detach()
    arr = np.asarray(
        raw.cpu().numpy() if hasattr(raw, "cpu") else raw, dtype=np.float64,
    )
    if arr.ndim == 0 or arr.size == 1:
        try:
            return float(arr.reshape(()))
        except (TypeError, ValueError):
            return None
    return np.ascontiguousarray(arr.reshape(-1))


def perceptron_wrapped_activation_scale(module) -> NodeScale | None:
    """activation_scale on a Perceptron / PerceptronMapper held by a ComputeOp, or None.

    Returns a scalar ``float`` (scalar/1-element scale) or a per-channel vector
    (``ttfs_theta_cotrain``); ``None`` when no scale is held.
    """
    if module is None:
        return None
    s = getattr(module, "activation_scale", None)
    if s is None:
        perceptron = getattr(module, "perceptron", None)
        if perceptron is not None:
            s = getattr(perceptron, "activation_scale", None)
    return _coerce_node_scale(s)


def _scalar_node_scale(scale: NodeScale) -> float:
    """Collapse a scalar-or-per-channel scale to a single scalar (mean for a vector)."""
    return float(np.mean(scale)) if isinstance(scale, np.ndarray) else float(scale)


def _aggregate_source_scales(src_scales: list[NodeScale]) -> float:
    """Combine ComputeOp input-source scales into one scalar rescale factor.

    Per-channel source scales reduce to their own mean first; the all-scalar
    path stays byte-identical to the legacy ``sum / len`` mean.
    """
    if not src_scales:
        return 1.0
    reduced = [_scalar_node_scale(s) for s in src_scales]
    return sum(reduced) / len(reduced)


def compute_node_output_scales(ir_graph: IRGraph) -> dict[int, NodeScale]:
    """Per-node output activation_scale for state-buffer / TTFS normalization.

    Values are scalar ``float`` for scalar/1-element scales and per-channel
    ``np.ndarray`` (len == out features) for ``ttfs_theta_cotrain`` nodes.
    """
    scales: dict[int, NodeScale] = {}
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            coerced = _coerce_node_scale(node.activation_scale)
            scales[node.id] = 1.0 if coerced is None else coerced
        elif isinstance(node, ComputeOp):
            module = (node.params or {}).get("module")
            wrapped_scale = perceptron_wrapped_activation_scale(module)
            if wrapped_scale is not None:
                scales[node.id] = _scalar_node_scale(wrapped_scale)
                continue
            src_scales: list[NodeScale] = []
            for src in node.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    src_scales.append(scales.get(src.node_id, 1.0))
            scales[node.id] = _aggregate_source_scales(src_scales)
    return scales


def compute_node_input_scales(ir_graph: IRGraph) -> dict[int, NodeScale]:
    """Input-rescale factors for ComputeOps (encoding path uses 1.0)."""
    out_scales = compute_node_output_scales(ir_graph)
    in_scales: dict[int, NodeScale] = {}
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            in_scales[node.id] = out_scales[node.id]
        elif isinstance(node, ComputeOp):
            src_scales: list[NodeScale] = []
            all_raw = True
            for src in node.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    all_raw = False
                    src_scales.append(out_scales.get(src.node_id, 1.0))
            if all_raw:
                in_scales[node.id] = 1.0
            else:
                in_scales[node.id] = _aggregate_source_scales(src_scales)
    return in_scales
