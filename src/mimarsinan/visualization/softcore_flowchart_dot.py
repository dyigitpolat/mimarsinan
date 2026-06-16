from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mimarsinan.mapping.platform.coalescing import coalescing_fragment_count
from mimarsinan.mapping.platform.mapping_structure import (
    compute_core_input_count,
    compute_fc_tiling_mode,
)
import operator

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    Conv1DPerceptronMapper,
    Conv2DPerceptronMapper,
    Mapper,
    ModelRepresentation,
    PerceptronMapper,
    StackMapper,
)


from mimarsinan.visualization.softcore_flowchart_estimate import HWEstimate, estimate_fc_cores

def _node_display_name(node: Any) -> str:
    if hasattr(node, "name"):
        try:
            return str(getattr(node, "name"))
        except Exception:
            pass
    if isinstance(node, PerceptronMapper):
        return f"Perceptron:{getattr(node.perceptron, 'name', '<unnamed>')}"
    return type(node).__name__


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")


def generate_softcore_flowchart_dot(
    mapper_repr: ModelRepresentation,
    *,
    input_shape: tuple[int, ...],
    max_axons: int,
    max_neurons: int,
    device: torch.device | None = None,
    hardware_bias: bool = False,
    allow_coalescing: bool = True,
) -> str:
    """
    Generate a Graphviz DOT flowchart for the mapper graph, annotated with estimated SoftCoreMapping cost.

    This does NOT require successful mapping (pooling ops can exist); it uses forward shape tracing + estimation.
    """

    if device is None:
        device = torch.device("cpu")

    # Trace forward shapes through the mapper graph (single batch)
    mapper_repr._ensure_exec_graph()  # noqa: SLF001
    exec_order: list[Mapper] = list(mapper_repr._exec_order)  # noqa: SLF001
    deps = mapper_repr._deps  # noqa: SLF001

    x0 = torch.randn(1, *input_shape, device=device)
    values: dict[Mapper, torch.Tensor] = {}
    for node in exec_order:
        d = deps.get(node, [])
        if len(d) == 0:
            values[node] = node.forward(x0)
        elif len(d) == 1:
            values[node] = node.forward(values[d[0]])
        else:
            values[node] = node.forward(tuple(values[dep] for dep in d))

    # Build DOT
    lines: list[str] = []
    lines.append("digraph SoftCoreFlowchart {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=record, fontname=\"Helvetica\", fontsize=10];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    node_id: dict[Mapper, str] = {}
    for idx, node in enumerate(exec_order):
        nid = f"n{idx}"
        node_id[node] = nid

        out_t = values.get(node)
        out_shape = None
        if out_t is not None and hasattr(out_t, "shape"):
            # drop batch dim if present
            if len(out_t.shape) >= 1:
                out_shape = tuple(out_t.shape[1:])

        hw = None
        hw_text = "HW: n/a"
        sw_text = "SW: n/a"

        _est_kw = dict(
            max_axons=int(max_axons),
            max_neurons=int(max_neurons),
            hardware_bias=hardware_bias,
            allow_coalescing=allow_coalescing,
        )

        # Estimate hardware usage for known mappable ops
        if isinstance(node, PerceptronMapper):
            p = node.perceptron
            in_f = int(p.layer.weight.shape[1])
            out_f = int(p.layer.weight.shape[0])
            inst = 1
            sw_text = f"SW perceptrons=1 (in_features={in_f}, out_features={out_f})"
            hw = _estimate_map_fc(
                in_features=in_f,
                out_features=out_f,
                instances=inst,
                has_bias=(p.layer.bias is not None),
                **_est_kw,
            )
        elif isinstance(node, Conv2DPerceptronMapper):
            k_h, k_w = node.kernel_size
            in_f = int(node.in_channels * k_h * k_w)
            out_f = int(node.out_channels)
            sw_text = f"SW perceptrons=1 (shared conv weights, patch={in_f}, out_channels={out_f})"
            if out_shape is not None and len(out_shape) == 3:
                _, h, w = out_shape
                inst = int(h * w)
            else:
                inst = 1
            hw = _estimate_map_fc(
                in_features=in_f,
                out_features=out_f,
                instances=inst,
                has_bias=bool(node.bias),
                **_est_kw,
            )
        elif isinstance(node, Conv1DPerceptronMapper):
            in_f = int(node.in_channels * node.kernel_size)
            out_f = int(node.out_channels)
            sw_text = f"SW perceptrons=1 (shared conv weights, patch={in_f}, out_channels={out_f})"
            if out_shape is not None and len(out_shape) == 2:
                _, l = out_shape
                inst = int(l)
            else:
                inst = 1
            hw = _estimate_map_fc(
                in_features=in_f,
                out_features=out_f,
                instances=inst,
                has_bias=bool(node.bias),
                **_est_kw,
            )
        elif (
            isinstance(node, ComputeOpMapper)
            and isinstance(getattr(node, "module", None), ComputeAdapter)
            and getattr(node.module, "fn", None) is operator.add
            and getattr(node.module, "_bound_count", 0) == 0
        ):
            # Add is mapped as a linear op (concat + identity weights). Estimate roughly.
            # For visualization purposes, treat it as 1 instance, axons=2*features, neurons=features.
            if out_shape is not None and len(out_shape) == 1:
                feat = int(out_shape[0])
                sw_text = f"SW perceptrons=0 (Add op)"
                hw = _estimate_map_fc(
                    in_features=2 * feat,
                    out_features=feat,
                    instances=1,
                    has_bias=False,
                    **_est_kw,
                )
        elif isinstance(node, StackMapper):
            sw_text = "SW stack (host-side)"

        if hw is not None:
            if hw.mappable:
                hw_text = f"HW cores≈{hw.cores_total}\n{hw.details}"
            else:
                hw_text = f"HW UNMAPPABLE: {hw.reason}\n{hw.details}"

        label = f"{_node_display_name(node)}\\n{sw_text}\\n{hw_text}"
        if out_shape is not None:
            label += f"\\nout_shape={out_shape}"

        lines.append(f'  {nid} [label="{_dot_escape(label)}"];')

    # Edges
    for node in exec_order:
        for dep in deps.get(node, []):
            lines.append(f"  {node_id[dep]} -> {node_id[node]};")

    lines.append("}")
    return "\n".join(lines)


def write_softcore_flowchart_dot(
    mapper_repr: ModelRepresentation,
    path: str,
    *,
    input_shape: tuple[int, ...],
    max_axons: int,
    max_neurons: int,
    device: torch.device | None = None,
    hardware_bias: bool = False,
    allow_coalescing: bool = True,
) -> None:
    dot = generate_softcore_flowchart_dot(
        mapper_repr,
        input_shape=input_shape,
        max_axons=max_axons,
        max_neurons=max_neurons,
        device=device,
        hardware_bias=hardware_bias,
        allow_coalescing=allow_coalescing,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(dot)
